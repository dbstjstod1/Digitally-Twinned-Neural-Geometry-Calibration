"""Accelerated differentiable forward projectors for AI-Geocal (Triton kernels).

Two drop-in replacements for `differentiable_forward_projector.sinoproj_rdsh_pinv_raycast_dominant`
(same call signature, same output (B, nv, nu), gradients w.r.t. `Pmat` and `geo_parameter`):

  sinoproj_raymarch_triton   SAME MODEL as the original: `n_samples` uniform samples per ray
                             across the ray/box segment, trilinear (align_corners=False, zero
                             padding), integral = step * sum. The per-ray geometry (pinhole ray
                             from P, slab intersection, step) stays in torch, exactly as before;
                             only the sampling loop is a fused Triton kernel with a hand-written
                             backward w.r.t. the per-ray (origin, direction, tmin, step). Nothing
                             of size (rays x samples) is ever materialised, so memory is O(rays)
                             instead of O(rays x samples) and the autograd graph no longer grows
                             with `n_samples` or the batch.

  sinoproj_joseph            LEAP's modular-beam JOSEPH line integral (one bilinear sample per
                             dominant-axis voxel plane, entry plane at half weight, length
                             dx*|r|/|r_a|) with the EXACT gradient of that kernel w.r.t. the 12
                             LEAP-form geometry numbers per view (source, module centre, row
                             vector, column vector), chained through a differentiable
                             P -> modular-beam decomposition written for THIS repo's detector
                             conventions (INTERNAL y/z swap, ureverse/vreverse, ROI/recon_type,
                             extra_u, stitch, principal point). The kernel is the transcription
                             of `projectors_Joseph.cu` (LLNL LEAP, MIT) from the author's
                             Flow_matching_motion_3D project (`fm3d/triton_leap_grad.py`,
                             2026-07-29..08-05), copied here so this repo has no dependency on
                             that project or on the LEAP library. `value_backend="leap"` lets a
                             gate/benchmark take the VALUE from the real libleapct instead
                             (only meaningful with the Joseph-pinned build: stock LEAP silently
                             runs its separable-footprint kernel on an axially-aligned orbit,
                             which is a different model from the gradient computed here).

Both kernels gather from a volume stored TWICE ([z,y,x] and [z,x,y]) so that neighbouring
detector pixels read neighbouring addresses whichever in-plane axis the ray is dominant in
(the coalescing fix measured at 12x on the Joseph kernel). The stacked copy is built once per
volume tensor and cached (2x volume VRAM).

Requirements: torch >= 2.1 with a matching triton (the `dudodp` torch 1.13 env cannot run
these; use a torch 2.x env). Joseph additionally needs cubic voxels (dx == dy == dz) and
imsx == imsy, which both shipped configs (2T, 4T) satisfy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from differentiable_forward_projector import RT_PARAM, _build_jk_grid, _apply_roi_mapping


# ---------------------------------------------------------------------------------------
# Stacked-volume cache: [z,y,x] followed by [z,x,y], flat fp32. Keyed by the volume tensor's
# storage pointer + shape, so the reference volume is transposed once per process.
# ---------------------------------------------------------------------------------------
_STACK_CACHE: dict = {}


def stacked_volume(vol: torch.Tensor) -> torch.Tensor:
    """vol (D,H,W) fp32 cuda -> flat (2*D*H*W,) [natural ; last-two-axes transposed]."""
    vol = vol.detach()
    key = (vol.data_ptr(), tuple(vol.shape), vol.device.index)
    hit = _STACK_CACHE.get(key)
    if hit is not None:
        return hit
    v = vol.contiguous().to(torch.float32)
    st = torch.cat((v.reshape(-1), v.transpose(-2, -1).contiguous().reshape(-1)))
    _STACK_CACHE.clear()          # one reference volume at a time; keep VRAM bounded
    _STACK_CACHE[key] = st
    return st


# ---------------------------------------------------------------------------------------
# Kernel 1: fused uniform-sample trilinear ray march (the original model)
# ---------------------------------------------------------------------------------------
if HAVE_TRITON:

    @triton.jit
    def _raymarch_kernel(vol_ptr, o_ptr, d_ptr, tmin_ptr, step_ptr, hit_ptr,
                         gout_ptr, out_ptr, grad_ptr,
                         R, S, W, H, D, HW,
                         dx, dy, dz, X0, Y0, Z0,
                         BLOCK: tl.constexpr, MODE: tl.constexpr):
        """One lane = one ray. Sample k sits at t_k = tmin + step*(k+0.5), p = o + d*t_k,
        voxel-index coordinate ix = (p.x - X0)/dx - 0.5 (== grid_sample align_corners=False),
        trilinear with zero padding.  MODE 0: out[r] = step * sum_k V(p_k).
        MODE 1: grad[r, 0:8] = d<gout, out>/d(o, d, tmin, step) for this ray."""
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = r < R
        ox = tl.load(o_ptr + r * 3 + 0, mask=m, other=0.0)
        oy = tl.load(o_ptr + r * 3 + 1, mask=m, other=0.0)
        oz = tl.load(o_ptr + r * 3 + 2, mask=m, other=0.0)
        ddx = tl.load(d_ptr + r * 3 + 0, mask=m, other=0.0)
        ddy = tl.load(d_ptr + r * 3 + 1, mask=m, other=0.0)
        ddz = tl.load(d_ptr + r * 3 + 2, mask=m, other=0.0)
        tmin = tl.load(tmin_ptr + r, mask=m, other=0.0)
        step = tl.load(step_ptr + r, mask=m, other=0.0)
        hit = tl.load(hit_ptr + r, mask=m, other=0) != 0
        live = m & hit
        gout = tl.zeros((BLOCK,), dtype=tl.float32)
        if MODE == 1:
            gout = tl.load(gout_ptr + r, mask=live, other=0.0)

        # coalescing: x-dominant rays read the [z,x,y] copy (y contiguous), others [z,y,x]
        xdom = tl.abs(ddx) > tl.abs(ddy)
        voff = tl.where(xdom, D * HW, 0).to(tl.int64)
        sa = tl.where(xdom, H, W).to(tl.int64)       # stride of the 'a' axis

        Ssum = tl.zeros((BLOCK,), dtype=tl.float32)
        gx = tl.zeros((BLOCK,), dtype=tl.float32)     # sum_k dV/dp
        gy = tl.zeros((BLOCK,), dtype=tl.float32)
        gz = tl.zeros((BLOCK,), dtype=tl.float32)
        tx = tl.zeros((BLOCK,), dtype=tl.float32)     # sum_k t_k dV/dp
        ty = tl.zeros((BLOCK,), dtype=tl.float32)
        tz = tl.zeros((BLOCK,), dtype=tl.float32)
        kd = tl.zeros((BLOCK,), dtype=tl.float32)     # sum_k (k+0.5) (dV/dp . d)

        for k in range(0, S):
            tk = tmin + step * (k.to(tl.float32) + 0.5)
            px = ox + ddx * tk
            py = oy + ddy * tk
            pz = oz + ddz * tk
            ix = (px - X0) / dx - 0.5
            iy = (py - Y0) / dy - 0.5
            iz = (pz - Z0) / dz - 0.5
            fx0 = tl.floor(ix)
            fy0 = tl.floor(iy)
            fz0 = tl.floor(iz)
            wx = ix - fx0
            wy = iy - fy0
            wz = iz - fz0
            i0 = fx0.to(tl.int32)
            j0 = fy0.to(tl.int32)
            k0 = fz0.to(tl.int32)
            okx0 = (i0 >= 0) & (i0 < W)
            okx1 = (i0 + 1 >= 0) & (i0 + 1 < W)
            oky0 = (j0 >= 0) & (j0 < H)
            oky1 = (j0 + 1 >= 0) & (j0 + 1 < H)
            okz0 = (k0 >= 0) & (k0 < D)
            okz1 = (k0 + 1 >= 0) & (k0 + 1 < D)
            # a = the strided in-plane axis, b = the contiguous one
            a0 = tl.where(xdom, i0, j0).to(tl.int64)
            b0 = tl.where(xdom, j0, i0).to(tl.int64)
            base = voff + k0.to(tl.int64) * HW + a0 * sa + b0
            oka0 = tl.where(xdom, okx0, oky0)
            oka1 = tl.where(xdom, okx1, oky1)
            okb0 = tl.where(xdom, oky0, okx0)
            okb1 = tl.where(xdom, oky1, okx1)
            c000 = tl.load(vol_ptr + base, mask=live & okz0 & oka0 & okb0, other=0.0)
            c001 = tl.load(vol_ptr + base + 1, mask=live & okz0 & oka0 & okb1, other=0.0)
            c010 = tl.load(vol_ptr + base + sa, mask=live & okz0 & oka1 & okb0, other=0.0)
            c011 = tl.load(vol_ptr + base + sa + 1, mask=live & okz0 & oka1 & okb1, other=0.0)
            c100 = tl.load(vol_ptr + base + HW, mask=live & okz1 & oka0 & okb0, other=0.0)
            c101 = tl.load(vol_ptr + base + HW + 1, mask=live & okz1 & oka0 & okb1, other=0.0)
            c110 = tl.load(vol_ptr + base + HW + sa, mask=live & okz1 & oka1 & okb0, other=0.0)
            c111 = tl.load(vol_ptr + base + HW + sa + 1, mask=live & okz1 & oka1 & okb1,
                           other=0.0)
            # (z, a, b) weights; map back to (x, y): a=x,b=y when xdom else a=y,b=x
            wa = tl.where(xdom, wx, wy)
            wb = tl.where(xdom, wy, wx)
            l00 = (1.0 - wa) * (1.0 - wb)
            l01 = (1.0 - wa) * wb
            l10 = wa * (1.0 - wb)
            l11 = wa * wb
            v0 = l00 * c000 + l01 * c001 + l10 * c010 + l11 * c011
            v1 = l00 * c100 + l01 * c101 + l10 * c110 + l11 * c111
            V = (1.0 - wz) * v0 + wz * v1
            Ssum += V
            if MODE == 1:
                # dV/dwa, dV/dwb, dV/dwz
                dva = (1.0 - wz) * ((1.0 - wb) * (c010 - c000) + wb * (c011 - c001)) \
                    + wz * ((1.0 - wb) * (c110 - c100) + wb * (c111 - c101))
                dvb = (1.0 - wz) * ((1.0 - wa) * (c001 - c000) + wa * (c011 - c010)) \
                    + wz * ((1.0 - wa) * (c101 - c100) + wa * (c111 - c110))
                dvz = v1 - v0
                dvx = tl.where(xdom, dva, dvb) / dx
                dvy = tl.where(xdom, dvb, dva) / dy
                dvz = dvz / dz
                gx += dvx
                gy += dvy
                gz += dvz
                tx += dvx * tk
                ty += dvy * tk
                tz += dvz * tk
                kd += (k.to(tl.float32) + 0.5) * (dvx * ddx + dvy * ddy + dvz * ddz)

        if MODE == 0:
            tl.store(out_ptr + r, tl.where(live, step * Ssum, 0.0), mask=m)
        if MODE == 1:
            gs = gout * step
            gd = gx * ddx + gy * ddy + gz * ddz
            tl.store(grad_ptr + r * 8 + 0, gs * gx, mask=m)
            tl.store(grad_ptr + r * 8 + 1, gs * gy, mask=m)
            tl.store(grad_ptr + r * 8 + 2, gs * gz, mask=m)
            tl.store(grad_ptr + r * 8 + 3, gs * tx, mask=m)
            tl.store(grad_ptr + r * 8 + 4, gs * ty, mask=m)
            tl.store(grad_ptr + r * 8 + 5, gs * tz, mask=m)
            tl.store(grad_ptr + r * 8 + 6, gs * gd, mask=m)
            tl.store(grad_ptr + r * 8 + 7, gout * (Ssum + step * kd), mask=m)


class _RayMarchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vol_st, o, d, tmin, step, hit, S, W, H, D, dx, dy, dz, X0, Y0, Z0, block):
        R = o.shape[0]
        out = torch.empty((R,), device=o.device, dtype=torch.float32)
        grid = (triton.cdiv(R, block),)
        _raymarch_kernel[grid](vol_st, o, d, tmin, step, hit, out, out, out,
                               R, S, W, H, D, H * W,
                               float(dx), float(dy), float(dz), float(X0), float(Y0), float(Z0),
                               BLOCK=block, MODE=0)
        ctx.save_for_backward(vol_st, o, d, tmin, step, hit)
        ctx.meta = (S, W, H, D, dx, dy, dz, X0, Y0, Z0, block)
        return out

    @staticmethod
    def backward(ctx, gout):
        vol_st, o, d, tmin, step, hit = ctx.saved_tensors
        S, W, H, D, dx, dy, dz, X0, Y0, Z0, block = ctx.meta
        R = o.shape[0]
        gout = gout.contiguous().to(torch.float32)
        grad = torch.empty((R, 8), device=o.device, dtype=torch.float32)
        grid = (triton.cdiv(R, block),)
        _raymarch_kernel[grid](vol_st, o, d, tmin, step, hit, gout, grad, grad,
                               R, S, W, H, D, H * W,
                               float(dx), float(dy), float(dz), float(X0), float(Y0), float(Z0),
                               BLOCK=block, MODE=1)
        return (None, grad[:, 0:3].contiguous(), grad[:, 3:6].contiguous(),
                grad[:, 6].contiguous(), grad[:, 7].contiguous(),
                None, None, None, None, None, None, None, None, None, None, None, None)


def _swap_yz(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([v[..., 0], v[..., 2], v[..., 1]], dim=-1)


def _ray_box_intersect(o, d, box_min, box_max, eps_dir: float = 1e-8):
    """Identical to differentiable_forward_projector._ray_box_intersect."""
    d_safe = torch.where(d.abs() < eps_dir, torch.full_like(d, eps_dir), d)
    inv_d = 1.0 / d_safe
    t0 = (box_min[None, :] - o) * inv_d
    t1 = (box_max[None, :] - o) * inv_d
    tmin = torch.max(torch.minimum(t0, t1), dim=-1).values
    tmax = torch.min(torch.maximum(t0, t1), dim=-1).values
    return tmin, tmax, tmax > tmin


def _detector_uv(*, nu, nv, du, dv, ureverse, vreverse, roi, recon_type, ori_nu, ori_nv,
                 stitch_u, stitch_v, device):
    """(u, v) mm of every output pixel, (nv*nu,) each -- the original's formulas verbatim."""
    jp2, kp2 = _build_jk_grid(nu, nv, device=device, dtype=torch.float32)
    jp, kp = _apply_roi_mapping(jp2, kp2, roi=roi, ori_nu=ori_nu, ori_nv=ori_nv, nu=nu, nv=nv,
                                recon_type=recon_type)
    extra_u = float(ori_nu - (roi.x + roi.width))
    if int(ureverse) == 1:
        u = (float(nu) - jp + 0.5 + extra_u) * float(du)
    else:
        u = (jp + 0.5 + stitch_u + extra_u) * float(du)
    if int(vreverse) == 1:
        v = (float(nv) - kp + 0.5 + stitch_v) * float(dv)
    else:
        v = (kp + 0.5 + stitch_v) * float(dv)
    return u, v


def sinoproj_raymarch_triton(
    *, smat, Pmat, geo_parameter, geo_stitch, nu, nv, du, dv, imsx, imsy, imsz, dx, dy, dz,
    X0, Y0, Z0, ureverse, vreverse, roi: RT_PARAM, recon_type, n_samples: int = 128,
    chunk_size: int = 8192, ori_nu: Optional[int] = None, ori_nv: Optional[int] = None,
    reg: float = 1e-6, align_corners: bool = False, block: int = 256,
) -> torch.Tensor:
    """Drop-in for `sinoproj_rdsh_pinv_raycast_dominant` (same model, fused kernel).

    `chunk_size` is accepted and ignored (nothing is chunked any more); `align_corners` must be
    False (the only value the original pipeline uses)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not importable; use the original projector")
    if align_corners:
        raise ValueError("sinoproj_raymarch_triton implements align_corners=False only")
    assert smat.ndim == 5 and smat.shape[0] == 1 and smat.shape[1] == 1
    device = smat.device
    ori_nu = nu if ori_nu is None else ori_nu
    ori_nv = nv if ori_nv is None else ori_nv
    Pm = Pmat.reshape(-1, 3, 4).to(torch.float32)
    B = Pm.shape[0]
    R = nu * nv
    vol_st = stacked_volume(smat[0, 0])

    box_min = torch.tensor([X0, Y0, Z0], device=device, dtype=torch.float32)
    box_max = torch.tensor([X0 + dx * imsx, Y0 + dy * imsy, Z0 + dz * imsz], device=device,
                           dtype=torch.float32)
    I3 = torch.eye(3, device=device, dtype=torch.float32)

    o_all, d_all, tmin_all, step_all, hit_all = [], [], [], [], []
    for b in range(B):
        P = Pm[b]
        A_reg = P[:, :3] + reg * I3
        t = P[:, 3]
        src_world = geo_parameter[b, 0:3].to(torch.float32)
        w_geo = geo_parameter[b, 5].to(torch.float32)
        u, v = _detector_uv(nu=nu, nv=nv, du=du, dv=dv, ureverse=ureverse, vreverse=vreverse,
                            roi=roi, recon_type=recon_type, ori_nu=ori_nu, ori_nv=ori_nv,
                            stitch_u=geo_stitch[b, 0].to(torch.float32),
                            stitch_v=geo_stitch[b, 1].to(torch.float32), device=device)
        cal = torch.stack([u * w_geo, v * w_geo, w_geo.expand_as(u)], dim=-1) - t[None, :]
        # the original solves A_reg x = cal per ray; one inverse + matmul is the same solution
        soln = cal @ torch.linalg.inv(A_reg).transpose(0, 1)
        vec = soln - src_world[None, :]
        vec = vec / torch.sqrt((vec * vec).sum(-1, keepdim=True) + 1e-12)
        d = _swap_yz(vec)                                    # unit already; the original
        d = d / torch.sqrt((d * d).sum(-1, keepdim=True) + 1e-12)  # renormalises the same way
        o = _swap_yz(src_world)[None, :].expand(R, 3)
        tmin, tmax, hit = _ray_box_intersect(o, d, box_min, box_max)
        step = (tmax - tmin).clamp_min(0.0) / float(n_samples)
        o_all.append(o); d_all.append(d); tmin_all.append(tmin); step_all.append(step)
        hit_all.append(hit)
    o = torch.cat(o_all).contiguous()
    d = torch.cat(d_all).contiguous()
    tmin = torch.cat(tmin_all).contiguous()
    step = torch.cat(step_all).contiguous()
    hit = torch.cat(hit_all).to(torch.int32).contiguous()
    proj = _RayMarchFn.apply(vol_st, o, d, tmin, step, hit, int(n_samples), imsx, imsy, imsz,
                             dx, dy, dz, X0, Y0, Z0, block)
    return proj.view(B, nv, nu).to(smat.dtype)


# ---------------------------------------------------------------------------------------
# Kernel 2: LEAP modular-beam JOSEPH forward + exact geometry gradient
# (transcription of LLNL LEAP `projectors_Joseph.cu`, MIT; from the author's
#  Flow_matching_motion_3D/fm3d/triton_leap_grad.py `_leap_joseph_kernel`, verbatim maths)
# ---------------------------------------------------------------------------------------
if HAVE_TRITON:

    @triton.jit
    def _leap_joseph_kernel(vol_ptr, gout_ptr, out_ptr,
                            src_ptr, mod_ptr, rowv_ptr, colv_ptr,
                            nv, nu, NP, D, HW,
                            dx, dz, du, dv, u0g, v0g, b0, z0,
                            BLOCK: tl.constexpr, MODE: tl.constexpr):
        """modularBeamJosephProjectorKernel + lineIntegral_Joseph_ZYX (in-plane dominant).
        Bilinear sample at every dominant-axis voxel plane; the entry-plane sample carries an
        extra 0.5 weight; length = dx * |r| / |r_a|.
        MODE 0: out = sinogram (V, nv, nu) fp32.   MODE 2: out = (V, 12) fp64 d/d(src, mod,
        rowv, colv) of <gout, sinogram>, block-reduced then atomically accumulated."""
        view = tl.program_id(1)
        pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pix < nv * nu
        m = pix // nu
        n = pix - m * nu

        px = tl.load(src_ptr + view * 3 + 0)
        py = tl.load(src_ptr + view * 3 + 1)
        pz = tl.load(src_ptr + view * 3 + 2)
        cx = tl.load(mod_ptr + view * 3 + 0)
        cy = tl.load(mod_ptr + view * 3 + 1)
        cz_ = tl.load(mod_ptr + view * 3 + 2)
        vx = tl.load(rowv_ptr + view * 3 + 0)
        vy = tl.load(rowv_ptr + view * 3 + 1)
        vz = tl.load(rowv_ptr + view * 3 + 2)
        ux = tl.load(colv_ptr + view * 3 + 0)
        uy = tl.load(colv_ptr + view * 3 + 1)
        uz = tl.load(colv_ptr + view * 3 + 2)

        t_r = m.to(tl.float32) * dv + v0g
        s = n.to(tl.float32) * du + u0g

        detx = cx + ux * s + vx * t_r
        dety = cy + uy * s + vy * t_r
        detz = cz_ + uz * s + vz * t_r
        rx = detx - px
        ry = dety - py
        rz = detz - pz

        ydom = tl.abs(ry) > tl.abs(rx)
        r_a = tl.where(ydom, ry, rx)
        r_b = tl.where(ydom, rx, ry)
        p_a = tl.where(ydom, py, px)
        p_b = tl.where(ydom, px, py)
        inv_ra = 1.0 / r_a
        j0 = tl.where(r_a > 0.0, 0, NP - 1)

        nrm = tl.sqrt(rx * rx + ry * ry + rz * rz)
        abs_ra = tl.abs(r_a)
        L = dx * nrm / abs_ra

        gout = tl.zeros((BLOCK,), dtype=tl.float32)
        if MODE == 2:
            gout = tl.load(gout_ptr + (view * nv * nu).to(tl.int64) + pix, mask=mask,
                           other=0.0)
        ghat = gout * L

        S_tot = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_a = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_b = tl.zeros((BLOCK,), dtype=tl.float32)
        bp_z = tl.zeros((BLOCK,), dtype=tl.float32)
        br_a = tl.zeros((BLOCK,), dtype=tl.float32)
        br_b = tl.zeros((BLOCK,), dtype=tl.float32)
        br_z = tl.zeros((BLOCK,), dtype=tl.float32)

        for j in range(0, NP):
            w_j = j * dx + b0
            lam = (w_j - p_a) * inv_ra
            cb = (p_b + lam * r_b - b0) / dx
            cz = (pz + lam * rz - z0) / dz
            ibf = tl.floor(cb)
            izf = tl.floor(cz)
            wb = cb - ibf
            wz = cz - izf
            ib = ibf.to(tl.int32)
            iz = izf.to(tl.int32)
            okc0 = (ib >= 0) & (ib < NP)
            okc1 = (ib + 1 >= 0) & (ib + 1 < NP)
            okz0 = (iz >= 0) & (iz < D)
            okz1 = (iz + 1 >= 0) & (iz + 1 < D)
            # x-dominant rays read the [z,x,y] copy so the cross-axis step is 1 for every ray
            voff = tl.where(ydom, 0, D * NP * NP).to(tl.int64)
            fl00 = voff + iz.to(tl.int64) * HW + (j * NP).to(tl.int64) + ib.to(tl.int64)
            f00 = tl.load(vol_ptr + fl00, mask=mask & okz0 & okc0, other=0.0)
            f10 = tl.load(vol_ptr + fl00 + 1, mask=mask & okz0 & okc1, other=0.0)
            f01 = tl.load(vol_ptr + fl00 + HW, mask=mask & okz1 & okc0, other=0.0)
            f11 = tl.load(vol_ptr + fl00 + 1 + HW, mask=mask & okz1 & okc1, other=0.0)
            S = (1.0 - wb) * (1.0 - wz) * f00 + wb * (1.0 - wz) * f10 \
                + (1.0 - wb) * wz * f01 + wb * wz * f11
            wgt = tl.where(j == j0, 0.5, 1.0)
            S_tot += wgt * S
            if MODE == 2:
                scale = ghat * wgt
                bcb = scale * ((1.0 - wz) * (f10 - f00) + wz * (f11 - f01))
                bcz = scale * ((1.0 - wb) * (f01 - f00) + wb * (f11 - f10))
                bp_b += bcb / dx
                bp_z += bcz / dz
                blam = bcb * r_b / dx + bcz * rz / dz
                br_b += bcb * lam / dx
                br_z += bcz * lam / dz
                bp_a -= blam * inv_ra
                br_a -= blam * lam * inv_ra

        if MODE == 0:
            tl.store(out_ptr + (view * nv * nu).to(tl.int64) + pix, L * S_tot, mask=mask)
        if MODE == 2:
            bL = gout * S_tot
            sgn_a = tl.where(r_a >= 0.0, 1.0, -1.0)
            br_a += bL * dx * (r_a / (abs_ra * nrm) - nrm * sgn_a / (r_a * r_a))
            br_b += bL * dx * r_b / (abs_ra * nrm)
            br_z += bL * dx * rz / (abs_ra * nrm)
            br_x = tl.where(ydom, br_b, br_a)
            br_y = tl.where(ydom, br_a, br_b)
            bp_x = tl.where(ydom, bp_b, bp_a)
            bp_y = tl.where(ydom, bp_a, bp_b)
            bp_x -= br_x
            bp_y -= br_y
            bp_z2 = bp_z - br_z
            bc_x = br_x
            bc_y = br_y
            bc_z = br_z
            bu_x = s * br_x
            bu_y = s * br_y
            bu_z = s * br_z
            bv_x = t_r * br_x
            bv_y = t_r * br_y
            bv_z = t_r * br_z
            ob = out_ptr + view * 12
            zero = tl.zeros((BLOCK,), dtype=tl.float32)
            tl.atomic_add(ob + 0, tl.sum(tl.where(mask, bp_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 1, tl.sum(tl.where(mask, bp_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 2, tl.sum(tl.where(mask, bp_z2, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 3, tl.sum(tl.where(mask, bc_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 4, tl.sum(tl.where(mask, bc_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 5, tl.sum(tl.where(mask, bc_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 6, tl.sum(tl.where(mask, bv_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 7, tl.sum(tl.where(mask, bv_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 8, tl.sum(tl.where(mask, bv_z, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 9, tl.sum(tl.where(mask, bu_x, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 10, tl.sum(tl.where(mask, bu_y, zero)).to(tl.float64), sem="relaxed")
            tl.atomic_add(ob + 11, tl.sum(tl.where(mask, bu_z, zero)).to(tl.float64), sem="relaxed")


def _joseph_launch(mode, vol_st, gout, out, arrs, *, nv, nu, D, H, W, dx, dz, du, dv, block):
    src, mod, rowv, colv = arrs
    V = src.shape[0]
    u0g = -0.5 * (nu - 1) * du
    v0g = -0.5 * (nv - 1) * dv
    b0 = -0.5 * (W - 1) * dx
    z0 = -0.5 * (D - 1) * dz
    grid = (triton.cdiv(nv * nu, block), V)
    _leap_joseph_kernel[grid](vol_st, gout if gout is not None else out, out,
                              src, mod, rowv, colv,
                              nv, nu, W, D, H * W,
                              float(dx), float(dz), float(du), float(dv),
                              float(u0g), float(v0g), float(b0), float(z0),
                              BLOCK=block, MODE=mode)


class _JosephFn(torch.autograd.Function):
    """(src, mod, rowv, colv) each (V,3) fp32 -> sinogram (V, nv, nu). Value from the Triton
    transcription (default) or from libleapct (`value_backend='leap'`); gradient always the
    exact Triton MODE 2."""

    @staticmethod
    def forward(ctx, vol_st, src, mod, rowv, colv, nv, nu, D, H, W, dx, dz, du, dv, block,
                value_backend, vol_raw):
        V = src.shape[0]
        arrs = tuple(a.detach().contiguous().to(torch.float32) for a in (src, mod, rowv, colv))
        if value_backend == "leap":
            out = _leap_library_value(vol_raw, arrs, nv=nv, nu=nu, dx=dx, dz=dz, du=du, dv=dv)
        else:
            out = torch.empty((V, nv, nu), device=src.device, dtype=torch.float32)
            _joseph_launch(0, vol_st, None, out, arrs, nv=nv, nu=nu, D=D, H=H, W=W,
                           dx=dx, dz=dz, du=du, dv=dv, block=block)
        ctx.save_for_backward(vol_st, *arrs)
        ctx.meta = (nv, nu, D, H, W, dx, dz, du, dv, block)
        return out

    @staticmethod
    def backward(ctx, gout):
        vol_st, src, mod, rowv, colv = ctx.saved_tensors
        nv, nu, D, H, W, dx, dz, du, dv, block = ctx.meta
        V = src.shape[0]
        g = torch.zeros((V, 12), device=src.device, dtype=torch.float64)
        _joseph_launch(2, vol_st, gout.contiguous().to(torch.float32), g,
                       (src, mod, rowv, colv), nv=nv, nu=nu, D=D, H=H, W=W,
                       dx=dx, dz=dz, du=du, dv=dv, block=block)
        g = g.to(torch.float32)
        return (None, g[:, 0:3].contiguous(), g[:, 3:6].contiguous(), g[:, 6:9].contiguous(),
                g[:, 9:12].contiguous(), None, None, None, None, None, None, None, None, None,
                None, None, None)


def modular_arrays_geocal(
    Pmat, geo_stitch, *, nu, nv, du, dv, ureverse, vreverse, roi: RT_PARAM, recon_type,
    ori_nu, ori_nv, imsx, imsy, imsz, dx, dy, dz, X0, Y0, Z0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable P (B,3,4) -> LEAP-form (src, mod, rowv, colv), (B,3) each, in the
    VOLUME frame (INTERNAL = P-world with y/z swapped) CENTRED on the volume box.

    P = K [R | t] with K = [[fu,0,un],[0,fv,vn],[0,0,1]] (any homogeneous scale). The pixel
    (m, n) of the OUTPUT array maps to detector mm (u, v) exactly as the original projector
    does (ureverse/vreverse, ROI/recon_type, extra_u, stitch), and the detector point is
        X = C + fu e_n + (u - un) e_u + (v - vn) (fu/fv) e_v,
    whose camera projection is (u, v) -- the same ray the original marches. The kernel
    parameterises X = mod + colv*(n du + u0g) + rowv*(m dv + v0g) with u0g = -(nu-1)du/2, so
    colv = +-e_u, rowv = +-(fu/fv) e_v and mod = X(0,0) - colv u0g - rowv v0g."""
    P = Pmat.reshape(-1, 3, 4).to(torch.float64)
    B = P.shape[0]
    dev = P.device
    # homogeneous freedom: scale each P so that its depth row (row 2 of M) is a unit vector
    sc = P[:, 2, :3].norm(dim=-1)                       # (B,)
    M = P[:, :, :3] / sc[:, None, None]
    p4 = P[:, :, 3] / sc[:, None]
    e_n = M[:, 2]
    un = (M[:, 0] * e_n).sum(-1)
    vn = (M[:, 1] * e_n).sum(-1)
    fu_vec = M[:, 0] - un[:, None] * e_n
    fv_vec = M[:, 1] - vn[:, None] * e_n
    fu = fu_vec.norm(dim=-1)
    fv = fv_vec.norm(dim=-1)
    e_u = fu_vec / fu[:, None]
    e_v = fv_vec / fv[:, None]
    C = torch.linalg.solve(M, -p4[..., None])[..., 0]

    ori_nu = nu if ori_nu is None else ori_nu
    ori_nv = nv if ori_nv is None else ori_nv
    extra_u = float(ori_nu - (roi.x + roi.width))
    if recon_type == 1:
        cj = -float(roi.x)
        ck = -float(ori_nv - (roi.height + roi.y))
    elif recon_type == 0:
        cj = 0.0
        ck = float(ori_nv - (roi.height + roi.y))
    else:
        cj = 0.0
        ck = 0.0
    st_u = geo_stitch[:, 0].to(torch.float64)
    st_v = geo_stitch[:, 1].to(torch.float64)
    if int(ureverse) == 1:
        sig_u = -1.0
        u_0 = (float(nu) - cj + 0.5 + extra_u) * float(du) + 0.0 * st_u
    else:
        sig_u = 1.0
        u_0 = (cj + 0.5 + extra_u) * float(du) + st_u * float(du)
    if int(vreverse) == 1:
        sig_v = -1.0
        v_0 = (float(nv) - ck + 0.5) * float(dv) + st_v * float(dv)
    else:
        sig_v = 1.0
        v_0 = (ck + 0.5) * float(dv) + st_v * float(dv)

    ratio = (fu / fv)[:, None]
    colv = sig_u * e_u
    rowv = sig_v * ratio * e_v
    det00 = C + fu[:, None] * e_n + (u_0 - un)[:, None] * e_u + (v_0 - vn)[:, None] * ratio * e_v
    u0g = -0.5 * (nu - 1) * float(du)
    v0g = -0.5 * (nv - 1) * float(dv)
    mod = det00 - colv * u0g - rowv * v0g

    # P-world -> volume frame (swap y/z), then centre on the box
    centre = torch.tensor([X0 + 0.5 * imsx * dx, Y0 + 0.5 * imsy * dy, Z0 + 0.5 * imsz * dz],
                          device=dev, dtype=torch.float64)
    src = _swap_yz(C) - centre
    mod = _swap_yz(mod) - centre
    return src, mod, _swap_yz(rowv), _swap_yz(colv)


def sinoproj_joseph(
    *, smat, Pmat, geo_parameter, geo_stitch, nu, nv, du, dv, imsx, imsy, imsz, dx, dy, dz,
    X0, Y0, Z0, ureverse, vreverse, roi: RT_PARAM, recon_type, n_samples: int = 0,
    chunk_size: int = 0, ori_nu: Optional[int] = None, ori_nv: Optional[int] = None,
    reg: float = 0.0, align_corners: bool = False, block: int = 256,
    value_backend: str = "triton",
) -> torch.Tensor:
    """Drop-in for `sinoproj_rdsh_pinv_raycast_dominant` with the LEAP Joseph line-integral
    model. `geo_parameter` is accepted for signature parity (the source is re-derived from P,
    which is what it holds); `n_samples`/`chunk_size`/`reg` are ignored."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not importable; use the original projector")
    assert smat.ndim == 5 and smat.shape[0] == 1 and smat.shape[1] == 1
    if abs(dx - dy) > 1e-9 or abs(dz - dx) > 1e-9:
        raise ValueError(f"the Joseph kernel needs cubic voxels, got dx={dx} dy={dy} dz={dz}")
    if imsx != imsy:
        raise ValueError(f"the Joseph kernel's in-plane slab loop needs imsx == imsy, got "
                         f"{imsx} x {imsy}")
    D, H, W = imsz, imsy, imsx
    vol_st = stacked_volume(smat[0, 0])
    src, mod, rowv, colv = modular_arrays_geocal(
        Pmat, geo_stitch, nu=nu, nv=nv, du=du, dv=dv, ureverse=ureverse, vreverse=vreverse,
        roi=roi, recon_type=recon_type, ori_nu=ori_nu, ori_nv=ori_nv, imsx=imsx, imsy=imsy,
        imsz=imsz, dx=dx, dy=dy, dz=dz, X0=X0, Y0=Y0, Z0=Z0)
    out = _JosephFn.apply(vol_st, src.to(torch.float32), mod.to(torch.float32),
                          rowv.to(torch.float32), colv.to(torch.float32),
                          nv, nu, D, H, W, dx, dz, du, dv, block, value_backend,
                          smat[0, 0] if value_backend == "leap" else None)
    return out.to(smat.dtype)


# ---------------------------------------------------------------------------------------
# Optional: the VALUE from the real LEAP library (gate / benchmark only)
# ---------------------------------------------------------------------------------------
_LEAP_MODEL = {}


def _leap_model(device):
    from leapctype import tomographicModels
    key = device.index
    m = _LEAP_MODEL.get(key)
    if m is None:
        m = tomographicModels()
        m.set_gpu(0 if key is None else key)
        if not hasattr(m, "set_forceJosephModular"):
            raise RuntimeError("this libleapct is UNPATCHED (no set_forceJosephModular): on an "
                               "axially-aligned orbit stock LEAP runs its SF kernel, a "
                               "different model from the Joseph gradient computed here")
        m.set_forceJosephModular(True)
        _LEAP_MODEL[key] = m
    return m


def _leap_library_value(vol, arrs, *, nv, nu, dx, dz, du, dv):
    """vol (D,H,W) fp32 cuda, arrs = (src, mod, rowv, colv) fp32 (V,3) in the centred volume
    frame -> (V, nv, nu) from libleapct's modular-beam Joseph projector."""
    D, H, W = vol.shape
    leap = _leap_model(vol.device)
    src, mod, rowv, colv = (a.detach().to("cpu", torch.float32).numpy().copy(order="C")
                            for a in arrs)
    V = src.shape[0]
    if not leap.set_modularbeam(V, nv, nu, dv, du, src, mod, rowv, colv):
        raise RuntimeError("LEAP rejected the modular geometry")
    if not leap.set_volume(W, H, D, dx, dz):
        raise RuntimeError("LEAP rejected the volume")
    leap.set_diameterFOV(1.0e5)
    g = torch.zeros((V, nv, nu), device=vol.device, dtype=torch.float32)
    leap.project_gpu(g, vol.contiguous().to(torch.float32))
    if float(g.abs().max()) == 0.0 and float(vol.abs().max()) != 0.0:
        raise RuntimeError("LEAP returned an all-zero projection (its CUDA allocation failed)")
    return g


# ---------------------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------------------
PROJECTORS = ("auto", "raymarch", "raymarch_triton", "joseph")


def get_projector(kind: str):
    """'joseph' (the default everywhere since the 2026-09-21 A/B) = LEAP Joseph model with
    exact geometry gradient; 'raymarch_triton' = the original model, fused kernel (use it for
    non-cubic voxels or imsx != imsy); 'raymarch' = the original grid_sample projector;
    'auto' = joseph when triton is importable, else the original."""
    kind = (kind or "joseph").lower()
    if kind == "auto":
        kind = "joseph" if (HAVE_TRITON and torch.cuda.is_available()) else "raymarch"
    if kind == "raymarch":
        from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant
        return sinoproj_rdsh_pinv_raycast_dominant
    if kind == "raymarch_triton":
        return sinoproj_raymarch_triton
    if kind == "joseph":
        return sinoproj_joseph
    raise ValueError(f"unknown projector {kind!r}; choose one of {PROJECTORS}")
