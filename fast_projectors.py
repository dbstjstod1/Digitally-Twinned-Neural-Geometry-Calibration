"""Differentiable Joseph forward projection for AI-Geocal.

The Triton kernel implements LEAP's modular-beam Joseph line integral and its
geometry gradient, chained through this repository's detector conventions.
It samples each dominant-axis voxel plane with bilinear interpolation, weights
the entry plane by one half, and scales by dx * |ray| / |ray_axis|.

Requires CUDA, PyTorch >= 2.1, and a matching Triton installation. Volumes must
have cubic voxels and equal x/y dimensions. A cached transposed copy doubles
the reference volume storage to keep reads coalesced for both in-plane axes.
No external LEAP installation is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


@dataclass
class RT_PARAM:
    """Detector region of interest in original detector pixels."""

    x: int
    y: int
    width: int
    height: int


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


def _swap_yz(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([v[..., 0], v[..., 2], v[..., 1]], dim=-1)


# ---------------------------------------------------------------------------------------
# LEAP modular-beam Joseph forward + exact geometry gradient
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
    """Project (src, mod, rowv, colv), with the kernel's exact geometry gradient."""

    @staticmethod
    def forward(ctx, vol_st, src, mod, rowv, colv, nv, nu, D, H, W, dx, dz, du, dv, block):
        V = src.shape[0]
        arrs = tuple(a.detach().contiguous().to(torch.float32) for a in (src, mod, rowv, colv))
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
                None)


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
) -> torch.Tensor:
    """Project a fixed volume to (views, nv, nu), with gradients through P and stitch.

    The source is derived from P. ``geo_parameter`` and the legacy sampling keywords
    ``n_samples``, ``chunk_size``, ``reg``, ``align_corners`` are accepted but unused.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Joseph projection requires PyTorch >= 2.1 and matching Triton.")
    if smat.ndim != 5 or tuple(smat.shape[:2]) != (1, 1):
        raise ValueError("smat must have shape (1, 1, imsz, imsy, imsx)")
    if tuple(smat.shape[2:]) != (imsz, imsy, imsx):
        raise ValueError("smat dimensions must match imsz, imsy, and imsx")
    if not smat.is_cuda:
        raise ValueError("Joseph projection requires a CUDA volume tensor")
    if Pmat.device != smat.device or geo_stitch.device != smat.device:
        raise ValueError("Pmat, geo_stitch, and smat must be on the same CUDA device")
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
                          nv, nu, D, H, W, dx, dz, du, dv, block)
    return out.to(smat.dtype)
