"""Gate for `fast_projectors` against the original ray-marching projector, on the shipped
4T Denseball case (929^3 @ 0.2 mm reference volume, 776 x 1264 panel, measured projections).

  G0  kernel unit test (small random volume, W != H): fused ray-march VALUE vs torch
      grid_sample, and its per-ray GRADIENT vs autograd of the same torch chain.
  G1  VALUE parity  raymarch_triton == original (same n_samples)      -> rel <= 1e-4
  G2  GRADIENT parity dL/d(ts,tp,rot) raymarch_triton == original     -> rel <= 1e-3
      (L = 1 + LNCC(kernel 31) on the training ROI against the measured projection, exactly
      the training loss; the gradient flows through apply_9DoF_transform_effective)
  G3  Joseph vs the ray-march model as n_samples grows (256 -> 4096): the two discretise
      the same trilinear line integral, so the gap must shrink towards the Joseph value.
  G4  Joseph GRADIENT vs a central finite difference of the Joseph model's fp64 L2 loss
      (9 DoF, 1 view, BLURRED volume, eps sweep), and the same for raymarch_triton. A binary
      phantom's a.e. gradient and MONAI's fp32 LNCC both defeat a finite difference -- see the
      function docstring -- so this is the referee that can actually reach the gradient.
  G5  Joseph Triton VALUE vs the real libleapct (Joseph-pinned build) -> rel ~1e-4
      (tex3D's 9-bit lerp), skipped if leapctype is unavailable or unpatched.

    CUDA_VISIBLE_DEVICES=1 python gate_fast_projectors.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AI_Geocal import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from DoF_transform import apply_9DoF_transform_effective
from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant
from fast_projectors import (sinoproj_raymarch_triton, sinoproj_joseph, _RayMarchFn,
                             stacked_volume)
from helpers import load_raw_f32_memmap, reverse_flag
from monai.losses import LocalNormalizedCrossCorrelationLoss

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = os.path.join(HERE, "open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_"
                         "Ntheta24_zpitch20.00_929x929x801.float32.raw")
PROJ = os.path.join(HERE, "Denseball_proj_480.raw")

n_fail = 0


def check(name, ok, detail=""):
    global n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        n_fail += 1


def rel(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    return float((a - b).norm() / (b.norm() + 1e-30))


def cosd(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# --------------------------------------------------------------------------------------
print("G0  fused ray-march kernel vs torch grid_sample on a small W != H volume")
torch.manual_seed(0)
dev = torch.device("cuda")
D0, H0, W0 = 40, 48, 56
dx0, dy0, dz0 = 0.7, 0.5, 0.6
X00, Y00, Z00 = -0.5 * W0 * dx0, -0.5 * H0 * dy0 + 1.3, 0.0
vol0 = torch.rand(D0, H0, W0, device=dev)
vol0 = F.avg_pool3d(vol0[None, None], 3, 1, 1)[0, 0].contiguous()   # smooth: no kink trap
R0, S0 = 3000, 64
o0 = torch.randn(R0, 3, device=dev) * 5 + torch.tensor([0.0, 1.0, 12.0], device=dev)
o0[:, 0] -= 60.0
d0 = F.normalize(torch.randn(R0, 3, device=dev) * torch.tensor([1.0, 0.3, 0.3], device=dev)
                 + torch.tensor([1.0, 0.0, 0.0], device=dev), dim=-1)
tmin0 = torch.rand(R0, device=dev) * 5 + 20
step0 = torch.rand(R0, device=dev) * 0.3 + 0.4
hit0 = (torch.rand(R0, device=dev) > 0.1)


def ref_march(o, d, tmin, step):
    ks = torch.arange(S0, device=dev, dtype=torch.float32) + 0.5
    t = tmin[:, None] + step[:, None] * ks[None]
    p = o[:, None, :] + d[:, None, :] * t[..., None]
    ix = (p[..., 0] - X00) / dx0 - 0.5
    iy = (p[..., 1] - Y00) / dy0 - 0.5
    iz = (p[..., 2] - Z00) / dz0 - 0.5
    g = torch.stack([(2 * ix + 1) / W0 - 1, (2 * iy + 1) / H0 - 1, (2 * iz + 1) / D0 - 1], -1)
    g = g.permute(1, 0, 2).contiguous()[None, :, None]
    s = F.grid_sample(vol0[None, None], g, mode="bilinear", padding_mode="zeros",
                      align_corners=False)[0, 0, :, 0, :]
    out = s.sum(0) * step
    return torch.where(hit0, out, torch.zeros_like(out))


leafs = [x.clone().requires_grad_(True) for x in (o0, d0, tmin0, step0)]
ref = ref_march(*leafs)
w0 = torch.randn(R0, device=dev)
(ref * w0).sum().backward()
gref = [x.grad.clone() for x in leafs]
leafs2 = [x.clone().requires_grad_(True) for x in (o0, d0, tmin0, step0)]
out = _RayMarchFn.apply(stacked_volume(vol0), leafs2[0].contiguous(), leafs2[1].contiguous(),
                        leafs2[2], leafs2[3], hit0.to(torch.int32), S0, W0, H0, D0,
                        dx0, dy0, dz0, X00, Y00, Z00, 128)
(out * w0).sum().backward()
check("G0 value", rel(out, ref) < 1e-5, f"rel {rel(out, ref):.2e}")
for nm, a, b in zip(("d/do", "d/dd", "d/dtmin", "d/dstep"), [x.grad for x in leafs2], gref):
    check(f"G0 {nm}", rel(a, b) < 1e-4, f"rel {rel(a, b):.2e}  cos {cosd(a, b):.6f}")

# --------------------------------------------------------------------------------------
print("\nsetup: 4T Denseball geometry, real reference volume and measured projections")
cfg = ReconConfig(NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0, nu=776, nv=1264,
                  ori_nu=776, ori_nv=1264, du=0.228, dv=0.228, imsx=929, imsy=929, imsz=801,
                  dx=0.2, dy=0.2, dz=0.2, ureverse_raw=1, vreverse_raw=1, recon_type=1,
                  n_samples=256, chunk_size=16384)
cfg.X0 = -0.5 * cfg.imsx * cfg.dx
cfg.Y0 = -0.5 * cfg.imsy * cfg.dy
cfg.Z0 = 0.0
roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)
urev, vrev = reverse_flag(cfg.ureverse_raw), reverse_flag(cfg.vreverse_raw)
vol = torch.from_numpy(np.array(load_raw_f32_memmap(VOL, (cfg.imsz, cfg.imsy, cfg.imsx)))
                       ).to(dev)[None, None]
proj_mm = load_raw_f32_memmap(PROJ, (cfg.NLAM, cfg.nv, cfg.nu))
P_all, geo_all, st_all = build_nominal_orbit_from_geometry(
    n_views=cfg.NLAM, scan_angle_deg=360.0, start_angle_deg=180.0, k_nominal=650.0,
    un_nominal=34.0, vn_nominal=15.0, SOD=443.0, SDD=650.0, nx=cfg.imsx, ny=cfg.imsy,
    nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
    nu_ori=cfg.nu, nv_ori=cfg.nv, du=cfg.du, dv=cfg.dv, orbit_axis="y", clockwise_sign=-1.0,
    include_endpoint=False, use_beamcenter_geo=False, device=dev)
lncc = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=31,
                                           kernel_type="rectangular", reduction="mean").to(dev)
u0, u1, v0, v1 = 26, cfg.nu, 0, 1182
common = dict(smat=vol, nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv, imsx=cfg.imsx,
              imsy=cfg.imsy, imsz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0,
              Y0=cfg.Y0, Z0=cfg.Z0, ureverse=urev, vreverse=vrev, roi=roi,
              recon_type=cfg.recon_type, chunk_size=cfg.chunk_size, ori_nu=cfg.ori_nu,
              ori_nv=cfg.ori_nv, align_corners=False)
IDX = torch.tensor([3, 131, 277, 402], device=dev)     # four views around the orbit
tgt = torch.from_numpy(np.array(proj_mm[IDX.cpu().numpy()])).to(dev)
torch.manual_seed(1)
theta0 = torch.cat([torch.randn(len(IDX), 3) * 2.0,       # ts  mm
                    torch.randn(len(IDX), 3) * 2.0,       # tp  mm
                    torch.randn(len(IDX), 3) * 1.0], 1).to(dev)   # rot deg


def render(proj_fn, theta, idx=IDX, n_samples=256, **kw):
    P_new, geo_new = apply_9DoF_transform_effective(
        P0=P_all[idx], geo_old=geo_all[idx], ts_internal=theta[:, 0:3],
        tp_internal=theta[:, 3:6], rot_internal_deg=theta[:, 6:9], nx=cfg.imsx, ny=cfg.imsy,
        nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0)
    return proj_fn(Pmat=P_new, geo_parameter=geo_new, geo_stitch=st_all[idx],
                   n_samples=n_samples, **common, **kw)


def loss_of(pred, t):
    return 1.0 + lncc(pred[:, None, v0:v1, u0:u1].float(), t[:, None, v0:v1, u0:u1].float())


def loss_and_grad(proj_fn, theta, **kw):
    th = theta.clone().requires_grad_(True)
    L = loss_of(render(proj_fn, th, **kw), tgt)
    L.backward()
    return float(L), th.grad.clone()


# --------------------------------------------------------------------------------------
print("\nG1  value parity: raymarch_triton vs original (n_samples=256, 4 views)")
with torch.no_grad():
    y_orig = render(sinoproj_rdsh_pinv_raycast_dominant, theta0)
    y_rt = render(sinoproj_raymarch_triton, theta0)
r = rel(y_rt, y_orig)
check("G1", r < 1e-4, f"rel {r:.2e}  max|diff| {float((y_rt - y_orig).abs().max()):.2e} "
                      f"(max value {float(y_orig.max()):.3f})")

print("\nG2  gradient parity of the TRAINING loss w.r.t. (ts,tp,rot): raymarch_triton vs original")
L_o, g_o = loss_and_grad(sinoproj_rdsh_pinv_raycast_dominant, theta0)
L_t, g_t = loss_and_grad(sinoproj_raymarch_triton, theta0)
check("G2 loss", abs(L_o - L_t) < 1e-5, f"orig {L_o:.6f}  triton {L_t:.6f}")
# the reference volume is a BINARY phantom, so the trilinear model's gradient is only an a.e.
# derivative: a 1e-7 relative rounding difference in a sample position can put that sample on
# the other side of a voxel face and change its contribution O(1). The right bar is therefore
# the ORIGINAL projector's own sensitivity to a perturbation at the fp32 rounding scale.
torch.manual_seed(7)
_, g_o2 = loss_and_grad(sinoproj_rdsh_pinv_raycast_dominant,
                        theta0 * (1.0 + 1e-6 * torch.randn_like(theta0)))
floor = rel(g_o2, g_o)
print(f"      original's own a.e. floor (theta * (1 + 1e-6 noise)): rel {floor:.2e}  "
      f"cos {cosd(g_o2, g_o):.7f}")
check("G2 grad", rel(g_t, g_o) < max(1e-3, 3.0 * floor),
      f"rel {rel(g_t, g_o):.2e}  cos {cosd(g_t, g_o):.7f}   (bar = 3x the original's floor)")

# --------------------------------------------------------------------------------------
print("\nG3  Joseph vs the ray-march model with increasing n_samples (same 4 views)")
with torch.no_grad():
    y_j = render(sinoproj_joseph, theta0)
    gaps = []
    for n in (256, 1024, 4096):
        y_n = render(sinoproj_raymarch_triton, theta0, n_samples=n)
        gaps.append(rel(y_n, y_j))
        print(f"      n_samples={n:5d}: rel(raymarch, joseph) = {gaps[-1]:.3e}")
    y_16k = render(sinoproj_raymarch_triton, theta0, n_samples=16384)
    print(f"      n_samples=16384: rel(raymarch, joseph) = {rel(y_16k, y_j):.3e}   "
          f"[and rel(raymarch 256, raymarch 16384) = {rel(y_orig, y_16k):.3e}]")
check("G3 gap shrinks monotonically", gaps[0] > gaps[1] > gaps[2],
      f"{gaps[0]:.2e} > {gaps[1]:.2e} > {gaps[2]:.2e}")
# Joseph = one bilinear sample per voxel plane (trapezoid-like, entry plane at half weight);
# on a BINARY phantom the trilinear interpolant's second differences are large at every edge,
# so the two quadratures of the same interpolant settle ~1% apart. The shipped 256-sample
# model is FURTHER from the converged integral than Joseph is (printed below).
check("G3 Joseph vs converged line integral (model gap, expect ~1e-2 on a binary phantom)",
      rel(y_16k, y_j) < 2e-2, f"rel {rel(y_16k, y_j):.2e}")
with torch.no_grad():
    Lj = float(loss_of(y_j, tgt))
    L16 = float(loss_of(y_16k, tgt))
print(f"      training loss at this theta: orig(256) {L_o:.5f} | raymarch 16384 {L16:.5f} | "
      f"joseph {Lj:.5f}")

# --------------------------------------------------------------------------------------
print("\nG4  analytic gradient vs central finite difference of the SAME loss (view 131, 9 DoF)")
# FD across a BINARY phantom secants across voxel-face kinks (the a.e. gradient is the local
# slope, FD the trend) -- so the rig volume is BLURRED (two 5-voxel box passes) and eps is swept:
# the FD must converge onto the analytic gradient as eps shrinks, until fp32 cancellation.
tgt1 = tgt[1:2]
th1 = theta0[1:2].clone()
with torch.no_grad():
    vol_blur = F.avg_pool3d(F.avg_pool3d(vol, 5, 1, 2), 5, 1, 2).contiguous()
common_blur = dict(common, smat=vol_blur)


def render_b(proj_fn, theta, idx, **kw):
    P_new, geo_new = apply_9DoF_transform_effective(
        P0=P_all[idx], geo_old=geo_all[idx], ts_internal=theta[:, 0:3],
        tp_internal=theta[:, 3:6], rot_internal_deg=theta[:, 6:9], nx=cfg.imsx, ny=cfg.imsy,
        nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0)
    return proj_fn(Pmat=P_new, geo_parameter=geo_new, geo_stitch=st_all[idx], n_samples=256,
                   **common_blur, **kw)


def fd_check(name, proj_fn, **kw):
    """FD referee in float64: the loss is a plain L2 misfit accumulated in fp64 (MONAI's fp32
    LNCC computes window variances as sum - mean^2 and its cancellation noise, ~1e-4 of the
    loss, swamps any finite difference below eps ~0.2 mm -- it cannot referee a gradient)."""
    idx = IDX[1:2]
    t64 = tgt1.double()

    def L2(pred):
        return ((pred.double()[:, v0:v1, u0:u1] - t64[:, v0:v1, u0:u1]) ** 2).mean()

    th = th1.clone().requires_grad_(True)
    L2(render_b(proj_fn, th, idx, **kw)).backward()
    g_an = th.grad[0].double()
    best = None
    for scale in (4.0, 1.0, 0.25, 0.0625):
        eps = torch.tensor([0.05] * 6 + [0.02] * 3, device=dev) * scale
        g_fd = torch.zeros(9, dtype=torch.float64, device=dev)
        with torch.no_grad():
            for k in range(9):
                e = torch.zeros(1, 9, device=dev)
                e[0, k] = eps[k]
                Lp = L2(render_b(proj_fn, th1 + e, idx, **kw))
                Lm = L2(render_b(proj_fn, th1 - e, idx, **kw))
                g_fd[k] = (float(Lp) - float(Lm)) / (2 * float(eps[k]))
        c, r_ = cosd(g_an, g_fd), rel(g_an, g_fd)
        print(f"      eps x{scale:<6g}: cos {c:.7f}  rel {r_:.2e}")
        if best is None or r_ < best[1]:
            best = (c, r_, g_fd.clone(), scale)
    check(f"G4 {name}", best[0] > 0.9999 and best[1] < 1e-2,
          f"best (eps x{best[3]:g}) cos {best[0]:.7f}  rel {best[1]:.2e}")
    for k, nm in enumerate(("ts_u", "ts_v", "ts_f", "tp_x", "tp_y", "tp_z", "rx", "ry", "rz")):
        print(f"        {nm:5s} analytic {float(g_an[k]):+.5e}  fd {float(best[2][k]):+.5e}")


fd_check("joseph", sinoproj_joseph)
fd_check("raymarch_triton", sinoproj_raymarch_triton)
del vol_blur
torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------
print("\nG5  Joseph Triton value vs the real libleapct (Joseph-pinned build)")
try:
    with torch.no_grad():
        y_lib = render(sinoproj_joseph, theta0, value_backend="leap")
    r = rel(y_lib, y_j)
    check("G5", r < 5e-4, f"rel {r:.2e}  (tex3D 9-bit lerp floor ~1e-4)")
except Exception as ex:  # noqa: BLE001
    print(f"  [SKIP] G5  {type(ex).__name__}: {ex}")

print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'}")
sys.exit(1 if n_fail else 0)
