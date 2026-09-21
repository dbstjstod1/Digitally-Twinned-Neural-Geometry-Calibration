"""Benchmark the projectors at the shipped 4T Denseball size (929^3 @ 0.2 mm, 776 x 1264 panel).

Per projector: forward only (1 view), forward + backward (1 view), forward + backward
(batch of 4, the training batch), and a FULL training step on a batch of 4 (nominal orbit ->
9-DoF transform -> projection -> LNCC loss on the training ROI -> backward -> Adam step on a
9-DoF leaf), with peak GPU memory. Medians of `reps` timed runs after one warm-up.

    CUDA_VISIBLE_DEVICES=1 python bench_projectors.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AI_Geocal import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from DoF_transform import apply_9DoF_transform_effective
from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant
from fast_projectors import sinoproj_raymarch_triton, sinoproj_joseph
from helpers import load_raw_f32_memmap, reverse_flag
from monai.losses import LocalNormalizedCrossCorrelationLoss

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = os.path.join(HERE, "open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_"
                         "Ntheta24_zpitch20.00_929x929x801.float32.raw")
PROJ = os.path.join(HERE, "Denseball_proj_480.raw")
dev = torch.device("cuda")
reps = 3

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
IDX4 = torch.tensor([3, 131, 277, 402], device=dev)
tgt4 = torch.from_numpy(np.array(proj_mm[IDX4.cpu().numpy()])).to(dev)


def render(proj_fn, theta, idx, n_samples, **kw):
    P_new, geo_new = apply_9DoF_transform_effective(
        P0=P_all[idx], geo_old=geo_all[idx], ts_internal=theta[:, 0:3],
        tp_internal=theta[:, 3:6], rot_internal_deg=theta[:, 6:9], nx=cfg.imsx, ny=cfg.imsy,
        nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0)
    return proj_fn(Pmat=P_new, geo_parameter=geo_new, geo_stitch=st_all[idx],
                   n_samples=n_samples, **common, **kw)


def timed(fn):
    fn()                                    # warm-up (Triton JIT, allocator)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), torch.cuda.max_memory_allocated() / 2 ** 30


def bench(name, proj_fn, n_samples=256, **kw):
    theta1 = torch.zeros(1, 9, device=dev)
    theta4 = torch.zeros(4, 9, device=dev)

    def fwd1():
        with torch.no_grad():
            render(proj_fn, theta1, IDX4[:1], n_samples, **kw)

    def fb1():
        th = theta1.clone().requires_grad_(True)
        y = render(proj_fn, th, IDX4[:1], n_samples, **kw)
        (y * y).mean().backward()

    def fb4():
        th = theta4.clone().requires_grad_(True)
        y = render(proj_fn, th, IDX4, n_samples, **kw)
        (y * y).mean().backward()

    leaf = torch.zeros(4, 9, device=dev, requires_grad=True)
    opt = torch.optim.Adam([leaf], lr=1e-3)

    def step4():
        opt.zero_grad(set_to_none=True)
        y = render(proj_fn, leaf, IDX4, n_samples, **kw)
        L = 1.0 + lncc(y[:, None, v0:v1, u0:u1].float(), tgt4[:, None, v0:v1, u0:u1].float())
        L.backward()
        opt.step()

    base = torch.cuda.memory_allocated() / 2 ** 30
    t_f1, m_f1 = timed(fwd1)
    t_b1, m_b1 = timed(fb1)
    t_b4, m_b4 = timed(fb4)
    t_s4, m_s4 = timed(step4)
    print(f"{name:28s} fwd 1v {t_f1 * 1e3:8.1f} ms | fwd+bwd 1v {t_b1 * 1e3:8.1f} ms "
          f"(peak {m_b1:5.2f} GB) | fwd+bwd 4v {t_b4 * 1e3:8.1f} ms (peak {m_b4:5.2f} GB) | "
          f"TRAIN STEP 4v {t_s4 * 1e3:8.1f} ms (peak {m_s4:5.2f} GB)   "
          f"[resident before: {base:.2f} GB]")
    return t_s4


print(f"GPU: {torch.cuda.get_device_name(0)}   volume {tuple(vol.shape[2:])}  panel "
      f"{cfg.nv}x{cfg.nu}  n_samples(raymarch)={cfg.n_samples}\n")
t = {}
t["orig"] = bench("original grid_sample (256)", sinoproj_rdsh_pinv_raycast_dominant)
t["rt256"] = bench("raymarch_triton (256)", sinoproj_raymarch_triton)
t["rt1024"] = bench("raymarch_triton (1024)", sinoproj_raymarch_triton, n_samples=1024)
t["joseph"] = bench("joseph (triton value)", sinoproj_joseph)
try:
    t["joseph_leap"] = bench("joseph (libleapct value)", sinoproj_joseph, value_backend="leap")
except Exception as ex:  # noqa: BLE001
    print(f"joseph (libleapct value): SKIP {type(ex).__name__}: {ex}")

print("\nspeed-up of the full training step (batch of 4) over the original:")
for k, v in t.items():
    print(f"   {k:12s} {t['orig'] / v:6.2f}x   -> 100 epochs x 120 batches = "
          f"{100 * 120 * v / 3600:5.2f} h")
