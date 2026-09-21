"""A/B of the fast projectors against the ORIGINAL training run: same recipe (4T Denseball,
view_step=8, 100 epochs, batch 4, lr 1e-3, seed 0), three solutions

    orig_vs8        original grid_sample ray march (logged run, RTX 3090)
    joseph          fast_projectors.sinoproj_joseph
    raymarch_triton fast_projectors.sinoproj_raymarch_triton (same model as the original)

plus orig_vs1 (the original run on all 480 views) as the best available reference, and the
nominal orbit (no correction) as the floor. Reports:

  1. recovered 9-DoF curves: RMS difference per DoF between solutions, against the yardstick
     "orig_vs8 vs orig_vs1" (the method's own spread across view_step)
  2. every solution re-scored on ALL 480 views with the ORIGINAL projector AND with joseph
     (per-view 1 + LNCC on the training ROI against the measured projection): mean / median /
     max and the number of 'stuck' views (> 0.72) -- projector-independent evidence
  3. loss history and epoch time

    CUDA_VISIBLE_DEVICES=1 python compare_ab_projectors.py
"""
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AI_Geocal import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from DoF_transform import apply_9DoF_transform_effective
from fast_projectors import get_projector
from helpers import load_raw_f32_memmap, reverse_flag
from monai.losses import LocalNormalizedCrossCorrelationLoss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result_denseball", "9DoF_analytic_nominalP_10")
RUNS = {
    "orig_vs8": os.path.join(RES, "no_initP_k_un_vn_SOD_SDD_vs8"),
    "orig_vs1": os.path.join(RES, "no_initP_k_un_vn_SOD_SDD_vs1"),
    "joseph": os.path.join(RES, "fastproj_joseph_vs8_ep100"),
    "raymarch_triton": os.path.join(RES, "fastproj_raymarch_triton_vs8_ep100"),
}
OUT = os.path.join(HERE, "result_denseball", "fastproj_diag_20260921")
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
DOF = ["ts_u", "ts_v", "ts_f", "tp_x", "tp_y", "tp_z", "rx", "ry", "rz"]


def load_motion(d):
    return np.concatenate([np.load(os.path.join(d, "motion_ts_mm.npy")),
                           np.load(os.path.join(d, "motion_tp_mm.npy")),
                           np.load(os.path.join(d, "motion_rot_deg.npy"))], 1)   # (480, 9)


motion = {k: load_motion(v) for k, v in RUNS.items() if os.path.exists(v)}
motion["nominal"] = np.zeros_like(motion["orig_vs8"])
hist = {k: list(csv.DictReader(open(os.path.join(v, "loss_history.csv"))))
        for k, v in RUNS.items() if os.path.exists(v)}

print("1. recovered 9-DoF curves: RMS difference over 480 views (mm / deg)")
pairs = [("orig_vs8", "orig_vs1"), ("joseph", "orig_vs8"), ("joseph", "orig_vs1"),
         ("raymarch_triton", "orig_vs8"), ("raymarch_triton", "orig_vs1"), ("joseph", "raymarch_triton")]
print("   pair                              " + "  ".join(f"{d:>6s}" for d in DOF))
for a, b in pairs:
    if a in motion and b in motion:
        rms = np.sqrt(((motion[a] - motion[b]) ** 2).mean(0))
        print(f"   {a:>15s} vs {b:<15s}  " + "  ".join(f"{x:6.3f}" for x in rms))
print("   amplitude (RMS of orig_vs1 itself)   " + "  ".join(f"{x:6.3f}" for x in np.sqrt((motion['orig_vs1'] ** 2).mean(0))))

# ---------------------------------------------------------------------------- re-score
cfg = ReconConfig(NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0, nu=776, nv=1264,
                  ori_nu=776, ori_nv=1264, du=0.228, dv=0.228, imsx=929, imsy=929, imsz=801,
                  dx=0.2, dy=0.2, dz=0.2, ureverse_raw=1, vreverse_raw=1, recon_type=1,
                  n_samples=256, chunk_size=16384)
cfg.X0, cfg.Y0, cfg.Z0 = -0.5 * cfg.imsx * cfg.dx, -0.5 * cfg.imsy * cfg.dy, 0.0
roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)
urev, vrev = reverse_flag(cfg.ureverse_raw), reverse_flag(cfg.vreverse_raw)
vol = torch.from_numpy(np.array(load_raw_f32_memmap(
    os.path.join(HERE, "open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_"
                       "Ntheta24_zpitch20.00_929x929x801.float32.raw"),
    (cfg.imsz, cfg.imsy, cfg.imsx)))).to(dev)[None, None]
proj_mm = load_raw_f32_memmap(os.path.join(HERE, "Denseball_proj_480.raw"), (cfg.NLAM, cfg.nv, cfg.nu))
P_all, geo_all, st_all = build_nominal_orbit_from_geometry(
    n_views=cfg.NLAM, scan_angle_deg=360.0, start_angle_deg=180.0, k_nominal=650.0,
    un_nominal=34.0, vn_nominal=15.0, SOD=443.0, SDD=650.0, nx=cfg.imsx, ny=cfg.imsy,
    nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
    nu_ori=cfg.nu, nv_ori=cfg.nv, du=cfg.du, dv=cfg.dv, orbit_axis="y", clockwise_sign=-1.0,
    include_endpoint=False, use_beamcenter_geo=False, device=dev)
lncc = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=31, kernel_type="rectangular",
                                           reduction="none").to(dev)
u0, u1, v0, v1 = 26, cfg.nu, 0, 1182
common = dict(smat=vol, nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv, imsx=cfg.imsx, imsy=cfg.imsy,
              imsz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
              ureverse=urev, vreverse=vrev, roi=roi, recon_type=cfg.recon_type,
              n_samples=cfg.n_samples, chunk_size=cfg.chunk_size, ori_nu=cfg.ori_nu,
              ori_nv=cfg.ori_nv, align_corners=False)


@torch.no_grad()
def score_all(proj_fn, theta_np, bs=8):
    theta = torch.from_numpy(theta_np).to(dev, torch.float32)
    out = np.zeros(cfg.NLAM, dtype=np.float64)
    for i0 in range(0, cfg.NLAM, bs):
        idx = torch.arange(i0, min(i0 + bs, cfg.NLAM), device=dev)
        P_new, geo_new = apply_9DoF_transform_effective(
            P0=P_all[idx], geo_old=geo_all[idx], ts_internal=theta[idx, 0:3],
            tp_internal=theta[idx, 3:6], rot_internal_deg=theta[idx, 6:9], nx=cfg.imsx,
            ny=cfg.imsy, nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0)
        pred = proj_fn(Pmat=P_new, geo_parameter=geo_new, geo_stitch=st_all[idx], **common)
        tgt = torch.from_numpy(np.array(proj_mm[i0:i0 + len(idx)])).to(dev)
        L = 1.0 + lncc(pred[:, None, v0:v1, u0:u1].float(), tgt[:, None, v0:v1, u0:u1].float())
        out[i0:i0 + len(idx)] = L.reshape(len(idx), -1).mean(1).double().cpu().numpy()
    return out


print("\n2. per-view loss (1 + LNCC, training ROI) on ALL 480 views, re-scored with a fixed projector")
scores = {}
for pj in ("raymarch", "joseph"):
    fn = get_projector(pj)
    for name, th in motion.items():
        t0 = time.perf_counter()
        s = score_all(fn, th)
        scores[(pj, name)] = s
        print(f"   scored by {pj:9s} | {name:16s} mean {s.mean():.5f}  median {np.median(s):.5f}  "
              f"max {s.max():.5f}  stuck(>0.72) {(s > 0.72).sum():3d}   [{time.perf_counter() - t0:5.1f} s]")
np.save(os.path.join(OUT, "rescore_per_view.npy"), {f"{a}|{b}": v for (a, b), v in scores.items()},
        allow_pickle=True)

print("\n3. training history (loss = mean over the epoch's 15 batches)")
for k, h in hist.items():
    last = np.mean([float(r["loss"]) for r in h[-5:]])
    ep_t = np.median([float(r["epoch_time_sec"]) for r in h[1:]])
    print(f"   {k:16s} epochs {len(h):3d}  final loss (mean last 5) {last:.5f}  "
          f"epoch time {ep_t:6.1f} s  total {float(h[-1]['elapsed_time_sec']) / 60:6.1f} min")

# ---------------------------------------------------------------------------- figure
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
cols = {"orig_vs8": "k", "orig_vs1": "gray", "joseph": "tab:green", "raymarch_triton": "tab:orange"}
for k in range(9):
    ax = axes[k // 4, k % 4]
    for name in ("orig_vs1", "orig_vs8", "raymarch_triton", "joseph"):
        if name in motion:
            ax.plot(motion[name][:, k], color=cols[name], lw=1.0, alpha=0.9 if name != "orig_vs1" else 0.5,
                    label=name)
    ax.set_title(DOF[k] + (" [mm]" if k < 6 else " [deg]"))
    ax.grid(alpha=0.3)
    if k == 0:
        ax.legend(fontsize=8)
ax = axes[2, 1]
for name in ("orig_vs8", "raymarch_triton", "joseph"):
    if name in hist:
        ax.plot([float(r["loss"]) for r in hist[name]], color=cols[name], label=name)
ax.set_title("training loss vs epoch (view_step 8)"); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
for j, pj in enumerate(("raymarch", "joseph")):
    ax = axes[2, 2 + j]
    for name in ("nominal", "orig_vs8", "raymarch_triton", "joseph"):
        s = scores[(pj, name)]
        ax.plot(s, lw=0.8, color=cols.get(name, "tab:red"), label=f"{name} (mean {s.mean():.4f})")
    ax.set_title(f"per-view loss, all 480 views, scored by {pj}", fontsize=10)
    ax.set_ylim(0.55, 0.95); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel("view")
fig.suptitle("AI-Geocal: recovered 9-DoF and re-scored loss -- original ray march vs Triton ray march vs Joseph "
             "(4T Denseball, view_step 8, 100 epochs)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ab_projectors_motion_and_loss.png"), dpi=100)
print("saved", os.path.join(OUT, "ab_projectors_motion_and_loss.png"))
