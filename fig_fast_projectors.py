"""Figure for the projector acceleration: (a) measured view vs the original ray-march render
vs the Joseph render and their difference, (b) 5-epoch smoke loss curves against the original
run's log, (c) training-step time and peak memory. Also smokes `Sample._export_projection_raw`
with the selected projector (2 views) so the export path is exercised.

    CUDA_VISIBLE_DEVICES=1 python fig_fast_projectors.py --smoke_dir <dir with smoke_joseph, smoke_rt>
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AI_Geocal import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from fast_projectors import get_projector
from helpers import load_raw_f32_memmap, reverse_flag
import Sample

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--smoke_dir", default=HERE)
ap.add_argument("--out_dir", default=os.path.join(HERE, "result_denseball", "fastproj_diag_20260921"))
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
dev = torch.device("cuda")

cfg = ReconConfig(NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0, nu=776, nv=1264,
                  ori_nu=776, ori_nv=1264, du=0.228, dv=0.228, imsx=929, imsy=929, imsz=801,
                  dx=0.2, dy=0.2, dz=0.2, ureverse_raw=1, vreverse_raw=1, recon_type=1,
                  n_samples=256, chunk_size=16384)
cfg.X0, cfg.Y0, cfg.Z0 = -0.5 * cfg.imsx * cfg.dx, -0.5 * cfg.imsy * cfg.dy, 0.0
roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)
urev, vrev = reverse_flag(cfg.ureverse_raw), reverse_flag(cfg.vreverse_raw)
VOL = os.path.join(HERE, "open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_"
                         "Ntheta24_zpitch20.00_929x929x801.float32.raw")
vol = torch.from_numpy(np.array(load_raw_f32_memmap(VOL, (cfg.imsz, cfg.imsy, cfg.imsx)))).to(dev)[None, None]
meas = np.array(load_raw_f32_memmap(os.path.join(HERE, "Denseball_proj_480.raw"),
                                    (cfg.NLAM, cfg.nv, cfg.nu))[131])
P_all, geo_all, st_all = build_nominal_orbit_from_geometry(
    n_views=cfg.NLAM, scan_angle_deg=360.0, start_angle_deg=180.0, k_nominal=650.0,
    un_nominal=34.0, vn_nominal=15.0, SOD=443.0, SDD=650.0, nx=cfg.imsx, ny=cfg.imsy,
    nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
    nu_ori=cfg.nu, nv_ori=cfg.nv, du=cfg.du, dv=cfg.dv, orbit_axis="y", clockwise_sign=-1.0,
    include_endpoint=False, use_beamcenter_geo=False, device=dev)
idx = torch.tensor([131, 300], device=dev)
common = dict(smat=vol, Pmat=P_all[idx], geo_parameter=geo_all[idx], geo_stitch=st_all[idx],
              nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv, imsx=cfg.imsx, imsy=cfg.imsy,
              imsz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz, X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
              ureverse=urev, vreverse=vrev, roi=roi, recon_type=cfg.recon_type,
              n_samples=cfg.n_samples, chunk_size=cfg.chunk_size, ori_nu=cfg.ori_nu,
              ori_nv=cfg.ori_nv, align_corners=False)
with torch.no_grad():
    y_o = get_projector("raymarch")(**common)[0].cpu().numpy()
    y_j = get_projector("joseph")(**common)[0].cpu().numpy()
    # export-path smoke: Sample._export_projection_raw with cfg.projector = joseph
    cfg.projector = "joseph"
    tmp = os.path.join(args.out_dir, "export_smoke_2views.raw")
    Sample._export_projection_raw(out_path=tmp, smat=vol, P_all=P_all[idx], geo_all=geo_all[idx],
                                  geo_stitch=st_all[idx], cfg=cfg, roi=roi, ureverse=urev,
                                  vreverse=vrev, proj_batch=2, use_amp_projector=False)
    exp = np.fromfile(tmp, dtype=np.float32).reshape(2, cfg.nv, cfg.nu)
    print(f"Sample export smoke: shape {exp.shape}, rel diff vs direct joseph call "
          f"{np.linalg.norm(exp[0] - y_j) / np.linalg.norm(y_j):.2e}")
    os.remove(tmp)

for nm, a in (("view131_measured", meas), ("view131_raymarch256", y_o), ("view131_joseph", y_j)):
    a.astype(np.float32).tofile(os.path.join(args.out_dir, f"{nm}_{cfg.nv}x{cfg.nu}.raw"))

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
vmax = float(np.percentile(y_o, 99.5))
img_axes = []
for i, (t, a, cm, vm) in enumerate((("measured (view 131)", meas, "gray", None),
                                    ("original ray march, 256 samples", y_o, "gray", vmax),
                                    ("Joseph (LEAP model, Triton)", y_j, "gray", vmax),
                                    ("|ray march 256 - Joseph|", np.abs(y_o - y_j), "magma", 0.1 * vmax))):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(a, cmap=cm, vmin=0, vmax=vm if vm is not None else float(np.percentile(a, 99.5)),
                   aspect="auto")
    ax.set_title(t, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
    img_axes.append(ax)
r = np.linalg.norm(y_o - y_j) / np.linalg.norm(y_j)
img_axes[3].set_xlabel(f"rel L2 {r:.2e} at the nominal orbit\n(the 256-sample march aliases the "
                       f"1.5 mm balls: 5 voxels per step)", fontsize=9)
print(f"view 131 nominal: rel L2 (raymarch256 vs joseph) = {r:.3e}")

# (b) loss curves
ax = fig.add_subplot(gs[1, 0:2])
ref = os.path.join(HERE, "result_denseball", "9DoF_analytic_nominalP_10",
                   "no_initP_k_un_vn_SOD_SDD_vs8", "loss_history.csv")
curves = []
if os.path.exists(ref):
    rows = list(csv.DictReader(open(ref)))
    curves.append(("original ray march (logged run, RTX 3090, 54 s/epoch)",
                   [float(x["loss"]) for x in rows][:12]))
for tag, lab in (("smoke_rt", "raymarch_triton (same model)"), ("smoke_joseph", "joseph")):
    f = os.path.join(args.smoke_dir, tag, "loss_history.csv")
    if os.path.exists(f):
        rows = list(csv.DictReader(open(f)))
        curves.append((f"{lab}, {np.mean([float(x['epoch_time_sec']) for x in rows[1:]]):.1f} s/epoch (A6000)",
                       [float(x["loss"]) for x in rows]))
for lab, c in curves:
    ax.plot(range(1, len(c) + 1), c, marker="o", label=lab)
ax.set_xlabel("epoch (view_step=8, 60 views, batch 4, seed 0)")
ax.set_ylabel("1 + LNCC loss")
ax.set_title("training loss trajectory")
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# (c) bench bars (numbers from bench_projectors.py, A6000)
ax = fig.add_subplot(gs[1, 2:4])
names = ["original\ngrid_sample", "raymarch\ntriton 256", "raymarch\ntriton 1024", "joseph\n(triton)", "joseph\n(libleapct value)"]
t_ms = [1615.3, 293.9, 1007.3, 98.0, 103.0]
mem = [18.86, 8.82, 8.82, 8.13, 8.13]
x = np.arange(len(names))
b1 = ax.bar(x - 0.2, t_ms, 0.4, label="train step, batch of 4 [ms]")
ax2 = ax.twinx()
b2 = ax2.bar(x + 0.2, mem, 0.4, color="tab:orange", label="peak GPU memory [GB]")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("ms per training step (4 views)"); ax2.set_ylabel("peak GB")
for xi, t in zip(x, t_ms):
    ax.text(xi - 0.2, t, f"{t:.0f}\n({t_ms[0] / t:.1f}x)", ha="center", va="bottom", fontsize=8)
ax.set_title("full-size 4T (929^3 volume, 776x1264 panel), RTX A6000")
ax.legend(handles=[b1, b2], fontsize=9, loc="upper right")
fig.suptitle("AI-Geocal projector acceleration: same-model Triton ray march and LEAP-Joseph model", fontsize=13)
fig.tight_layout()
out = os.path.join(args.out_dir, "fast_projectors_summary.png")
fig.savefig(out, dpi=110)
print("saved", out)
