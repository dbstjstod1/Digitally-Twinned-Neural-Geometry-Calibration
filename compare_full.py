"""
Full MLP vs direct per-view 9-DoF comparison over ALL views.

Reads the merged direct results and the MLP exports, recomputes MLP per-view
loss (same loss/ROI/projector), and writes:
  comparison_summary.txt
  comparison_plots/convergence.png       (MLP per-epoch vs direct per-iter)
  comparison_plots/per_view_loss.png     (final loss vs view + histogram)
  comparison_plots/motion_params.png     (ts/tp/rot per view, MLP vs direct)
into the direct output directory.
"""
import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.losses import LocalNormalizedCrossCorrelationLoss

from AI_Geocal import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from helpers import load_raw_f32_memmap, reverse_flag
from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant
from models.MotionNetHash import MotionNetHash_9DoF

BASE = "result_denseball/9DoF_analytic_nominalP_10"
MLP_DIR = f"{BASE}/no_initP_k_un_vn_SOD_SDD_vs1"
DIRECT_DIR = f"{BASE}/direct_param_vs1"
CKPT = f"{MLP_DIR}/motion_model_ep0100.pth"
VOLUME = "./open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw"
PROJ = "./Denseball_proj_480.raw"
OUTP = f"{DIRECT_DIR}/comparison_plots"


def read_mlp_epoch_curve():
    ep, loss = [], []
    with open(f"{MLP_DIR}/loss_history.csv") as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); loss.append(float(r["loss"]))
    return np.array(ep), np.array(loss)


def main():
    os.makedirs(OUTP, exist_ok=True)
    device = torch.device("cuda")

    cfg = ReconConfig(
        NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0,
        nu=776, nv=1264, ori_nu=776, ori_nv=1264, du=0.228, dv=0.228,
        imsx=929, imsy=929, imsz=801, dx=0.2, dy=0.2, dz=0.2,
        ureverse_raw=1, vreverse_raw=1, recon_type=1, n_samples=256, chunk_size=16384,
    )
    cfg.X0 = -0.5 * cfg.imsx * cfg.dx; cfg.Y0 = -0.5 * cfg.imsy * cfg.dy; cfg.Z0 = 0.0
    roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)
    V = cfg.NLAM

    # ---- load direct merged results ----
    direct_curves = np.load(f"{DIRECT_DIR}/loss_curves.npy")          # (V, n_iters)
    direct_final = direct_curves[:, -1]
    direct_mean_iter = np.load(f"{DIRECT_DIR}/mean_loss_vs_iter.npy")  # (n_iters,)
    d_ts = np.load(f"{DIRECT_DIR}/motion_ts_mm.npy")
    d_tp = np.load(f"{DIRECT_DIR}/motion_tp_mm.npy")
    d_rot = np.load(f"{DIRECT_DIR}/motion_rot_deg.npy")

    m_ts = np.load(f"{MLP_DIR}/motion_ts_mm.npy")
    m_tp = np.load(f"{MLP_DIR}/motion_tp_mm.npy")
    m_rot = np.load(f"{MLP_DIR}/motion_rot_deg.npy")
    mlp_ep, mlp_curve = read_mlp_epoch_curve()

    # ---- recompute MLP per-view loss (same metric) ----
    vol = torch.from_numpy(np.asarray(
        load_raw_f32_memmap(VOLUME, (cfg.imsz, cfg.imsy, cfg.imsx))).copy()).to(device, torch.float32)
    smat = vol[None, None]
    proj_mm = load_raw_f32_memmap(PROJ, (cfg.NLAM, cfg.nv, cfg.nu))
    urev, vrev = reverse_flag(cfg.ureverse_raw), reverse_flag(cfg.vreverse_raw)
    P_nom, geo_nom, st = build_nominal_orbit_from_geometry(
        n_views=V, scan_angle_deg=cfg.ScanAngle_deg, start_angle_deg=cfg.StartAngle_deg,
        k_nominal=650.0, un_nominal=34.0, vn_nominal=15.0, SOD=443.0, SDD=650.0,
        nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
        X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, nu_ori=cfg.ori_nu, nv_ori=cfg.ori_nv,
        du=cfg.du, dv=cfg.dv, orbit_axis="y", clockwise_sign=-1.0,
        include_endpoint=False, use_beamcenter_geo=False, device=device, dtype=torch.float32)
    lncc = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=2, kernel_size=31, kernel_type="rectangular", reduction="mean").to(device)
    u0, u1, v0, v1 = 26, cfg.nu, 0, 1182

    model = MotionNetHash_9DoF(n_views=V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)["model_state"])
    model.eval()
    idx = torch.arange(V, device=device)
    with torch.no_grad():
        p9 = model(idx)
        ts, tp, rot, _ = motion9_to_ts_tp_rot(p9, ts_max_mm=10.0, tp_max_mm=10.0, rot_max_deg=10.0)

    mlp_final = np.zeros(V, dtype=np.float64)
    with torch.no_grad():
        for b in range(0, V, 8):
            sl = slice(b, min(b + 8, V))
            P_new, geo_new = apply_9DoF_transform_effective(
                P0=P_nom[sl], geo_old=geo_nom[sl], ts_internal=ts[sl],
                tp_internal=tp[sl], rot_internal_deg=rot[sl],
                nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, use_inverse_right_multiply=0)
            pred = sinoproj_rdsh_pinv_raycast_dominant(
                smat=smat, Pmat=P_new, geo_parameter=geo_new, geo_stitch=st[sl],
                nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv,
                imsx=cfg.imsx, imsy=cfg.imsy, imsz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, ureverse=urev, vreverse=vrev, roi=roi,
                recon_type=cfg.recon_type, n_samples=cfg.n_samples, chunk_size=cfg.chunk_size,
                ori_nu=cfg.ori_nu, ori_nv=cfg.ori_nv, align_corners=False)
            tgt = torch.from_numpy(np.asarray(proj_mm[b:min(b+8, V)]).copy()).to(device, torch.float32)
            for k in range(pred.shape[0]):
                mlp_final[b + k] = 1.0 + float(lncc(
                    pred[k:k+1].unsqueeze(1)[:, :, v0:v1, u0:u1].float(),
                    tgt[k:k+1].unsqueeze(1)[:, :, v0:v1, u0:u1].float()))
            print(f"  MLP render {min(b+8, V)}/{V}", flush=True)

    # ---- summary ----
    stuck = 0.72
    lines = []
    lines.append(f"Comparison over all {V} views (identical loss / ROI / projector)\n")
    lines.append(f"{'':14}{'MLP':>12}{'direct':>12}")
    lines.append(f"{'mean loss':14}{mlp_final.mean():12.5f}{direct_final.mean():12.5f}")
    lines.append(f"{'std loss':14}{mlp_final.std():12.5f}{direct_final.std():12.5f}")
    lines.append(f"{'median loss':14}{np.median(mlp_final):12.5f}{np.median(direct_final):12.5f}")
    lines.append(f"{'max loss':14}{mlp_final.max():12.5f}{direct_final.max():12.5f}")
    lines.append(f"{'views >'+str(stuck):14}{int((mlp_final>stuck).sum()):12d}{int((direct_final>stuck).sum()):12d}")
    lines.append(f"\ndirect better on {(direct_final<mlp_final).sum()}/{V} views; "
                 f"MLP better on {(mlp_final<direct_final).sum()}/{V}")
    # timing
    try:
        with open(f"{DIRECT_DIR}/training_time.txt") as f:
            lines.append("\n[direct timing]\n" + f.read())
    except FileNotFoundError:
        pass
    try:
        with open(f"{MLP_DIR}/training_time.txt") as f:
            lines.append("[MLP timing]\n" + f.read())
    except FileNotFoundError:
        pass
    summary = "\n".join(lines)
    print(summary, flush=True)
    with open(f"{DIRECT_DIR}/comparison_summary.txt", "w") as f:
        f.write(summary + "\n")
    np.save(f"{DIRECT_DIR}/mlp_per_view_loss.npy", mlp_final)

    # ---- plots ----
    # 1) convergence (separate x-axes via normalized progress)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(mlp_ep, mlp_curve, label="MLP (per-epoch mean)", color="C0")
    ax[0].plot(np.linspace(1, mlp_ep[-1], len(direct_mean_iter)), direct_mean_iter,
               label="direct (per-iter mean)", color="C1")
    ax[0].set_xlabel("optimization progress (rescaled)"); ax[0].set_ylabel("1 + LNCC loss")
    ax[0].set_title("Convergence"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(direct_mean_iter, color="C1"); ax[1].set_xlabel("per-view iteration")
    ax[1].set_ylabel("mean 1+LNCC"); ax[1].set_title("direct: mean loss vs per-view iter")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUTP}/convergence.png", dpi=130); plt.close(fig)

    # 2) per-view final loss + histogram
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(mlp_final, label=f"MLP (mean {mlp_final.mean():.4f})", color="C0", lw=0.9)
    ax[0].plot(direct_final, label=f"direct (mean {direct_final.mean():.4f})", color="C1", lw=0.9)
    ax[0].axhline(stuck, color="r", ls="--", lw=0.7, label=f"stuck thr {stuck}")
    ax[0].set_xlabel("view index"); ax[0].set_ylabel("final 1+LNCC")
    ax[0].set_title("Per-view final loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    bins = np.linspace(min(mlp_final.min(), direct_final.min()),
                       max(mlp_final.max(), direct_final.max()), 40)
    ax[1].hist(mlp_final, bins=bins, alpha=0.6, label="MLP", color="C0")
    ax[1].hist(direct_final, bins=bins, alpha=0.6, label="direct", color="C1")
    ax[1].set_xlabel("final 1+LNCC"); ax[1].set_ylabel("# views")
    ax[1].set_title("Final-loss distribution"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUTP}/per_view_loss.png", dpi=130); plt.close(fig)

    # 3) motion params
    names = ["ts (mm)", "tp (mm)", "rot (deg)"]
    comps = ["x/u", "y/v", "z/f"]
    arrs_m = [m_ts, m_tp, m_rot]; arrs_d = [d_ts, d_tp, d_rot]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for r in range(3):
        for c in range(3):
            a = axes[r, c]
            a.plot(arrs_m[r][:, c], color="C0", lw=0.8, label="MLP")
            a.plot(arrs_d[r][:, c], color="C1", lw=0.8, label="direct")
            a.set_title(f"{names[r]} [{comps[c]}]"); a.grid(alpha=0.3)
            if r == 2: a.set_xlabel("view index")
            if r == 0 and c == 0: a.legend(fontsize=8)
    fig.suptitle("Recovered 9-DoF motion parameters: MLP vs direct")
    fig.tight_layout(); fig.savefig(f"{OUTP}/motion_params.png", dpi=130); plt.close(fig)

    print(f"\n[plots] saved to {OUTP}/", flush=True)


if __name__ == "__main__":
    main()
