"""
Direct per-view 9-DoF calibration (baseline for comparison with the MLP /
hash-encoded MotionNet in AI_Geocal.py).

Key differences vs AI_Geocal.train_motion_hash_model:
  * NO neural network. The 9-DoF parameters are stored DIRECTLY as
    nn.Parameter(1, 9) and updated by gradient descent.
  * NOT a stochastic mini-batch update over shuffled views. Instead each view
    is solved INDEPENDENTLY ("view by view"): for view v we spin up its own
    Adam optimizer and iterate n_iters times, matching ONLY that view's
    measured projection. Views do not share any parameters.

Everything else (analytic nominal orbit, 9-DoF effective transform,
differentiable ray-march projector, LNCC loss, bounds) is identical to
AI_Geocal so the comparison is apples-to-apples.

Loss history and timing are recorded in the same spirit as AI_Geocal:
  loss_history.csv / .npy   per-view final loss + per-view/elapsed time
  loss_curves.npy           (V, n_iters) full per-iteration loss curve
  mean_loss_vs_iter.npy     (n_iters,) mean loss across views vs iteration
                            (directly comparable to the MLP per-epoch curve)
  training_time.txt         total wall-clock summary
  motion_p9_raw.npy / motion_ts_mm.npy / motion_tp_mm.npy / motion_rot_deg.npy
  P_nominal_analytic.npy / geo_nominal_analytic.npy

This script processes a contiguous view range [view_start, view_stop) so that
several instances can split the work across GPUs; merge_direct_param() then
stitches the shards back together.
"""

import os
import csv
import time
import numpy as np
import torch

from monai.losses import LocalNormalizedCrossCorrelationLoss

# Reuse the exact building blocks from the MLP pipeline.
from AI_Geocal import (
    ReconConfig,
    RT_PARAM,
    build_nominal_orbit_from_geometry,
)
from helpers import load_raw_f32_memmap, reverse_flag
from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant  # noqa: F401
from fast_projectors import get_projector


def _build_nominal(cfg, *, k_nominal, un_nominal, vn_nominal, SOD, SDD,
                   nominal_orbit_axis, nominal_clockwise_sign,
                   nominal_include_endpoint, use_beamcenter_geo, device):
    cfg.X0 = -0.5 * cfg.imsx * cfg.dx + cfg.x_pos
    cfg.Y0 = -0.5 * cfg.imsy * cfg.dy + cfg.y_pos
    cfg.Z0 = 0.0

    nu_ori_use = int(cfg.ori_nu) if (cfg.ori_nu is not None) else int(cfg.nu)
    nv_ori_use = int(cfg.ori_nv) if (cfg.ori_nv is not None) else int(cfg.nv)

    return build_nominal_orbit_from_geometry(
        n_views=cfg.NLAM,
        scan_angle_deg=float(cfg.ScanAngle_deg),
        start_angle_deg=float(cfg.StartAngle_deg),
        k_nominal=float(k_nominal),
        un_nominal=float(un_nominal),
        vn_nominal=float(vn_nominal),
        SOD=float(SOD),
        SDD=float(SDD),
        nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz,
        dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
        X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
        nu_ori=nu_ori_use, nv_ori=nv_ori_use,
        du=float(cfg.du), dv=float(cfg.dv),
        orbit_axis=nominal_orbit_axis,
        clockwise_sign=nominal_clockwise_sign,
        include_endpoint=nominal_include_endpoint,
        use_beamcenter_geo=use_beamcenter_geo,
        device=device,
        dtype=torch.float32,
    )


def train_direct_param_model(
    *,
    cfg: ReconConfig,
    volume_path: str,
    proj_meas_path: str,
    roi: RT_PARAM,
    out_dir: str,
    n_iters: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    ts_max_mm: float = 10.0,
    tp_max_mm: float = 10.0,
    rot_max_deg: float = 10.0,
    view_step: int = 1,
    view_start: int = 0,
    view_stop: int = None,
    # Nominal analytic geometry (must match the MLP run for comparison).
    k_nominal: float = 650.0,
    un_nominal: float = 34.0,
    vn_nominal: float = 15.0,
    SOD: float = 443.0,
    SDD: float = 650.0,
    nominal_orbit_axis: str = "y",
    nominal_clockwise_sign: float = -1.0,
    nominal_include_endpoint: bool = False,
    use_beamcenter_geo: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load volume + measured projections ----
    vol_mm = load_raw_f32_memmap(volume_path, (cfg.imsz, cfg.imsy, cfg.imsx))
    volume = torch.from_numpy(np.asarray(vol_mm).copy()).to(device=device, dtype=torch.float32)
    smat = volume[None, None]

    proj_mm = load_raw_f32_memmap(proj_meas_path, (cfg.NLAM, cfg.nv, cfg.nu))

    urev = reverse_flag(cfg.ureverse_raw)
    vrev = reverse_flag(cfg.vreverse_raw)

    # ---- Analytic nominal baseline (identical to MLP run) ----
    P_nominal_all, geo_nominal_all, geo_stitch = _build_nominal(
        cfg,
        k_nominal=k_nominal, un_nominal=un_nominal, vn_nominal=vn_nominal,
        SOD=SOD, SDD=SDD,
        nominal_orbit_axis=nominal_orbit_axis,
        nominal_clockwise_sign=nominal_clockwise_sign,
        nominal_include_endpoint=nominal_include_endpoint,
        use_beamcenter_geo=use_beamcenter_geo,
        device=device,
    )

    np.save(os.path.join(out_dir, "P_nominal_analytic.npy"),
            P_nominal_all.detach().cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "geo_nominal_analytic.npy"),
            geo_nominal_all.detach().cpu().numpy().astype(np.float32))

    # ---- Loss ----
    lncc = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=2, kernel_size=31, kernel_type="rectangular", reduction="mean",
    ).to(device)

    V = cfg.NLAM
    if view_stop is None:
        view_stop = V
    view_stop = min(int(view_stop), V)
    all_views = list(range(0, V, max(int(view_step), 1)))
    view_indices = [v for v in all_views if view_start <= v < view_stop]

    # ROI for similarity loss (same as AI_Geocal).
    u0, u1 = 26, cfg.nu
    v0, v1 = 0, 1182
    u0 = max(0, min(u0, cfg.nu)); u1 = max(0, min(u1, cfg.nu))
    v0 = max(0, min(v0, cfg.nv)); v1 = max(0, min(v1, cfg.nv))

    print(f"[direct] views {view_indices[0]}..{view_indices[-1]} "
          f"({len(view_indices)} views), n_iters={n_iters}, lr={lr}", flush=True)
    print(f"[ROI] u={u0}:{u1}, v={v0}:{v1}", flush=True)

    n_sel = len(view_indices)
    p9_raw_out = np.zeros((n_sel, 9), dtype=np.float32)
    ts_out = np.zeros((n_sel, 3), dtype=np.float32)
    tp_out = np.zeros((n_sel, 3), dtype=np.float32)
    rot_out = np.zeros((n_sel, 3), dtype=np.float32)
    loss_curves = np.zeros((n_sel, n_iters), dtype=np.float32)
    view_time = np.zeros((n_sel,), dtype=np.float64)
    final_loss = np.zeros((n_sel,), dtype=np.float64)
    init_loss = np.zeros((n_sel,), dtype=np.float64)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for j, v in enumerate(view_indices):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_view = time.perf_counter()

        # Per-view independent 9-DoF parameter (raw, pre-tanh). Init 0 -> nominal.
        p9 = torch.nn.Parameter(torch.zeros(1, 9, device=device, dtype=torch.float32))
        opt = torch.optim.Adam([p9], lr=lr)

        P0 = P_nominal_all[v:v + 1]
        geo_b = geo_nominal_all[v:v + 1]
        st_b = geo_stitch[v:v + 1]

        tgt_np = np.asarray(proj_mm[v:v + 1]).copy()
        tgt = torch.from_numpy(tgt_np).to(device=device, dtype=torch.float32)
        tgt_4d = tgt.unsqueeze(1)[:, :, v0:v1, u0:u1].float()

        for it in range(n_iters):
            ts, tp, rot_deg, _ = motion9_to_ts_tp_rot(
                p9, ts_max_mm=ts_max_mm, tp_max_mm=tp_max_mm, rot_max_deg=rot_max_deg,
            )
            P_new, geo_new = apply_9DoF_transform_effective(
                P0=P0, geo_old=geo_b,
                ts_internal=ts, tp_internal=tp, rot_internal_deg=rot_deg,
                nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz,
                dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
                use_inverse_right_multiply=0,
            )
            pred = get_projector(cfg.projector)(
                smat=smat, Pmat=P_new, geo_parameter=geo_new, geo_stitch=st_b,
                nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv,
                imsx=cfg.imsx, imsy=cfg.imsy, imsz=cfg.imsz,
                dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0,
                ureverse=urev, vreverse=vrev, roi=roi, recon_type=cfg.recon_type,
                n_samples=cfg.n_samples, chunk_size=cfg.chunk_size,
                ori_nu=cfg.ori_nu, ori_nv=cfg.ori_nv, align_corners=False,
            )
            pred_4d = pred.unsqueeze(1)[:, :, v0:v1, u0:u1].float()
            loss = 1.0 + lncc(pred_4d, tgt_4d)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            lv = float(loss.item())
            loss_curves[j, it] = lv
            if it == 0:
                init_loss[j] = lv

        final_loss[j] = lv

        with torch.no_grad():
            ts, tp, rot_deg, raw = motion9_to_ts_tp_rot(
                p9, ts_max_mm=ts_max_mm, tp_max_mm=tp_max_mm, rot_max_deg=rot_max_deg,
            )
        p9_raw_out[j] = raw.detach().cpu().numpy().astype(np.float32)[0]
        ts_out[j] = ts.detach().cpu().numpy().astype(np.float32)[0]
        tp_out[j] = tp.detach().cpu().numpy().astype(np.float32)[0]
        rot_out[j] = rot_deg.detach().cpu().numpy().astype(np.float32)[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        view_time[j] = time.perf_counter() - t_view
        elapsed = time.perf_counter() - t_start

        print(f"[view {v:04d}] init={init_loss[j]:.6e} -> final={final_loss[j]:.6e} "
              f"| view_time={view_time[j]:.2f}s | elapsed={elapsed/60.0:.2f}min", flush=True)

    total_time = time.perf_counter() - t_start

    # ---- Save shard outputs ----
    shard = {
        "view_idx": np.asarray(view_indices, dtype=np.int64),
        "p9_raw": p9_raw_out,
        "ts_mm": ts_out,
        "tp_mm": tp_out,
        "rot_deg": rot_out,
        "loss_curves": loss_curves,
        "view_time_sec": view_time,
        "final_loss": final_loss,
        "init_loss": init_loss,
        "n_iters": np.int64(n_iters),
        "total_time_sec": np.float64(total_time),
    }
    shard_path = os.path.join(out_dir, f"shard_{view_start:04d}_{view_stop:04d}.npz")
    np.savez(shard_path, **shard)
    print(f"[direct] saved shard: {shard_path}  total_time={total_time/60.0:.2f}min", flush=True)
    return shard_path


def merge_direct_param(out_dir: str, n_views: int):
    """Stitch all shard_*.npz in out_dir into full per-view arrays + logs."""
    import glob
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    if not shards:
        raise SystemExit(f"[merge] no shards in {out_dir}")

    parts = [np.load(s) for s in shards]
    n_iters = int(parts[0]["n_iters"])

    idx = np.concatenate([p["view_idx"] for p in parts])
    order = np.argsort(idx)
    idx_sorted = idx[order]

    def _cat(key):
        return np.concatenate([p[key] for p in parts], axis=0)[order]

    p9_raw = _cat("p9_raw")
    ts_mm = _cat("ts_mm")
    tp_mm = _cat("tp_mm")
    rot_deg = _cat("rot_deg")
    loss_curves = _cat("loss_curves")
    view_time = _cat("view_time_sec")
    final_loss = _cat("final_loss")
    init_loss = _cat("init_loss")
    total_time = float(sum(float(p["total_time_sec"]) for p in parts))  # summed shard wall time
    wall_time = float(max(float(p["total_time_sec"]) for p in parts))   # parallel wall-clock approx

    np.save(os.path.join(out_dir, "motion_p9_raw.npy"), p9_raw.astype(np.float32))
    np.save(os.path.join(out_dir, "motion_ts_mm.npy"), ts_mm.astype(np.float32))
    np.save(os.path.join(out_dir, "motion_tp_mm.npy"), tp_mm.astype(np.float32))
    np.save(os.path.join(out_dir, "motion_rot_deg.npy"), rot_deg.astype(np.float32))
    np.save(os.path.join(out_dir, "loss_curves.npy"), loss_curves.astype(np.float32))

    # Mean loss across views vs iteration (comparable to MLP per-epoch curve).
    mean_loss_vs_iter = loss_curves.mean(axis=0)
    np.save(os.path.join(out_dir, "mean_loss_vs_iter.npy"),
            mean_loss_vs_iter.astype(np.float64))

    # Per-view loss history (analog of AI_Geocal loss_history.csv).
    rows = []
    cum = 0.0
    for k in range(len(idx_sorted)):
        cum += float(view_time[k])
        rows.append({
            "view": int(idx_sorted[k]),
            "init_loss": float(init_loss[k]),
            "final_loss": float(final_loss[k]),
            "n_iters": int(n_iters),
            "view_time_sec": float(view_time[k]),
            "elapsed_time_sec": float(cum),
        })
    csv_path = os.path.join(out_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["view", "init_loss", "final_loss",
                                          "n_iters", "view_time_sec", "elapsed_time_sec"])
        w.writeheader(); w.writerows(rows)
    np.save(os.path.join(out_dir, "loss_history.npy"),
            np.asarray([[r["view"], r["final_loss"], r["view_time_sec"], r["elapsed_time_sec"]]
                        for r in rows], dtype=np.float64))

    with open(os.path.join(out_dir, "training_time.txt"), "w") as f:
        f.write(f"method: direct_per_view_9dof\n")
        f.write(f"n_views: {len(idx_sorted)}\n")
        f.write(f"n_iters_per_view: {n_iters}\n")
        f.write(f"sum_shard_compute_time_sec: {total_time:.6f}\n")
        f.write(f"sum_shard_compute_time_min: {total_time/60.0:.6f}\n")
        f.write(f"parallel_wallclock_time_sec: {wall_time:.6f}\n")
        f.write(f"parallel_wallclock_time_min: {wall_time/60.0:.6f}\n")
        f.write(f"mean_final_loss: {float(final_loss.mean()):.6e}\n")
        f.write(f"mean_init_loss: {float(init_loss.mean()):.6e}\n")

    print(f"[merge] {len(idx_sorted)} views | mean final loss={float(final_loss.mean()):.6e} "
          f"| sum compute={total_time/60.0:.2f}min | wallclock~{wall_time/60.0:.2f}min", flush=True)
    print(f"[merge] wrote motion_*.npy, loss_curves.npy, mean_loss_vs_iter.npy, "
          f"loss_history.csv/.npy, training_time.txt in {out_dir}", flush=True)
