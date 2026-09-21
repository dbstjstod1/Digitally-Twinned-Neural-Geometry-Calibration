import os
import numpy as np
import torch

from fast_projectors import sinoproj_joseph
from geometry import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from helpers import load_raw_f32_memmap, reverse_flag
from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
import time
import csv
from monai.losses import LocalNormalizedCrossCorrelationLoss


def train_motion_hash_model(
    *,
    cfg: ReconConfig,
    volume_path: str,
    proj_meas_path: str,
    roi: RT_PARAM,
    out_dir: str = "./debug_train_hash",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    use_amp: bool = True,
    seed: int = 0,
    ts_max_mm: float = 5.0,
    tp_max_mm: float = 5.0,
    rot_max_deg: float = 5.0,
    loss_type: str = "lncc",
    save_every: int = 1,
    view_step: int = 1,
    # Nominal analytic geometry
    k_nominal: float = 651.0,
    un_nominal: float = 33.0,
    vn_nominal: float = 16.0,
    SOD: float = 393.0,
    SDD: float = 651.0,
    nominal_orbit_axis: str = "y",
    nominal_clockwise_sign: float = -1.0,
    nominal_include_endpoint: bool = False,
    use_beamcenter_geo: bool = False,
    preload_projections: bool = True,
):
    """
    English comments only.

    Train MotionNetHash_9DoF using an analytic nominal P-matrix baseline
    generated only from k, un, vn, SOD, and SDD.

    Pure 9-DoF model (ts, tp, rot only). The intrinsic skew DoF is removed:
    apply_9DoF_transform_effective no longer takes a skew argument.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Joseph calibration requires an NVIDIA CUDA GPU; see README.md.")
    from models.MotionNetHash import MotionNetHash_9DoF

    device = torch.device("cuda")
    os.makedirs(out_dir, exist_ok=True)

    # Match origin convention.
    cfg.X0 = -0.5 * cfg.imsx * cfg.dx + cfg.x_pos
    cfg.Y0 = -0.5 * cfg.imsy * cfg.dy + cfg.y_pos
    cfg.Z0 = 0.0

    # ------------------------------------------------------------
    # Load volume.
    # ------------------------------------------------------------
    vol_mm = load_raw_f32_memmap(volume_path, (cfg.imsz, cfg.imsy, cfg.imsx))
    volume_np = np.asarray(vol_mm).copy()
    volume = torch.from_numpy(volume_np).to(device=device, dtype=torch.float32)
    smat = volume.detach()[None, None]  # (1,1,D,H,W)

    # ------------------------------------------------------------
    # Load measured projections.
    # ------------------------------------------------------------
    proj_mm = load_raw_f32_memmap(proj_meas_path, (cfg.NLAM, cfg.nv, cfg.nu))

    # Keep every measured view resident on the GPU (4T: 480 x 1264 x 776 fp32 = 1.9 GB) so a
    # batch is a slice instead of a memmap read + host-to-device copy per step; falls back to
    # the memmap when it does not fit.
    proj_gpu = None
    if preload_projections and device.type == "cuda":
        try:
            proj_gpu = torch.from_numpy(np.ascontiguousarray(proj_mm)).to(device=device, dtype=torch.float32)
            print(f"[train] measured projections resident on GPU ({proj_gpu.numel() * 4 / 2**30:.2f} GB)")
        except RuntimeError as ex:
            print(f"[train] projections stay memmapped ({ex})")
            proj_gpu = None

    print("[train] projector = Joseph (Triton)")

    urev = reverse_flag(cfg.ureverse_raw)
    vrev = reverse_flag(cfg.vreverse_raw)

    # ------------------------------------------------------------
    # Build analytic nominal P/geo without initial P.
    # ------------------------------------------------------------
    nu_ori_use = int(cfg.ori_nu) if (cfg.ori_nu is not None) else int(cfg.nu)
    nv_ori_use = int(cfg.ori_nv) if (cfg.ori_nv is not None) else int(cfg.nv)

    P_nominal_all, geo_nominal_all, geo_stitch = build_nominal_orbit_from_geometry(
        n_views=cfg.NLAM,
        scan_angle_deg=float(cfg.ScanAngle_deg),
        start_angle_deg=float(cfg.StartAngle_deg),
        k_nominal=float(k_nominal),
        un_nominal=float(un_nominal),
        vn_nominal=float(vn_nominal),
        SOD=float(SOD),
        SDD=float(SDD),
        nx=cfg.imsx,
        ny=cfg.imsy,
        nz=cfg.imsz,
        dx=cfg.dx,
        dy=cfg.dy,
        dz=cfg.dz,
        X0=cfg.X0,
        Y0=cfg.Y0,
        Z0=cfg.Z0,
        nu_ori=nu_ori_use,
        nv_ori=nv_ori_use,
        du=float(cfg.du),
        dv=float(cfg.dv),
        orbit_axis=nominal_orbit_axis,
        clockwise_sign=nominal_clockwise_sign,
        include_endpoint=nominal_include_endpoint,
        use_beamcenter_geo=use_beamcenter_geo,
        device=device,
        dtype=torch.float32,
    )

    print("[nominal geometry]")
    print(f"  k={k_nominal}, un={un_nominal}, vn={vn_nominal}")
    print(f"  SOD={SOD}, SDD={SDD}")
    print(f"  orbit_axis={nominal_orbit_axis}, clockwise_sign={nominal_clockwise_sign}")
    print(f"  use_beamcenter_geo={use_beamcenter_geo}")

    # Save nominal geometry for debugging.
    np.save(
        os.path.join(out_dir, "P_nominal_analytic.npy"),
        P_nominal_all.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "geo_nominal_analytic.npy"),
        geo_nominal_all.detach().cpu().numpy().astype(np.float32),
    )

    # ------------------------------------------------------------
    # Loss.
    # ------------------------------------------------------------
    if loss_type.lower() != "lncc":
        raise ValueError("This script currently supports only loss_type='lncc'.")

    lncc = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=2,
        kernel_size=31,
        kernel_type="rectangular",
        reduction="mean",
    ).to(device)

    # ------------------------------------------------------------
    # Motion model.
    # ------------------------------------------------------------
    motion_model = MotionNetHash_9DoF(n_views=cfg.NLAM).to(device)

    opt = torch.optim.Adam(motion_model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    V = cfg.NLAM

    # Train on a subset of views only.
    view_step = max(int(view_step), 1)
    train_idx = torch.arange(0, V, view_step, device=device)
    n_train = int(train_idx.numel())
    print(f"[train] using {n_train}/{V} views (view_step={view_step})")

    # ROI coordinates for similarity loss - 11floor.
    u0, u1 = 26, cfg.nu
    v0, v1 = 0, 1182

    # Safety clamp for detector size.
    u0 = max(0, min(u0, cfg.nu))
    u1 = max(0, min(u1, cfg.nu))
    v0 = max(0, min(v0, cfg.nv))
    v1 = max(0, min(v1, cfg.nv))

    print(f"[ROI] u={u0}:{u1}, v={v0}:{v1}")

    # ------------------------------------------------------------
    # Training log setup.
    # ------------------------------------------------------------
    loss_history = []

    loss_csv_path = os.path.join(out_dir, "loss_history.csv")
    loss_npy_path = os.path.join(out_dir, "loss_history.npy")
    time_txt_path = os.path.join(out_dir, "training_time.txt")

    if device.type == "cuda":
        torch.cuda.synchronize()

    train_start_time = time.perf_counter()

    # ------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------
    for ep in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_start_time = time.perf_counter()

        motion_model.train()

        perm = train_idx[torch.randperm(n_train, device=device)]
        n_batches = int(np.ceil(n_train / batch_size))
        ep_loss = 0.0

        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            if idx.numel() == 0:
                continue

            # Use analytic nominal baseline.
            P0 = P_nominal_all[idx]
            geo_b = geo_nominal_all[idx]
            st_b = geo_stitch[idx]

            if proj_gpu is not None:
                tgt = proj_gpu[idx]
            else:
                tgt_np = np.asarray(proj_mm[idx.detach().cpu().numpy()]).copy()
                tgt = torch.from_numpy(tgt_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Predict raw motion parameters from view indices.
                p9_raw = motion_model(idx)

                # Map raw outputs to bounded parameter ranges.
                ts, tp, rot_deg, _ = motion9_to_ts_tp_rot(
                    p9_raw,
                    ts_max_mm=ts_max_mm,
                    tp_max_mm=tp_max_mm,
                    rot_max_deg=rot_max_deg,
                )

                # Apply effective 9DoF correction on analytic nominal baseline.
                with torch.cuda.amp.autocast(enabled=False):
                    P_new, geo_new = apply_9DoF_transform_effective(
                        P0=P0,
                        geo_old=geo_b,
                        ts_internal=ts,
                        tp_internal=tp,
                        rot_internal_deg=rot_deg,
                        nx=cfg.imsx,
                        ny=cfg.imsy,
                        nz=cfg.imsz,
                        dx=cfg.dx,
                        dy=cfg.dy,
                        dz=cfg.dz,
                        X0=cfg.X0,
                        Y0=cfg.Y0,
                        Z0=cfg.Z0,
                        use_inverse_right_multiply=0,
                    )

                # Differentiable Joseph forward projection.
                pred = sinoproj_joseph(
                    smat=smat,
                    Pmat=P_new,
                    geo_parameter=geo_new,
                    geo_stitch=st_b,
                    nu=cfg.nu,
                    nv=cfg.nv,
                    du=cfg.du,
                    dv=cfg.dv,
                    imsx=cfg.imsx,
                    imsy=cfg.imsy,
                    imsz=cfg.imsz,
                    dx=cfg.dx,
                    dy=cfg.dy,
                    dz=cfg.dz,
                    X0=cfg.X0,
                    Y0=cfg.Y0,
                    Z0=cfg.Z0,
                    ureverse=urev,
                    vreverse=vrev,
                    roi=roi,
                    recon_type=cfg.recon_type,
                    ori_nu=cfg.ori_nu,
                    ori_nv=cfg.ori_nv,
                )

                # Similarity loss.
                with torch.cuda.amp.autocast(enabled=False):
                    pred_4d = pred.unsqueeze(1)
                    tgt_4d = tgt.unsqueeze(1)

                    loss = (
                        1.0
                        + lncc(
                            pred_4d[:, :, v0:v1, u0:u1].float(),
                            tgt_4d[:, :, v0:v1, u0:u1].float(),
                        )
                    )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ep_loss += float(loss.item())

        ep_loss /= max(n_batches, 1)

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time_sec = time.perf_counter() - epoch_start_time
        elapsed_time_sec = time.perf_counter() - train_start_time

        loss_history.append(
            {
                "epoch": ep,
                "loss": float(ep_loss),
                "epoch_time_sec": float(epoch_time_sec),
                "elapsed_time_sec": float(elapsed_time_sec),
            }
        )

        print(
            f"[epoch {ep:04d}/{epochs:04d}] "
            f"loss={ep_loss:.6e}, "
            f"epoch_time={epoch_time_sec:.2f}s, "
            f"elapsed={elapsed_time_sec / 60.0:.2f}min"
        )

        # Save loss history every epoch.
        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "loss", "epoch_time_sec", "elapsed_time_sec"],
            )
            writer.writeheader()
            writer.writerows(loss_history)

        np.save(
            loss_npy_path,
            np.asarray(
                [
                    [x["epoch"], x["loss"], x["epoch_time_sec"], x["elapsed_time_sec"]]
                    for x in loss_history
                ],
                dtype=np.float64,
            ),
        )

        if save_every > 0 and (ep % save_every == 0 or ep == epochs):
            ckpt_path = os.path.join(out_dir, f"motion_model_ep{ep:04d}.pth")
            torch.save(
                {
                    "epoch": ep,
                    "model_state": motion_model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": cfg.__dict__,
                    "roi": roi.__dict__,
                    "training": {
                        "epochs": epochs, "batch_size": batch_size, "lr": lr,
                        "seed": seed, "view_step": view_step, "use_amp": use_amp,
                        "volume_path": str(volume_path),
                        "proj_meas_path": str(proj_meas_path),
                    },
                    "k_nominal": k_nominal,
                    "un_nominal": un_nominal,
                    "vn_nominal": vn_nominal,
                    "SOD": SOD,
                    "SDD": SDD,
                    "ts_max_mm": ts_max_mm,
                    "tp_max_mm": tp_max_mm,
                    "rot_max_deg": rot_max_deg,
                    "dof_mode": "9DoF",
                    "nominal_orbit_axis": nominal_orbit_axis,
                    "nominal_clockwise_sign": nominal_clockwise_sign,
                    "nominal_include_endpoint": nominal_include_endpoint,
                    "use_beamcenter_geo": use_beamcenter_geo,
                    "loss_history": loss_history,
                    "loss_csv_path": loss_csv_path,
                    "loss_npy_path": loss_npy_path,
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")

    # ------------------------------------------------------------
    # Save total training time.
    # ------------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_train_time_sec = time.perf_counter() - train_start_time

    with open(time_txt_path, "w") as f:
        f.write(f"total_train_time_sec: {total_train_time_sec:.6f}\n")
        f.write(f"total_train_time_min: {total_train_time_sec / 60.0:.6f}\n")
        f.write(f"total_train_time_hour: {total_train_time_sec / 3600.0:.6f}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"view_step: {view_step}\n")
        f.write(f"n_train_views: {n_train}\n")
        f.write(f"n_batches_per_epoch: {n_batches}\n")
        f.write(f"loss_csv_path: {loss_csv_path}\n")
        f.write(f"loss_npy_path: {loss_npy_path}\n")

    print(
        f"[training done] total_time="
        f"{total_train_time_sec:.2f}s "
        f"({total_train_time_sec / 60.0:.2f}min)"
    )
    print(f"Saved: {loss_csv_path}")
    print(f"Saved: {loss_npy_path}")
    print(f"Saved: {time_txt_path}")

    # ------------------------------------------------------------
    # Export learned motion.
    # ------------------------------------------------------------
    motion_model.eval()
    with torch.no_grad():
        idx = torch.arange(V, device=device)
        p9_raw = motion_model(idx)
        ts_all, tp_all, rot_all, _ = motion9_to_ts_tp_rot(
            p9_raw,
            ts_max_mm=ts_max_mm,
            tp_max_mm=tp_max_mm,
            rot_max_deg=rot_max_deg,
        )

    np.save(
        os.path.join(out_dir, "motion_p9_raw.npy"),
        p9_raw.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "motion_ts_mm.npy"),
        ts_all.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "motion_tp_mm.npy"),
        tp_all.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "motion_rot_deg.npy"),
        rot_all.detach().cpu().numpy().astype(np.float32),
    )

    print(f"Saved: {out_dir}/motion_p9_raw.npy")
    print(f"Saved: {out_dir}/motion_ts_mm.npy")
    print(f"Saved: {out_dir}/motion_tp_mm.npy")
    print(f"Saved: {out_dir}/motion_rot_deg.npy")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    from run_single_viewstep import main

    main()
