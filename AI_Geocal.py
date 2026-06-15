import os
import time
import csv
import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional, Tuple

from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant
from helpers import (
    load_raw_f32_memmap,
    reverse_flag,
    recompute_geo_like_C_numpy,
    recompute_geo_like_C_numpy_beamcenter,
)
from DoF_transform import apply_9DoF_transform_effective, motion10_to_ts_tp_rot_skew
from models.MotionNetHash import MotionNetHash_10DoF
from monai.losses import LocalNormalizedCrossCorrelationLoss


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class RT_PARAM:
    x: int
    y: int
    width: int
    height: int


@dataclass
class ReconConfig:
    # English comments only.
    NLAM: int = 800
    ScanAngle_deg: float = 360.0
    StartAngle_deg: float = -4.8

    nu: int = 776
    nv: int = 1264
    du: float = 0.228
    dv: float = 0.228

    ori_nu: Optional[int] = None
    ori_nv: Optional[int] = None

    imsx: int = 1002
    imsy: int = 1002
    imsz: int = 982
    dx: float = 0.2
    dy: float = 0.2
    dz: float = 0.2

    # Volume origin offset
    x_pos: float = 0.0
    y_pos: float = 0.0
    z_pos: float = 0.0

    ureverse_raw: int = -1
    vreverse_raw: int = -1
    recon_type: int = 1

    # Projector controls
    n_samples: int = 128
    chunk_size: int = 8192

    # Computed world origin for voxel grid
    X0: float = 0.0
    Y0: float = 0.0
    Z0: float = 0.0


# ============================================================
# Nominal P builder without initial P dependency
# ============================================================

def build_nominal_orbit_from_geometry(
    *,
    n_views: int,
    scan_angle_deg: float,
    start_angle_deg: float,
    k_nominal: float,
    un_nominal: float,
    vn_nominal: float,
    SOD: float,
    SDD: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    X0: float,
    Y0: float,
    Z0: float,
    nu_ori: int,
    nv_ori: int,
    du: float,
    dv: float,
    orbit_axis: str = "y",
    clockwise_sign: float = 1.0,
    include_endpoint: bool = False,
    use_beamcenter_geo: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.

    Build nominal projection matrices without using initial P matrices.

    Returns:
        P_nominal_flat:
            (V,12) nominal projection matrices.
        geo_nominal:
            (V,7) geometry parameters.
        geo_stitch:
            (V,2) zero stitch offsets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V = int(n_views)
    orbit_axis = orbit_axis.lower()

    # ------------------------------------------------------------
    # Step 1. Define isocenter in WORLD coordinates.
    # ------------------------------------------------------------
    cx_i = float(X0 + dx * (0.5 * nx))
    cy_i = float(Y0 + dy * (0.5 * ny))

    # Keep the same z0 convention as the existing projector/gantry setup.
    # Do not use Z0 + dz * (0.5 * nz).
    cz_i = float(Z0)

    # Match the existing convention:
    # INTERNAL(x,y,z) = WORLD(x,z,y)
    iso_world = torch.tensor(
        [cx_i, cz_i, cy_i],
        device=device,
        dtype=dtype,
    )

    print("[analytic nominal Pmat center]")
    print(f"  X0, Y0, Z0       = {X0:.6f}, {Y0:.6f}, {Z0:.6f}")
    print(f"  cx_i, cy_i, cz_i = {cx_i:.6f}, {cy_i:.6f}, {cz_i:.6f}")
    print(f"  iso_world        = {iso_world.detach().cpu().numpy()}")

    # ------------------------------------------------------------
    # Step 2. Build angular positions.
    # ------------------------------------------------------------
    if include_endpoint and V > 1:
        step_deg = float(scan_angle_deg) / float(V - 1)
    else:
        step_deg = float(scan_angle_deg) / float(V)

    angles_deg = (
        float(start_angle_deg)
        + torch.arange(V, device=device, dtype=dtype) * (float(clockwise_sign) * step_deg)
    )
    theta = angles_deg * (torch.pi / 180.0)

    # ------------------------------------------------------------
    # Step 3. Build source trajectory.
    # ------------------------------------------------------------
    source = torch.zeros((V, 3), device=device, dtype=dtype)

    if orbit_axis == "y":
        # Source rotates on the XZ plane around WORLD-y axis.
        source[:, 0] = iso_world[0] + float(SOD) * torch.sin(theta)
        source[:, 1] = iso_world[1]
        source[:, 2] = iso_world[2] - float(SOD) * torch.cos(theta)

        up_world = torch.tensor(
            [0.0, 1.0, 0.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).repeat(V, 1)

    elif orbit_axis == "x":
        # Source rotates on the YZ plane around WORLD-x axis.
        source[:, 0] = iso_world[0]
        source[:, 1] = iso_world[1] + float(SOD) * torch.sin(theta)
        source[:, 2] = iso_world[2] - float(SOD) * torch.cos(theta)

        up_world = torch.tensor(
            [1.0, 0.0, 0.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).repeat(V, 1)

    elif orbit_axis == "z":
        # Source rotates on the XY plane around WORLD-z axis.
        source[:, 0] = iso_world[0] + float(SOD) * torch.sin(theta)
        source[:, 1] = iso_world[1] - float(SOD) * torch.cos(theta)
        source[:, 2] = iso_world[2]

        up_world = torch.tensor(
            [0.0, 0.0, 1.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).repeat(V, 1)

    else:
        raise ValueError(f"Unsupported orbit_axis: {orbit_axis}")

    # ------------------------------------------------------------
    # Step 4. Build camera/detector coordinate frame.
    #
    # Convention:
    #   Xc = R Xw + t
    #   source maps to camera origin
    #   central ray direction maps to +Z camera axis
    # ------------------------------------------------------------
    z_cam_world = iso_world.view(1, 3) - source
    z_cam_world = z_cam_world / (
        torch.linalg.norm(z_cam_world, dim=1, keepdim=True) + 1e-12
    )

    x_cam_world = torch.cross(up_world, z_cam_world, dim=1)
    x_cam_world = x_cam_world / (
        torch.linalg.norm(x_cam_world, dim=1, keepdim=True) + 1e-12
    )

    y_cam_world = torch.cross(z_cam_world, x_cam_world, dim=1)
    y_cam_world = y_cam_world / (
        torch.linalg.norm(y_cam_world, dim=1, keepdim=True) + 1e-12
    )

    # Rows of R are camera axes expressed in WORLD coordinates.
    R = torch.stack(
        [x_cam_world, y_cam_world, z_cam_world],
        dim=1,
    )

    # t = -R C
    t = -torch.bmm(R, source[:, :, None])[:, :, 0]

    # ------------------------------------------------------------
    # Step 5. Build nominal intrinsic matrix K.
    # ------------------------------------------------------------
    K = torch.zeros((V, 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = float(k_nominal)
    K[:, 1, 1] = float(k_nominal)
    K[:, 0, 2] = float(un_nominal)
    K[:, 1, 2] = float(vn_nominal)
    K[:, 2, 2] = 1.0

    # ------------------------------------------------------------
    # Step 6. Build projection matrices.
    # ------------------------------------------------------------
    E = torch.zeros((V, 3, 4), device=device, dtype=dtype)
    E[:, :, :3] = R
    E[:, :, 3] = t

    P_nominal = torch.bmm(K, E)
    P_nominal_flat = P_nominal.reshape(V, 12)

    # ------------------------------------------------------------
    # Step 7. Recompute geo from generated P.
    # ------------------------------------------------------------
    P_np = P_nominal_flat.detach().cpu().numpy().astype(np.float32)

    geo_old_np = np.zeros((V, 7), dtype=np.float32)
    geo_old_np[:, 6] = np.arange(V, dtype=np.float32)

    if use_beamcenter_geo:
        geo_np = recompute_geo_like_C_numpy_beamcenter(
            P_12=P_np,
            geo_old_7=geo_old_np,
            nu_ori=int(nu_ori),
            nv_ori=int(nv_ori),
            du=float(du),
            dv=float(dv),
        )
    else:
        geo_np = recompute_geo_like_C_numpy(
            P_12=P_np,
            geo_old_7=geo_old_np,
            nu_ori=int(nu_ori),
            nv_ori=int(nv_ori),
            du=float(du),
            dv=float(dv),
            iter_num=100,
            step_mm=0.1,
            step_mag=0.75,
            isocenter_method="circlefit",
            y_mode="mean",
            outlier_reject=False,
        )

    geo_nominal = torch.from_numpy(geo_np).to(device=device, dtype=dtype)

    # No gantry file dependency, so stitch offsets are zero.
    geo_stitch = torch.zeros((V, 2), device=device, dtype=dtype)

    # Keep SDD as explicit input for logging/debugging.
    _ = float(SDD)

    return P_nominal_flat, geo_nominal, geo_stitch


# ============================================================
# Train motion model
# ============================================================

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
    skew_max: float = 1.0,
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
):
    """
    English comments only.

    Train MotionNetHash_10DoF using an analytic nominal P-matrix baseline
    generated only from k, un, vn, SOD, and SDD.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    motion_model = MotionNetHash_10DoF(n_views=cfg.NLAM).to(device)

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

            tgt_np = np.asarray(proj_mm[idx.detach().cpu().numpy()]).copy()
            tgt = torch.from_numpy(tgt_np).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Predict raw motion parameters from view indices.
                p10_raw = motion_model(idx)

                # Map raw outputs to bounded parameter ranges.
                ts, tp, rot_deg, skew, _ = motion10_to_ts_tp_rot_skew(
                    p10_raw,
                    ts_max_mm=ts_max_mm,
                    tp_max_mm=tp_max_mm,
                    rot_max_deg=rot_max_deg,
                    skew_max=skew_max,
                )

                # Apply effective 10DoF correction on analytic nominal baseline.
                with torch.cuda.amp.autocast(enabled=False):
                    P_new, geo_new = apply_9DoF_transform_effective(
                        P0=P0,
                        geo_old=geo_b,
                        ts_internal=ts,
                        tp_internal=tp,
                        rot_internal_deg=rot_deg,
                        skew=skew,
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

                # Forward projection.
                pred = sinoproj_rdsh_pinv_raycast_dominant(
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
                    n_samples=cfg.n_samples,
                    chunk_size=cfg.chunk_size,
                    ori_nu=cfg.ori_nu,
                    ori_nv=cfg.ori_nv,
                    align_corners=False,
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
                    "k_nominal": k_nominal,
                    "un_nominal": un_nominal,
                    "vn_nominal": vn_nominal,
                    "SOD": SOD,
                    "SDD": SDD,
                    "ts_max_mm": ts_max_mm,
                    "tp_max_mm": tp_max_mm,
                    "rot_max_deg": rot_max_deg,
                    "skew_max": skew_max,
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
        p10_raw = motion_model(idx)
        ts_all, tp_all, rot_all, skew_all, _ = motion10_to_ts_tp_rot_skew(
            p10_raw,
            ts_max_mm=ts_max_mm,
            tp_max_mm=tp_max_mm,
            rot_max_deg=rot_max_deg,
            skew_max=skew_max,
        )

    np.save(
        os.path.join(out_dir, "motion_p10_raw.npy"),
        p10_raw.detach().cpu().numpy().astype(np.float32),
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
    np.save(
        os.path.join(out_dir, "motion_skew.npy"),
        skew_all.detach().cpu().numpy().astype(np.float32),
    )

    print(f"Saved: {out_dir}/motion_p10_raw.npy")
    print(f"Saved: {out_dir}/motion_ts_mm.npy")
    print(f"Saved: {out_dir}/motion_tp_mm.npy")
    print(f"Saved: {out_dir}/motion_rot_deg.npy")
    print(f"Saved: {out_dir}/motion_skew.npy")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    cfg_2T = ReconConfig(
        NLAM=600,
        ScanAngle_deg=240.0,
        StartAngle_deg=-4.8,
        nu=904,
        nv=724,
        ori_nu=1120,
        ori_nv=724,
        du=0.2,
        dv=0.2,
        imsx=780,
        imsy=780,
        imsz=632,
        dx=0.15,
        dy=0.15,
        dz=0.15,
        ureverse_raw=1,
        vreverse_raw=1,
        recon_type=0,
        n_samples=256,
        chunk_size=16384,
    )

    cfg_4T = ReconConfig(
        NLAM=480,
        ScanAngle_deg=360.0,
        StartAngle_deg=180.0,
        nu=776,
        nv=1264,
        ori_nu=776,
        ori_nv=1264,
        du=0.228,
        dv=0.228,
        imsx=929,
        imsy=929,
        imsz=801,
        dx=0.2,
        dy=0.2,
        dz=0.2,
        ureverse_raw=1,
        vreverse_raw=1,
        recon_type=1,
        n_samples=256,
        chunk_size=16384,
    )

    roi = RT_PARAM(
        x=0,
        y=0,
        width=cfg_4T.nu,
        height=cfg_4T.nv,
    )

    train_motion_hash_model(
        cfg=cfg_4T,
        volume_path="./open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw",
        proj_meas_path="./Denseball_proj_480.raw",
        roi=roi,
        out_dir="./result_denseball/10DoF_analytic_nominalP_10/no_initP_k_un_vn_SOD_SDD",
        epochs=100,
        batch_size=4,
        lr=1e-3,
        use_amp=False,
        seed=0,
        ts_max_mm=5.0 + 5,
        tp_max_mm=5.0 + 5,
        rot_max_deg=5.0 + 5,
        skew_max=5.0,
        loss_type="lncc",
        save_every=5,
        view_step=1,

        # --------------------------------------------------------
        # Analytic nominal geometry parameters.
        # Change these values to match the actual system.
        # --------------------------------------------------------
        k_nominal=650.0 + 0,
        un_nominal=34.0 + 0,
        vn_nominal=15.0 + 0,
        SOD=443 + 0.0,
        SDD=650.0,

        nominal_orbit_axis="y",
        nominal_clockwise_sign=-1.0,
        nominal_include_endpoint=False,

        # False: projected-isocenter based geo[3:6]
        # True : true-beam-center based geo[3:6]
        use_beamcenter_geo=False,
    )