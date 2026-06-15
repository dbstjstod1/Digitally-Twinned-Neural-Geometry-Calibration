import os
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
    write_gantry_file_33_recompute_PI_and_recompute_geoC,
)
from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
from models.MotionNetHash import MotionNetHash_9DoF


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

    x_pos: float = 0.0
    y_pos: float = 0.0
    z_pos: float = 0.0

    ureverse_raw: int = -1
    vreverse_raw: int = -1
    recon_type: int = 1

    n_samples: int = 128
    chunk_size: int = 8192

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
    clockwise_sign: float = -1.0,
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
    # Step 4. Build source-detector coordinate frame.
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


# ---------------------------
# Projection export helper
# ---------------------------

@torch.no_grad()
def _export_projection_raw(
    *,
    out_path: str,
    smat: torch.Tensor,
    P_all: torch.Tensor,
    geo_all: torch.Tensor,
    geo_stitch: torch.Tensor,
    cfg: ReconConfig,
    roi: RT_PARAM,
    ureverse: int,
    vreverse: int,
    proj_batch: int,
    use_amp_projector: bool,
) -> None:
    """
    English comments only.
    Save forward projections to a raw float32 file.
    """
    V = P_all.shape[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        for i0 in range(0, V, proj_batch):
            i1 = min(i0 + proj_batch, V)

            P_b = P_all[i0:i1]
            geo_b = geo_all[i0:i1]
            st_b = geo_stitch[i0:i1]

            with torch.cuda.amp.autocast(enabled=(use_amp_projector and P_all.device.type == "cuda")):
                pred = sinoproj_rdsh_pinv_raycast_dominant(
                    smat=smat,
                    Pmat=P_b,
                    geo_parameter=geo_b,
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
                    ureverse=ureverse,
                    vreverse=vreverse,
                    roi=roi,
                    recon_type=cfg.recon_type,
                    n_samples=cfg.n_samples,
                    chunk_size=cfg.chunk_size,
                    ori_nu=cfg.ori_nu,
                    ori_nv=cfg.ori_nv,
                    align_corners=False,
                )

            pred.detach().float().cpu().numpy().astype(np.float32).tofile(f)

            print(
                f"[projection export] {i0:04d}:{i1:04d} / {V:04d} "
                f"saved to {out_path}"
            )


# ---------------------------
# Main export function
# ---------------------------

@torch.no_grad()
def export_aligned_projections_and_gantry(
    *,
    cfg: ReconConfig,
    volume_path: str,
    roi: RT_PARAM,
    ckpt_path: str,
    out_dir: str,
    out_proj_raw: str = "Projections_aligned_9DoF.raw",
    out_proj_raw_before: str = "Projections_before_9DoF.raw",
    out_gantry_dat: str = "Gantry_updated_9DoF.dat",
    out_gantry_nominal_dat: str = "Gantry_nominal_9DoF.dat",
    ts_max_mm: float = 5.0,
    tp_max_mm: float = 5.0,
    rot_max_deg: float = 5.0,
    apply_batch: int = 64,
    proj_batch: int = 2,
    k_nominal: float = 650.0,
    un_nominal: float = 34.0,
    vn_nominal: float = 15.0,
    SOD: float = 443.0,
    SDD: float = 650.0,
    nominal_orbit_axis: str = "y",
    nominal_clockwise_sign: float = -1.0,
    nominal_include_endpoint: bool = False,
    use_beamcenter_geo: bool = False,
    use_amp_projector: bool = False,
    save_motion_npy: bool = True,
    save_before_raw: bool = True,
    save_nominal_gantry: bool = True,
    isocenter_method: str = "oppositepair",
    y_mode: str = "mean",
    outlier_reject: bool = False,
) -> None:
    """
    English comments only.

    Save:
      1) BEFORE projection using analytic nominal baseline
      2) AFTER projection using learned effective 9DoF correction
      3) Nominal gantry from analytic nominal P
      4) Updated gantry from AFTER projection matrices

    Pure 9-DoF model (ts, tp, rot only). The intrinsic skew DoF is removed:
    apply_9DoF_transform_effective no longer takes a skew argument.

    This version does not require an initial gantry file.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match origin convention.
    cfg.X0 = -0.5 * cfg.imsx * cfg.dx + cfg.x_pos
    cfg.Y0 = -0.5 * cfg.imsy * cfg.dy + cfg.y_pos
    cfg.Z0 = 0.0

    nu_ori_use = int(cfg.ori_nu) if (cfg.ori_nu is not None) else int(cfg.nu)
    nv_ori_use = int(cfg.ori_nv) if (cfg.ori_nv is not None) else int(cfg.nv)

    # ------------------------------------------------------------
    # Step 0. Load volume.
    # ------------------------------------------------------------
    vol_mm = load_raw_f32_memmap(volume_path, (cfg.imsz, cfg.imsy, cfg.imsx))
    volume = torch.from_numpy(np.asarray(vol_mm).copy()).to(device=device, dtype=torch.float32)
    smat = volume[None, None]

    urev = reverse_flag(cfg.ureverse_raw)
    vrev = reverse_flag(cfg.vreverse_raw)
    V = int(cfg.NLAM)

    # ------------------------------------------------------------
    # Step 1. Build analytic nominal baseline.
    # ------------------------------------------------------------
    P_before_nominal, geo_before_nominal, geo_stitch = build_nominal_orbit_from_geometry(
        n_views=V,
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

    print("[analytic nominal geometry]")
    print(f"  k={k_nominal}, un={un_nominal}, vn={vn_nominal}")
    print(f"  SOD={SOD}, SDD={SDD}")
    print(f"  orbit_axis={nominal_orbit_axis}, clockwise_sign={nominal_clockwise_sign}")
    print(f"  use_beamcenter_geo={use_beamcenter_geo}")

    np.save(
        os.path.join(out_dir, "P_before_nominal_analytic.npy"),
        P_before_nominal.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "geo_before_nominal_analytic.npy"),
        geo_before_nominal.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "geo_stitch_analytic.npy"),
        geo_stitch.detach().cpu().numpy().astype(np.float32),
    )

    # ------------------------------------------------------------
    # Step 1.5. Save nominal gantry if requested.
    # ------------------------------------------------------------
    if save_nominal_gantry:
        nominal_gantry_out_path = os.path.join(out_dir, out_gantry_nominal_dat)

        write_gantry_file_33_recompute_PI_and_recompute_geoC(
            nominal_gantry_out_path,
            P_12=P_before_nominal.detach().cpu().numpy().astype(np.float32),
            geo_old_7=geo_before_nominal.detach().cpu().numpy().astype(np.float32),
            stitch_2=geo_stitch.detach().cpu().numpy().astype(np.float32),
            nu_ori=nu_ori_use,
            nv_ori=nv_ori_use,
            du=float(cfg.du),
            dv=float(cfg.dv),
            iso_iter_num=100,
            iso_step_mm=0.1,
            iso_step_mag=0.75,
            isocenter_method=isocenter_method,
            y_mode=y_mode,
            outlier_reject=outlier_reject,
        )

    # ------------------------------------------------------------
    # Step 2. Load trained motion model and predict motion.
    # ------------------------------------------------------------
    motion_model = MotionNetHash_9DoF(n_views=cfg.NLAM).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    motion_model.load_state_dict(ckpt["model_state"])
    motion_model.eval()

    all_idx = torch.arange(V, device=device)
    p9_raw = motion_model(all_idx)

    ts_all, tp_all, rot_all, _ = motion9_to_ts_tp_rot(
        p9_raw,
        ts_max_mm=ts_max_mm,
        tp_max_mm=tp_max_mm,
        rot_max_deg=rot_max_deg,
    )

    if save_motion_npy:
        np.save(os.path.join(out_dir, "p9_raw.npy"), p9_raw.detach().cpu().numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "ts_mm.npy"), ts_all.detach().cpu().numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "tp_mm.npy"), tp_all.detach().cpu().numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "rot_deg.npy"), rot_all.detach().cpu().numpy().astype(np.float32))

    # ------------------------------------------------------------
    # Step 3. Apply residual effective 9DoF on analytic nominal baseline.
    # ------------------------------------------------------------
    P_new_list = []
    geo_new_list = []

    for i0 in range(0, V, apply_batch):
        i1 = min(i0 + apply_batch, V)

        P0 = P_before_nominal[i0:i1]
        geo0 = geo_before_nominal[i0:i1]

        Pn, geon = apply_9DoF_transform_effective(
            P0=P0,
            geo_old=geo0,
            ts_internal=ts_all[i0:i1],
            tp_internal=tp_all[i0:i1],
            rot_internal_deg=rot_all[i0:i1],
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

        P_new_list.append(Pn.detach())
        geo_new_list.append(geon.detach())

        print(f"[apply 9DoF] {i0:04d}:{i1:04d} / {V:04d}")

    P_new = torch.cat(P_new_list, dim=0)
    geo_new = torch.cat(geo_new_list, dim=0)

    np.save(
        os.path.join(out_dir, "P_updated_9DoF.npy"),
        P_new.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        os.path.join(out_dir, "geo_updated_9DoF.npy"),
        geo_new.detach().cpu().numpy().astype(np.float32),
    )

    # ------------------------------------------------------------
    # Step 4. Save updated gantry using AFTER matrices.
    # ------------------------------------------------------------
    gantry_out_path = os.path.join(out_dir, out_gantry_dat)

    write_gantry_file_33_recompute_PI_and_recompute_geoC(
        gantry_out_path,
        P_12=P_new.detach().cpu().numpy().astype(np.float32),
        geo_old_7=geo_new.detach().cpu().numpy().astype(np.float32),
        stitch_2=geo_stitch.detach().cpu().numpy().astype(np.float32),
        nu_ori=nu_ori_use,
        nv_ori=nv_ori_use,
        du=float(cfg.du),
        dv=float(cfg.dv),
        iso_iter_num=100,
        iso_step_mm=0.1,
        iso_step_mag=0.75,
        isocenter_method=isocenter_method,
        y_mode=y_mode,
        outlier_reject=outlier_reject,
    )

    # ------------------------------------------------------------
    # Step 5. Export BEFORE projections.
    # ------------------------------------------------------------
    if save_before_raw:
        proj_before_path = os.path.join(out_dir, out_proj_raw_before)
        _export_projection_raw(
            out_path=proj_before_path,
            smat=smat,
            P_all=P_before_nominal,
            geo_all=geo_before_nominal,
            geo_stitch=geo_stitch,
            cfg=cfg,
            roi=roi,
            ureverse=urev,
            vreverse=vrev,
            proj_batch=proj_batch,
            use_amp_projector=use_amp_projector,
        )

    # ------------------------------------------------------------
    # Step 6. Export AFTER projections.
    # ------------------------------------------------------------
    proj_after_path = os.path.join(out_dir, out_proj_raw)
    _export_projection_raw(
        out_path=proj_after_path,
        smat=smat,
        P_all=P_new,
        geo_all=geo_new,
        geo_stitch=geo_stitch,
        cfg=cfg,
        roi=roi,
        ureverse=urev,
        vreverse=vrev,
        proj_batch=proj_batch,
        use_amp_projector=use_amp_projector,
    )

    if save_nominal_gantry:
        print(f"[OK] NOMINAL gantry saved: {nominal_gantry_out_path}")

    print(f"[OK] Updated gantry saved: {gantry_out_path}")

    if save_before_raw:
        print(
            f"[OK] BEFORE projections saved: {proj_before_path} "
            f"(shape: {V} x {cfg.nv} x {cfg.nu}, float32)"
        )

    print(
        f"[OK] ALIGNED projections saved: {proj_after_path} "
        f"(shape: {V} x {cfg.nv} x {cfg.nu}, float32)"
    )


# ---------------------------
# Example usage
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
    
    export_aligned_projections_and_gantry(
        cfg=cfg_4T,
        volume_path="./open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw",
        roi=roi,
        ckpt_path="./result_denseball/9DoF_analytic_nominalP_10/no_initP_k_un_vn_SOD_SDD/motion_model_ep0055.pth",
        out_dir="./result_denseball/9DoF_analytic_nominalP_10/no_initP_k_un_vn_SOD_SDD_export",
        out_proj_raw_before="Projections_before_9DoF.raw",
        out_proj_raw="Projections_aligned_9DoF.raw",
        out_gantry_dat="Gantry_updated_9DoF.dat",
        out_gantry_nominal_dat="Gantry_nominal_9DoF.dat",
        ts_max_mm=5.0 + 5,
        tp_max_mm=5.0 + 5,
        rot_max_deg=5.0 + 5,
        apply_batch=64,
        proj_batch=2,

        # --------------------------------------------------------
        # Must match the analytic nominal geometry used in training.
        # --------------------------------------------------------
        k_nominal=650.0 + 0,
        un_nominal=34.0 + 0,
        vn_nominal=15.0 + 0,
        SOD=443.0 + 0,
        SDD=650.0,
        nominal_orbit_axis="y",
        nominal_clockwise_sign=-1.0,
        nominal_include_endpoint=False,
        use_beamcenter_geo=False,

        use_amp_projector=False,
        save_motion_npy=True,
        save_before_raw=True,
        save_nominal_gantry=True,

        # For analytic nominal orbit, circlefit is usually safer than oppositepair
        # unless the scan has exact 180-degree opposite-view pairs.
        isocenter_method="circlefit",
        y_mode="mean",
        outlier_reject=False,
    )