import os
import numpy as np
import torch

from fast_projectors import sinoproj_joseph
from geometry import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from helpers import load_raw_f32_memmap, reverse_flag
from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
from dataclasses import fields
from helpers import write_gantry_file_33_recompute_PI_and_recompute_geoC


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
                pred = sinoproj_joseph(
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
                    ori_nu=cfg.ori_nu,
                    ori_nv=cfg.ori_nv,
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

    Saved checkpoint geometry, ROI, and motion bounds take precedence over the
    supplied defaults. No initial gantry file is required.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    full_detector_roi = (roi.x, roi.y, roi.width, roi.height) == (0, 0, cfg.nu, cfg.nv)
    config_fields = {field.name for field in fields(ReconConfig)}
    saved_cfg = {key: value for key, value in ckpt.get("cfg", {}).items()
                 if key in config_fields}
    cfg = ReconConfig(**{**cfg.__dict__, **saved_cfg})
    if "roi" in ckpt:
        roi = RT_PARAM(**ckpt["roi"])
    elif full_detector_roi:
        roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)
    k_nominal = ckpt.get("k_nominal", k_nominal)
    un_nominal = ckpt.get("un_nominal", un_nominal)
    vn_nominal = ckpt.get("vn_nominal", vn_nominal)
    SOD = ckpt.get("SOD", SOD)
    SDD = ckpt.get("SDD", SDD)
    nominal_orbit_axis = ckpt.get("nominal_orbit_axis", nominal_orbit_axis)
    nominal_clockwise_sign = ckpt.get("nominal_clockwise_sign", nominal_clockwise_sign)
    nominal_include_endpoint = ckpt.get("nominal_include_endpoint", nominal_include_endpoint)
    use_beamcenter_geo = ckpt.get("use_beamcenter_geo", use_beamcenter_geo)
    ts_max_mm = ckpt.get("ts_max_mm", ts_max_mm)
    tp_max_mm = ckpt.get("tp_max_mm", tp_max_mm)
    rot_max_deg = ckpt.get("rot_max_deg", rot_max_deg)

    if not torch.cuda.is_available():
        raise RuntimeError("Joseph calibration requires an NVIDIA CUDA GPU; see README.md.")
    from models.MotionNetHash import MotionNetHash_9DoF

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")

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
    from run_single_sample import main

    main()
