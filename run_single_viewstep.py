"""
Run AI_Geocal training for a SINGLE view_step (given as argv[1]).

out_dir = .../no_initP_k_un_vn_SOD_SDD_vs{view_step}

Pin to a GPU from the outside via CUDA_VISIBLE_DEVICES so several of these
can run in parallel across GPUs. Mirrors the cfg_4T setup from
AI_Geocal.__main__.
"""

import os
import sys

from AI_Geocal import ReconConfig, RT_PARAM, train_motion_hash_model


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run_single_viewstep.py <view_step>")
    vs = int(sys.argv[1])

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

    roi = RT_PARAM(x=0, y=0, width=cfg_4T.nu, height=cfg_4T.nv)

    volume_path = "./open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw"
    proj_meas_path = "./Denseball_proj_480.raw"

    base_out = "./result_denseball/9DoF_analytic_nominalP_10"
    out_dir = os.path.join(base_out, f"no_initP_k_un_vn_SOD_SDD_vs{vs}")

    print("=" * 80, flush=True)
    print(f"[RUN] view_step={vs} -> out_dir={out_dir} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    print("=" * 80, flush=True)

    train_motion_hash_model(
        cfg=cfg_4T,
        volume_path=volume_path,
        proj_meas_path=proj_meas_path,
        roi=roi,
        out_dir=out_dir,
        epochs=100,
        batch_size=4,
        lr=1e-3,
        use_amp=False,
        seed=0,
        ts_max_mm=5.0 + 5,
        tp_max_mm=5.0 + 5,
        rot_max_deg=5.0 + 5,
        loss_type="lncc",
        save_every=5,
        view_step=vs,

        # Analytic nominal geometry parameters (match AI_Geocal.__main__).
        k_nominal=650.0 + 0,
        un_nominal=34.0 + 0,
        vn_nominal=15.0 + 0,
        SOD=443 + 0.0,
        SDD=650.0,
        nominal_orbit_axis="y",
        nominal_clockwise_sign=-1.0,
        nominal_include_endpoint=False,
        use_beamcenter_geo=False,
    )

    print(f"[RUN] done view_step={vs}", flush=True)


if __name__ == "__main__":
    main()
