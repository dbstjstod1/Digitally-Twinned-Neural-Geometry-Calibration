"""
Export aligned projections + gantry for a SINGLE view_step (argv[1]).

Loads the LATEST checkpoint (highest epoch) from the training out_dir:
    .../no_initP_k_un_vn_SOD_SDD_vs{vs}/motion_model_ep*.pth
and writes export results to:
    .../no_initP_k_un_vn_SOD_SDD_vs{vs}_export

Pin to a GPU from the outside via CUDA_VISIBLE_DEVICES.
Mirrors the cfg_4T setup from Sample.__main__; nominal geometry and
*_max bounds must match the training run.
"""

import os
import re
import sys
import glob

from Sample import ReconConfig, RT_PARAM, export_aligned_projections_and_gantry


def _latest_ckpt(train_dir: str) -> str:
    cands = glob.glob(os.path.join(train_dir, "motion_model_ep*.pth"))
    if not cands:
        raise SystemExit(f"[sample] no checkpoint found in {train_dir}")

    def _ep(p):
        m = re.search(r"ep(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    best = max(cands, key=_ep)
    print(f"[sample] using checkpoint: {best} (ep={_ep(best)})", flush=True)
    return best


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run_single_sample.py <view_step>")
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

    base_out = "./result_denseball/9DoF_analytic_nominalP_10"
    train_dir = os.path.join(base_out, f"no_initP_k_un_vn_SOD_SDD_vs{vs}")
    out_dir = os.path.join(base_out, f"no_initP_k_un_vn_SOD_SDD_vs{vs}_export")

    ckpt_path = _latest_ckpt(train_dir)

    print("=" * 80, flush=True)
    print(f"[SAMPLE] view_step={vs} -> out_dir={out_dir} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    print("=" * 80, flush=True)

    export_aligned_projections_and_gantry(
        cfg=cfg_4T,
        volume_path=volume_path,
        roi=roi,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        out_proj_raw_before="Projections_before_9DoF.raw",
        out_proj_raw="Projections_aligned_9DoF.raw",
        out_gantry_dat="Gantry_updated_9DoF.dat",
        out_gantry_nominal_dat="Gantry_nominal_9DoF.dat",
        ts_max_mm=5.0 + 5,
        tp_max_mm=5.0 + 5,
        rot_max_deg=5.0 + 5,
        apply_batch=64,
        proj_batch=2,

        # Must match the analytic nominal geometry used in training.
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
        isocenter_method="circlefit",
        y_mode="mean",
        outlier_reject=False,
    )

    print(f"[SAMPLE] done view_step={vs}", flush=True)


if __name__ == "__main__":
    main()
