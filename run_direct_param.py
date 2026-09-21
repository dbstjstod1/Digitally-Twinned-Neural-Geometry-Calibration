"""
CLI runner for the direct per-view 9-DoF baseline.

  python run_direct_param.py run   --start S --stop E --niters N --lr LR
  python run_direct_param.py merge

Uses the cfg_4T setup (matches AI_Geocal / Sample __main__) and view_step=1.
out_dir = result_denseball/9DoF_analytic_nominalP_10/direct_param_vs1
"""

import sys
import argparse

from AI_Geocal_direct import (
    ReconConfig, RT_PARAM, train_direct_param_model, merge_direct_param,
)

BASE_OUT = "./result_denseball/9DoF_analytic_nominalP_10"
OUT_DIR = f"{BASE_OUT}/direct_param_vs1"

VOLUME = "./open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw"
PROJ = "./Denseball_proj_480.raw"


def make_cfg():
    return ReconConfig(
        NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0,
        nu=776, nv=1264, ori_nu=776, ori_nv=1264, du=0.228, dv=0.228,
        imsx=929, imsy=929, imsz=801, dx=0.2, dy=0.2, dz=0.2,
        ureverse_raw=1, vreverse_raw=1, recon_type=1,
        n_samples=256, chunk_size=16384,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=480)
    ap.add_argument("--niters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--outdir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    cfg = make_cfg()
    roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)

    if args.mode == "run":
        train_direct_param_model(
            cfg=cfg, volume_path=VOLUME, proj_meas_path=PROJ, roi=roi,
            out_dir=args.outdir,
            n_iters=args.niters, lr=args.lr, seed=0,
            ts_max_mm=10.0, tp_max_mm=10.0, rot_max_deg=10.0,
            view_step=1, view_start=args.start, view_stop=args.stop,
            k_nominal=650.0, un_nominal=34.0, vn_nominal=15.0,
            SOD=443.0, SDD=650.0,
            nominal_orbit_axis="y", nominal_clockwise_sign=-1.0,
            nominal_include_endpoint=False, use_beamcenter_geo=False,
        )
    else:
        merge_direct_param(args.outdir, n_views=cfg.NLAM)


if __name__ == "__main__":
    main()
