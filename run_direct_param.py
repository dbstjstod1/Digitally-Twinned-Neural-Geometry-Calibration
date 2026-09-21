"""Run or merge the independent per-view Joseph calibration baseline."""

import argparse
from pathlib import Path

from configs.denseball import (
    DIRECT_OUT_DIR, DIRECT_RECIPE, MOTION_BOUNDS, NOMINAL_GEOMETRY,
    PROJECTIONS_PATH, VOLUME_PATH, make_config, make_roi,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "merge"])
    parser.add_argument("--volume", type=Path, default=VOLUME_PATH)
    parser.add_argument("--projections", type=Path, default=PROJECTIONS_PATH)
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", type=Path, default=DIRECT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=480)
    parser.add_argument("--niters", type=int, default=DIRECT_RECIPE["n_iters"])
    parser.add_argument("--lr", type=float, default=DIRECT_RECIPE["lr"])
    parser.add_argument("--seed", type=int, default=DIRECT_RECIPE["seed"])
    args = parser.parse_args()
    if args.mode == "run":
        if not 0 <= args.start < args.stop <= 480:
            parser.error("view range must satisfy 0 <= start < stop <= 480")
        if args.niters < 1 or args.lr <= 0:
            parser.error("niters and lr must be positive")
        for path in (args.volume, args.projections):
            if not path.is_file():
                parser.error(f"input file does not exist: {path}")

    from AI_Geocal_direct import train_direct_param_model, merge_direct_param

    cfg = make_config()
    if args.mode == "run":
        train_direct_param_model(
            cfg=cfg, volume_path=str(args.volume), proj_meas_path=str(args.projections),
            roi=make_roi(cfg), out_dir=str(args.out_dir), n_iters=args.niters,
            lr=args.lr, seed=args.seed, view_step=1,
            view_start=args.start, view_stop=args.stop,
            **MOTION_BOUNDS, **NOMINAL_GEOMETRY,
        )
    else:
        merge_direct_param(str(args.out_dir), n_views=cfg.NLAM)


if __name__ == "__main__":
    main()
