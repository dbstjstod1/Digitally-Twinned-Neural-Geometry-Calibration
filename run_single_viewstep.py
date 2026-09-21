"""Train the Joseph neural calibration model on the shared Denseball preset."""

import argparse
from pathlib import Path

from configs.denseball import (
    MOTION_BOUNDS, NOMINAL_GEOMETRY, PROJECTIONS_PATH, TRAINING_RECIPE,
    VOLUME_PATH, make_config, make_roi, training_dir,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view_step", type=int, nargs="?", default=1)
    parser.add_argument("--volume", type=Path, default=VOLUME_PATH)
    parser.add_argument("--projections", type=Path, default=PROJECTIONS_PATH)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=TRAINING_RECIPE["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_RECIPE["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAINING_RECIPE["lr"])
    parser.add_argument("--seed", type=int, default=TRAINING_RECIPE["seed"])
    args = parser.parse_args()
    if min(args.view_step, args.epochs, args.batch_size) < 1 or args.lr <= 0:
        parser.error("view_step, epochs, batch-size and lr must be positive")
    for path in (args.volume, args.projections):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")

    from AI_Geocal import train_motion_hash_model

    cfg = make_config()
    recipe = dict(TRAINING_RECIPE)
    recipe.update(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    train_motion_hash_model(
        cfg=cfg, volume_path=str(args.volume), proj_meas_path=str(args.projections),
        roi=make_roi(cfg), out_dir=str(args.out_dir or training_dir(args.view_step)),
        view_step=args.view_step, **recipe, **MOTION_BOUNDS, **NOMINAL_GEOMETRY,
    )


if __name__ == "__main__":
    main()
