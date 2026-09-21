"""Export before/after projections and gantry geometry from a Joseph checkpoint."""

import argparse
from pathlib import Path
import re

from configs.denseball import (
    MOTION_BOUNDS, NOMINAL_GEOMETRY, VOLUME_PATH, make_config, make_roi, training_dir,
)


def _latest_ckpt(train_dir):
    candidates = []
    for path in Path(train_dir).glob("motion_model_ep*.pth"):
        match = re.fullmatch(r"motion_model_ep(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no motion_model_ep*.pth checkpoint in {train_dir}")
    return max(candidates)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view_step", type=int, nargs="?", default=1)
    parser.add_argument("--volume", type=Path, default=VOLUME_PATH)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--checkpoint", type=Path)
    selection.add_argument("--train-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proj-batch", type=int, default=2)
    args = parser.parse_args()
    if args.view_step < 1 or args.proj_batch < 1:
        parser.error("view_step and proj-batch must be positive")
    if not args.volume.is_file():
        parser.error(f"input volume does not exist: {args.volume}")
    train_dir = args.train_dir or (args.checkpoint.parent if args.checkpoint else training_dir(args.view_step))
    try:
        checkpoint = args.checkpoint or _latest_ckpt(train_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    from Sample import export_aligned_projections_and_gantry

    cfg = make_config()
    export_aligned_projections_and_gantry(
        cfg=cfg, volume_path=str(args.volume), roi=make_roi(cfg),
        ckpt_path=str(checkpoint), out_dir=str(args.out_dir or train_dir / "export"),
        apply_batch=64, proj_batch=args.proj_batch, use_amp_projector=False,
        save_motion_npy=True, save_before_raw=True, save_nominal_gantry=True,
        isocenter_method="circlefit", y_mode="mean", outlier_reject=False,
        **MOTION_BOUNDS, **NOMINAL_GEOMETRY,
    )


if __name__ == "__main__":
    main()
