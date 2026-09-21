"""Plot the nine learned geometry parameters exported by training or sampling."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_motion(result_dir):
    names = ("ts_mm.npy", "tp_mm.npy", "rot_deg.npy")
    for prefix in ("", "motion_"):
        paths = [result_dir / f"{prefix}{name}" for name in names]
        if all(path.is_file() for path in paths):
            arrays = [np.load(path, allow_pickle=False) for path in paths]
            break
    else:
        raise FileNotFoundError(
            f"{result_dir}: expected ts_mm.npy, tp_mm.npy, rot_deg.npy "
            "(or all three with the motion_ prefix)."
        )

    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] != 3:
        raise ValueError(f"Expected motion arrays with shape (views, 3); got {shape}.")
    if any(array.shape != shape or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Motion arrays must have matching (views, 3) shapes and finite values.")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Training or sampling output directory.")
    parser.add_argument("--out-dir", type=Path, help="Plot destination (default: RESULT_DIR/motion_plots_intrinsic_extrinsic).")
    args = parser.parse_args()

    ts, tp, rot = load_motion(args.result_dir)
    out_dir = args.out_dir or args.result_dir / "motion_plots_intrinsic_extrinsic"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = (
        (ts, (r"$\Delta c_u$", r"$\Delta c_v$", r"$\Delta f$"),
         "Intrinsic translation (mm)", "intrinsic_parameters_cu_cv_f.png"),
        (tp, (r"$\Delta t_x$", r"$\Delta t_y$", r"$\Delta t_z$"),
         "Extrinsic translation (mm)", "extrinsic_parameters_translation.png"),
        (rot, (r"$\Delta r_x$", r"$\Delta r_y$", r"$\Delta r_z$"),
         "Extrinsic rotation (deg)", "extrinsic_parameters_rotation.png"),
    )
    for values, labels, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        for column, label in enumerate(labels):
            ax.plot(np.arange(len(values)), values[:, column], linewidth=2, label=label)
        ax.set_xlabel("View index", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
