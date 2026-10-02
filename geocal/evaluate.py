"""Evaluate projection geometry after fitting; GT never enters training."""

import argparse
import json
from pathlib import Path
import numpy as np
from .coordinates import centered_source_positions
from .parameters import effective_parameters_from_pmat
from .pipeline import project_points, sha256


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(x.mean()),
        std=float(x.std(ddof=0)),
        rms=float(np.sqrt(np.mean(x * x))),
        maximum=float(x.max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    inp, run = args.input_dir, args.run_dir
    meta = json.loads((inp / "input.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    if provenance["input_sha256"] != sha256(inp / "input.json"):
        raise ValueError("Run belongs to different prepared inputs")
    for name in (
        "P_truth_pixel.npy",
        "P_truth_world_mm.npy",
        "landmarks.json",
        "P_nominal_pixel.npy",
        "P_nominal_world_mm.npy",
    ):
        if sha256(inp / name) != meta["sha256"][name]:
            raise ValueError(f"Changed evaluation input: {name}")
    points = np.array(
        [
            p["xyz_mm"]
            for p in json.loads((inp / "landmarks.json").read_text())["landmarks"]
        ]
    )
    truth = np.load(inp / "P_truth_world_mm.npy")
    nominal = np.load(inp / "P_nominal_world_mm.npy")
    gt_uv = project_points(np.load(inp / "P_truth_pixel.npy"), points)
    gt_source = centered_source_positions(truth)
    gt_motion = effective_parameters_from_pmat(truth, nominal)["parameters_9"]
    recipe = json.loads((run / "recipe.json").read_text())
    train_mask = np.arange(len(truth)) % recipe["view_step"] == 0
    results = dict(
        rpe_definition="Euclidean detector-pixel error over fixed view/ball pairs; population std (ddof=0)",
        source_definition="Euclidean source error per view in centered physical xyz (mm)",
        methods={},
    )
    for name, folder, stem in [
        ("nominal", inp, "nominal"),
        ("proposed", run, "proposed"),
    ]:
        p = np.load(folder / f"P_{stem}_world_mm.npy")
        pixel = np.load(folder / f"P_{stem}_pixel.npy")
        rpe = np.linalg.norm(project_points(pixel, points) - gt_uv, axis=-1)
        source = centered_source_positions(p)
        params = effective_parameters_from_pmat(p, nominal)
        results["methods"][name] = dict(
            rpe_px=summary(rpe),
            training_rpe_px=summary(rpe[train_mask]),
            heldout_rpe_px=summary(rpe[~train_mask]) if (~train_mask).any() else None,
            source_error_mm=summary(np.linalg.norm(source - gt_source, axis=-1)),
            parameters9_rmse=np.sqrt(
                np.mean((params["parameters_9"] - gt_motion) ** 2, axis=0)
            ).tolist(),
            parameter_names=params["parameter_names"],
        )
        np.save(run / f"{name}_rpe_per_view_ball.npy", rpe)
        np.savetxt(
            run / f"{name}_source_xyz_mm.csv",
            source,
            delimiter=",",
            header="x_mm,y_mm,z_mm",
            comments="",
        )
        np.savetxt(
            run / f"{name}_parameters9.csv",
            params["parameters_9"],
            delimiter=",",
            header=",".join(params["parameter_names"]),
            comments="",
        )
    np.savetxt(
        run / "gt_source_xyz_mm.csv",
        gt_source,
        delimiter=",",
        header="x_mm,y_mm,z_mm",
        comments="",
    )
    np.savetxt(
        run / "gt_parameters9.csv",
        gt_motion,
        delimiter=",",
        header=",".join(params["parameter_names"]),
        comments="",
    )
    (run / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
