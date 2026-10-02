"""Generate circular or sineSpin validation inputs from a fixed ball phantom."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from .coordinates import geometry_to_pmat
from .fov import require_box_fov
from .landmarks import extract
from .noise import poisson_noisy_projections
from .pipeline import load_volume, project, project_points, sha256
from .trajectories import build_validation_family, resolve_validation_config


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference_roi(labels, nominal_pixel, rows, cols, target_hash, voxel):
    corners = []
    for bead in labels["landmarks"]:
        bounds = np.array(bead["threshold_bbox_edge_xyz_mm"])
        bounds[0] -= 2 * voxel
        bounds[1] += 2 * voxel
        corners.extend(product(*zip(bounds[0], bounds[1])))
    corners = np.asarray(corners)
    uv = project_points(nominal_pixel, corners)
    lower = np.floor(uv.min(1) - 20).astype(int)
    upper = np.ceil(uv.max(1) + 20).astype(int) + 1
    size = (upper - lower).max(0)
    lo = np.floor((lower + upper - size) / 2).astype(int)
    boxes = np.c_[lo, lo + size]
    if not (
        np.all(boxes[:, :2] >= 0)
        and np.all(boxes[:, 2] <= cols)
        and np.all(boxes[:, 3] <= rows)
    ):
        raise ValueError("Nominal/reference ROI does not fit the detector")
    return (
        dict(
            definition="Known fixed reference bead bounding boxes plus two-voxel halo, projected using nominal only; 20px margin and common crop shape; frozen before fitting",
            target_sha256=target_hash,
            detector_shape_vu=[rows, cols],
            boxes_xyxy=boxes.tolist(),
            crop_shape_vu=size[::-1].tolist(),
        ),
        corners,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["circular", "sinespin"], required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("examples/trajectories.json")
    )
    parser.add_argument("--shape-zyx", type=int, nargs=3, default=[651, 643, 643])
    parser.add_argument("--voxel-mm", type=float, default=0.4)
    parser.add_argument("--expected-balls", type=int, default=35)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    import torch
    from .leap import configure_leap

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    config = resolve_validation_config(json.loads(args.config.read_text()))
    if config["perturbation_model"] != "gantry_pivot_rotation":
        parser.error("This reproduction uses the fixed offset-pivot acquisition")
    family, _ = build_validation_family(args.family, config)
    nominal, truth = family["nominal"], family["validation"]
    shape, voxel = tuple(args.shape_zyx), args.voxel_mm
    fov = require_box_fov({"nominal": nominal, "truth": truth}, shape, voxel)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    labels, audit = extract(
        args.volume,
        shape_zyx=shape,
        voxel_mm=voxel,
        expected_components=args.expected_balls,
    )
    save_json(out / "landmarks.json", labels)
    save_json(out / "phantom_audit.json", audit)
    volume = load_volume(args.volume, device, shape)
    ct = configure_leap(truth, shape, voxel, device)
    target = torch.empty(
        (truth.n_views, truth.detector_rows, truth.detector_cols), device=device
    )
    ct.project_gpu(target, volume)
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(target).all()) or float(target.max()) <= 0:
        raise RuntimeError("Invalid independent projections")
    clean = target.cpu().numpy()
    noisy, noise, _ = poisson_noisy_projections(clean, i0=44000.0, seed=0)
    np.save(out / "target_projections.npy", noisy)
    for name, g in [("nominal", nominal), ("truth", truth)]:
        np.save(out / f"P_{name}_world_mm.npy", geometry_to_pmat(g))
        np.save(out / f"P_{name}_pixel.npy", g.projection_matrices())
    # Cross-check independent CUDA generation against the fitting projector.
    probe = np.unique(
        np.linspace(0, truth.n_views - 1, min(12, truth.n_views)).round().astype(int)
    )
    oracle = []
    with torch.no_grad():
        for batch in np.array_split(probe, 3):
            p = torch.as_tensor(geometry_to_pmat(truth)[batch], device=device)
            oracle.append(project(volume, p, truth, voxel=voxel).cpu().numpy())
    floor = float(
        np.linalg.norm((np.concatenate(oracle) - clean[probe]).astype(np.float64))
        / np.linalg.norm(clean[probe].astype(np.float64))
    )
    if floor > 0.01:
        raise RuntimeError(
            f"Independent projector relative discrepancy {floor} exceeds 1%"
        )
    roi, corners = reference_roi(
        labels,
        nominal.projection_matrices(),
        truth.detector_rows,
        truth.detector_cols,
        sha256(out / "target_projections.npy"),
        voxel,
    )
    save_json(out / "loss_roi.json", roi)
    # GT verifies coverage after the ROI is frozen; it never changes its size.
    boxes = np.asarray(roi["boxes_xyxy"])
    uv = project_points(truth.projection_matrices(), corners)
    margins = np.stack(
        (
            uv[..., 0] - boxes[:, None, 0] + 0.5,
            boxes[:, None, 2] - 0.5 - uv[..., 0],
            uv[..., 1] - boxes[:, None, 1] + 0.5,
            boxes[:, None, 3] - 0.5 - uv[..., 1],
        ),
        -1,
    )
    if margins.min() <= 0:
        raise ValueError("Frozen nominal ROI clips GT bead support")
    files = [
        "target_projections.npy",
        "P_nominal_world_mm.npy",
        "P_nominal_pixel.npy",
        "P_truth_world_mm.npy",
        "P_truth_pixel.npy",
        "landmarks.json",
        "loss_roi.json",
    ]
    save_json(
        out / "input.json",
        dict(
            family=args.family,
            trajectory=config,
            volume=dict(
                shape_zyx=list(shape), voxel_mm=voxel, sha256=sha256(args.volume)
            ),
            detector=dict(
                detector_rows=truth.detector_rows,
                detector_cols=truth.detector_cols,
                pixel_height=truth.pixel_height,
                pixel_width=truth.pixel_width,
            ),
            sha256={name: sha256(out / name) for name in files},
            noise=noise,
            fov=fov,
            independent_projector_relative_l2=floor,
            roi_minimum_margin_px=float(margins.min()),
            leap_library_sha256=sha256(Path(ct.libprojectors._name).resolve()),
            source_sha256={
                str(p.relative_to(Path(__file__).parent)): sha256(p)
                for p in Path(__file__).parent.rglob("*.py")
            },
        ),
    )
    print(f"Prepared {args.family}: {out}", flush=True)


if __name__ == "__main__":
    main()
