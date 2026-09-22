"""Extract fixed 3-D bead landmarks from the supplied Denseball reference raw.

This is a CPU-only evaluation helper. Landmarks must not be used to initialize
or optimize an image-based geometry-calibration experiment. The first voxel
centre is explicit: the default centres the raw array at the sineSpin isocentre
and does not apply the historical Denseball coordinate swap or half-cone shift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from configs.denseball import VOLUME_PATH


def extract_denseball_landmarks(
    volume_path: str | Path = VOLUME_PATH,
    *,
    shape_zyx: tuple[int, int, int] = (801, 929, 929),
    voxel_size_mm: float = 0.2,
    first_voxel_xyz_mm: tuple[float, float, float] | None = None,
    threshold: float = 0.05,
) -> dict:
    """Find isolated high-attenuation components and their contrast centroids.

    Components are identified with 26-connectivity above ``threshold``. For
    each bead, subtract the local cylinder background sampled three slices
    beyond the occupied ring, and include its low-intensity interpolation tails
    in a two-voxel expanded bounding box. Only one occupied z-band is labelled
    at a time, avoiding a multi-gigabyte integer label volume.
    """
    path = Path(volume_path)
    shape = np.asarray(shape_zyx, dtype=int)
    if shape.shape != (3,) or np.any(shape <= 0) or voxel_size_mm <= 0:
        raise ValueError("A positive 3-D shape and voxel pitch are required.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("A positive finite bead threshold is required.")
    expected_size = int(np.prod(shape, dtype=np.int64)) * 4
    if path.stat().st_size != expected_size:
        raise ValueError(f"Raw size {path.stat().st_size} != expected {expected_size}.")
    origin = (-0.5 * (shape[::-1] - 1) * voxel_size_mm
              if first_voxel_xyz_mm is None else np.asarray(first_voxel_xyz_mm, dtype=float))
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("first_voxel_xyz_mm must contain three finite values.")
    volume = np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))
    occupied = np.zeros(shape[0], dtype=bool)
    hasher = hashlib.sha256()
    minimum, maximum, nonzero = np.inf, -np.inf, 0
    for z, section in enumerate(volume):
        hasher.update(memoryview(section).cast("B"))
        minimum = min(minimum, float(section.min()))
        maximum = max(maximum, float(section.max()))
        nonzero += int(np.count_nonzero(section))
        occupied[z] = bool(np.any(section > threshold))
    if not occupied.any():
        raise ValueError("No bead material was found above the threshold.")
    edges = np.diff(np.r_[False, occupied, False].astype(np.int8))
    bands = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    beads = []
    for band_id, (lo, hi) in enumerate(bands):
        labels, count = ndimage.label(volume[lo:hi] > threshold,
                                      structure=np.ones((3, 3, 3), dtype=bool))
        for component_id, bbox in enumerate(ndimage.find_objects(labels), 1):
            if bbox is None:
                continue
            component_size = int(np.count_nonzero(labels[bbox] == component_id))
            if component_size < 8:
                raise ValueError("Tiny high-attenuation component found; inspect the input.")
            b0 = np.array([lo + bbox[0].start, bbox[1].start, bbox[2].start])
            b1 = np.array([lo + bbox[0].stop, bbox[1].stop, bbox[2].stop])
            b0 = np.maximum(b0 - 2, 0)
            b1 = np.minimum(b1 + 2, shape)
            zs, ys, xs = (slice(int(a), int(b)) for a, b in zip(b0, b1))
            before, after = max(0, int(lo) - 3), min(int(shape[0]) - 1, int(hi) + 2)
            background = 0.5 * (volume[before, ys, xs].astype(np.float64)
                                + volume[after, ys, xs].astype(np.float64))
            weights = np.maximum(volume[zs, ys, xs].astype(np.float64) - background, 0.0)
            mass = float(weights.sum())
            center_local = np.asarray(ndimage.center_of_mass(weights))
            center_zyx = b0 + center_local
            center_xyz_mm = origin + voxel_size_mm * center_zyx[::-1]
            beads.append({
                "ring_id": band_id,
                "index_zyx": center_zyx.tolist(),
                "xyz_mm": center_xyz_mm.tolist(),
                "threshold_voxel_count": component_size,
                "integrated_contrast_mm2": mass * voxel_size_mm ** 3,
            })
        del labels
    # Stable ordering establishes fixed correspondences; no nearest-neighbour
    # rematching is allowed when a candidate camera matrix is evaluated.
    beads.sort(key=lambda bead: (bead["ring_id"], float(np.arctan2(
        bead["index_zyx"][1] - 0.5 * (shape[1] - 1),
        bead["index_zyx"][2] - 0.5 * (shape[2] - 1)))))
    for index, bead in enumerate(beads):
        bead["landmark_id"] = index
    centers = np.asarray([bead["xyz_mm"] for bead in beads])
    ring_counts = [sum(bead["ring_id"] == ring_id for bead in beads)
                   for ring_id in range(len(bands))]
    ring_z = [float(np.mean([bead["xyz_mm"][2] for bead in beads
                            if bead["ring_id"] == ring_id]))
              for ring_id in range(len(bands))]
    return {
        "volume_filename": path.name,
        "volume_sha256": hasher.hexdigest(),
        "volume_bytes": expected_size,
        "shape_zyx": shape.tolist(),
        "voxel_size_mm": voxel_size_mm,
        "first_voxel_xyz_mm": origin.tolist(),
        "coordinate_rule": "xyz_mm = first_voxel_xyz_mm + voxel_size_mm * index_zyx[::-1]",
        "coordinate_permutation_applied": False,
        "attenuation_min_max_mm_inverse": [minimum, maximum],
        "nonzero_voxel_count": nonzero,
        "threshold_mm_inverse": threshold,
        "centroid_method": "positive contrast above adjacent ring-free cylinder background",
        "landmark_count": len(beads),
        "ring_counts": ring_counts,
        "ring_z_mm": ring_z,
        "landmark_bounds_xyz_mm": [centers.min(axis=0).tolist(), centers.max(axis=0).tolist()],
        "use": "held-out geometric evaluation only; not calibration supervision",
        "symmetry_warning": "24 near-equally-spaced beads per ring admit 15-degree rotational aliases; preserve landmark identities",
        "landmarks": beads,
    }


def project_landmarks(pmat: np.ndarray, xyz_mm: np.ndarray) -> np.ndarray:
    """Project fixed world points with physical-world-to-pixel-centre matrices.

    Homogeneous division makes the result invariant to nonzero scalar gauges
    of P. It does not align the phantom or rematch symmetric bead identities.
    """
    matrices = np.asarray(pmat, dtype=np.float64)
    points = np.asarray(xyz_mm, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 4):
        raise ValueError("pmat must have shape [views, 3, 4].")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz_mm must have shape [landmarks, 3].")
    scale = np.max(np.abs(matrices), axis=(1, 2), keepdims=True)
    matrices = np.divide(matrices, scale, out=np.zeros_like(matrices), where=scale > 0)
    homogeneous = np.c_[points, np.ones(len(points))]
    camera = np.einsum("vij,nj->vni", matrices, homogeneous)
    valid = np.abs(camera[..., 2]) > 1e-12
    result = np.full(camera.shape[:2] + (2,), np.nan)
    np.divide(camera[..., :2], camera[..., 2, None], out=result, where=valid[..., None])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, default=VOLUME_PATH)
    parser.add_argument("--output", type=Path,
                        default=Path("result_sinespin/denseball_calibration/input/landmarks.json"))
    parser.add_argument("--first-voxel-xyz-mm", type=float, nargs=3)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    result = extract_denseball_landmarks(args.volume, threshold=args.threshold,
                                        first_voxel_xyz_mm=args.first_voxel_xyz_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "landmarks"}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
