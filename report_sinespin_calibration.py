"""CPU-only independent geometry evaluation of saved sineSpin calibration P.

This script reads completed matrix snapshots and never changes training,
selects a checkpoint, aligns the phantom, or rematches symmetric landmarks.
The experiment's final, fixed epoch must be used for the main comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration_geometry import centered_source_positions, pmat_to_pixel
from denseball_landmarks import project_landmarks


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Evaluation values must be nonempty and finite.")
    return {"rms": float(np.sqrt(np.mean(values ** 2))),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "maximum": float(values.max())}


def describe_data(metadata: dict, landmarks: dict) -> dict:
    """Derive labels from the recorded input, without assuming a bead layout."""
    volume = metadata["volume"]
    if (landmarks["volume_sha256"] != volume["sha256"]
            or landmarks["shape_zyx"] != volume["shape_zyx"]
            or landmarks["voxel_size_mm"] != volume["voxel_mm"]):
        raise ValueError("Landmarks do not describe the prepared phantom and voxel grid.")
    count = len(landmarks["landmarks"])
    if count == 0 or landmarks.get("landmark_count", count) != count:
        raise ValueError("Landmark count is empty or inconsistent.")
    noise = metadata.get("noise")
    if noise:
        i0 = noise.get("i0_photons_per_detector_pixel_per_view")
        if i0 is not None:
            noise_label = f"Poisson quantum noise, I0={i0:g} photons/pixel/view"
            target_label = f"Poisson target\nI0={i0:g}"
        else:
            noise_label = "Projection noise as recorded in input metadata"
            target_label = "Recorded noisy target"
    elif "noiseless" in metadata.get("generator", "").lower():
        noise_label, target_label = "Noiseless synthetic projections", "Target: independent Joseph"
    else:
        noise_label, target_label = "No added noise recorded", "Recorded projection target"
    return {
        "phantom_label": f"Ball phantom ({count} beads)",
        "phantom_filename": Path(volume["path"]).name,
        "noise_label": noise_label,
        "target_label": target_label,
        "noise": noise,
        "symmetry_caution": landmarks.get("symmetry_warning",
            "No phantom symmetry correction is applied; fixed landmark IDs are retained."),
    }


def evaluate_saved_run(input_dir: Path, run_dir: Path) -> dict:
    metadata = json.loads((input_dir / "experiment.json").read_text())
    experiment = json.loads((run_dir / "experiment.json").read_text())
    landmark_file = input_dir / "landmarks.json"
    landmarks = json.loads(landmark_file.read_text())
    description = describe_data(metadata, landmarks)
    if experiment["input"] != metadata:
        raise ValueError("Run input metadata differs from the requested prepared input.")
    xyz = np.asarray([bead["xyz_mm"] for bead in landmarks["landmarks"]])
    truth_p = np.load(input_dir / "P_truth_pixel.npy")
    truth_source = np.load(input_dir / "truth_geometry.npz")["source_positions"]
    rows, cols = metadata["truth"]["detector_shape_vu"]
    dv, du = metadata["truth"]["pixel_vu_mm"]
    true_q = project_landmarks(truth_p, xyz)
    depth = np.einsum("vj,nj->vn", truth_p[:, 2], np.c_[xyz, np.ones(len(xyz))])
    visible = ((true_q[..., 0] >= -0.5) & (true_q[..., 0] <= cols - 0.5)
               & (true_q[..., 1] >= -0.5) & (true_q[..., 1] <= rows - 0.5)
               & (depth > 0))
    true_azimuth = np.arctan2(truth_source[:, 0], -truth_source[:, 1])
    true_elevation = np.arctan2(truth_source[:, 2], np.linalg.norm(truth_source[:, :2], axis=1))
    view_step = int(experiment["recipe"]["view_step"])
    training_views = np.arange(len(truth_p)) % view_step == 0
    history = {}
    with (run_dir / "loss_history.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            history[int(row["epoch"])] = {name: float(value) for name, value in row.items()}
    paths = [(0, input_dir / "P_nominal_world_mm.npy")]
    paths += [(int(path.stem[7:]), path) for path in sorted(run_dir.glob("P_epoch*.npy"))]
    snapshots = []
    for epoch, path in paths:
        matrix = np.load(path)
        estimate_q = project_landmarks(pmat_to_pixel(matrix, du=du, dv=dv), xyz)
        error = np.linalg.norm(estimate_q - true_q, axis=-1)
        source = centered_source_positions(matrix)
        azimuth_error = np.rad2deg(np.angle(np.exp(1j * (
            np.arctan2(source[:, 0], -source[:, 1]) - true_azimuth))))
        elevation_error = np.rad2deg(np.arctan2(source[:, 2], np.linalg.norm(source[:, :2], axis=1))
                                    - true_elevation)
        source_error = np.linalg.norm(source - truth_source, axis=-1)
        z_error = source[:, 2] - truth_source[:, 2]
        snapshot = {
            "epoch": epoch,
            "matrix_filename": path.name,
            "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training_loss": history.get(epoch, {}).get("loss"),
            "visible_ball_reprojection_error_px": distribution(error[visible]),
            "training_view_ball_reprojection_error_px": distribution(error[visible & training_views[:, None]]),
            "source_position_error_mm": distribution(source_error),
            "source_z_error_mm": distribution(np.abs(z_error)),
            "source_elevation_error_deg": distribution(np.abs(elevation_error)),
            "source_azimuth_error_deg": distribution(np.abs(azimuth_error)),
            "source_azimuth_error_signed_range_deg": [float(azimuth_error.min()), float(azimuth_error.max())],
            "source_z_mm": source[:, 2].tolist(),
            "per_view_ball_rmse_px": np.sqrt((error ** 2 * visible).sum(axis=1) / visible.sum(axis=1)).tolist(),
        }
        if np.any(~training_views):
            snapshot["untrained_view_ball_reprojection_error_px"] = distribution(error[visible & ~training_views[:, None]])
        snapshots.append(snapshot)
    return {
        "run_name": run_dir.name,
        "data_description": description,
        "recipe": experiment["recipe"],
        "latest_completed_snapshot_epoch": snapshots[-1]["epoch"],
        "fixed_final_epoch": int(experiment["recipe"]["epochs"]),
        "fixed_final_epoch_available": snapshots[-1]["epoch"] == int(experiment["recipe"]["epochs"]),
        "volume_sha256": landmarks["volume_sha256"],
        "landmarks_sha256": hashlib.sha256(landmark_file.read_bytes()).hexdigest(),
        "landmark_count": len(xyz),
        "visible_landmark_view_pairs": int(visible.sum()),
        "all_landmark_view_pairs": int(visible.size),
        "training_view_count": int(training_views.sum()),
        "untrained_view_count": int((~training_views).sum()),
        "evaluation_rule": "Fixed physical phantom, fixed landmark IDs, fixed true visibility; no gauge alignment, bead rematching, or checkpoint selection",
        "symmetry_caution": description["symmetry_caution"],
        "truth_source_z_mm": truth_source[:, 2].tolist(),
        "snapshots": snapshots,
    }


def save_figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshots = report["snapshots"]
    epochs = [item["epoch"] for item in snapshots]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [item["visible_ball_reprojection_error_px"]["rms"] for item in snapshots], "o-")
    axes[0, 0].set(ylabel="Fixed-ID bead reprojection RMSE (pixel)", xlabel="Epoch")
    axes[0, 1].plot(epochs, [item["source_position_error_mm"]["rms"] for item in snapshots], "o-")
    axes[0, 1].set(ylabel="Source position RMSE (mm)", xlabel="Epoch")
    axes[1, 0].plot(report["truth_source_z_mm"], color="black", label="True sineSpin")
    axes[1, 0].plot(snapshots[0]["source_z_mm"], "--", label="Circular input")
    for item in snapshots[1:]:
        final = item is snapshots[-1]
        axes[1, 0].plot(item["source_z_mm"], alpha=1 if final else 0.18,
                        label=f"Epoch {item['epoch']}" if final else None)
    axes[1, 0].set(xlabel="View index", ylabel="Source z (mm)")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [item["source_elevation_error_deg"]["rms"] for item in snapshots], "o-", label="Elevation")
    axes[1, 1].plot(epochs, [item["source_azimuth_error_deg"]["rms"] for item in snapshots], "o-", label="Azimuth")
    axes[1, 1].set(xlabel="Epoch", ylabel="Source angular RMSE (degree)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    status = "final epoch" if report["fixed_final_epoch_available"] else "training in progress"
    description = report["data_description"]
    fig.suptitle(f"{report['run_name']}: independent geometry audit ({status})\n"
                 f"{description['phantom_label']}; {description['noise_label']}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _check_geometry_against_runner(matrix, truth_p, truth_source, xyz, visible, du, dv, saved):
    """Recompute geometry from arrays, independently of the GPU runner's metrics."""
    q = project_landmarks(pmat_to_pixel(matrix, du=du, dv=dv), xyz)
    error = np.linalg.norm(q - project_landmarks(truth_p, xyz), axis=-1)
    source = centered_source_positions(matrix)
    recomputed = {
        "ball_reprojection_error_px": distribution(error[visible]),
        "source_position_error_mm": distribution(np.linalg.norm(source - truth_source, axis=-1)),
        "source_z_rmse_mm": float(np.sqrt(np.mean((source[:, 2] - truth_source[:, 2]) ** 2))),
    }
    if int(saved["visible_ball_view_pairs"]) != int(visible.sum()):
        raise ValueError("Runner visibility differs from the fixed true-geometry mask.")
    for category in ("ball_reprojection_error_px", "source_position_error_mm"):
        for name, value in recomputed[category].items():
            np.testing.assert_allclose(value, saved[category][name], rtol=1e-8, atol=1e-8,
                                       err_msg=f"Runner {category}.{name} does not match saved P.")
    np.testing.assert_allclose(recomputed["source_z_rmse_mm"], saved["source_z_rmse_mm"],
                               rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(source, saved["source_xyz_mm"], rtol=1e-10, atol=1e-8)
    return recomputed


def _compact_result(result: dict, geometry: dict) -> dict:
    """Keep scalar scores; omit the runner's per-view arrays and source positions."""
    compact = {
        "projection_relative_l2": {split: score["relative_l2"]
                                   for split, score in result["projection"].items()},
        "projection_view_counts": {split: score["views"]
                                   for split, score in result["projection"].items()},
        "lncc_loss_for_this_recipe": result["lncc_loss"],
        **geometry,
    }
    clean_scores = {split: score["relative_l2_to_clean"]
                    for split, score in result["projection"].items() if "relative_l2_to_clean" in score}
    if clean_scores:
        compact["projection_relative_l2_to_clean"] = clean_scores
    return compact


def compare_completed_runs(input_dir: Path, run_dirs: list[Path], summary_path: Path,
                           comparison_dir: Path) -> dict:
    """Validate final epoch 100 artifacts before publishing their comparison.

    Loading checkpoints is CPU-only. Projection scores are the saved complete
    GPU evaluations, while geometry scores are recomputed from the saved P
    arrays. No snapshot is selected using these scores.
    """
    if len(run_dirs) < 2 or len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError("Comparison requires at least two distinct completed run directories.")
    # Check completion before expensive data hashing or writing any outputs.
    for run in run_dirs:
        required = ("experiment.json", "metrics.json", "checkpoint.pt", "P_epoch0100.npy",
                    "P_optimized_world_mm.npy", "P_optimized_pixel.npy", "motion9.npy",
                    "projection_examples.npz", "loss_history.csv")
        for name in required:
            if not (run / name).is_file():
                raise ValueError(f"Run {run.name} is incomplete: missing {name}.")
        experiment = json.loads((run / "experiment.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        if experiment["recipe"]["epochs"] != 100 or metrics["epoch"] != 100:
            raise ValueError(f"Run {run.name} must contain its prespecified final epoch 100.")
    metadata = json.loads((input_dir / "experiment.json").read_text())
    landmarks = json.loads((input_dir / "landmarks.json").read_text())
    description = describe_data(metadata, landmarks)
    input_hashes = {name: sha256(input_dir / name) for name in (
        "experiment.json", "landmarks.json", "P_nominal_world_mm.npy", "P_nominal_pixel.npy",
        "P_truth_world_mm.npy", "P_truth_pixel.npy", "truth_geometry.npz", "nominal_geometry.npz",
        "target_projections.npy")}
    expected = {"target_projections.npy": metadata["target_sha256"],
                "P_nominal_world_mm.npy": metadata["nominal_pmat_sha256"],
                "P_truth_world_mm.npy": metadata["truth_pmat_sha256"]}
    for name, key in (("clean_projections.npy", "clean_sha256"),
                      ("photon_counts.npy", "photon_counts_sha256")):
        if key in metadata:
            input_hashes[name] = sha256(input_dir / name)
            expected[name] = metadata[key]
    if metadata.get("noise"):
        if not {"clean_sha256", "photon_counts_sha256"} <= metadata.keys():
            raise ValueError("Noisy input lacks the recorded clean-projection and photon-count hashes.")
        input_hashes["photon_noise.json"] = sha256(input_dir / "photon_noise.json")
        if json.loads((input_dir / "photon_noise.json").read_text()) != metadata["noise"]:
            raise ValueError("Photon-noise metadata differs from the prepared experiment.")
    if "preparation_source_sha256" in metadata:
        archive = input_dir / metadata["preparation_source_archive"]
        for name, expected_hash in metadata["preparation_source_sha256"].items():
            if sha256(archive / name) != expected_hash:
                raise ValueError(f"Archived preparation source changed: {name}.")
    for name, value in expected.items():
        if input_hashes[name] != value:
            raise ValueError(f"Prepared input has changed: {name}.")
    volume_path = Path(metadata["volume"]["path"])
    volume_hash = sha256(volume_path)
    if volume_hash != metadata["volume"]["sha256"] or volume_hash != landmarks["volume_sha256"]:
        raise ValueError("Raw phantom does not match the projections and landmarks.")
    xyz = np.asarray([bead["xyz_mm"] for bead in landmarks["landmarks"]])
    truth_p = np.load(input_dir / "P_truth_pixel.npy")
    truth_geometry = np.load(input_dir / "truth_geometry.npz")
    truth_source = truth_geometry["source_positions"]
    theta = truth_geometry["theta_deg"]
    rows, cols = metadata["truth"]["detector_shape_vu"]
    dv, du = metadata["truth"]["pixel_vu_mm"]
    q = project_landmarks(truth_p, xyz)
    depth = np.einsum("vj,nj->vn", truth_p[:, 2], np.c_[xyz, np.ones(len(xyz))])
    visible = ((q[..., 0] >= -0.5) & (q[..., 0] <= cols - 0.5)
               & (q[..., 1] >= -0.5) & (q[..., 1] <= rows - 0.5) & (depth > 0))
    target_memmap = np.load(input_dir / "target_projections.npy", mmap_mode="r")
    reports, summaries, examples = [], [], []
    common_recipe = None
    common_source_hashes = None
    nominal_summary = oracle_summary = None
    import torch
    for run in run_dirs:
        experiment = json.loads((run / "experiment.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        recipe = experiment["recipe"]
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != 100 or checkpoint["recipe"] != recipe or metrics["recipe"] != recipe:
            raise ValueError(f"Run {run.name}: checkpoint, metrics, and final recipe disagree.")
        if checkpoint["input_sha256"] != input_hashes["experiment.json"] or experiment["input"] != metadata:
            raise ValueError(f"Run {run.name} used different prepared inputs.")
        if checkpoint["history"][-1]["epoch"] != 100:
            raise ValueError(f"Run {run.name}: final checkpoint history is incomplete.")
        del checkpoint
        invariant_recipe = {key: value for key, value in recipe.items() if key != "loss_levels"}
        if common_recipe is None:
            common_recipe = invariant_recipe
        elif invariant_recipe != common_recipe:
            raise ValueError("The comparison must change only loss_levels, with one shared seed.")
        source_hashes = experiment["source_sha256"]
        for name, expected_hash in source_hashes.items():
            if sha256(ROOT / name) != expected_hash:
                raise ValueError(f"Source changed after run {run.name}: {name}.")
        if common_source_hashes is None:
            common_source_hashes = source_hashes
        elif source_hashes != common_source_hashes:
            raise ValueError("The compared runs used different source code.")
        artifact_hashes = {name: sha256(run / name) for name in (
            "experiment.json", "metrics.json", "checkpoint.pt", "P_epoch0100.npy",
            "P_optimized_world_mm.npy", "P_optimized_pixel.npy", "motion9.npy",
            "projection_examples.npz", "loss_history.csv")}
        for name, expected_hash in metrics["artifact_sha256"].items():
            if artifact_hashes[name] != expected_hash:
                raise ValueError(f"Run {run.name} artifact changed: {name}.")
        if metrics["landmarks_sha256"] != input_hashes["landmarks.json"] or metrics["landmark_count"] != len(xyz):
            raise ValueError(f"Run {run.name} used different landmark evaluation inputs.")
        final_p = np.load(run / "P_optimized_world_mm.npy")
        np.testing.assert_array_equal(final_p, np.load(run / "P_epoch0100.npy"),
                                      err_msg="Final P differs from the fixed epoch 100 P.")
        np.testing.assert_allclose(np.load(run / "P_optimized_pixel.npy"),
                                   pmat_to_pixel(final_p, du=du, dv=dv), rtol=1e-12, atol=1e-9)
        computed = {}
        for name, matrix in (("nominal", np.load(input_dir / "P_nominal_world_mm.npy")),
                             ("optimized", final_p),
                             ("oracle", np.load(input_dir / "P_truth_world_mm.npy"))):
            computed[name] = _check_geometry_against_runner(
                matrix, truth_p, truth_source, xyz, visible, du, dv, metrics["results"][name]["geometry"])
            if metadata.get("noise") and any("relative_l2_to_clean" not in score
                    for score in metrics["results"][name]["projection"].values()):
                raise ValueError(f"Run {run.name} lacks a clean-reference score for {name}.")
        audit = evaluate_saved_run(input_dir, run)
        if not audit["fixed_final_epoch_available"]:
            raise ValueError(f"Run {run.name} does not end at its fixed final epoch.")
        reports.append(audit)
        archive = np.load(run / "projection_examples.npz")
        sample = {key: archive[key] for key in archive.files}
        archive.close()
        np.testing.assert_array_equal(sample["target"], target_memmap[sample["indices"]])
        if examples:
            np.testing.assert_array_equal(sample["indices"], examples[0]["indices"])
            np.testing.assert_array_equal(sample["target"], examples[0]["target"])
            np.testing.assert_allclose(sample["nominal"], examples[0]["nominal"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(sample["oracle"], examples[0]["oracle"], rtol=1e-6, atol=1e-6)
        examples.append(sample)
        final_audit = audit["snapshots"][-1]
        summaries.append({
            "run_name": run.name,
            "loss_levels": recipe["loss_levels"],
            "epoch": 100,
            **_compact_result(metrics["results"]["optimized"], computed["optimized"]),
            "source_elevation_error_deg": final_audit["source_elevation_error_deg"],
            "source_azimuth_error_deg": final_audit["source_azimuth_error_deg"],
            "source_azimuth_error_signed_range_deg": final_audit["source_azimuth_error_signed_range_deg"],
            "artifact_sha256": artifact_hashes,
        })
        if nominal_summary is None:
            nominal_summary = _compact_result(metrics["results"]["nominal"], computed["nominal"])
            oracle_summary = _compact_result(metrics["results"]["oracle"], computed["oracle"])
            # These losses have different definitions for different loss_levels.
            nominal_summary.pop("lncc_loss_for_this_recipe")
            oracle_summary.pop("lncc_loss_for_this_recipe")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_figures(reports, examples, theta, comparison_dir)
    summary = {
        "experiment": f"{description['phantom_label']}: circular initialization to sineSpin geometry calibration",
        "scope": f"Single-seed numerical feasibility; {description['noise_label']}; same sampled phantom and Joseph discretization in generation and fitting",
        "noise": description["noise"],
        "training": common_recipe,
        "comparison_variable": "Only image-loss downsampling levels differ; geometry model, initialization, data, seed, and final epoch are shared",
        "validation": {
            "final_epoch": 100,
            "checkpoint_selection": "Fixed final epoch; no truth-based checkpoint selection",
            "geometry": "CPU rederived from saved P and checked against the runner for nominal, optimized, and oracle",
            "projection": "Saved complete GPU evaluation; relative_l2 uses the prepared target, with separate clean-reference scores when available; saved selected targets checked against input",
            "correspondence": f"Fixed {len(xyz)} landmark IDs and true visibility; no nearest-neighbour rematching or phantom alignment",
            "visible_landmark_view_pairs": int(visible.sum()),
            "all_landmark_view_pairs": int(visible.size),
            "training_views": reports[0]["training_view_count"],
            "untrained_views": reports[0]["untrained_view_count"],
        },
        "limitations": [
            "One phantom and one seed; no repeated-seed robustness conclusion.",
            f"{description['noise_label']}; no measured scanner poses, scatter, or motion acquisition.",
            "Same reference voxel grid and Joseph discretization; no avoidance of inverse crime is claimed.",
            "All measured views are used for training when view_step is 1; this does not test untrained-view interpolation.",
            "Compared loss definitions differ; their raw LNCC loss values are not a common accuracy metric.",
            "Comparative runtime is not evaluated by this report.",
        ],
        "phantom": {"filename": volume_path.name, "shape_zyx": landmarks["shape_zyx"],
                    "voxel_size_mm": landmarks["voxel_size_mm"], "landmark_count": len(xyz),
                    "sha256": volume_hash},
        "nominal_geometry": metadata["nominal"],
        "true_geometry": metadata["truth"],
        "source_sha256": common_source_hashes,
        "preparation_source_sha256": metadata.get("preparation_source_sha256"),
        "report_script_sha256": sha256(Path(__file__).resolve()),
        "independent_leap_library_sha256": metadata["leap_sha256"],
        "input_sha256": input_hashes,
        "nominal": nominal_summary,
        "oracle": oracle_summary,
        "runs": summaries,
        "figure_sha256": {name: sha256(comparison_dir / name)
                          for name in ("geometry_recovery.png", "projection_comparison.png")},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def save_comparison_figures(reports: list[dict], examples: list[dict], theta: np.ndarray,
                            output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(theta, reports[0]["truth_source_z_mm"], "k", linewidth=2, label="True sineSpin")
    initial = reports[0]["snapshots"][0]
    axes[0, 0].plot(theta, initial["source_z_mm"], "--", color="0.55", label="Circular input")
    axes[0, 1].semilogy(theta, np.maximum(initial["per_view_ball_rmse_px"], 1e-5),
                       "--", color="0.55", label="Circular input")
    for index, report in enumerate(reports):
        snapshots = report["snapshots"]
        final = snapshots[-1]
        color = f"C{index}"
        label = report["run_name"]
        axes[0, 0].plot(theta, final["source_z_mm"], color=color, alpha=.85, label=label)
        axes[0, 1].semilogy(theta, np.maximum(final["per_view_ball_rmse_px"], 1e-5), color=color, label=label)
        epochs = [item["epoch"] for item in snapshots]
        axes[1, 0].semilogy(epochs, [item["visible_ball_reprojection_error_px"]["rms"] for item in snapshots],
                           "o-", color=color, label=label)
        axes[1, 1].semilogy(epochs, [item["source_position_error_mm"]["rms"] for item in snapshots],
                           "o-", color=color, label=label)
    axes[0, 0].set(xlabel="Scan angle (degree)", ylabel="Source z (mm)")
    axes[0, 1].set(xlabel="Scan angle (degree)", ylabel="Fixed-ID bead RMSE (pixel)")
    axes[1, 0].set(xlabel="Epoch", ylabel="Fixed-ID bead RMSE (pixel)")
    axes[1, 1].set(xlabel="Epoch", ylabel="Source position RMSE (mm)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    description = reports[0]["data_description"]
    fig.suptitle(f"{description['phantom_label']}: circular initialization to sineSpin P-matrix calibration\n"
                 f"Final epoch 100; {description['noise_label']}; image-only fitting")
    fig.savefig(output_dir / "geometry_recovery.png", dpi=160)
    plt.close(fig)
    reference = examples[0]
    chosen = np.array([1, 2, 4]) if len(reference["indices"]) >= 5 else np.arange(len(reference["indices"]))
    n_rows = 2 + 2 * len(reports)
    fig, axes = plt.subplots(n_rows, len(chosen), figsize=(12, 2.6 * n_rows),
                             constrained_layout=True, squeeze=False)
    vmax = float(np.quantile(reference["target"], .999))
    vmin = min(0., float(np.quantile(reference["target"], .001)))
    emax = max(float(np.quantile(np.abs(reference["nominal"] - reference["target"]), .995)), .1)
    labels = [description["target_label"], "Circular input"]
    labels += [f"{report['run_name']}\nfinal epoch 100" for report in reports]
    labels += [f"{report['run_name']}\nresidual to target" for report in reports]
    for column, selected in enumerate(chosen):
        panels = [reference["target"][selected], reference["nominal"][selected]]
        panels += [sample["optimized"][selected] for sample in examples]
        panels += [sample["optimized"][selected] - reference["target"][selected] for sample in examples]
        for row, panel in enumerate(panels):
            residual = row >= 2 + len(reports)
            axes[row, column].imshow(panel, origin="lower", cmap="coolwarm" if residual else "gray",
                                      vmin=-emax if residual else vmin, vmax=emax if residual else vmax)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        view = int(reference["indices"][selected])
        axes[0, column].set_title(f"View {view}, angle {theta[view]:.1f}°")
    for axis, label in zip(axes[:, 0], labels):
        axis.set_ylabel(label, fontsize=9)
    fig.suptitle(f"Saved {description['phantom_label'].lower()} projections at fixed views\n"
                 "Shared attenuation window; shared residual window based on circular-input error")
    fig.savefig(output_dir / "projection_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("result_sinespin/denseball_calibration/input"))
    parser.add_argument("--run-dir", type=Path, default=Path("result_sinespin/denseball_calibration/baseline_seed0"))
    parser.add_argument("--compare", type=Path, nargs="+", metavar="RUN_DIR",
                        help="Compare completed fixed-epoch-100 runs; does not accept incomplete runs")
    parser.add_argument("--summary-json", type=Path, default=Path("docs/sinespin_calibration_summary.json"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("result_sinespin/denseball_calibration/comparison"))
    args = parser.parse_args()
    if args.compare:
        if len(args.compare) < 2:
            parser.error("--compare requires at least two completed run directories")
        summary = compare_completed_runs(args.input_dir, args.compare, args.summary_json, args.comparison_dir)
        print(json.dumps({"epoch": 100, "nominal_ball_rmse_px": summary["nominal"]["ball_reprojection_error_px"]["rms"],
                          "runs": [{"name": run["run_name"],
                                    "ball_rmse_px": run["ball_reprojection_error_px"]["rms"],
                                    "source_rmse_mm": run["source_position_error_mm"]["rms"]}
                                   for run in summary["runs"]]}, indent=2))
        print(f"Wrote {args.summary_json}")
        return
    report = evaluate_saved_run(args.input_dir, args.run_dir)
    output = args.run_dir / "independent_geometry_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    save_figure(report, args.run_dir / "checkpoint_geometry_audit.png")
    for snapshot in report["snapshots"]:
        print(json.dumps({"epoch": snapshot["epoch"], "loss": snapshot["training_loss"],
                          "ball_rmse_px": snapshot["visible_ball_reprojection_error_px"]["rms"],
                          "source_rmse_mm": snapshot["source_position_error_mm"]["rms"],
                          "elevation_rmse_deg": snapshot["source_elevation_error_deg"]["rms"],
                          "azimuth_rmse_deg": snapshot["source_azimuth_error_deg"]["rms"]}))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
