"""Display actual prepared ball-phantom calibration inputs or completed fits.

The default preview is CPU-only. Optional --run-dirs validates two completed
fixed-epoch experiments, then forward projects their saved matrices on GPU 1.
No optimization is performed. Full detector images and fixed-ID bead crops
share physical/intensity scales; numerical pixel values are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ball_phantom_fov import centered_box_corners
from denseball_landmarks import project_landmarks

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "result_sinespin/ball_calibration"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(input_dir):
    metadata = json.loads((input_dir / "experiment.json").read_text())
    noise = json.loads((input_dir / "photon_noise.json").read_text())
    fov = json.loads((input_dir / "fov_audit.json").read_text())
    landmarks = json.loads((input_dir / "landmarks.json").read_text())
    probe = dict(np.load(input_dir / "projection_probe.npz"))
    required = {"indices", "target", "noisy_target", "photon_counts", "nominal", "oracle"}
    if not required.issubset(probe):
        raise ValueError(f"Prepared probe is missing {sorted(required-set(probe))}")
    indices = np.asarray(probe["indices"], dtype=int)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("Prepared probe view indices must be unique")
    for name, filename in (("target", "clean_projections.npy"),
                           ("noisy_target", "target_projections.npy"),
                           ("photon_counts", "photon_counts.npy")):
        full = np.load(input_dir / filename, mmap_mode="r")
        np.testing.assert_array_equal(probe[name], full[indices], err_msg=f"Probe {name} differs from prepared data")
    geometry = dict(np.load(input_dir / "truth_geometry.npz"))
    truth_p = np.load(input_dir / "P_truth_pixel.npy")
    nominal_p = np.load(input_dir / "P_nominal_pixel.npy")
    return metadata, noise, fov, landmarks, probe, geometry, truth_p, nominal_p


def make_preview(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.spatial import ConvexHull

    metadata, noise, fov, landmarks, probe, geometry, truth_p, nominal_p = load_inputs(args.input_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = metadata["truth"]["detector_shape_vu"]
    dv, du = metadata["truth"]["pixel_vu_mm"]
    extent = [-cols*du/2, cols*du/2, -rows*dv/2, rows*dv/2]
    theta, tilt = geometry["theta_deg"], geometry["tilt_deg"]
    indices = probe["indices"].astype(int)
    chosen = [int(np.argmax(tilt)), int(fov["truth"]["worst_margin_view"]),
              int(np.argmin(tilt)), min(413, len(theta)-1)]
    selected = list(dict.fromkeys(chosen))
    if any(view not in indices for view in selected):
        raise ValueError(f"Prepared circular probe lacks requested views {sorted(set(selected)-set(indices))}")
    positions = [int(np.flatnonzero(indices == view)[0]) for view in selected]
    reason_list = ["Positive tilt peak", "Smallest full-box FOV margin", "Negative tilt peak", "Prior Denseball failure angle"]
    reasons = {view: "; ".join(text for index, text in zip(chosen, reason_list) if index == view) for view in selected}
    i0 = float(noise["i0_photons_per_detector_pixel_per_view"])
    vmin = float(args.vmin) if args.vmin is not None else 0.
    vmax = float(args.vmax) if args.vmax is not None else float(np.quantile(
        np.concatenate((probe["target"].ravel(), probe["nominal"].ravel())), .9995))
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError("Display intensity limits must be finite and increasing")
    residual = probe["nominal"] - probe["target"]
    residual_max = max(float(np.quantile(np.abs(residual[positions]), .995)), 1e-6)
    corners = centered_box_corners(metadata["volume"]["shape_zyx"], metadata["volume"]["voxel_mm"])
    corner_uv = project_landmarks(truth_p, corners)
    uv_to_mm = lambda uv: (np.asarray(uv)-[(cols-1)/2, (rows-1)/2])*[du, dv]
    footprint = uv_to_mm(corner_uv)
    box_label = " x ".join(f"{size:.1f}" for size in metadata["volume"]["box_extent_xyz_mm"])
    xyz = np.asarray([item["xyz_mm"] for item in landmarks["landmarks"]])
    true_q, nominal_q = project_landmarks(truth_p, xyz), project_landmarks(nominal_p, xyz)
    error = np.linalg.norm(true_q[selected]-nominal_q[selected], axis=-1)
    pair_distances = np.linalg.norm(true_q[selected, :, None]-true_q[selected, None, :], axis=-1)
    pair_distances[:, np.arange(len(xyz)), np.arange(len(xyz))] = np.inf
    isolated = pair_distances.min(axis=2).min(axis=0) > 10
    crop_half_px = args.crop_width_mm/(2*np.array([du, dv]))
    inside = ((true_q[selected]-crop_half_px >= [-.5, -.5])
              & (true_q[selected]+crop_half_px <= [cols-.5, rows-.5])).all(axis=(0, 2))
    eligible = isolated & inside
    if not eligible.any():
        raise ValueError("No isolated fixed-ID bead crop fits all selected views; reduce --crop-width-mm")
    bead = int(np.argmax(np.where(eligible, error.mean(axis=0), -np.inf)))
    bead_id = int(landmarks["landmarks"][bead]["landmark_id"])

    def show(axis, values, *, difference=False):
        image = axis.imshow(values, origin="lower", extent=extent, interpolation="nearest",
                            cmap="RdBu_r" if difference else "gray",
                            vmin=-residual_max if difference else vmin,
                            vmax=residual_max if difference else vmax)
        axis.set_xlim(extent[:2]); axis.set_ylim(extent[2:])
        axis.set_xlabel("u (mm)"); axis.set_ylabel("v (mm)")
        return image

    ncols, nrows = 4, int(np.ceil(len(indices)/4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.4*nrows), layout="constrained", squeeze=False)
    for position, view in enumerate(indices):
        axis = axes.flat[position]
        image = show(axis, probe["noisy_target"][position])
        hull = ConvexHull(footprint[view]); polygon = np.r_[hull.vertices, hull.vertices[0]]
        axis.plot(footprint[view, polygon, 0], footprint[view, polygon, 1], "--", color="#00d8de", lw=.65)
        axis.set_title(f"View {view} | theta {theta[view]:+.2f} deg | tilt {tilt[view]:+.2f} deg", fontsize=10)
    for axis in list(axes.flat)[len(indices):]:
        axis.set_visible(False)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.6, pad=.01, label="Observed line integral; shared window")
    fig.suptitle(f"Actual small ball phantom: Poisson I0 = {i0:,.0f}, seed {noise['seed']}\n"
                 f"Complete detector at every shown angle; dashed cyan = projected full {box_label} mm voxel box", fontsize=13)
    fig.savefig(args.out_dir / "projection_contact_sheet.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(4, len(selected), figsize=(17, 13.5), layout="constrained", squeeze=False)
    row_labels = ["True sineSpin / clean", f"True sineSpin / Poisson {i0:,.0f}",
                  "Circular initial forward projection", "Circular minus clean sineSpin"]
    for j, (view, position) in enumerate(zip(selected, positions)):
        centre = uv_to_mm(true_q[view, bead])
        values = [probe["target"][position], probe["noisy_target"][position], probe["nominal"][position], residual[position]]
        for i in range(4):
            handle = show(axes[i, j], values[i], difference=i == 3)
            if i == 0: main_handle = handle
            if i == 3: difference_handle = handle
            axes[i, j].add_patch(Rectangle(centre-args.crop_width_mm/2, args.crop_width_mm, args.crop_width_mm,
                                           fill=False, edgecolor="#ffb000", lw=.8))
            if j == 0: axes[i, j].set_ylabel(row_labels[i] + "\nv (mm)", fontsize=10)
            if i == 0:
                axes[i, j].set_title(f"View {view} | theta {theta[view]:+.2f} / tilt {tilt[view]:+.2f} deg\n{reasons[view]}", fontsize=10)
    fig.colorbar(main_handle, ax=axes[:3].ravel().tolist(), shrink=.6, pad=.01, label="Line integral; shared window")
    fig.colorbar(difference_handle, ax=axes[3].ravel().tolist(), shrink=.8, pad=.01, label="Signed geometry difference")
    fig.suptitle("Actual input comparison at the tilt extrema and diagnostic angles\n"
                 f"Entire detector retained; yellow boxes identify bead ID {bead_id} crops. No optimization has been applied.", fontsize=13)
    fig.savefig(args.out_dir / "projection_key_views.png", dpi=155); plt.close(fig)

    crop_values = []
    for position, view in zip(positions, selected):
        col, row = np.rint(true_q[view, bead]).astype(int)
        rv, ru = int(np.ceil(args.crop_width_mm/(2*dv))), int(np.ceil(args.crop_width_mm/(2*du)))
        crop_values.append(probe["target"][position, row-rv:row+rv+1, col-ru:col+ru+1].ravel())
    zoom_min, zoom_max = [float(value) for value in np.quantile(np.concatenate(crop_values), [.01, .999])]
    if zoom_max <= zoom_min: zoom_min, zoom_max = vmin, vmax
    norm = lambda image: np.clip((image-zoom_min)/(zoom_max-zoom_min), 0, 1)
    fig, axes = plt.subplots(4, len(selected), figsize=(15, 15), layout="constrained", squeeze=False)
    view_reports = []
    for j, (view, position) in enumerate(zip(selected, positions)):
        centre = uv_to_mm(true_q[view, bead]); nominal_centre = uv_to_mm(nominal_q[view, bead])
        true = norm(probe["target"][position]); initial = norm(probe["nominal"][position])
        overlay = np.stack((initial, true, np.maximum(initial, true)), axis=-1)
        for i, name in enumerate(("target", "noisy_target", "nominal")):
            image = axes[i, j].imshow(probe[name][position], origin="lower", extent=extent,
                                      cmap="gray", vmin=zoom_min, vmax=zoom_max, interpolation="nearest")
        axes[3, j].imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
        for i, axis in enumerate(axes[:, j]):
            axis.plot(*centre, "o", mfc="none", mec="#00e68a", ms=10, mew=1)
            if i >= 2:
                axis.plot(*nominal_centre, "+", color="#ff782e", ms=11, mew=1.5)
            axis.set_xlim(centre[0]-args.crop_width_mm/2, centre[0]+args.crop_width_mm/2)
            axis.set_ylim(centre[1]-args.crop_width_mm/2, centre[1]+args.crop_width_mm/2)
            axis.set_xlabel("u (mm)")
            if j == 0:
                axis.set_ylabel(["Clean true data", "Poisson observation", "Circular initialization", "True/circular intensity overlay"][i]+"\nv (mm)")
        axes[0, j].set_title(f"View {view} | bead ID {bead_id}\ntheta {theta[view]:+.2f} / tilt {tilt[view]:+.2f} deg", fontsize=10)
        delta = (probe["noisy_target"][position] - probe["target"][position]).astype(np.float64)
        geometry_delta = residual[position].astype(np.float64)
        view_reports.append(dict(view=int(view), theta_deg=float(theta[view]), tilt_deg=float(tilt[view]), selection=reasons[view],
                                 crop_center_uv_mm=centre.tolist(), nominal_same_bead_center_uv_mm=nominal_centre.tolist(),
                                 same_bead_initial_error_px=float(np.linalg.norm(true_q[view, bead]-nominal_q[view, bead])),
                                 noise_rmse=float(np.sqrt(np.mean(delta**2))), circular_minus_truth_rmse=float(np.sqrt(np.mean(geometry_delta**2))),
                                 circular_relative_l2=float(np.linalg.norm(geometry_delta)/np.linalg.norm(probe["target"][position].astype(np.float64))),
                                 full_box_fov=fov["truth"]["per_view"][view]))
    fig.colorbar(image, ax=axes[:3].ravel().tolist(), shrink=.5, pad=.01, label="Line integral; shared crop window")
    fig.suptitle(f"Actual bead ID {bead_id} at four angles ({args.crop_width_mm:g} x {args.crop_width_mm:g} mm crops)\n"
                 "Green circle = true centre; orange cross = SAME-ID circular projection (display/evaluation only).\n"
                 "Overlay: cyan = clean true intensities, magenta = circular intensities, white = overlap.", fontsize=12)
    fig.savefig(args.out_dir / "projection_feature_crops.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for name, label in (("nominal", "Circular nominal"), ("truth", "sineSpin truth")):
        report = fov[name]
        axes[0].plot(theta, [min(item["detector_edge_margins_mm"]) for item in report["per_view"]], label=label)
        axes[1].plot(theta, [item["min_camera_depth_mm"] for item in report["per_view"]], label=label)
    axes[0].axhline(0, color="black", lw=.8)
    for view in selected: axes[0].axvline(theta[view], color="gray", lw=.6, ls=":")
    axes[0].set(xlabel="Scan theta (degree)", ylabel="Smallest full-box detector edge margin (mm)")
    axes[1].set(xlabel="Scan theta (degree)", ylabel="Smallest box-corner camera depth (mm)")
    for axis in axes:
        axis.legend(); axis.grid(alpha=.2)
    fig.suptitle(f"Full {box_label} mm box: all {len(theta)} views checked, no crop or rescale\n"
                 f"Circle passed: {fov['nominal']['passed']} | sineSpin passed: {fov['truth']['passed']}; finite-detector containment, not Tuy completeness")
    fig.savefig(args.out_dir / "full_box_fov_margins.png", dpi=160); plt.close(fig)

    summary = dict(input_dir=str(args.input_dir.resolve()), volume=metadata["volume"], noise=noise,
                   contact_sheet_views=indices.tolist(), key_views=view_reports,
                   full_detector_extent_uv_mm=extent, intensity_window=[vmin, vmax],
                   full_window_rule="Fixed for all full panels; default 0 to combined clean/nominal probe 99.95th percentile; stored numerical samples unchanged",
                   clean_fraction_above_display_max=float(np.mean(probe["target"] > vmax)),
                   noisy_fraction_below_display_min=float(np.mean(probe["noisy_target"] < vmin)),
                   geometry_difference_window=[-residual_max, residual_max],
                   feature_crop_width_mm=args.crop_width_mm, crop_intensity_window=[zoom_min, zoom_max], crop_landmark_id=bead_id,
                   crop_rule="One fixed-ID bead: largest mean circular displacement across selected views, restricted to beads >10px from other true beads and with full crops inside detector; evaluation/display only",
                   prepared_inputs_verified="Probe clean/noisy/count arrays equal the same view slices of the complete prepared arrays exactly",
                   source_sha256={name: sha256(args.input_dir/name) for name in
                                  ("experiment.json", "photon_noise.json", "fov_audit.json", "landmarks.json", "projection_probe.npz")},
                   interpretation="Input preview only; circular-to-sineSpin differences are initial mismatch, not calibration results")
    (args.out_dir / "preview_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Saved actual full-detector projections, fixed-ID bead crops and FOV margins in {args.out_dir}", flush=True)


def validate_completed_runs(input_dir, run_dirs, metadata):
    """Check final artifacts before allowing any GPU work or cache reuse."""
    if len(run_dirs) != 2 or run_dirs[0].resolve() == run_dirs[1].resolve():
        raise ValueError("Provide exactly two distinct completed run directories")
    # Reject incomplete experiments before torch import or large input hashing.
    for run in run_dirs:
        for filename in ("experiment.json", "metrics.json", "checkpoint.pt", "motion9.npy",
                         "P_optimized_world_mm.npy", "P_optimized_pixel.npy"):
            if not (run/filename).is_file():
                raise ValueError(f"Run {run.name} is incomplete: missing {filename}")
    input_names = {"experiment.json": None, "landmarks.json": None,
                   "target_projections.npy": metadata["target_sha256"],
                   "clean_projections.npy": metadata["clean_sha256"],
                   "P_nominal_world_mm.npy": metadata["nominal_pmat_sha256"],
                   "P_truth_world_mm.npy": metadata["truth_pmat_sha256"],
                   "P_truth_pixel.npy": None, "truth_geometry.npz": None,
                   "projection_probe.npz": None}
    input_hashes = {name: sha256(input_dir/name) for name in input_names}
    for name, expected in input_names.items():
        if expected is not None and input_hashes[name] != expected:
            raise ValueError(f"Prepared input changed: {name}")
    volume_path = Path(metadata["volume"]["path"])
    if sha256(volume_path) != metadata["volume"]["sha256"]:
        raise ValueError("Reference volume differs from the prepared experiment")
    from calibration_geometry import pmat_to_pixel
    import torch
    records, matrices, common_recipe = [], [], None
    dv, du = metadata["truth"]["pixel_vu_mm"]
    for run in run_dirs:
        experiment = json.loads((run/"experiment.json").read_text())
        metrics = json.loads((run/"metrics.json").read_text())
        recipe = experiment["recipe"]
        epoch = int(recipe["epochs"])
        if epoch <= 0 or metrics["epoch"] != epoch or metrics["recipe"] != recipe:
            raise ValueError(f"Run {run.name} has no matching prespecified final-epoch metrics")
        if experiment["input"] != metadata or metrics["landmarks_sha256"] != input_hashes["landmarks.json"]:
            raise ValueError(f"Run {run.name} used different prepared input or landmarks")
        invariant = {key: value for key, value in recipe.items() if key != "loss_levels"}
        if common_recipe is None:
            common_recipe = invariant
        elif invariant != common_recipe:
            raise ValueError("The two displayed experiments must differ only in loss_levels")
        artifacts = metrics["artifact_sha256"]
        for filename in ("P_optimized_world_mm.npy", "P_optimized_pixel.npy", "motion9.npy", "checkpoint.pt"):
            if artifacts[filename] != sha256(run/filename):
                raise ValueError(f"Run {run.name} final artifact changed: {filename}")
        for filename, expected in experiment["source_sha256"].items():
            if sha256(ROOT/filename) != expected:
                raise ValueError(f"Run {run.name} source changed after training: {filename}")
        checkpoint = torch.load(run/"checkpoint.pt", map_location="cpu", weights_only=False)
        if (checkpoint["epoch"] != epoch or checkpoint["recipe"] != recipe
                or checkpoint["input_sha256"] != input_hashes["experiment.json"]
                or checkpoint["history"][-1]["epoch"] != epoch):
            raise ValueError(f"Run {run.name} checkpoint is not its matching final fixed epoch")
        del checkpoint
        snapshot = run/f"P_epoch{epoch:04d}.npy"
        if not snapshot.is_file():
            raise ValueError(f"Run {run.name} is missing its final matrix snapshot")
        matrix = np.load(run/"P_optimized_world_mm.npy")
        if matrix.shape != (metadata["truth"]["views"], 3, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Run {run.name} contains invalid final matrices")
        np.testing.assert_array_equal(matrix, np.load(snapshot), err_msg="Final P differs from the fixed final snapshot")
        np.testing.assert_allclose(pmat_to_pixel(matrix, du=du, dv=dv), np.load(run/"P_optimized_pixel.npy"),
                                   rtol=1e-10, atol=1e-9, err_msg="Final world/pixel matrices disagree")
        records.append(dict(run_dir=str(run.resolve()), run_name=run.name, epoch=epoch, recipe=recipe,
                            experiment_sha256=sha256(run/"experiment.json"), metrics_sha256=sha256(run/"metrics.json"),
                            final_snapshot_sha256=sha256(snapshot), artifact_sha256=artifacts,
                            source_sha256=experiment["source_sha256"],
                            final_projection_metrics=metrics["results"]["optimized"]["projection"],
                            final_geometry_metrics=metrics["results"]["optimized"]["geometry"]))
        matrices.append(matrix)
    return records, matrices, input_hashes


def make_fit_comparison(args):
    """Project four diagnostic views of two final saved fits, then plot/cache."""
    metadata, noise, fov, landmarks, probe, geometry, truth_p, nominal_p = load_inputs(args.input_dir)
    records, matrices, input_hashes = validate_completed_runs(args.input_dir, args.run_dirs, metadata)
    rows, cols = metadata["truth"]["detector_shape_vu"]
    dv, du = metadata["truth"]["pixel_vu_mm"]
    theta, tilt = geometry["theta_deg"], geometry["tilt_deg"]
    selected = list(dict.fromkeys([int(np.argmax(tilt)), int(fov["truth"]["worst_margin_view"]),
                                   int(np.argmin(tilt)), min(413, len(theta)-1)]))
    if any(view not in probe["indices"] for view in selected):
        raise ValueError("Prepared probe lacks a requested diagnostic view")
    positions = [int(np.flatnonzero(probe["indices"] == view)[0]) for view in selected]
    signature = dict(input_sha256=input_hashes, volume_sha256=metadata["volume"]["sha256"],
                     shape_zyx=metadata["volume"]["shape_zyx"], voxel_mm=metadata["volume"]["voxel_mm"],
                     views=selected, runs=[{key: item[key] for key in
                                           ("run_dir", "epoch", "recipe", "artifact_sha256", "source_sha256")}
                                          for item in records])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir/"final_projection_arrays.npz"
    provenance_path = args.out_dir/"final_projection_provenance.json"
    if cache_path.exists() or provenance_path.exists():
        if not cache_path.exists() or not provenance_path.exists():
            raise ValueError("Incomplete final-projection cache; choose another --out-dir")
        provenance = json.loads(provenance_path.read_text())
        if provenance["signature"] != signature or provenance["projection_npz_sha256"] != sha256(cache_path):
            raise ValueError("Final-projection cache belongs to different input, matrices or projector code")
        arrays = dict(np.load(cache_path))
        print("Reusing validated final-view forward projections; no GPU work required", flush=True)
    else:
        import os
        if args.gpu != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            raise ValueError("Final-view FP requires --gpu 1 with CUDA_VISIBLE_DEVICES unset (physical GPU 1)")
        import torch
        from run_sinespin_calibration import load_volume, project
        from sinespin_geometry import SineSpinGeometry
        g = SineSpinGeometry(kind=metadata["truth"]["kind"],
                             source_positions=geometry["source_positions"], module_centers=geometry["module_centers"],
                             row_vectors=geometry["row_vectors"], col_vectors=geometry["col_vectors"],
                             detector_rows=rows, detector_cols=cols, pixel_height=dv, pixel_width=du,
                             theta_deg=theta, tilt_deg=tilt, sod_mm=metadata["truth"]["sod_mm"],
                             sdd_mm=metadata["truth"]["sdd_mm"], scan_angle_deg=metadata["truth"]["scan_angle_deg"])
        np.testing.assert_allclose(g.projection_matrices(), truth_p, rtol=1e-12, atol=1e-9)
        device = torch.device("cuda:1"); torch.cuda.set_device(device)
        shape, spacing = tuple(metadata["volume"]["shape_zyx"]), float(metadata["volume"]["voxel_mm"])
        volume = load_volume(metadata["volume"]["path"], device, shape)
        arrays = dict(indices=np.asarray(selected), clean=probe["target"][positions], noisy=probe["noisy_target"][positions],
                      circular=probe["nominal"][positions], oracle=probe["oracle"][positions])
        with torch.no_grad():
            for index, matrix in enumerate(matrices):
                pmat = torch.as_tensor(matrix[selected], device=device)
                values = project(volume, pmat, g, voxel=spacing).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite final-view projection")
                arrays[f"run_{index}"] = values
                print(f"Projected {records[index]['run_name']}: {len(selected)} views, physical GPU 1", flush=True)
        del volume, pmat
        torch.cuda.empty_cache()
        np.savez_compressed(cache_path, **arrays)
        provenance = dict(signature=signature, projection_npz_sha256=sha256(cache_path),
                          projector="Existing differentiable Triton Joseph at saved final P; no fitting",
                          physical_gpu=1, torch_version=torch.__version__,
                          clean_reference="Independent LEAP Joseph data before Poisson noise",
                          noisy_reference="Actual prepared post-log Poisson observations used in training",
                          oracle="Saved true-P Triton forward projection; numerical projector comparison only")
        provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    np.testing.assert_array_equal(arrays["indices"], selected)
    np.testing.assert_array_equal(arrays["clean"], probe["target"][positions])
    np.testing.assert_array_equal(arrays["noisy"], probe["noisy_target"][positions])
    render_fit_comparison(args, metadata, noise, landmarks, records, matrices, arrays, theta, tilt,
                          truth_p, nominal_p, provenance)


def render_fit_comparison(args, metadata, noise, landmarks, records, matrices, arrays, theta, tilt,
                          truth_p, nominal_p, provenance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from calibration_geometry import pmat_to_pixel

    rows, cols = metadata["truth"]["detector_shape_vu"]
    dv, du = metadata["truth"]["pixel_vu_mm"]
    extent = [-cols*du/2, cols*du/2, -rows*dv/2, rows*dv/2]
    selected = arrays["indices"].astype(int)
    xyz = np.asarray([item["xyz_mm"] for item in landmarks["landmarks"]])
    true_q, nominal_q = project_landmarks(truth_p, xyz), project_landmarks(nominal_p, xyz)
    fitted_q = [project_landmarks(pmat_to_pixel(matrix, du=du, dv=dv), xyz) for matrix in matrices]
    distances = np.linalg.norm(true_q[selected, :, None]-true_q[selected, None, :], axis=-1)
    distances[:, np.arange(len(xyz)), np.arange(len(xyz))] = np.inf
    crop_half_px = args.crop_width_mm/(2*np.array([du, dv]))
    eligible = (distances.min(axis=2).min(axis=0) > 10)
    eligible &= ((true_q[selected]-crop_half_px >= [-.5, -.5])
                 & (true_q[selected]+crop_half_px <= [cols-.5, rows-.5])).all(axis=(0, 2))
    if not eligible.any():
        raise ValueError("No full fixed-ID bead crop fits; reduce --crop-width-mm")
    initial_error = np.linalg.norm(true_q[selected]-nominal_q[selected], axis=-1)
    bead = int(np.argmax(np.where(eligible, initial_error.mean(axis=0), -np.inf)))
    bead_id = int(landmarks["landmarks"][bead]["landmark_id"])
    uv_to_mm = lambda uv: (np.asarray(uv)-[(cols-1)/2, (rows-1)/2])*[du, dv]
    vmin = float(args.vmin) if args.vmin is not None else 0.
    # Match the initial preview window; never normalize one fit independently.
    probe = np.load(args.input_dir/"projection_probe.npz")
    vmax = float(args.vmax) if args.vmax is not None else float(np.quantile(
        np.concatenate((probe["target"].ravel(), probe["nominal"].ravel())), .9995))
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError("Display intensity limits must be finite and increasing")
    differences = [arrays[f"run_{index}"]-arrays["clean"] for index in range(2)]
    emax = max(float(np.quantile(np.abs(np.concatenate(differences)), .995)), 1e-6)
    run_labels = [f"{record['run_name']}\nfinal epoch {record['epoch']}" for record in records]
    fig, axes = plt.subplots(5, len(selected), figsize=(17, 16.5), layout="constrained", squeeze=False)
    for j, view in enumerate(selected):
        centre = uv_to_mm(true_q[view, bead])
        panels = [arrays["noisy"][j], arrays["run_0"][j], arrays["run_1"][j], differences[0][j], differences[1][j]]
        for i, values in enumerate(panels):
            difference = i >= 3
            handle = axes[i, j].imshow(values, origin="lower", extent=extent, interpolation="nearest",
                                       cmap="RdBu_r" if difference else "gray",
                                       vmin=-emax if difference else vmin, vmax=emax if difference else vmax)
            if difference: error_handle = handle
            else: value_handle = handle
            axes[i, j].set_xlim(extent[:2]); axes[i, j].set_ylim(extent[2:])
            axes[i, j].set_xlabel("u (mm)")
            axes[i, j].add_patch(Rectangle(centre-args.crop_width_mm/2, args.crop_width_mm, args.crop_width_mm,
                                           fill=False, edgecolor="#ffb000", lw=.7))
            if j == 0:
                axes[i, j].set_ylabel(([f"Observed Poisson I0={noise['i0_photons_per_detector_pixel_per_view']:,.0f}"]
                                      +run_labels+["First fit minus CLEAN truth", "Second fit minus CLEAN truth"])[i]+"\nv (mm)", fontsize=9)
            if i == 0:
                axes[i, j].set_title(f"View {view} | theta {theta[view]:+.2f} deg\ntilt {tilt[view]:+.2f} deg", fontsize=10)
    fig.colorbar(value_handle, ax=axes[:3].ravel().tolist(), pad=.01, shrink=.6, label="Line integral; one shared window")
    fig.colorbar(error_handle, ax=axes[3:].ravel().tolist(), pad=.01, shrink=.7, label="Fit minus independent CLEAN truth")
    bounds = records[0]["recipe"]["bounds"]
    fig.suptitle("Actual forward projections from the final calibrated P matrices; complete detector retained\n"
                 f"Observed data have Poisson noise. Signed residuals use the independent clean reference. Bounds: {bounds}", fontsize=12)
    fig.savefig(args.out_dir/"final_projection_key_views.png", dpi=150); plt.close(fig)

    crops = []
    for j, view in enumerate(selected):
        col, row = np.rint(true_q[view, bead]).astype(int)
        rv, ru = int(np.ceil(args.crop_width_mm/(2*dv))), int(np.ceil(args.crop_width_mm/(2*du)))
        crops.append(arrays["clean"][j, row-rv:row+rv+1, col-ru:col+ru+1].ravel())
    zoom_min, zoom_max = [float(value) for value in np.quantile(np.concatenate(crops), [.01, .999])]
    if zoom_max <= zoom_min: zoom_min, zoom_max = vmin, vmax
    norm = lambda values: np.clip((values-zoom_min)/(zoom_max-zoom_min), 0, 1)
    fig, axes = plt.subplots(5, len(selected), figsize=(15, 18.5), layout="constrained", squeeze=False)
    view_reports = []
    for j, view in enumerate(selected):
        centre = uv_to_mm(true_q[view, bead])
        for i, key in enumerate(("noisy", "run_0", "run_1")):
            value_handle = axes[i, j].imshow(arrays[key][j], origin="lower", extent=extent,
                                            vmin=zoom_min, vmax=zoom_max, cmap="gray", interpolation="nearest")
        true = norm(arrays["clean"][j])
        for index in range(2):
            predicted = norm(arrays[f"run_{index}"][j])
            axes[index+3, j].imshow(np.stack((predicted, true, np.maximum(predicted, true)), axis=-1),
                                    origin="lower", extent=extent, interpolation="nearest")
        for i, axis in enumerate(axes[:, j]):
            axis.plot(*centre, "o", mfc="none", mec="#00e68a", ms=10, mew=1)
            if i > 0:
                index = (i-1) if i < 3 else i-3
                axis.plot(*uv_to_mm(fitted_q[index][view, bead]), "+", color="#ff782e", ms=11, mew=1.4)
            axis.set_xlim(centre[0]-args.crop_width_mm/2, centre[0]+args.crop_width_mm/2)
            axis.set_ylim(centre[1]-args.crop_width_mm/2, centre[1]+args.crop_width_mm/2)
            axis.set_xlabel("u (mm)")
            if j == 0:
                axis.set_ylabel((["Observed Poisson data"]+run_labels+["First fit / CLEAN intensity overlay", "Second fit / CLEAN intensity overlay"])[i]+"\nv (mm)", fontsize=9)
        axes[0, j].set_title(f"View {view} | same bead ID {bead_id}\ntheta {theta[view]:+.2f} / tilt {tilt[view]:+.2f} deg", fontsize=10)
        item = dict(view=int(view), theta_deg=float(theta[view]), tilt_deg=float(tilt[view]),
                    bead_id=bead_id, truth_bead_uv_mm=centre.tolist(), fits=[])
        for index in range(2):
            pred = arrays[f"run_{index}"][j].astype(np.float64)
            one = dict(run_name=records[index]["run_name"],
                       same_bead_error_px=float(np.linalg.norm(fitted_q[index][view, bead]-true_q[view, bead])),
                       fitted_bead_uv_mm=uv_to_mm(fitted_q[index][view, bead]).tolist())
            for target in ("clean", "noisy", "oracle"):
                ref = arrays[target][j].astype(np.float64); delta = pred-ref
                one[f"relative_l2_to_{target}"] = float(np.linalg.norm(delta)/np.linalg.norm(ref))
                one[f"rmse_to_{target}"] = float(np.sqrt(np.mean(delta**2)))
            item["fits"].append(one)
        view_reports.append(item)
    fig.colorbar(value_handle, ax=axes[:3].ravel().tolist(), pad=.01, shrink=.6, label="Line integral; shared crop window")
    fig.suptitle(f"Final calibrated projections: SAME bead ID {bead_id}, {args.crop_width_mm:g} x {args.crop_width_mm:g} mm\n"
                 "Green circle = true bead centre; orange cross = final same-ID centre.\n"
                 "Overlay: cyan = independent CLEAN truth, magenta = final projection, white = overlap. Labels are evaluation only.", fontsize=12)
    fig.savefig(args.out_dir/"final_projection_bead_crops.png", dpi=150); plt.close(fig)
    summary = dict(runs=records, key_views=view_reports, volume=metadata["volume"], noise=noise,
                   intensity_window=[vmin, vmax], difference_window=[-emax, emax], crop_intensity_window=[zoom_min, zoom_max],
                   full_detector_extent_uv_mm=extent, crop_width_mm=args.crop_width_mm, crop_landmark_id=bead_id,
                   selection_rule="Same views and bead as the input preview; selected from nominal-vs-truth geometry, not optimized errors",
                   projection_npz_sha256=provenance["projection_npz_sha256"], renderer_sha256=sha256(Path(__file__)),
                   interpretation="Actual deterministic final-P projections. Both fits trained on the same noisy observations; CLEAN residuals isolate geometry/numerical error from observation noise.")
    (args.out_dir/"final_projection_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Saved completed-fit full detector and fixed-ID bead comparison in {args.out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE / "input")
    parser.add_argument("--out-dir", type=Path, default=BASE / "input_preview")
    parser.add_argument("--crop-width-mm", type=float, default=40.)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--run-dirs", type=Path, nargs=2,
                        help="Optional two completed final runs to compare; uses short GPU forward projections then caches them")
    parser.add_argument("--gpu", type=int, default=1, help="Physical GPU for optional final-run FP; only 1 is authorized")
    args = parser.parse_args()
    if not np.isfinite(args.crop_width_mm) or args.crop_width_mm <= 0:
        parser.error("--crop-width-mm must be finite and positive")
    if args.run_dirs:
        make_fit_comparison(args)
    else:
        make_preview(args)


if __name__ == "__main__":
    main()
