"""CQ500 Joseph data -> circular FDK and Sine Spin Grangeat reconstruction.

The strict Grangeat result contains NaN where required sampled Radon-plane
data are missing. The separately named partial result inserts zero for missing
angular contributions without renormalization; it is not an exact reconstruction.
No LS update, nonnegative clipping, or fitted intensity scaling is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sim_sinespin_recon import configure_leap, geometry_record, save_json
from sinespin_geometry import build_icono_orbit


ARMS = ("circular_200", "circular_220", "sinespin_220")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_array(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def resample_reference(reference, valid, source_voxel, target_voxel):
    """Sample identical physical voxel-centre coordinates on a coarser grid."""
    from scipy.ndimage import affine_transform

    if reference.shape != valid.shape or not np.isfinite(target_voxel) or target_voxel <= 0:
        raise ValueError("Invalid reference, coverage mask, or target voxel size")
    target_size = np.asarray(reference.shape) * source_voxel / target_voxel
    if not np.allclose(target_size, np.round(target_size), atol=1e-9, rtol=0):
        raise ValueError("--voxel must exactly divide each input physical extent")
    shape = tuple(np.round(target_size).astype(int))
    if min(shape) < 2:
        raise ValueError("Target grid must contain at least two voxels per axis")
    if source_voxel == target_voxel:
        return np.asarray(reference, dtype=np.float32), np.asarray(valid, dtype=bool)
    ratio = target_voxel / source_voxel
    offset = (np.asarray(reference.shape)-1)/2 - ratio*(np.asarray(shape)-1)/2
    kwargs = dict(matrix=np.eye(3)*ratio, offset=offset, output_shape=shape,
                  order=1, mode="constant", cval=0., prefilter=False)
    sampled = affine_transform(reference, output=np.float32, **kwargs)
    sampled_valid = affine_transform(np.asarray(valid, dtype=np.float32), output=np.float32, **kwargs) >= 1.-1e-6
    return sampled, sampled_valid


def arm_geometries(detector_bin, detector_padding_factor=1):
    if (not isinstance(detector_padding_factor, (int, np.integer))
            or detector_padding_factor < 1):
        raise ValueError("detector_padding_factor must be a positive integer")
    geometries = {"circular_200": build_icono_orbit("circular", detector_bin=detector_bin),
            "circular_220": build_icono_orbit("circular", detector_bin=detector_bin,
                                               n_views=546, scan_angle_deg=220),
            "sinespin_220": build_icono_orbit("sinespin", detector_bin=detector_bin)}
    return {name: replace(g, detector_rows=g.detector_rows*detector_padding_factor,
                          detector_cols=g.detector_cols*detector_padding_factor)
            for name, g in geometries.items()}


def geometry_digest(geometry):
    digest = hashlib.sha256()
    for value in (*geometry.modular_arrays(dtype=np.float64), geometry.theta_deg, geometry.tilt_deg):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def visibility_masks(geometries, shape, voxel):
    z, y, x = [(np.arange(n)-(n-1)/2)*voxel for n in shape]
    X, Y = np.meshgrid(x, y, indexing="xy")
    masks = {}
    for name, geometry in geometries.items():
        limits = geometry.longitudinal_intervals(np.stack((X, Y), axis=-1))
        masks[name] = (z[:, None, None] >= limits[None, :, :, 0]) & (z[:, None, None] <= limits[None, :, :, 1])
    return masks, (z, y, x)


def score_regions(reconstruction, reference, masks, mu_water):
    """CPU HU errors with explicit requested/evaluated counts; no NaN filling."""
    result = {}
    finite = np.isfinite(reconstruction) & np.isfinite(reference)
    for name, mask in masks.items():
        requested = int(np.count_nonzero(mask))
        selected = mask & finite
        count = int(np.count_nonzero(selected))
        values = dict(requested_voxels=requested, evaluated_voxels=count,
                      finite_fraction=count/requested if requested else None)
        if count:
            error = (np.asarray(reconstruction[selected], dtype=np.float64)
                     - reference[selected]) * (1000. / mu_water)
            values.update(rmse_hu=float(np.sqrt(np.mean(error*error))),
                          mean_error_hu=float(error.mean()), mae_hu=float(np.abs(error).mean()),
                          dark_error_fraction_below_minus50_hu=float(np.mean(error < -50.)))
        else:
            values.update(rmse_hu=None, mean_error_hu=None, mae_hu=None,
                          dark_error_fraction_below_minus50_hu=None)
        result[name] = values
    return result


def make_figures(out, reference, reconstructions, partial, coverage, known, masks, axes, mu_water, patient,
                 *, estimate=None, estimate_coverage=None, source_hull=None, detector_padding_factor=1):
    """Full reference-head framing; gray explicitly marks unsupported regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cq500_figures import _edges, _occupied_bounds

    z, y, x = axes
    xi, yi = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))
    bounds = _occupied_bounds((reference > .5*mu_water) & known, axes, 15.)
    planes = [("Sagittal", y, lambda a: a[:, :, xi], "y, posterior [mm]", 1),
              ("Coronal", x, lambda a: a[:, yi, :], "x, left [mm]", 2)]
    labels = {"truth": "CQ500 reference", "circular_200": "Circular 200° / 496: FDK",
              "circular_220": "Circular 220° / 546: FDK"}
    names = ("truth",) + tuple(reconstructions)
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#777777")
    modes = (("strict", "partial", "estimate") if estimate is not None else ("strict", "partial")) if "sinespin_220" in reconstructions else ("strict",)
    for mode in modes:
        volumes = {"truth": reference, **reconstructions}
        if "sinespin_220" in volumes:
            volumes["sinespin_220"] = {"partial": partial, "strict": reconstructions["sinespin_220"], "estimate": estimate}[mode]
        fig, panels = plt.subplots(2, len(names), figsize=(4.25*len(names), 9), constrained_layout=True, squeeze=False)
        for row, (plane, horizontal, slicer, xlabel, dim) in enumerate(planes):
            for col, name in enumerate(names):
                volume = volumes[name]
                values = np.asarray(slicer(volume), dtype=float)*(1000./mu_water)-1000.
                if mode != "estimate":
                    valid = slicer(known).copy()
                    if name.startswith("circular"):
                        valid &= slicer(masks[name])
                    values[~valid] = np.nan
                panel = panels[row, col]
                panel.imshow(values, origin="lower", extent=(*_edges(horizontal), *_edges(z)),
                             cmap=cmap, vmin=-110, vmax=210, aspect="equal")
                if name == "sinespin_220" and mode == "partial":
                    incomplete = slicer(coverage) < 1.-1e-6
                    overlay = np.zeros(incomplete.shape + (4,), dtype=float)
                    overlay[incomplete] = [.5, .5, .5, .45]
                    panel.imshow(overlay, origin="lower", extent=(*_edges(horizontal), *_edges(z)), aspect="equal")
                if mode == "estimate" and source_hull is not None:
                    hull_slice = slicer(source_hull)
                    if np.any(hull_slice) and not np.all(hull_slice):
                        panel.contour(horizontal, z, hull_slice, levels=[.5], colors=["cyan"], linewidths=.8)
                panel.set_xlim(*bounds[dim]); panel.set_ylim(*bounds[0])
                panel.set_xlabel(xlabel)
                if col == 0: panel.set_ylabel(f"{plane}: z, superior [mm]")
                sine_titles = {"strict": "Sine Spin: strict Grangeat support",
                               "partial": "Sine Spin: PARTIAL diagnostic",
                               "estimate": "Sine Spin: finite-detector estimate"}
                title = sine_titles[mode] if name == "sinespin_220" else labels[name]
                panel.set_title(title)
        if mode == "partial":
            subtitle = "Gray tint: missing Radon angles; zero contributions, no renormalization. Partial image is not exact."
            filename = "head_grangeat_partial_diagnostic.png"
        elif mode == "strict":
            subtitle = "Gray: unavailable source CT, detector data, or complete sampled Radon support. FDK is approximate."
            filename = "head_grangeat_strict.png"
        else:
            subtitle = ("Truncated-data estimate, not exact; no IR. Cyan: Sine Spin source convex-hull boundary (necessary only).")
            filename = "head_grangeat_estimate.png"
        detector_label = ("nominal detector" if detector_padding_factor == 1
                          else f"DIAGNOSTIC {detector_padding_factor}× enlarged detector; not the paper's panel")
        heading = f"CQ500CT{patient} · Joseph data → FDK / Grangeat · HU [-110,210]"
        if len(names) < 3:
            from textwrap import fill
            heading += "\n" + detector_label
            subtitle = fill(subtitle, width=76)
        else:
            heading += " · " + detector_label
        fig.suptitle(heading + "\n" + subtitle)
        fig.savefig(Path(out)/filename, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("result_sinespin/cq500_centered/input"))
    parser.add_argument("--out-dir", type=Path, default=Path("result_sinespin/cq500_grangeat"))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--voxel", type=float, default=2.)
    parser.add_argument("--detector-bin", type=int, default=2)
    parser.add_argument("--detector-padding-factor", type=int, default=1,
                        help="Diagnostic enlargement: multiply detector rows and columns only; preserve pixel pitch and all poses. Default 1 is the nominal panel.")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="Selected protocols; diagnostic enlarged-panel controls can use --arms sinespin_220.")
    parser.add_argument("--n-polar", type=int, default=64)
    parser.add_argument("--n-azimuth", type=int, default=256)
    parser.add_argument("--radon-step-mm", type=float, default=1.)
    parser.add_argument("--line-step-mm", type=float)
    parser.add_argument("--resume", action="store_true", help="Reuse saved Joseph projections only if all physics/input/library hashes match; recompute reconstruction.")
    args = parser.parse_args()
    if min(args.voxel, args.detector_bin, args.detector_padding_factor, args.n_polar, args.n_azimuth, args.radon_step_mm) <= 0:
        parser.error("Grid, quadrature, and binning values must be positive")
    if args.line_step_mm is not None and args.line_step_mm <= 0:
        parser.error("--line-step-mm must be positive")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicate protocols")
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    if (out/"experiment.json").exists() and not args.resume:
        raise FileExistsError("Output experiment exists; use --resume or another --out-dir")
    input_paths = {name: args.input_dir/name for name in
                   ("head_metadata.json", "reference_mu.npy", "reference_valid.npy", "forward_mu.npy")}
    inputs = json.loads(input_paths["head_metadata.json"].read_text())
    input_hashes = {name: file_sha256(path) for name, path in input_paths.items()}
    reference, known = resample_reference(np.load(input_paths["reference_mu.npy"], mmap_mode="r"),
                                         np.load(input_paths["reference_valid.npy"], mmap_mode="r"),
                                         float(inputs["recon_voxel_mm"]), args.voxel)
    mu_water = float(inputs["mu_water_per_mm"])
    geometries = arm_geometries(args.detector_bin, args.detector_padding_factor)
    geometries = {name: geometries[name] for name in args.arms}
    masks, axes = visibility_masks(geometries, reference.shape, args.voxel)
    save_array(out/"reference_mu.npy", reference); save_array(out/"reference_valid.npy", known)
    run = dict(input=inputs, input_sha256=input_hashes, selected_arms=list(geometries),
               reconstruction_grid=dict(shape_zyx=list(reference.shape), voxel_mm=args.voxel,
                                        resampling="Linear physical-centre sampling of the saved reference; no intensity fit"),
               geometry={name: geometry_record(g) for name, g in geometries.items()},
               detector=dict(padding_factor=args.detector_padding_factor,
                             diagnostic_enlargement=args.detector_padding_factor != 1,
                             definition="Rows and columns multiplied by factor; pixel pitch, source/detector poses and scan protocol unchanged",
                             nominal_panel_extent_vu_mm=[292.908, 397.936]),
               methods=dict(forward=f"LEAP Joseph on independently sampled {inputs['forward_voxel_mm']:g}-mm input grid",
                            circular="Circular short-scan FDK; approximate cone-beam reconstruction",
                            sinespin="Grangeat: cone data to Radon derivatives to plane inversion",
                            strict="NaN when any required sampled Radon-plane contribution is missing",
                            partial="Missing angular contributions are zero, without renormalization; not exact",
                            finite_detector_estimate="Zero continuation of detector data including boundary derivatives; truncated-data estimate, not exact or a manufacturer implementation",
                            n_polar=args.n_polar, n_azimuth=args.n_azimuth,
                            radon_step_mm=args.radon_step_mm, line_step_mm=args.line_step_mm),
               assumptions=["SOD/SDD 750/1200 mm and nominal orbit; not measured manufacturer poses",
                            "No added noise, scatter, HU clipping above air, or posthoc intensity scaling",
                            "Finite input CT coverage; manufacturer processing is not available"], results={})
    import torch
    if not torch.cuda.is_available() or args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {args.gpu} is unavailable")
    device = torch.device("cuda", args.gpu); torch.cuda.set_device(device)
    fine_np = np.load(input_paths["forward_mu.npy"], mmap_mode="r")
    fine = None
    run["runtime"] = dict(gpu=args.gpu, gpu_name=torch.cuda.get_device_name(device), torch=torch.__version__)
    for name, geometry in geometries.items():
        np.savez(out/f"{name}_geometry.npz", source_positions=geometry.source_positions,
                 module_centers=geometry.module_centers, row_vectors=geometry.row_vectors,
                 col_vectors=geometry.col_vectors, theta_deg=geometry.theta_deg, tilt_deg=geometry.tilt_deg,
                 P_pixel=geometry.projection_matrices())
        ct = configure_leap(geometry, fine_np.shape, float(inputs["forward_voxel_mm"]), device)
        library = Path(ct.libprojectors._name).resolve()
        library_hash = file_sha256(library)
        run["runtime"].update(leap_library=str(library), leap_sha256=library_hash)
        signature = dict(input_sha256=input_hashes, forward_shape_zyx=list(fine_np.shape),
                         forward_voxel_mm=float(inputs["forward_voxel_mm"]), geometry=geometry_record(geometry),
                         geometry_sha256=geometry_digest(geometry), leap_sha256=library_hash, projector="Joseph",
                         projector_configuration_sha256=file_sha256(Path(__file__).with_name("sim_sinespin_recon.py")))
        projection_path, metadata_path = out/f"projections_{name}.npy", out/f"projections_{name}.json"
        reused = False
        if args.resume and projection_path.exists() and metadata_path.exists():
            cached = json.loads(metadata_path.read_text())
            if cached["signature"] != signature or cached["projection_sha256"] != file_sha256(projection_path):
                raise ValueError(f"{name}: cached projections do not match input, geometry, operator, or content")
            reused = True
        if not reused:
            if fine is None: fine = torch.from_numpy(np.array(fine_np)).to(device)
            projections = torch.zeros((geometry.n_views, geometry.detector_rows, geometry.detector_cols), device=device)
            start = time.perf_counter(); ct.project_gpu(projections, fine); torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-start
            if not bool(torch.isfinite(projections).all()) or float(projections.max()) <= 0:
                raise RuntimeError(f"{name}: invalid Joseph projection data")
            save_array(projection_path, projections.cpu().numpy())
            cached = dict(signature=signature, projection_seconds=elapsed, projection_sha256=file_sha256(projection_path))
            save_json(metadata_path, cached)
            del projections
        run["results"][name] = dict(projection_seconds=cached["projection_seconds"], reused_projections=reused)
        print(f"[{name}] {'reused verified' if reused else 'saved fresh'} Joseph projections", flush=True)
        del ct
    del fine, fine_np
    torch.cuda.empty_cache()
    from circular_fdk import circular_fdk
    from grangeat_recon import reconstruct_grangeat
    run["source_sha256"] = {filename: file_sha256(Path(__file__).with_name(filename)) for filename in
                             ("run_grangeat_head.py", "circular_fdk.py", "grangeat_recon.py", "sinespin_geometry.py")}
    save_json(out/"experiment.json", run)
    reconstructions = {}
    partial = coverage = estimate = estimate_coverage = source_hull = None
    for name in geometries:
        data = np.load(out/f"projections_{name}.npy", mmap_mode="r")
        start = time.perf_counter()
        if name.startswith("circular"):
            result = circular_fdk(data, geometries[name], reference.shape, args.voxel, gpu=args.gpu)
            if isinstance(result, dict):
                reconstruction = np.asarray(result["reconstruction"], dtype=np.float32)
                diagnostics = result.get("diagnostics", {})
            else:
                reconstruction, diagnostics = np.asarray(result, dtype=np.float32), {}
        else:
            result = reconstruct_grangeat(data, geometries[name], reference.shape, args.voxel,
                                          n_polar=args.n_polar, n_azimuth=args.n_azimuth,
                                          radon_step_mm=args.radon_step_mm, line_step_mm=args.line_step_mm,
                                          gpu=args.gpu, progress=print)
            reconstruction = np.asarray(result["reconstruction"], dtype=np.float32)
            partial = np.asarray(result["partial_reconstruction"], dtype=np.float32)
            coverage = np.asarray(result["coverage"], dtype=np.float32)
            estimate = np.asarray(result["finite_detector_estimate"], dtype=np.float32)
            estimate_coverage = np.asarray(result["estimate_plane_coverage"], dtype=np.float32)
            source_hull = np.asarray(result["source_hull_mask"], dtype=bool)
            if partial.shape != reference.shape or coverage.shape != reference.shape:
                raise ValueError("Grangeat partial/coverage shapes differ from the requested grid")
            if not np.isfinite(partial).all() or not np.isfinite(coverage).all() or coverage.min() < -1e-6 or coverage.max() > 1.+1e-6:
                raise ValueError("Invalid partial reconstruction or angular coverage")
            if np.any(np.isfinite(reconstruction) & (coverage < 1.-1e-6)):
                raise ValueError("Strict Grangeat result contains values outside complete sampled support")
            if any(value.shape != reference.shape for value in (estimate, estimate_coverage, source_hull)):
                raise ValueError("Grangeat finite-detector estimate/coverage/hull shapes differ from the requested grid")
            if np.any(np.isfinite(reconstruction) & ~source_hull):
                raise ValueError("Strict Grangeat result contains values outside the source-position convex hull")
            if (not np.isfinite(estimate).all() or not np.isfinite(estimate_coverage).all()
                    or estimate_coverage.min() < -1e-6 or estimate_coverage.max() > 1.+1e-6):
                raise ValueError("Invalid finite-detector estimate or plane coverage")
            save_array(out/"recon_sinespin_220_partial.npy", partial)
            save_array(out/"coverage_sinespin_220.npy", coverage)
            save_array(out/"finite_detector_estimate_sinespin_220.npy", estimate)
            save_array(out/"estimate_plane_coverage_sinespin_220.npy", estimate_coverage)
            save_array(out/"source_hull_mask_sinespin_220.npy", source_hull)
            diagnostics = result.get("diagnostics", {})
        if reconstruction.shape != reference.shape:
            raise ValueError(f"{name}: reconstruction shape differs from reference")
        reconstructions[name] = reconstruction
        save_array(out/f"recon_{name}.npy", reconstruction)
        run["results"][name].update(reconstruction_seconds=time.perf_counter()-start, diagnostics=diagnostics)
        save_json(out/"experiment.json", run)
        print(f"[{name}] reconstruction saved", flush=True)
    common_detector = known & np.logical_and.reduce(list(masks.values()))
    common_strict = common_detector & np.logical_and.reduce([np.isfinite(r) for r in reconstructions.values()])
    z, y, x = axes; X, Y = np.meshgrid(x, y, indexing="xy")
    truth_hu = reference*(1000./mu_water)-1000.
    roi = inputs["fixed_skullbase_region"]
    soft = (truth_hu >= roi["soft_tissue_hu"][0]) & (truth_hu <= roi["soft_tissue_hu"][1])
    skullbase = (((z >= roi["z_mm"][0]) & (z < roi["z_mm"][1]))[:, None, None]
                 & ((X*X+Y*Y) <= roi["radius_mm"]**2)[None, :, :] & soft)
    tissue = dict(head=truth_hu > -500, soft_tissue=soft, skullbase_soft=skullbase, bone=truth_hu >= 300)
    run["scoring_definition"] = dict(common_detector="Acquired CT and every-view detector visibility in all selected arms",
                                    strict_common="Common detector mask and finite strict reconstructions in every arm",
                                    partial="Diagnostic only; missing-angle zero contributions are not exact",
                                    finite_detector_estimate="Approximate zero-continued detector data; scored separately, never certified exact",
                                    fixed_skullbase_region=roi, common_detector_voxels=int(common_detector.sum()),
                                    strict_common_voxels=int(common_strict.sum()))
    for name in geometries:
        run["results"][name]["acquired_ct_regions"] = score_regions(reconstructions[name], reference,
                                                     {k:v & known for k,v in tissue.items()}, mu_water)
        run["results"][name]["common_detector_regions"] = score_regions(reconstructions[name], reference,
                                                     {k:v & common_detector for k,v in tissue.items()}, mu_water)
        run["results"][name]["strict_common_regions"] = score_regions(reconstructions[name], reference,
                                                     {k:v & common_strict for k,v in tissue.items()}, mu_water)
    if "sinespin_220" in reconstructions:
        run["partial_diagnostic_regions"] = score_regions(partial, reference,
                                                         {k:v & common_detector for k,v in tissue.items()}, mu_water)
        run["finite_detector_estimate"] = dict(
            method="Grangeat inversion after zero continuation including detector-boundary derivatives; approximation under truncation",
            acquired_ct_regions=score_regions(estimate, reference, {k:v & known for k,v in tissue.items()}, mu_water),
            common_detector_regions=score_regions(estimate, reference, {k:v & common_detector for k,v in tissue.items()}, mu_water),
            source_hull_common_detector_regions=score_regions(estimate, reference,
                                                              {k:v & common_detector & source_hull for k,v in tissue.items()}, mu_water),
            source_hull_is_only_necessary_not_sufficient=True,
            source_hull_acquired_head_voxels=int(np.count_nonzero(source_hull & known & tissue["head"])),
            mean_plane_coverage_over_acquired_head=float(estimate_coverage[known & tissue["head"]].mean()))
        run["sinespin_coverage"] = dict(mean_over_acquired_head=float(coverage[known & tissue["head"]].mean()),
                                        strict_finite_voxels=int(np.isfinite(reconstructions["sinespin_220"]).sum()))
    save_json(out/"metrics.json", run)
    make_figures(out, reference, reconstructions, partial, coverage, known, masks, axes, mu_water, inputs["patient"],
                 estimate=estimate, estimate_coverage=estimate_coverage, source_hull=source_hull,
                 detector_padding_factor=args.detector_padding_factor)
    print(f"[done] {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
