"""Reproduce noiseless Gaussian checks of the numerical Grangeat pipeline.

Run on physical GPU 1 with the project's NumPy/SciPy/Numba environment::

    CUDA_VISIBLE_DEVICES=1 python validate_grangeat.py

The fixed translated, anisotropic Gaussian has analytic cone projections and
Radon derivatives. No fitted gain, iterative reconstruction, noise, or clipping
is applied. A Gaussian is not compactly supported; its negligible detector-edge
tails are still checked by the reconstruction's truncation diagnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from grangeat_phantom import gaussian_cone_projection, gaussian_density, gaussian_radon
from grangeat_recon import (
    grangeat_derivatives_gpu, grangeat_view_numpy, hemisphere_quadrature,
    inverse_radon_gpu, reconstruct_grangeat,
)
from sinespin_geometry import build_icono_orbit


CENTER = np.array([4., -3., 5.])
COVARIANCE = np.array([[144., 20., 0.], [20., 100., 10.], [0., 10., 81.]])
AMPLITUDE = .02
SHAPE = (49, 49, 49)
VOXEL_MM = 2.
DEFAULT_LEVELS = ((546, 8, 12, 48, 2.), (1092, 4, 24, 96, 1.), (2184, 2, 32, 128, .5))


def parse_level(text):
    try:
        fields = text.split(',')
        if len(fields) != 5:
            raise ValueError
        views, detector_bin, polar, azimuth = map(int, fields[:4])
        dp = float(fields[4])
        if views < 2 or detector_bin < 1 or polar < 2 or azimuth < 4 or not np.isfinite(dp) or dp <= 0:
            raise ValueError
        return views, detector_bin, polar, azimuth, dp
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use views,detector_bin,polar,azimuth,positive_dp_mm') from exc


def analytic_projections(geometry):
    """Evaluate cone-ray integrals view by view, without a sampled volume forward projector."""
    g = geometry
    u = (np.arange(g.detector_cols) - (g.detector_cols - 1) / 2) * g.pixel_width
    v = (np.arange(g.detector_rows) - (g.detector_rows - 1) / 2) * g.pixel_height
    output = np.empty((g.n_views, g.detector_rows, g.detector_cols), dtype=np.float32)
    normals = g.normals
    for i in range(g.n_views):
        rays = (g.sdd_mm * normals[i] + u[None, :, None] * g.col_vectors[i]
                + v[:, None, None] * g.row_vectors[i])
        rays /= np.linalg.norm(rays, axis=-1)[..., None]
        output[i] = gaussian_cone_projection(g.source_positions[i], rays, CENTER, COVARIANCE, AMPLITUDE)
    return output


def _relative_l2(actual, expected):
    return float(np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected))


def stage1_check(projections, geometry, gpu):
    """Compare useful R' samples against both an independent CPU line integral and the analytic value."""
    g = geometry
    selected = np.unique(np.linspace(0, g.n_views - 1, 5).round().astype(int))
    subset = replace(g, source_positions=g.source_positions[selected], module_centers=g.module_centers[selected],
                     row_vectors=g.row_vectors[selected], col_vectors=g.col_vectors[selected],
                     theta_deg=g.theta_deg[selected], tilt_deg=g.tilt_deg[selected])
    blocks = []
    for source in subset.source_positions:
        direction = source - CENTER
        distance = np.linalg.norm(direction)
        direction /= distance
        helper = np.eye(3)[np.argmin(np.abs(direction))]
        first = np.cross(direction, helper)
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        block = []
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            transverse = np.cos(angle) * first + np.sin(angle) * second
            for offset in (-24., -12., -6., 6., 12., 24.):
                fraction = offset / distance
                block.append(np.sqrt(1 - fraction ** 2) * transverse + fraction * direction)
        blocks.append(np.asarray(block))
    normals = np.concatenate(blocks)
    values, complete = grangeat_derivatives_gpu(projections[selected], subset, normals, gpu=gpu)
    actual, oracle, analytic, valid = [], [], [], []
    cursor = 0
    for i, block in enumerate(blocks):
        count = len(block)
        actual.extend(values[cursor:cursor + count, i])
        valid.extend(complete[cursor:cursor + count, i])
        oracle.extend(grangeat_view_numpy(projections[selected[i]], subset, i, block))
        positions = block @ subset.source_positions[i]
        analytic.extend(gaussian_radon(block, positions, CENTER, COVARIANCE, AMPLITUDE, derivative=1))
        cursor += count
    actual, oracle, analytic = map(np.asarray, (actual, oracle, analytic))
    return dict(samples=int(len(actual)), selected_view_indices=selected.tolist(),
                finite_samples=int(np.isfinite(actual).sum()), detector_complete_fraction=float(np.mean(valid)),
                gpu_vs_analytic_relative_l2=_relative_l2(actual, analytic),
                cpu_vs_analytic_relative_l2=_relative_l2(oracle, analytic),
                gpu_vs_cpu_relative_l2=_relative_l2(actual, oracle),
                gpu_vs_cpu_max_absolute=float(np.max(np.abs(actual - oracle))),
                analytic_max_absolute=float(np.max(np.abs(analytic))))


def validate_level(level, truth, output, gpu):
    views, detector_bin, polar, azimuth, dp = level
    identifier = f'v{views}_bin{detector_bin}_n{polar}x{azimuth}_dp{dp:g}'
    print(f'[{identifier}] preparing analytic cone projections', flush=True)
    start = time.perf_counter()
    geometry = build_icono_orbit('sinespin', detector_bin=detector_bin, n_views=views)
    projections = analytic_projections(geometry)
    projection_seconds = time.perf_counter() - start
    print(f'[{identifier}] Stage 1: analytic/CPU/GPU Radon-derivative comparison', flush=True)
    stage1 = stage1_check(projections, geometry, gpu)
    normals, weights = hemisphere_quadrature(polar, azimuth)
    offsets = np.arange(-90., 90. + dp / 2, dp)
    analytic_second = gaussian_radon(normals[:, None, :], offsets[None, :], CENTER, COVARIANCE,
                                     AMPLITUDE, derivative=2)
    print(f'[{identifier}] direct inversion of analytic R double-prime', flush=True)
    direct, direct_coverage = inverse_radon_gpu(analytic_second, normals, weights, offsets, SHAPE, VOXEL_MM, gpu=gpu)
    result = reconstruct_grangeat(projections, geometry, SHAPE, VOXEL_MM, n_polar=polar,
                                 n_azimuth=azimuth, radon_step_mm=dp, gpu=gpu,
                                 progress=lambda message: print(f'[{identifier}] {message}', flush=True))
    roi = truth > AMPLITUDE * .01
    finite = np.isfinite(result['reconstruction'])
    valid_roi = roi & finite
    report = dict(level=identifier, views=views, detector_bin=detector_bin,
                  detector_shape=[geometry.detector_rows, geometry.detector_cols],
                  detector_pixel_mm=[geometry.pixel_height, geometry.pixel_width],
                  polar=polar, azimuth=azimuth, dp_mm=dp,
                  projection_seconds=projection_seconds, stage1=stage1,
                  direct_inverse_relative_l2=_relative_l2(direct, truth),
                  direct_inverse_coverage_min=float(direct_coverage.min()),
                  pipeline_relative_l2=_relative_l2(result['reconstruction'][valid_roi], truth[valid_roi]) if valid_roi.any() else None,
                  roi_voxels=int(roi.sum()), valid_roi_voxels=int(valid_roi.sum()),
                  roi_coverage=float(finite[roi].mean()),
                  peak_ratio=float(np.nanmax(result['reconstruction']) / truth.max()) if finite.any() else None,
                  total_seconds=time.perf_counter() - start, diagnostics=result['diagnostics'])
    np.savez_compressed(output / f'{identifier}.npz', truth=truth.astype(np.float32), direct_inverse=direct,
                        reconstruction=result['reconstruction'], partial_reconstruction=result['partial_reconstruction'],
                        coverage=result['coverage'])
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', type=int, default=1, help='Physical GPU index (default: 1)')
    parser.add_argument('--levels', type=parse_level, nargs='+', default=DEFAULT_LEVELS)
    parser.add_argument('--output-dir', type=Path, default=Path('result_sinespin/grangeat_validation'))
    parser.add_argument('--summary-json', type=Path, default=Path('docs/grangeat_validation.json'))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = [(np.arange(n) - (n - 1) / 2) * VOXEL_MM for n in SHAPE]
    z, y, x = np.meshgrid(*coordinates, indexing='ij')
    truth = gaussian_density(np.stack((x, y, z), axis=-1), CENTER, COVARIANCE, AMPLITUDE)
    summary = dict(method='Analytic translated anisotropic Gaussian validation',
                   reference='https://doi.org/10.1007/BFb0084509',
                   center_xyz_mm=CENTER.tolist(), covariance_mm2=COVARIANCE.tolist(), amplitude=AMPLITUDE,
                   shape_zyx=list(SHAPE), voxel_mm=VOXEL_MM, noise=False, fitted_gain=False,
                   positivity_clipping=False, reconstruction_iterations=0,
                   roi_definition='analytic density > 1% of Gaussian amplitude; finite measured-plane reconstruction',
                   scan_angle_deg=220., tilt_amplitude_deg=10., sod_mm=750., sdd_mm=1200., levels=[])
    for level in args.levels:
        report = validate_level(level, truth, args.output_dir, args.gpu)
        summary['levels'].append(report)
        (args.output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    metrics = [entry['pipeline_relative_l2'] for entry in summary['levels']]
    summary['pipeline_error_decreased_at_each_level'] = (
        all(a is not None and b is not None and b < a for a, b in zip(metrics[:-1], metrics[1:]))
        if len(metrics) > 1 else None)
    if len(metrics) > 1:
        shared = truth > AMPLITUDE * .01
        reconstructions = []
        for entry in summary['levels']:
            with np.load(args.output_dir / (entry['level'] + '.npz')) as arrays:
                reconstructions.append(arrays['reconstruction'])
                shared &= np.isfinite(arrays['reconstruction'])
        summary['common_roi_voxels'] = int(shared.sum())
        summary['common_roi_relative_l2'] = [_relative_l2(recon[shared], truth[shared]) if shared.any() else None
                                            for recon in reconstructions]
    payload = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    (args.output_dir / 'metrics.json').write_text(payload)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(payload)
    print(f'Wrote {args.output_dir / "metrics.json"} and {args.summary_json}', flush=True)


if __name__ == '__main__':
    main()
