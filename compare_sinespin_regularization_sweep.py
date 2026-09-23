"""Compare fixed-epoch intrinsic-only prior strengths against the same seed1 run.

Uses the audited pairwise reporter for provenance and independent geometry
measurements. GT is evaluation-only; no trajectory fit or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from calibration_gauge import PARAMETER_NAMES, effective_parameters_from_pmat
from calibration_geometry import pixel_to_pmat
from denseball_landmarks import project_landmarks
from compare_sinespin_regularization import (
    ROOT, RESULTS, read_json, sha256, validate_data, load_run,
    regularization_config, validate_initialization, validate_image_loss_sources, evaluate,
)


def make_figures(out, theta, truth, cameras, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#2469b2', '#8a45b7', '#e18123', '#15804a']
    labels = [r'$\Delta u$ [mm]', r'$\Delta f$ [mm]', r'$\Delta v$ [mm]',
              r'$t_x$ (effective) [mm]', r'$t_y$ (effective) [mm]', r'$t_z$ (effective) [mm]',
              r'$r_x$ [degree]', r'$r_y$ [degree]', r'$r_z$ [degree]']
    names = [f"lambda={item['regularization_config']['intrinsic_weight']:g}" for item in summaries]
    for errors in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, layout='constrained')
        for j, ax in enumerate(axes.flat):
            for i, camera in enumerate(cameras):
                y = camera['parameter_error9' if errors else 'parameters9'][:, j]
                ax.plot(theta, y, color=colors[i % len(colors)], lw=1., label=names[i])
            ax.plot(theta, np.zeros_like(theta) if errors else truth[:, j], color='black',
                    ls='--', lw=1.3, label='GT')
            ax.set_title(('Error in ' if errors else '')+labels[j], fontsize=11)
            ax.grid(alpha=.2)
            if j >= 6:
                ax.set_xlabel('Scan angle [degree]')
        handles, names_here = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names_here, loc='outside lower center', ncol=len(handles))
        fig.suptitle(('Canonical parameter errors relative to GT' if errors else 'Canonical effective 9DoF')+
                     '\nIntrinsic-only L2; seed 1, fixed epoch 100; data-driven axis ranges; no pose alignment')
        fig.savefig(out/('canonical_errors9.png' if errors else 'canonical_parameters9.png'), dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for i, camera in enumerate(cameras):
        color = colors[i % len(colors)]
        axes[0, 0].plot(theta, camera['source_error_mm'], lw=.9, color=color, label=names[i])
        axes[0, 1].plot(theta, camera['per_view_bead_rms_px'], lw=.9, color=color, label=names[i])
        width = .8/len(cameras)
        offset = (i-(len(cameras)-1)/2)*width
        group = summaries[i]['prior_groups']
        axes[1, 0].bar(np.arange(2)+offset,
                       [group[k]['canonical_gt_error_component_rms'] for k in ('intrinsic', 'translation')],
                       width=width, color=color, label=names[i])
        axes[1, 1].bar(i, group['rotation']['canonical_gt_error_component_rms'], color=color)
    axes[0, 0].set(xlabel='Scan angle [degree]', ylabel='Source distance to GT [mm]')
    axes[0, 1].set(xlabel='Scan angle [degree]', ylabel='Fixed-ID bead RMS [pixel]')
    axes[1, 0].set_xticks([0, 1], ['Intrinsic', 'Translation'])
    axes[1, 0].set_ylabel('GT error RMS over views and 3 components [mm]')
    axes[1, 1].set_xticks(range(len(names)), names)
    axes[1, 1].set_ylabel('Euler GT error RMS over views and 3 components [degree]')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0, 0].legend(ncol=2, fontsize=9)
    axes[0, 1].legend(ncol=2, fontsize=9)
    fig.suptitle('Intrinsic-only regularization: complete geometry and parameter errors\n'
                 'Translation and rotation have zero direct penalty; original fixed phantom frame')
    fig.savefig(out/'geometry_comparison.png', dpi=170)
    plt.close(fig)


def compare_sweep(baseline, paths, folder, out):
    baseline, folder, out = map(Path, (baseline, folder, out))
    paths = list(map(Path, paths))
    acquisition = read_json(folder/'experiment.json')
    hashes = validate_data(folder, acquisition)
    original = load_run(baseline, acquisition, hashes)
    regularized = [load_run(path, acquisition, hashes) for path in paths]
    regularized.sort(key=lambda r: regularization_config(r['recipe'])['intrinsic_weight'])
    loaded = [original]+regularized
    baseline_config = regularization_config(original['recipe'])
    if any(baseline_config[k+'_weight'] != 0 for k in ('intrinsic', 'translation', 'rotation')):
        raise ValueError('Baseline must have all regularization weights zero')
    recipe = {k: v for k, v in original['recipe'].items() if k != 'regularization'}
    core_sources = ('calibration_geometry.py', 'fast_projectors.py', 'DoF_transform.py',
                    'models/MotionNetHash.py', 'models/hash_encoder.py')
    controls, weights = [], [0.]
    for run in regularized:
        current = {k: v for k, v in run['recipe'].items() if k != 'regularization'}
        if current != recipe:
            raise ValueError('All training recipes must match except regularization')
        config = regularization_config(run['recipe'])
        weight = config['intrinsic_weight']
        expected = dict(intrinsic_weight=weight, translation_weight=0., rotation_weight=0.,
                        intrinsic_scale_mm=10., translation_scale_mm=10., rotation_scale_deg=15.)
        if config != expected or not np.isfinite(weight) or weight <= 0 or weight in weights:
            raise ValueError('Require distinct positive intrinsic-only weights with 10/10/15 scales')
        weights.append(weight)
        for key in ('train_views', 'heldout_views'):
            if run['experiment'][key] != original['experiment'][key]:
                raise ValueError(f'Changed control: {key}')
        versions = {}
        for key in ('torch', 'monai'):
            old, new = original['experiment'].get(key), run['experiment'].get(key)
            if old is not None and new is not None and old != new:
                raise ValueError(f'Changed recorded library version: {key}')
            versions[key] = dict(baseline=old, regularized=new,
                                 verified_equal=old is not None and new is not None)
        for name in core_sources:
            if run['source_sha256'][name] != original['source_sha256'][name]:
                raise ValueError(f'Changed geometry/projector/model source: {name}')
        if run['source_sha256'] != regularized[0]['source_sha256']:
            raise ValueError('Regularized runs must use identical archived training sources')
        controls.append(dict(path=str(run['path']), intrinsic_weight=weight,
                             recorded_library_versions=versions,
                             initialization=validate_initialization(baseline, run['path']),
                             image_loss=validate_image_loss_sources(baseline, run['path'], recipe['loss_config'])))
    dv, du = acquisition['truth']['pixel_vu_mm']
    truth_pixel = np.load(folder/'P_truth_pixel.npy')
    nominal = pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'), du=du, dv=dv, dtype=np.float64)
    truth_world = pixel_to_pmat(truth_pixel, du=du, dv=dv, dtype=np.float64)
    for name, matrix in (('P_nominal_world_mm.npy', nominal), ('P_truth_world_mm.npy', truth_world)):
        np.testing.assert_allclose(np.load(folder/name), matrix, rtol=2e-7, atol=2e-5)
    truth = effective_parameters_from_pmat(truth_world, nominal)
    with np.load(folder/'truth_geometry.npz') as geometry:
        theta = geometry['theta_deg']
        np.testing.assert_allclose(truth['source_xyz_mm'], geometry['source_positions'], atol=1e-8, rtol=1e-11)
    labels = read_json(folder/'landmarks.json')
    if (labels['volume_sha256'] != acquisition['volume']['sha256']
            or labels['voxel_size_mm'] != acquisition['volume']['voxel_mm']
            or labels['shape_zyx'] != acquisition['volume']['shape_zyx']):
        raise ValueError('Landmark physical reference differs from acquisition')
    points = np.array([item['xyz_mm'] for item in labels['landmarks']])
    uv = project_landmarks(truth_pixel, points)
    depth = np.einsum('vj,nj->vn', truth_pixel[:, 2], np.c_[points, np.ones(len(points))])
    rows, cols = acquisition['truth']['detector_shape_vu']
    visible = ((uv[..., 0] >= -.5) & (uv[..., 0] <= cols-.5) &
               (uv[..., 1] >= -.5) & (uv[..., 1] <= rows-.5) & (depth > 0))
    if not visible.all():
        raise ValueError('This study expects all 35 reference beads visible in all 546 views')
    summaries, cameras, table = [], [], []
    for run, weight in zip(loaded, weights):
        summary, camera = evaluate(run, truth, nominal, truth_pixel, points, visible, du, dv)
        summaries.append(summary); cameras.append(camera)
        row = dict(lambda_intrinsic=weight, image_loss=summary['image_loss'],
                   regularization_loss=summary['regularization_loss'], total_objective=summary['objective_loss'],
                   bead_rms_px=summary['bead_error_px']['rms'],
                   worst_view_bead_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
                   source_rms_mm=summary['source_error_mm']['rms'],
                   source_max_mm=summary['source_error_mm']['maximum_absolute'],
                   views_bead_rms_below_1px=summary['views_bead_rms_below_1px'])
        for key, unit in (('intrinsic', 'mm'), ('translation', 'mm'), ('rotation', 'deg')):
            row[f'{key}_gt_component_rms_{unit}'] = summary['prior_groups'][key]['canonical_gt_error_component_rms']
        table.append(row)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    np.savez(out/'regularization_sweep.npz', theta_deg=theta, weights=weights,
             truth_parameters9=truth['parameters_9'],
             **{f'run{i}_{name}': value for i, c in enumerate(cameras) for name, value in c.items()})
    make_figures(out, theta, truth['parameters_9'], cameras, summaries)
    report = dict(scope='Intrinsic-only weights at fixed seed1/100 epochs, no new pose alignment',
                  weights=weights, runs=summaries, table=table, controls=controls,
                  identical_recipe_except_regularization=recipe,
                  input_sha256=hashes, noise=acquisition['noise'], parameter_names=list(PARAMETER_NAMES),
                  limitations=['GT is evaluation-only, with no GT-based checkpoint choice.',
                               'Only intrinsic corrections receive a direct penalty; all nine parameters still train.',
                               'A smaller intrinsic correction does not imply a more accurate complete P.',
                               'One initialization and one noise realization; no general optimum claimed.',
                               'Total objectives differ; compare image-only losses and independent geometry errors.'],
                  source_sha256={name: sha256(ROOT/name) for name in
                                 ('compare_sinespin_regularization_sweep.py', 'compare_sinespin_regularization.py',
                                  'calibration_gauge.py', 'calibration_geometry.py', 'denseball_landmarks.py')},
                  artifact_sha256={name: sha256(out/name) for name in
                                   ('summary.csv', 'regularization_sweep.npz', 'canonical_parameters9.png',
                                    'canonical_errors9.png', 'geometry_comparison.png')})
    (out/'regularization_sweep.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=RESULTS/'loss_signed_lncc31_seed1')
    parser.add_argument('--runs', type=Path, nargs='+', default=[RESULTS/name for name in
                        ('reg_intrinsic001_seed1', 'reg_intrinsic01_seed1', 'reg_intrinsic1_seed1')])
    parser.add_argument('--input-dir', type=Path, default=RESULTS/'input')
    parser.add_argument('--out-dir', type=Path, default=RESULTS/'regularization_sweep')
    args = parser.parse_args()
    report = compare_sweep(args.baseline, args.runs, args.input_dir, args.out_dir)
    print(json.dumps(report['table'], indent=2))
    print(f'Wrote {args.out_dir}')


if __name__ == '__main__':
    main()
