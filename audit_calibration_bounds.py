"""CPU-only audit of saved effective 9-DoF ranges and local tanh attenuation.

Truth is used only for this post-hoc diagnostic. No parameters, images or
checkpoints are changed; completed snapshots retain their original epoch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from calibration_geometry import centered_source_positions, pmat_to_pixel
from denseball_landmarks import project_landmarks
from DoF_transform import apply_9DoF_transform_effective


PARAMETERS = [
    ('ts_x', 'principal point u', 'mm'),
    ('ts_y', 'shared focal length', 'mm'),
    ('ts_z', 'principal point v', 'mm'),
    ('tp_x', 'effective rigid translation x', 'mm'),
    ('tp_y', 'effective rigid translation y', 'mm'),
    ('tp_z', 'effective rigid translation z', 'mm'),
    ('rx', 'effective internal Euler x', 'degree'),
    ('ry', 'effective internal Euler y', 'degree'),
    ('rz', 'effective internal Euler z', 'degree'),
]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def apply_saved_motion(nominal, motion, metadata):
    shape = metadata['volume']['shape_zyx']
    voxel = metadata['volume']['voxel_mm']
    tensor = torch.tensor(motion, dtype=torch.float32)
    with torch.no_grad():
        matrix, _ = apply_9DoF_transform_effective(
            torch.tensor(nominal, dtype=torch.float32), torch.zeros((len(motion), 7)),
            tensor[:, :3], tensor[:, 3:6], tensor[:, 6:],
            nx=shape[2], ny=shape[1], nz=shape[0], dx=voxel, dy=voxel, dz=voxel,
            X0=-shape[2]*voxel/2, Y0=-shape[1]*voxel/2, Z0=-shape[0]*voxel/2,
            use_inverse_right_multiply=0)
    return matrix.numpy().reshape(-1, 3, 4)


def reprojection(matrix, xyz, truth_q, visible, du, dv):
    q = project_landmarks(pmat_to_pixel(matrix, du=du, dv=dv), xyz)
    error = np.linalg.norm(q-truth_q, axis=-1)
    if not np.isfinite(error).all() or np.any(visible.sum(axis=1) == 0):
        raise ValueError('Every view needs finite projections and visible fixed-ID landmarks.')
    return np.sqrt(np.sum(error**2*visible, axis=1)/visible.sum(axis=1))


def audit(input_dir, run_dirs, out_dir, *, require_final_epoch=None, expected_bounds=None):
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    run_dirs = [Path(path) for path in run_dirs]
    if not run_dirs or len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError('Supply one or more distinct run directories.')
    metadata = json.loads((input_dir/'experiment.json').read_text())
    input_hash = digest(input_dir/'experiment.json')
    landmarks_path = input_dir/'landmarks.json'
    landmarks = json.loads(landmarks_path.read_text())
    if landmarks['volume_sha256'] != metadata['volume']['sha256']:
        raise ValueError('Landmarks belong to another reference volume.')
    if 'landmarks_sha256' in metadata and digest(landmarks_path) != metadata['landmarks_sha256']:
        raise ValueError('Landmarks have changed since target preparation.')
    if metadata['nominal']['kind'] != 'circular' or metadata['truth']['kind'] != 'sinespin':
        raise ValueError('The analytical oracle in this audit requires circular -> sineSpin geometry.')
    expected_origin = -.5*(np.asarray(metadata['volume']['shape_zyx'])[::-1]-1)*metadata['volume']['voxel_mm']
    np.testing.assert_allclose(metadata['volume']['voxel_center_origin_xyz_mm'], expected_origin,
                               atol=1e-10, rtol=0, err_msg='This audit assumes a volume box centred at isocentre.')
    for filename, key in [('P_nominal_world_mm.npy', 'nominal_pmat_sha256'),
                          ('P_truth_world_mm.npy', 'truth_pmat_sha256')]:
        if digest(input_dir/filename) != metadata[key]:
            raise ValueError(f'{filename} has changed since target preparation.')
    geometry = np.load(input_dir/'truth_geometry.npz')
    azimuth = geometry['theta_deg']
    tilt = geometry['tilt_deg']
    nominal = np.load(input_dir/'P_nominal_world_mm.npy')
    truth_pixel = np.load(input_dir/'P_truth_pixel.npy')
    truth_source = geometry['source_positions']
    xyz = np.asarray([x['xyz_mm'] for x in landmarks['landmarks']])
    identifiers = [x['landmark_id'] for x in landmarks['landmarks']]
    if (xyz.shape != (landmarks['landmark_count'], 3) or not np.isfinite(xyz).all()
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Expected finite landmarks with unique fixed IDs.')
    if nominal.shape != (len(azimuth), 3, 4) or truth_pixel.shape != nominal.shape:
        raise ValueError('Geometry and P matrix view counts differ.')
    nominal_source = centered_source_positions(nominal)
    truth_q = project_landmarks(truth_pixel, xyz)
    rows, cols = metadata['truth']['detector_shape_vu']
    dv, du = metadata['truth']['pixel_vu_mm']
    truth_world = np.load(input_dir/'P_truth_world_mm.npy')
    recovered_truth_pixel = pmat_to_pixel(truth_world, du=du, dv=dv)
    np.testing.assert_allclose(project_landmarks(recovered_truth_pixel, xyz), truth_q,
                               atol=2e-4, rtol=0, err_msg='Truth P coordinate systems disagree.')
    np.testing.assert_allclose(centered_source_positions(truth_world), truth_source,
                               atol=5e-4, rtol=0, err_msg='Truth source geometry disagrees with its P matrices.')
    depth = np.einsum('vj,nj->vn', truth_pixel[:, 2], np.c_[xyz, np.ones(len(xyz))])
    visible = ((truth_q[..., 0] >= -.5) & (truth_q[..., 0] <= cols-.5)
               & (truth_q[..., 1] >= -.5) & (truth_q[..., 1] <= rows-.5) & (depth > 0))
    oracle = np.zeros((len(azimuth), 9), dtype=np.float64)
    oracle[:, 6:] = Rotation.from_rotvec(np.deg2rad(tilt)[:, None]*geometry['col_vectors']).as_euler('xyz', degrees=True)
    oracle_p = apply_saved_motion(nominal, oracle, metadata)
    oracle_error = reprojection(oracle_p, xyz, truth_q, visible, du, dv)
    if oracle_error.max() > .005:
        raise ValueError('The stated circular-to-sineSpin oracle does not reproduce the prepared truth.')
    noise = metadata.get('noise')
    noise_label = ('No added noise' if noise is None else
                   f"Poisson I0={noise['i0_photons_per_detector_pixel_per_view']:g}, seed={noise['seed']}")
    result = {
        'purpose': 'Post-hoc CPU diagnostic only; truth is not a training input or initializer',
        'noise_of_audited_data': metadata['generator'],
        'noise': noise,
        'figure_data_label': f"{landmarks['landmark_count']} fixed balls; {noise_label}",
        'landmark_count': landmarks['landmark_count'],
        'visible_ball_view_pairs': int(visible.sum()),
        'all_ball_view_pairs': int(visible.size),
        'required_final_epoch': require_final_epoch,
        'expected_bounds_ts_tp_rot': expected_bounds,
        'input_sha256': {name: digest(input_dir/name) for name in (
            'experiment.json', 'landmarks.json', 'P_nominal_world_mm.npy',
            'P_truth_world_mm.npy', 'P_truth_pixel.npy', 'truth_geometry.npz')},
        'audit_source_sha256': {name: digest(Path(__file__).parent/name) for name in (
            'audit_calibration_bounds.py', 'calibration_geometry.py', 'DoF_transform.py')},
        'parameter_order': [dict(name=n, meaning=m, unit=u) for n, m, u in PARAMETERS],
        'mapping': {
            'bounded_parameter': 'bound * tanh(raw)',
            'normalized_tanh_derivative': '1 - (bounded_parameter / bound)^2; multiply by bound for d(parameter)/d(raw)',
            'ts_internal_to_K': 'ts_x -> u0; ts_y -> fu and fv together; ts_z -> v0; all in detector millimetres',
            'source_position': 'C_new = R_object.T @ (C_nominal - tp_world) for this centred zero pivot, with tp_world=(tp_x,tp_z,tp_y). K-only ts does not move the ideal camera centre.',
            'source_bounds': 'The ts/tp limits are not bounds on physical source xyz. Source motion from rotation scales with source distance; the physical displacements below are computed from P.',
            'nominal_source_isocentre_distance_mm': metadata['nominal']['sod_mm'],
        },
        'azimuth_deg': azimuth.tolist(), 'tilt_deg': tilt.tolist(),
        'truth_oracle': {
            'rotation_construction': 'Object rotation +tilt about nominal detector column, converted to internal xyz Euler; ts=tp=0',
            'minimum': oracle.min(0).tolist(), 'maximum': oracle.max(0).tolist(),
            'maximum_absolute': np.abs(oracle).max(0).tolist(),
            'per_view_ball_reprojection_rmse_px_maximum': float(oracle_error.max()),
            'parameters': oracle.tolist(),
        },
        'runs': [],
    }
    for run_dir in run_dirs:
        experiment = json.loads((run_dir/'experiment.json').read_text())
        checkpoint_hash = digest(run_dir/'checkpoint.pt')
        checkpoint = torch.load(run_dir/'checkpoint.pt', map_location='cpu', weights_only=False)
        epoch = int(checkpoint['epoch'])
        recipe = checkpoint['recipe']
        if (checkpoint['input_sha256'] != input_hash or experiment['input'] != metadata
                or experiment['recipe'] != recipe):
            raise ValueError(f'{run_dir.name}: checkpoint, run recipe and prepared input disagree.')
        if not checkpoint['history'] or int(checkpoint['history'][-1]['epoch']) != epoch:
            raise ValueError('Checkpoint history does not reach its recorded epoch.')
        if require_final_epoch is not None and (epoch != require_final_epoch or recipe['epochs'] != require_final_epoch):
            raise ValueError(f'{run_dir.name}: expected completed fixed epoch {require_final_epoch}, found {epoch}/{recipe["epochs"]}.')
        snapshot_path = run_dir/f'P_epoch{epoch:04d}.npy'
        motion_hash = digest(run_dir/'motion9.npy')
        matrix = np.load(snapshot_path)
        motion = np.load(run_dir/'motion9.npy').astype(np.float64)
        if motion.shape != oracle.shape or not np.isfinite(motion).all():
            raise ValueError('Invalid saved 9-DoF arrays')
        bounds = np.repeat([recipe['bounds'][k] for k in ('ts_max_mm', 'tp_max_mm', 'rot_max_deg')], 3)
        if not np.isfinite(bounds).all() or not (bounds > 0).all():
            raise ValueError('Expected positive effective 9-DoF bounds')
        if expected_bounds is not None and not np.array_equal(bounds[::3], expected_bounds):
            raise ValueError(f'{run_dir.name}: recorded bounds differ from explicitly required bounds.')
        if require_final_epoch is not None:
            final_matrix = np.load(run_dir/'P_optimized_world_mm.npy')
            np.testing.assert_array_equal(final_matrix, matrix, err_msg='Final P differs from the requested epoch snapshot.')
            metrics = json.loads((run_dir/'metrics.json').read_text())
            if metrics['recipe'] != recipe:
                raise ValueError('Final metrics do not describe the checkpoint recipe.')
        fractions = np.abs(motion)/bounds
        derivative = 1-fractions**2
        if (fractions > 1+1e-6).any():
            raise ValueError('Saved bounded parameters exceed their recorded range')
        reapplied = apply_saved_motion(nominal, motion, metadata)
        # CPU/GPU float32 QR can differ, so compare projected coordinates directly.
        saved_q = project_landmarks(pmat_to_pixel(matrix, du=du, dv=dv), xyz)
        reapplied_q = project_landmarks(pmat_to_pixel(reapplied, du=du, dv=dv), xyz)
        consistency = np.linalg.norm(saved_q-reapplied_q, axis=-1)
        if consistency.max() > .005:
            raise ValueError('motion9.npy is inconsistent with the saved checkpoint matrix')
        errors = reprojection(matrix, xyz, truth_q, visible, du, dv)
        source = centered_source_positions(matrix)
        source_displacement = np.linalg.norm(source-nominal_source, axis=-1)
        truth_displacement = np.linalg.norm(truth_source-nominal_source, axis=-1)
        per_parameter = []
        for j, (name, meaning, unit) in enumerate(PARAMETERS):
            per_parameter.append(dict(
                name=name, meaning=meaning, unit=unit, bound=float(bounds[j]),
                minimum=float(motion[:, j].min()), maximum=float(motion[:, j].max()),
                maximum_absolute_bound_fraction=float(fractions[:, j].max()),
                view_fraction_at_or_above_95_percent=float(np.mean(fractions[:, j] >= .95)),
                view_fraction_at_or_above_99_percent=float(np.mean(fractions[:, j] >= .99)),
                normalized_tanh_derivative_minimum=float(derivative[:, j].min()),
                normalized_tanh_derivative_median=float(np.median(derivative[:, j])),
                normalized_tanh_derivative_p05=float(np.quantile(derivative[:, j], .05)),
                oracle_maximum_absolute_bound_fraction=float(np.abs(oracle[:, j]).max()/bounds[j]),
                oracle_normalized_tanh_derivative_minimum=float(1-(np.abs(oracle[:, j]).max()/bounds[j])**2),
            ))
        worst = []
        for i in np.argsort(errors)[-10:][::-1]:
            j = int(fractions[i].argmax())
            worst.append(dict(view=int(i), azimuth_deg=float(azimuth[i]), tilt_deg=float(tilt[i]),
                              ball_reprojection_rmse_px=float(errors[i]), source_error_mm=float(np.linalg.norm(source[i]-truth_source[i])),
                              largest_bound_fraction=float(fractions[i, j]), most_attenuated_parameter=PARAMETERS[j][0],
                              minimum_normalized_tanh_derivative=float(derivative[i].min()),
                              parameter_values=motion[i].tolist(), oracle_parameter_values=oracle[i].tolist()))
        most = np.unravel_index(np.argmax(fractions), fractions.shape)
        run = dict(
            name=run_dir.name, path=str(run_dir), checkpoint_epoch=epoch,
            requested_final_epoch=int(recipe['epochs']), completed_fixed_final_epoch=epoch == int(recipe['epochs']),
            loss_levels=recipe['loss_levels'], bounds=recipe['bounds'],
            any_view_parameter_at_or_above_95_percent=bool(np.any(fractions >= .95)),
            any_view_parameter_at_or_above_99_percent=bool(np.any(fractions >= .99)),
            most_bound_attenuated_view=dict(view=int(most[0]), parameter=PARAMETERS[most[1]][0],
                azimuth_deg=float(azimuth[most[0]]), bound_fraction=float(fractions[most]),
                normalized_tanh_derivative=float(derivative[most]), ball_reprojection_rmse_px=float(errors[most[0]])),
            oracle_strictly_inside_all_bounds=bool((np.abs(oracle) < bounds).all()),
            parameters=per_parameter, worst_ten_views=worst,
            source_z_min_max_mm=[float(source[:, 2].min()), float(source[:, 2].max())],
            truth_source_z_min_max_mm=[float(truth_source[:, 2].min()), float(truth_source[:, 2].max())],
            physical_source_displacement_from_nominal_mm=dict(
                maximum=float(source_displacement.max()), rms=float(np.sqrt(np.mean(source_displacement**2))),
                truth_maximum=float(truth_displacement.max()), truth_rms=float(np.sqrt(np.mean(truth_displacement**2)))),
            mean_per_view_rmse_px=float(errors.mean()), maximum_per_view_rmse_px=float(errors.max()),
            artifact_sha256={p.name: digest(p) for p in (snapshot_path, run_dir/'motion9.npy', run_dir/'checkpoint.pt')},
            motion_snapshot_consistency_max_projected_error_px=float(consistency.max()),
            per_view_ball_rmse_px=errors.tolist(), per_view_maximum_bound_fraction=fractions.max(1).tolist(),
            per_view_minimum_normalized_tanh_derivative=derivative.min(1).tolist(),
            parameter_values=motion.tolist(),
        )
        if (digest(run_dir/'checkpoint.pt') != checkpoint_hash or digest(run_dir/'motion9.npy') != motion_hash):
            raise ValueError('Training artifacts changed during the audit; run again on stable completed artifacts.')
        result['runs'].append(run)
    result['interpretation'] = [
        'Worst-view errors must be compared to same-view parameter fractions; a source-height plateau alone does not diagnose a parameter bound.',
        'A true solution inside the range establishes representability. It does not exclude optimization slowdown from tanh, parameter coupling, image ambiguities or noise.',
        f'Data: {noise_label}. Noise metadata are read from the prepared input linked to every audited checkpoint.',
    ]
    for run in result['runs']:
        most = run['most_bound_attenuated_view']
        worst = run['worst_ten_views'][0]
        result['interpretation'].append(
            f"{run['name']}: saved epoch {run['checkpoint_epoch']}/{run['requested_final_epoch']}; "
            f"any parameter at >=95% bound: {run['any_view_parameter_at_or_above_95_percent']}; "
            f"true parameters strictly inside all bounds: {run['oracle_strictly_inside_all_bounds']}; "
            f"minimum normalized tanh derivative {most['normalized_tanh_derivative']:.6g} at view {most['view']}. "
            f"Worst-error view {worst['view']} instead has minimum normalized derivative "
            f"{worst['minimum_normalized_tanh_derivative']:.6g} and maximum bound fraction {worst['largest_bound_fraction']:.6g}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save_figures(result, out_dir)
    return result


def save_figures(result, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angle = np.asarray(result['azimuth_deg'])
    oracle = np.asarray(result['truth_oracle']['parameters'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for j, axis in enumerate(axes.flat):
        name, meaning, unit = PARAMETERS[j]
        axis.plot(angle, oracle[:, j], color='black', linewidth=1.7, label='True representable values (audit only)')
        all_bounds = []
        for run in result['runs']:
            axis.plot(angle, np.asarray(run['parameter_values'])[:, j], linewidth=1.1,
                      label=f"{run['name']} / epoch {run['checkpoint_epoch']}")
            all_bounds.append(run['parameters'][j]['bound'])
        for bound in sorted(set(all_bounds)):
            axis.axhline(bound, color='red', linestyle=':', alpha=.55)
            axis.axhline(-bound, color='red', linestyle=':', alpha=.55)
        axis.set(title=f'{name}: {meaning}', xlabel='Azimuth (degree)', ylabel=unit)
        axis.grid(alpha=.18)
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f"{result['figure_data_label']}: all nine bounded effective parameters\nRed dotted lines: called bounds; true values are diagnostic only", fontsize=14)
    fig.savefig(out_dir/'parameter_ranges.png', dpi=170)
    plt.close(fig)
    if len(result['runs']) == 1:
        run = result['runs'][0]
        values = np.asarray(run['parameter_values'])
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
        for j, axis in enumerate(axes.flat):
            name, meaning, unit = PARAMETERS[j]
            axis.plot(angle, oracle[:, j], color='black', linewidth=1.65,
                      label='True representable values (audit only)')
            axis.plot(angle, values[:, j], color='tab:blue', linewidth=1.25,
                      label=f"Estimated / epoch {run['checkpoint_epoch']}")
            low, high = min(values[:, j].min(), oracle[:, j].min()), max(values[:, j].max(), oracle[:, j].max())
            margin = max((high-low)*.18, .01)
            axis.set(ylim=(low-margin, high+margin), title=f'{name}: {meaning}',
                     xlabel='Azimuth (degree)', ylabel=unit)
            bound = run['parameters'][j]['bound']
            axis.text(.98, .96, f'Bound: ±{bound:g} {unit}', transform=axis.transAxes,
                      ha='right', va='top', color='firebrick', fontsize=9,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=.82))
            axis.grid(alpha=.18)
        axes[0, 0].legend(fontsize=8, loc='lower left')
        fig.suptitle(f"{run['name']} / epoch {run['checkpoint_epoch']}: estimated nine parameters\n"
                     f"{result['figure_data_label']} — axes zoomed to curves; bounds are numerical annotations", fontsize=14)
        fig.savefig(out_dir/'parameter_estimates.png', dpi=170)
        plt.close(fig)
        with (out_dir/'parameter_estimates.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['view', 'azimuth_deg', 'tilt_deg']+
                            [f"{name}_{'deg' if unit == 'degree' else unit}" for name, _, unit in PARAMETERS])
            for view in range(len(angle)):
                writer.writerow([view, angle[view], result['tilt_deg'][view], *values[view]])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for run in result['runs']:
        label=f"{run['name']} / epoch {run['checkpoint_epoch']}"
        axes[0].plot(angle, run['per_view_ball_rmse_px'], label=label)
        axes[1].plot(angle, run['per_view_maximum_bound_fraction'], label=label)
        axes[2].plot(angle, run['per_view_minimum_normalized_tanh_derivative'], label=label)
        worst = run['worst_ten_views'][0]
        axes[0].scatter([worst['azimuth_deg']], [worst['ball_reprojection_rmse_px']], s=40)
        axes[0].annotate(f"view {worst['view']}", (worst['azimuth_deg'], worst['ball_reprojection_rmse_px']), xytext=(5, 5), textcoords='offset points')
    axes[0].set(ylabel='Fixed-ID bead RMSE (pixel)', title='Error peaks versus parameter attenuation at the same measured views')
    axes[1].axhline(.95, color='red', linestyle=':', label='95% of bound')
    axes[1].set(ylabel='Maximum |parameter / bound|', ylim=(0, 1.05))
    axes[2].set(ylabel='Minimum 1 - (parameter / bound)^2', xlabel='Azimuth (degree)', ylim=(0, 1.05))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=9)
    fig.savefig(out_dir/'error_vs_bound_fraction.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('result_sinespin/ball_calibration')
    baseline = base/'baseline_rot15_seed0'
    parser.add_argument('--input-dir', type=Path, default=base/'input')
    parser.add_argument('--run-dirs', type=Path, nargs='+', default=[baseline])
    parser.add_argument('--out-dir', type=Path, default=baseline/'parameter_audit')
    parser.add_argument('--require-final-epoch', type=int,
                        help='Require this fixed epoch and matching completed final artifacts for every run.')
    parser.add_argument('--expected-bounds', type=float, nargs=3, metavar=('TS_MM', 'TP_MM', 'ROT_DEG'),
                        help='Reject runs whose recorded bounds differ from these explicit expected values.')
    args = parser.parse_args()
    result = audit(args.input_dir, args.run_dirs, args.out_dir,
                   require_final_epoch=args.require_final_epoch, expected_bounds=args.expected_bounds)
    for run in result['runs']:
        print(run['name'], 'epoch', run['checkpoint_epoch'], 'most attenuated', run['most_bound_attenuated_view'])
        print('worst view', run['worst_ten_views'][0])
    print('Saved', args.out_dir)


if __name__ == '__main__':
    main()
