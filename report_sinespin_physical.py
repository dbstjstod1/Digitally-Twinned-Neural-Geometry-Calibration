"""Plot source-centred physical camera quantities and errors against sineSpin GT.

This is a read-only analysis of frozen cameras. GT is used for reporting, never
to fit camera parameters or a new alignment. The existing single phantom pose
is an optional second reporting frame, displayed alongside the original frame.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from calibration_gauge import apply_rigid_frame
from denseball_landmarks import project_landmarks
from physical_camera import (compose_physical_camera, compose_physical_nine,
                             decompose_physical_camera, physical_camera_residuals)
from report_sinespin_pose import ROOT, DEFAULT_RUN, read_json, sha256, stats


NAMES = ['source_x_mm', 'source_y_mm', 'source_z_mm',
         'orientation_x_deg', 'orientation_y_deg', 'orientation_z_deg',
         'f_mm', 'cu_mm', 'cv_mm']
COLORS = {'truth': '#151515', 'estimated': '#2469b2', 'phantom_aligned': '#dc7b19'}
LEGEND = {'truth': 'GT', 'estimated': 'Estimated, original frame',
          'phantom_aligned': 'One phantom-pose frame change'}


def display_values(cameras):
    """Euler tracks are for absolute display only; errors use SO(3) logarithms."""
    values = {}
    for name, camera in cameras.items():
        q = camera['Q_camera_to_physical']
        angles = Rotation.from_matrix(q).as_euler('xyz', degrees=False)
        if np.max(np.abs(angles[:, 1])) > np.deg2rad(89.):
            raise ValueError('Absolute Euler display is near gimbal lock; use relative rotations.')
        angles = np.rad2deg(np.unwrap(angles, axis=0))
        if name != 'truth':
            # Choose a constant whole-turn display branch near GT, without
            # modifying the rotation matrix or using GT in reconstruction.
            angles -= 360. * np.round((angles[0]-values['truth'][0, 3:6])/360.)
        np.testing.assert_allclose(Rotation.from_euler('xyz', angles, degrees=True).as_matrix(),
                                   q, atol=2e-14, rtol=0)
        values[name] = np.column_stack((camera['source_xyz_mm'], angles,
                                       camera['intrinsics_f_cu_cv_mm']))
    return values


def plot_nine(out, theta, values, errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source_labels = [rf'$C_{axis}$ [mm]' for axis in 'xyz']
    angle_labels = [rf'$\mathrm{{Euler}}_{axis}(Q)$ [degree]' for axis in 'xyz']
    intrinsic_labels = ['$f$ [mm]', '$c_u$ [mm, detector-edge origin]',
                        '$c_v$ [mm, detector-edge origin]']
    error_labels = ([rf'$\Delta C_{axis}$ [mm]' for axis in 'xyz'] +
                    [rf'$\omega_{axis}$ [degree, physical axes]' for axis in 'xyz'] +
                    ['$\Delta f$ [mm]', '$\Delta c_u$ [mm]', '$\Delta c_v$ [mm]'])
    for kind, series, labels in (
            ('gt_errors', errors, error_labels),
            ('values', values, source_labels+angle_labels+intrinsic_labels)):
        fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharex=True, layout='constrained')
        for j, ax in enumerate(axes.flat):
            for name in ('estimated', 'phantom_aligned', 'truth'):
                ax.plot(theta, series[name][:, j], color=COLORS[name],
                        ls='--' if name == 'truth' else '-',
                        lw=1.45 if name == 'truth' else 1.05)
            ax.set_title(labels[j], fontsize=11)
            ax.grid(alpha=.22)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            if j >= 6:
                ax.set_xlabel('Scan angle [degree]')
            if kind == 'gt_errors':
                limit = max(float(np.max(np.abs(series[name][:, j])))
                            for name in ('estimated', 'phantom_aligned'))
                ax.set_ylim(-max(limit*1.12, .001), max(limit*1.12, .001))
        handles = [Line2D([], [], color=COLORS[n], ls='--' if n == 'truth' else '-',
                          label=LEGEND[n]) for n in ('truth', 'estimated', 'phantom_aligned')]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=10)
        if kind == 'gt_errors':
            title = ('Physical camera errors relative to GT: source (3), orientation (3), intrinsics (3)\n'
                     r'$\omega=\log(Q_{est}Q_{GT}^{T})^\vee$; GT = 0; original P retained, no fit toward GT')
        else:
            title = ('Source-centred physical camera decomposition: signed LNCC31, seed 1, epoch 100\n'
                     'Source xyz; camera-to-physical Euler xyz (Rz Ry Rx); shared focal and principal point')
        fig.suptitle(title, fontsize=13)
        fig.savefig(out/f'physical_parameters9_{kind}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'result_sinespin/ball_calibration/input')
    parser.add_argument('--pose-audit', type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    run, folder = args.run_dir, args.input_dir
    out = args.out_dir or run/'pose_report'
    pose_file = args.pose_audit or run/'pose_audit/phantom_pose.json'
    saved, acquisition, pose = map(read_json, (run/'metrics.json', folder/'experiment.json', pose_file))
    if saved['epoch'] != 100:
        raise ValueError('Require the completed fixed epoch 100')
    artifact_hashes = {name: sha256(run/name) for name in
                       ('P_optimized_world_mm.npy', 'P_optimized_pixel.npy', 'motion9.npy')}
    for name in ('P_optimized_world_mm.npy', 'motion9.npy'):
        if artifact_hashes[name] != saved['artifact_sha256'][name]:
            raise ValueError(f'Changed final artifact: {name}')
    if (pose['inputs']['estimated_pixel_pmat_sha256'] != artifact_hashes['P_optimized_pixel.npy']
            or pose['inputs']['target_sha256'] != acquisition['target_sha256']
            or pose['inputs']['landmarks_sha256'] != sha256(folder/'landmarks.json')):
        raise ValueError('Phantom pose does not correspond to current frozen inputs')
    flags = pose['invariants']
    if not (flags['rigid_transform_count'] == 1 and not flags['scale_fitted']
            and not flags['per_view_transforms'] and not flags['true_cameras_used_for_pose_estimation']):
        raise ValueError('Only one previously measured phantom rigid transform is allowed')
    dv, du = acquisition['truth']['pixel_vu_mm']
    h = np.asarray(pose['H_reconstruction_to_reference'])
    true_pixel = np.load(folder/'P_truth_pixel.npy')
    estimate = np.load(run/'P_optimized_world_mm.npy').astype(float)
    estimated_pixel = pmat_to_pixel(estimate, du=du, dv=dv)
    matrices = dict(truth=pixel_to_pmat(true_pixel, du=du, dv=dv, dtype=np.float64),
                    estimated=estimate,
                    phantom_aligned=pixel_to_pmat(apply_rigid_frame(estimated_pixel, h),
                                                 du=du, dv=dv, dtype=np.float64))
    cameras = {name: decompose_physical_camera(p) for name, p in matrices.items()}
    theta = np.load(folder/'truth_geometry.npz')['theta_deg']
    landmarks = np.asarray([p['xyz_mm'] for p in read_json(folder/'landmarks.json')['landmarks']])
    half = np.asarray(acquisition['volume']['box_extent_xyz_mm'])/2
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    check_points = np.concatenate((landmarks, corners))
    errors, validation, metrics, arrays = {}, {}, {}, {'theta_deg': theta, 'H': h}
    for name, camera in cameras.items():
        p = matrices[name]
        rebuilt = compose_physical_camera(camera['source_xyz_mm'], camera['Q_camera_to_physical'],
                                          camera['K_mm'], projective_scale=camera['projective_scale'])
        pixel = pmat_to_pixel(p, du=du, dv=dv)
        rebuilt_pixel = pmat_to_pixel(rebuilt, du=du, dv=dv)
        roundtrip = float(np.max(np.abs(project_landmarks(pixel, check_points)-
                                        project_landmarks(rebuilt_pixel, check_points))))
        if roundtrip > 1e-8:
            raise ValueError(f'Physical camera roundtrip changed projections: {name}')
        # Exact full K is preserved. Nine-variable shared-focal/zero-skew
        # representation differs slightly for float32 exported matrices.
        nine = np.column_stack((camera['source_xyz_mm'], np.zeros_like(camera['source_xyz_mm']),
                                camera['intrinsics_f_cu_cv_mm']))
        model_p = compose_physical_nine(nine, reference_Q=camera['Q_camera_to_physical'])
        model_pixel = pmat_to_pixel(model_p, du=du, dv=dv)
        model_error = float(np.max(np.abs(project_landmarks(pixel, check_points)-
                                          project_landmarks(model_pixel, check_points))))
        validation[name] = dict(exact_K_roundtrip_max_coordinate_error_px=roundtrip,
                                shared_focal_zero_skew_max_coordinate_error_px=model_error,
                                focal_anisotropy_max_mm=float(np.max(np.abs(camera['focal_anisotropy_mm']))),
                                skew_max_mm=float(np.max(np.abs(camera['skew_mm']))))
        residual = physical_camera_residuals(camera, cameras['truth'])
        errors[name] = residual['residuals9']
        bead_error = np.linalg.norm(project_landmarks(pixel, landmarks)-
                                    project_landmarks(true_pixel, landmarks), axis=-1)
        metrics[name] = dict(source_error_mm=stats(residual['source_error_mm']),
                             orientation_error_deg=stats(residual['orientation_error_deg']),
                             fixed_reference_bead_error_px=stats(bead_error),
                             component_rms={n: float(np.sqrt(np.mean(errors[name][:, j]**2)))
                                            for j, n in enumerate(NAMES)})
        for key in ('source_xyz_mm', 'Q_camera_to_physical', 'K_mm', 'intrinsics_f_cu_cv_mm'):
            arrays[f'{name}_{key}'] = camera[key]
        arrays[f'{name}_gt_errors9'] = errors[name]
    np.testing.assert_allclose(cameras['phantom_aligned']['Q_camera_to_physical'],
                               h[:3, :3] @ cameras['estimated']['Q_camera_to_physical'], atol=2e-14)
    values = display_values(cameras)
    arrays.update({f'{name}_display_values9': v for name, v in values.items()})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'physical_geometry.npz', **arrays)
    with (out/'physical_parameters9.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['view', 'theta_deg', 'frame']+[f'value_{n}' for n in NAMES]+
                        [f'error_{n}' for n in NAMES])
        for name in cameras:
            for view, angle in enumerate(theta):
                writer.writerow([view, angle, name, *values[name][view], *errors[name][view]])
    plot_nine(out, theta, values, errors)
    report = dict(
        scope='Source-centred re-expression of frozen seed1 P; no refit or fitting toward GT',
        order=NAMES, views=len(theta), landmarks=len(landmarks),
        conventions=dict(
            Q='Camera-to-physical basis columns: detector +u, detector -v, source-to-detector normal',
            exact_physical_formula='P_detector_mm = K_mm diag(1,-1,1) Q.T [I|-C]',
            world_conversion='Existing WORLD=(physical x,z,y); existing detector half-pixel convention retained',
            absolute_orientation='Euler xyz(Q), Rz Ry Rx, continuous display branch only',
            orientation_error='Rotation vector of Q_est Q_GT.T, in fixed physical xyz axes, degrees; not Euler differences',
            intrinsics='f=(K00+K11)/2; cu=K02, cv=K12, mm from original detector-edge coordinate origin',
            gt_use='Reference subtraction and comparison only; no GT camera fit or new alignment'),
        input_sha256={**artifact_hashes, 'P_truth_pixel.npy': sha256(folder/'P_truth_pixel.npy'),
                      'landmarks.json': sha256(folder/'landmarks.json'),
                      'experiment.json': sha256(folder/'experiment.json'), 'phantom_pose.json': sha256(pose_file)},
        validation=validation, metrics=metrics, H_reconstruction_to_reference=h.tolist(),
        limitations=['Physical reparameterization does not remove parameter coupling or improve the estimated P.',
                     'One independently measured phantom rigid frame is secondary; original-frame errors remain visible.',
                     'Full K is retained for exact reconstruction; nine-variable shared-focal/zero-skew approximation is measured separately.',
                     'Bead reprojection accuracy is evaluated within the known phantom, not the full scanner FOV.'],
        source_sha256={n: sha256(ROOT/n) for n in
                       ('report_sinespin_physical.py', 'physical_camera.py', 'calibration_gauge.py', 'calibration_geometry.py')},
        artifact_sha256={n: sha256(out/n) for n in
                         ('physical_parameters9_gt_errors.png', 'physical_parameters9_values.png',
                          'physical_parameters9.csv', 'physical_geometry.npz')})
    for name, digest in artifact_hashes.items():
        if sha256(run/name) != digest:
            raise ValueError(f'Frozen training artifact changed during reporting: {name}')
    (out/'physical_geometry_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(validation=validation, metrics={n: metrics[n]['source_error_mm']
                                                         for n in ('estimated', 'phantom_aligned')}), indent=2))
    print(f'Wrote {out}/physical_parameters9_*.png')


if __name__ == '__main__':
    main()
