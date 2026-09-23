"""Report final source trajectories and canonical effective 9DoF in physical units.

An optional, separately measured sparse-phantom pose changes one common frame.
It never fits a transform to GT cameras or removes per-view parameter errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from calibration_gauge import (PARAMETER_NAMES, apply_rigid_frame, compose_effective_parameters,
                               effective_parameters_from_pmat, transform_points)
from denseball_landmarks import project_landmarks
from trajectory_viewer import write_trajectory_html

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / 'result_sinespin/ball_calibration/loss_signed_lncc31_seed1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(rms=float(np.sqrt(np.mean(a*a))), mean=float(a.mean()),
                p95=float(np.quantile(a, .95)), maximum=float(a.max()))


def parameter_errors(estimate, truth):
    d = estimate-truth
    d[:, 6:] = (d[:, 6:]+180.) % 360.-180.
    return d


def metrics(camera, truth, pixel, true_pixel, points):
    error = parameter_errors(camera['parameters_9'], truth['parameters_9'])
    rotation = camera['R_physical'] @ truth['R_physical'].transpose(0, 2, 1)
    rotation_error = np.rad2deg(Rotation.from_matrix(rotation).magnitude())
    point_error = np.linalg.norm(project_landmarks(pixel, points)-project_landmarks(true_pixel, points), axis=-1)
    return dict(source_error_mm=stats(np.linalg.norm(camera['source_xyz_mm']-truth['source_xyz_mm'], axis=1)),
                camera_orientation_error_deg=stats(rotation_error),
                fixed_reference_bead_error_px=stats(point_error),
                parameter_error={name: stats(np.abs(error[:, j])) for j, name in enumerate(PARAMETER_NAMES)},
                shared_focal_mismatch_max_mm=float(np.abs(camera['shared_focal_mismatch_mm']).max()),
                skew_max_mm=float(np.abs(camera['skew_mm']).max()),
                effective_model_relative_residual_max=float(camera['effective_model_relative_residual'].max()))


def local_identifiability(nominal, parameters, points, du, dv):
    """Fixed known 3-D points; no jointly movable points and no loss Hessian claim."""
    units = np.array([10.]*6+[15.]*3)
    h = 1e-5
    derivatives = []
    for j in range(9):
        delta = np.zeros_like(parameters); delta[:, j] = h*units[j]
        plus = pmat_to_pixel(compose_effective_parameters(nominal, parameters+delta), du=du, dv=dv)
        minus = pmat_to_pixel(compose_effective_parameters(nominal, parameters-delta), du=du, dv=dv)
        derivatives.append(((project_landmarks(plus, points)-project_landmarks(minus, points))/(2*h)).reshape(len(parameters), -1))
    jacobian = np.stack(derivatives, axis=-1)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = (singular > singular[:, :1]*1e-8).sum(axis=1)
    normalized = jacobian / np.linalg.norm(jacobian, axis=1, keepdims=True)
    gram = normalized.transpose(0, 2, 1) @ normalized
    return singular, gram, dict(
        definition='Jacobian of 35 fixed-ID known 3-D bead projections; each view separately; not the image-loss Hessian',
        parameter_increment_units=units.tolist(), central_step_in_normalized_units=h,
        rank_relative_tolerance=1e-8, minimum_rank=int(rank.min()), maximum_rank=int(rank.max()),
        minimum_singular_value=float(singular[:, -1].min()),
        condition_number_min_max=[float((singular[:, 0]/singular[:, -1]).min()), float((singular[:, 0]/singular[:, -1]).max())],
        caveat='Full local rank excludes a continuous exact local gauge here; ill-conditioning still permits strong near-cancellation. No global uniqueness or noise covariance claim.')


def make_figures(out, theta, cameras, singular, gram, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = dict(truth='black', estimated='#2469b2', phantom_aligned='#dc7b19')
    names = dict(truth='GT', estimated='Estimated, original frame', phantom_aligned='One phantom-pose frame change')
    fig = plt.figure(figsize=(14, 7), layout='constrained')
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.], wspace=.14)
    ax = fig.add_subplot(outer[0], projection='3d')
    for name, camera in cameras.items():
        xyz = camera['source_xyz_mm']
        ax.plot(*xyz.T, color=colors[name], label=names[name], lw=2 if name=='truth' else 1.2,
                ls='--' if name=='truth' else '-')
    all_sources = np.concatenate([c['source_xyz_mm'] for c in cameras.values()])
    lo, hi = all_sources.min(0), all_sources.max(0)
    extent = (hi-lo)*1.05
    center = (lo+hi)/2
    # Matplotlib normalizes a supplied ndarray in place: preserve mm extents.
    ax.set_box_aspect(extent.copy())
    ax.set_xlim(center[0]-extent[0]/2, center[0]+extent[0]/2)
    ax.set_ylim(center[1]-extent[1]/2, center[1]+extent[1]/2)
    ax.set_zlim(center[2]-extent[2]/2, center[2]+extent[2]/2)
    ax.set(xlabel='Physical x [mm]', ylabel='Physical y [mm]', zlabel='Physical z [mm]')
    ax.view_init(32, -65); ax.legend(loc='upper left', fontsize=8)
    ax.set_title('3-D source trajectory; equal physical axis scale')
    axs = outer[1].subgridspec(3, 1).subplots(sharex=True)
    for j, a in enumerate(axs):
        for name in ('estimated', 'phantom_aligned'):
            error = cameras[name]['source_xyz_mm']-cameras['truth']['source_xyz_mm']
            a.plot(theta, error[:, j], color=colors[name], lw=.9, label=names[name])
        a.axhline(0, color='black', lw=.7); a.grid(alpha=.2)
        a.set_ylabel(f'{"xyz"[j]} error [mm]')
    axs[0].legend(fontsize=8); axs[-1].set_xlabel('Scan angle [degree]')
    fig.suptitle(title+'\nSource computed from complete P; registration measured from reconstructed beads only')
    fig.savefig(out/'source_trajectory_3d.png', dpi=170); plt.close(fig)

    labels = [r'$\Delta u$ (intrinsic) [mm]', r'$\Delta f$ (shared focal) [mm]',
              r'$\Delta v$ (intrinsic) [mm]', r'$t_x$ (effective) [mm]',
              r'$t_y$ (effective) [mm]', r'$t_z$ (effective) [mm]',
              r'$r_x$ (Euler xyz) [degree]', r'$r_y$ (Euler xyz) [degree]', r'$r_z$ (Euler xyz) [degree]']
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, layout='constrained')
    for j, ax in enumerate(axs.flat):
        for name in ('estimated', 'phantom_aligned', 'truth'):
            ax.plot(theta, cameras[name]['parameters_9'][:, j], color=colors[name],
                    lw=1.4 if name=='truth' else 1., ls='--' if name=='truth' else '-', label=names[name])
        ax.set_title(labels[j], fontsize=11); ax.grid(alpha=.25)
        if j < 6:
            ax.set_ylim(-10., 10.)
            ax.set_yticks([-10., -5., 0., 5., 10.])
        if j>=6: ax.set_xlabel('Scan angle [degree]')
    fig.legend([Line2D([], [], color=colors[n], ls='--' if n=='truth' else '-') for n in names],
               list(names.values()), loc='outside lower center', ncol=3)
    fig.suptitle('All nine parameters derived independently from P, using the same circular nominal frame\nIntrinsic and translation axes: +/-10 mm; rotation axes unchanged; intrinsic corrections are not Cartesian source shifts')
    fig.savefig(out/'canonical_parameters9.png', dpi=170); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axs[0].semilogy(theta, singular[:, 0]/singular[:, -1]); axs[0].grid(alpha=.25)
    axs[0].set(xlabel='Scan angle [degree]', ylabel='Scaled Jacobian condition number',
               title='All 9 directions identifiable locally, but coupled')
    view = min(409, len(theta)-1)
    im = axs[1].imshow(gram[view], vmin=-1, vmax=1, cmap='coolwarm')
    tick = ['du','df','dv','tx','ty','tz','rx','ry','rz']
    axs[1].set_xticks(range(9), tick); axs[1].set_yticks(range(9), tick)
    axs[1].set_title(f'View {view}: normalized derivative inner products')
    fig.colorbar(im, ax=axs[1], label='1 or -1: nearly parallel sensitivity')
    fig.suptitle('Fixed reference beads: near-cancellation is not an exact gauge to subtract')
    fig.savefig(out/'parameter_coupling.png', dpi=170); plt.close(fig)


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
    acquisition = read_json(folder/'experiment.json')
    saved = read_json(run/'metrics.json')
    if saved['epoch'] != 100: raise ValueError('Require the completed fixed epoch 100')
    for name in ('P_optimized_world_mm.npy', 'motion9.npy'):
        if sha256(run/name) != saved['artifact_sha256'][name]: raise ValueError(f'Changed final artifact: {name}')
    if sha256(folder/'landmarks.json') != saved['landmarks_sha256']: raise ValueError('Changed reference landmarks')
    landmarks = read_json(folder/'landmarks.json')
    points = np.array([p['xyz_mm'] for p in landmarks['landmarks']])
    pose = read_json(pose_file)
    if pose['inputs']['estimated_pixel_pmat_sha256'] != sha256(run/'P_optimized_pixel.npy'):
        raise ValueError('Phantom pose came from different fitted cameras')
    if pose['inputs']['landmarks_sha256'] != saved['landmarks_sha256'] or pose['inputs']['target_sha256'] != acquisition['target_sha256']:
        raise ValueError('Phantom pose came from different observations or reference landmarks')
    if not (pose['invariants']['rigid_transform_count']==1 and not pose['invariants']['scale_fitted']
            and not pose['invariants']['per_view_transforms'] and not pose['invariants']['true_cameras_used_for_pose_estimation']):
        raise ValueError('Only a single phantom-derived rigid transform is supported')
    h = np.array(pose['H_reconstruction_to_reference'], dtype=float)
    dv, du = acquisition['truth']['pixel_vu_mm']
    nominal = pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'), du=du, dv=dv, dtype=np.float64)
    truth_pixel = np.load(folder/'P_truth_pixel.npy')
    truth = pixel_to_pmat(truth_pixel, du=du, dv=dv, dtype=np.float64)
    estimate = np.load(run/'P_optimized_world_mm.npy').astype(float)
    pixel = pmat_to_pixel(estimate, du=du, dv=dv)
    aligned_pixel = apply_rigid_frame(pixel, h)
    aligned = pixel_to_pmat(aligned_pixel, du=du, dv=dv, dtype=np.float64)
    cameras = {name: effective_parameters_from_pmat(p, nominal) for name, p in
               (('truth', truth), ('estimated', estimate), ('phantom_aligned', aligned))}
    expected_source = transform_points(cameras['estimated']['source_xyz_mm'], h)
    np.testing.assert_allclose(cameras['phantom_aligned']['source_xyz_mm'], expected_source, rtol=1e-11, atol=1e-9)
    moved_points = transform_points(points, h)
    projection_invariance = float(np.abs(project_landmarks(aligned_pixel, moved_points)-project_landmarks(pixel, points)).max())
    if projection_invariance > 1e-8: raise ValueError('Rigid frame change did not preserve rays')
    singular, gram, identifiability = local_identifiability(nominal, cameras['estimated']['parameters_9'], points, du, dv)
    with np.load(folder/'truth_geometry.npz') as geom: theta = geom['theta_deg']
    report = dict(scope='Post-hoc canonical 9DoF and ONE independently measured phantom-frame comparison; no optimization of P',
                  input_sha256={name: sha256(folder/name) for name in ('experiment.json', 'landmarks.json', 'P_nominal_pixel.npy', 'P_truth_pixel.npy')},
                  final_p_sha256=sha256(run/'P_optimized_world_mm.npy'), phantom_pose_file=str(pose_file),
                  phantom_pose_sha256=sha256(pose_file), H_reconstruction_to_reference=h.tolist(),
                  parameter_names=list(PARAMETER_NAMES), convention='ts=(du,df,dv); tp=physical/internal xyz; Euler xyz means Rz Ry Rx; same circular nominal P; positive-focal RQ',
                  estimated=metrics(cameras['estimated'], cameras['truth'], pixel, truth_pixel, points),
                  phantom_aligned=metrics(cameras['phantom_aligned'], cameras['truth'], aligned_pixel, truth_pixel, points),
                  canonical_vs_saved_motion9_max_abs=float(np.abs(cameras['estimated']['parameters_9']-np.load(run/'motion9.npy')).max()),
                  canonical_vs_saved_motion9_max_abs_by_parameter={name:float(value) for name,value in zip(PARAMETER_NAMES, np.abs(cameras['estimated']['parameters_9']-np.load(run/'motion9.npy')).max(axis=0))},
                  paired_frame_projection_invariance_max_px=projection_invariance,
                  intrinsic_change_from_rigid_frame_max_mm=float(np.abs(cameras['estimated']['parameters_9'][:, :3]-cameras['phantom_aligned']['parameters_9'][:, :3]).max()),
                  identifiability=identifiability,
                  limitations=['The known fixed attenuation volume already fixes the metric/global pose frame.',
                               'Sparse reconstructed bead pose is a secondary frame convention, not a fitted correction to true cameras.',
                               'Weakly observable parameter tradeoffs cannot be deleted without changing the camera rays.',
                               'Gauge alignment is one rigid transform only; no scale, per-view alignment or GT-source Procrustes fit.'])
    report['source_sha256'] = {n:sha256(ROOT/n) for n in ('report_sinespin_pose.py','calibration_gauge.py','calibration_geometry.py','trajectory_viewer.py')}
    out.mkdir(parents=True, exist_ok=True)
    (out/'canonical_geometry_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez(out/'canonical_geometry.npz', theta_deg=theta, H_reconstruction_to_reference=h,
             singular_values=singular, derivative_inner_products=gram,
             **{f'{name}_parameters9':c['parameters_9'] for name,c in cameras.items()},
             **{f'{name}_source_xyz_mm':c['source_xyz_mm'] for name,c in cameras.items()})
    config=saved['recipe']['loss_config']
    title=f"{config['name']} {config['kernel_size']}, seed {saved['recipe']['seed']}, fixed epoch {saved['epoch']}"
    make_figures(out, theta, cameras, singular, gram, title)
    write_trajectory_html(out/'source_trajectory_3d.html', theta_deg=theta,
                          truth_xyz=cameras['truth']['source_xyz_mm'], estimated_xyz=cameras['estimated']['source_xyz_mm'],
                          aligned_xyz=cameras['phantom_aligned']['source_xyz_mm'],
                          title=title+': 3-D source trajectory',
                          registration_note='One rigid frame change measured from sparse reconstructed phantom beads. Fixed-volume calibration already fixes global gauge; retain original-frame errors.')
    print(json.dumps({name:report[name]['source_error_mm'] for name in ('estimated','phantom_aligned')}, indent=2))
    print(f'Wrote {out}')


if __name__=='__main__': main()
