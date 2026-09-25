"""Compare K correction bounds at a fixed 200 epochs from nominal initialization."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from calibration_gauge import PARAMETER_NAMES, effective_parameters_from_pmat
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from compare_sinespin_regularization import validate_data, load_run, evaluate
from denseball_landmarks import project_landmarks
from physical_camera import decompose_physical_camera
from run_sinespin_calibration import apply_motion, sha256
from spline_motion_model import BSplineMotion9, cubic_bspline_basis

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'result_spline9_scale2/ball_calibration'
OUT = BASE / 'kbound_comparison'
SPECS = [
    ('K bound +/-10 mm', BASE/'bspline20_joint200_seed1', 10., '#777777'),
    ('K bound +/-3 mm', BASE/'bspline20_joint200_kbound3_seed1', 3., '#126bbb'),
]


def audit_spline(run, meta, points):
    """Validate actual initialized/final coefficients and reconstructed rays."""
    path = run['path']
    bounds = run['recipe']['bounds']
    initial = torch.load(path/'initial_model.pt', map_location='cpu', weights_only=True)
    assert torch.count_nonzero(initial['raw_coefficients']).item() == 0
    assert not np.count_nonzero(np.load(path/'initial_motion9.npy'))
    model = BSplineMotion9(meta['truth']['views'], 20, **bounds)
    np.testing.assert_array_equal(initial['scales'], model.scales)
    checkpoint = torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    archive = np.load(path/'spline_coefficients.npz')
    for name, value in [('raw_coefficients', model.raw_coefficients.detach().numpy()),
                        ('physical_coefficients', model.physical_coefficients().detach().numpy()),
                        ('basis', model.basis.numpy()), ('scales', model.scales.numpy())]:
        if name == 'physical_coefficients':
            # CPU and CUDA tanh differ by float32 roundoff; stored raw values
            # and basis must still be identical.
            np.testing.assert_allclose(archive[name], value, atol=2e-6, rtol=1e-6)
        else:
            np.testing.assert_array_equal(archive[name], value)
    scales = model.scales.numpy()
    assert np.all(abs(archive['physical_coefficients']) <= scales)
    assert np.all(abs(run['motion']) <= scales + 1e-6)
    np.testing.assert_allclose(archive['basis'] @ archive['physical_coefficients'], run['motion'],
                               atol=2e-6, rtol=1e-6)
    with torch.no_grad():
        p, motion = apply_motion(torch.from_numpy(np.load(BASE/'input/P_nominal_world_mm.npy')),
            model(torch.arange(meta['truth']['views'])), bounds, physical=True,
            shape=tuple(meta['volume']['shape_zyx']), voxel=meta['volume']['voxel_mm'])
    np.testing.assert_allclose(motion.numpy(), run['motion'], atol=2e-6, rtol=1e-6)
    dv, du = meta['truth']['pixel_vu_mm']
    ray_delta = abs(project_landmarks(pmat_to_pixel(p.numpy(), du=du, dv=dv), points) -
                    project_landmarks(np.load(path/'P_optimized_pixel.npy'), points)).max()
    assert ray_delta < 1e-3, ray_delta
    return dict(initial_coefficients_zero=True, coefficient_and_view_bounds_verified=True,
        checkpoint_cpu_ray_max_abs_difference_px=float(ray_delta),
        cpu_physical_coefficient_max_abs_difference=float(abs(archive['physical_coefficients']-model.physical_coefficients().detach().numpy()).max()),
        coefficient_max_abs9=np.max(abs(archive['physical_coefficients']), axis=0).tolist(),
        motion_max_abs9=np.max(abs(run['motion']), axis=0).tolist(),
        coefficient_fraction_above_95pct_bound9=np.mean(abs(archive['physical_coefficients']) >= .95*scales, axis=0).tolist(),
        view_fraction_above_95pct_bound9=np.mean(abs(run['motion']) >= .95*scales, axis=0).tolist(),
        coefficient_archive_sha256=sha256(path/'spline_coefficients.npz'))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    folder = BASE/'input'
    meta = json.loads((folder/'experiment.json').read_text())
    hashes = validate_data(folder, meta)
    dv, du = meta['truth']['pixel_vu_mm']
    nominal = pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'), du=du, dv=dv, dtype=np.float64)
    truth_pixel = np.load(folder/'P_truth_pixel.npy')
    truth_p = pixel_to_pmat(truth_pixel, du=du, dv=dv, dtype=np.float64)
    truth = effective_parameters_from_pmat(truth_p, nominal)
    gt = np.load(folder/'spline_motion9.npy')
    np.testing.assert_allclose(truth['parameters_9'], gt, atol=1e-9, rtol=0)
    points = np.array([p['xyz_mm'] for p in json.loads((folder/'landmarks.json').read_text())['landmarks']])
    uv = project_landmarks(truth_pixel, points)
    rows, cols = meta['truth']['detector_shape_vu']
    visible = (uv[:,:,0]>=-.5)&(uv[:,:,0]<cols-.5)&(uv[:,:,1]>=-.5)&(uv[:,:,1]<rows-.5)
    assert visible.all()
    theta = np.load(folder/'truth_geometry.npz')['theta_deg']
    runs, summaries, arrays, checks, table, recipes = [], [], [], [], [], []
    for label, path, bound, color in SPECS:
        run = load_run(path, meta, hashes, required_epochs=200,
                       required_bounds=dict(ts_max_mm=bound, tp_max_mm=10., rot_max_deg=15.))
        assert not run['recipe']['regularization']['active']
        summary, array = evaluate(run, truth, nominal, truth_pixel, points, visible, du, dv)
        check = audit_spline(run, meta, points)
        runs.append(run); summaries.append(summary); arrays.append(array); checks.append(check)
        recipe = copy.deepcopy(run['recipe'])
        recipe['bounds'].pop('ts_max_mm')
        recipe['regularization'].pop('intrinsic_scale_mm')
        recipes.append(recipe)
        table.append(dict(label=label, intrinsic_bound_mm=bound,
            intrinsic_rms_mm=summary['prior_groups']['intrinsic']['canonical_gt_error_component_rms'],
            translation_rms_mm=summary['prior_groups']['translation']['canonical_gt_error_component_rms'],
            rotation_rms_deg=summary['prior_groups']['rotation']['canonical_gt_error_component_rms'],
            source_rms_mm=summary['source_error_mm']['rms'],
            bead_rms_px=summary['bead_error_px']['rms'],
            worst_view_bead_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
            image_loss=summary['image_loss'],
            parameter_rms9=np.sqrt(np.mean(array['parameter_error9']**2, axis=0)).tolist()))
    assert recipes[0] == recipes[1], 'Recipe differs beyond K bound / inactive prior scale'
    assert runs[0]['source_sha256'] == runs[1]['source_sha256'], 'Training sources differ'
    assert runs[0]['experiment']['train_views'] == runs[1]['experiment']['train_views']
    for name in ('initial_motion9.npy', 'P_initial_world_mm.npy'):
        np.testing.assert_array_equal(np.load(runs[0]['path']/name), np.load(runs[1]['path']/name))
    states = [torch.load(r['path']/'initial_model.pt', map_location='cpu', weights_only=True) for r in runs]
    for name in ('raw_coefficients', 'basis', 'knots'):
        np.testing.assert_array_equal(states[0][name], states[1][name])
    # Representation audit is post-hoc; coefficients never initialize the optimizer.
    basis, _ = cubic_bspline_basis(len(gt), 20)
    fit = np.linalg.lstsq(basis, gt, rcond=None)[0]
    representation = dict(evaluation_only=True, GT_max_abs9=abs(gt).max(0).tolist(),
        unconstrained_GT_fit_coefficient_max_abs9=abs(fit).max(0).tolist(),
        GT_fit_within_3mm_K_coefficient_bounds=bool(np.all(abs(fit[:,:3]) < 3.)),
        GT_fit_parameter_rms9=np.sqrt(np.mean((basis@fit-gt)**2, axis=0)).tolist())
    pad_v, pad_u = meta['detector_padding_vu']
    shift = np.array([[1.,0.,-pad_u*du],[0.,1.,-pad_v*dv],[0.,0.,1.]])
    def physical(p):
        c = decompose_physical_camera(shift@p)
        angle = Rotation.from_matrix(c['Q_camera_to_physical']).as_euler('xyz', degrees=True)
        return np.c_[c['intrinsics_f_cu_cv_mm'], c['source_xyz_mm'],
                     np.rad2deg(np.unwrap(np.deg2rad(angle), axis=0))]
    absolute_gt, absolute_nominal = physical(truth_p), physical(nominal)
    physical_runs = [physical(r['p']) for r in runs]
    k_limits = [(0., 2*float(np.median(absolute_nominal[:,0]))),
                (0., (cols-2*pad_u)*du), (0., (rows-2*pad_v)*dv)]
    for kind, truth_values, nominal_values, values, names in (
        ('geometry_components9', absolute_gt, absolute_nominal, physical_runs,
         ['f [mm]','cu [mm]','cv [mm]','Source x [mm]','Source y [mm]','Source z [mm]',
          'Camera x [degree]','Camera y [degree]','Camera z [degree]']),
        ('canonical_parameters9', gt, np.zeros_like(gt), [a['parameters9'] for a in arrays], PARAMETER_NAMES)):
        fig, axes = plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
        for j, ax in enumerate(axes.flat):
            ax.plot(theta, nominal_values[:,j], ':', color='#b38b45', label='Nominal', lw=.8)
            for spec, value in zip(SPECS, values):
                ax.plot(theta, value[:,j], color=spec[3], label=spec[0], lw=1.2)
            ax.plot(theta, truth_values[:,j], '--', color='black', label='GT', lw=1.2)
            ax.set_title(names[j]); ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if kind == 'geometry_components9' and j < 3: ax.set_ylim(k_limits[j])
            if kind == 'canonical_parameters9' and j < 6: ax.set_ylim(-10,10)
            if j >= 6: ax.set_xlabel('Scan angle [degree]')
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside lower center', ncol=4)
        fig.suptitle('K correction bound: +/-10 vs +/-3 mm; same zero initialization and data\n'
                     'Joint B20, 200 epochs, signed LNCC31; translation +/-10 mm / rotation +/-15 deg')
        fig.savefig(OUT/(kind+'.png'),dpi=160); plt.close(fig)
    convergence = {}
    fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for spec, run in zip(SPECS, runs):
        records = []
        for saved in sorted(run['path'].glob('P_epoch*.npy')):
            epoch = int(saved.stem.removeprefix('P_epoch'))
            if epoch > 200: continue
            p = np.load(saved)
            c = effective_parameters_from_pmat(p, nominal)
            error = c['parameters_9'] - gt
            err_uv = project_landmarks(pmat_to_pixel(p,du=du,dv=dv),points)-uv
            records.append(dict(epoch=epoch,
                intrinsic_rms_mm=float(np.sqrt(np.mean(error[:,:3]**2))),
                source_rms_mm=float(np.sqrt(np.mean(np.sum((c['source_xyz_mm']-truth['source_xyz_mm'])**2,axis=1)))),
                bead_rms_px=float(np.sqrt(np.mean(np.sum(err_uv**2,axis=-1))))))
        convergence[spec[0]] = records
        for ax, key, title in zip(axes.flat, ('intrinsic_rms_mm','source_rms_mm','bead_rms_px'),
                                 ('K component RMS [mm]','Source RMS [mm]','Bead reprojection RMS [pixel]')):
            ax.plot([r['epoch'] for r in records],[r[key] for r in records],color=spec[3],label=spec[0])
            ax.set(xlabel='Epoch',ylabel=title)
        history = np.genfromtxt(run['path']/'loss_history.csv',delimiter=',',names=True)
        axes[1,1].plot(history['epoch'],history['image_loss'],color=spec[3],label=spec[0])
    axes[1,1].set(xlabel='Epoch',ylabel='Training signed LNCC31')
    axes[0,0].legend()
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('Fixed final epoch 200; checkpoint GT metrics are post-hoc evaluation only')
    fig.savefig(OUT/'convergence.png',dpi=160); plt.close(fig)
    record = dict(table=table, runs=summaries, checks=checks, convergence=convergence,
        representation_diagnostic=representation, input_sha256=hashes,
        identical_initial_motion_and_P=True, identical_recipe_except_K_bound=True,
        training_sources_identical=True, k_axis_limits_mm=k_limits,
        limitations=[
            'Single seed, fixed 200 epochs, joint updates; no additional L2 penalty.',
            'Changing bound in bound*tanh(raw) also changes physical step scale and saturation; not a pure feasible-set comparison.',
            'The 3mm choice used knowledge of simulated +/-2mm GT amplitude as a range prior; no per-view GT or fitted coefficients enter training.',
            'Historical 10mm baseline migrated GPUs with validated shuffle replay; floating point results across devices are not claimed bitwise identical.',
            'Absolute-scale axes do not measure recovery of the much smaller spline motion amplitude.'],
        report_source_sha256=sha256(Path(__file__)),
        loader_source_sha256=sha256(ROOT/'compare_sinespin_regularization.py'))
    (OUT/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez(OUT/'comparison.npz',theta_deg=theta,truth=gt,physical_truth=absolute_gt,
             **{f'run{i}_{key}':v for i,a in enumerate(arrays) for key,v in a.items()})
    for kind in ('geometry_components9','canonical_parameters9','convergence'):
        shutil.copy2(OUT/(kind+'.png'),ROOT/'docs'/('spline_kbound_'+kind+'.png'))
    shutil.copy2(OUT/'comparison.json',ROOT/'docs/spline_kbound_comparison.json')
    print(json.dumps(dict(table=table,checks=checks,representation=representation),indent=2))


if __name__ == '__main__':
    report()
