"""CPU comparison of fixed-epoch signed-LNCC runs with/without an intrinsic prior.

GT enters only this report. All geometry uses the original fixed phantom frame;
there is no new pose fit, parameter cancellation, scale fit or checkpoint choice.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from calibration_gauge import PARAMETER_NAMES, effective_parameters_from_pmat
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from denseball_landmarks import project_landmarks

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'result_sinespin/ball_calibration'
GROUPS = (('intrinsic', slice(0, 3), 'mm'), ('translation', slice(3, 6), 'mm'),
          ('rotation', slice(6, 9), 'degree'))
COLORS = dict(truth='black', baseline='#2469b2', regularized='#8a45b7')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Statistics require nonempty finite values.')
    return dict(rms=float(np.sqrt(np.mean(values ** 2))),
                mean_absolute=float(np.abs(values).mean()),
                p95_absolute=float(np.quantile(np.abs(values), .95)),
                maximum_absolute=float(np.abs(values).max()))


def exact_array(a, b, label):
    """Compare dtype, shape and element bytes, including signed zeros."""
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise ValueError(f'{label} differ; comparison is not controlled.')


def validate_initialization(baseline, run):
    import torch
    states = [torch.load(path / 'initial_model.pt', map_location='cpu', weights_only=True)
              for path in (baseline, run)]
    if (not all(isinstance(state, dict) and state for state in states)
            or states[0].keys() != states[1].keys()):
        raise ValueError('Initial model state dictionaries differ or are invalid.')
    tensor_hashes = {}
    for name, value in states[0].items():
        other = states[1][name]
        if not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor):
            raise ValueError(f'Invalid initial model tensor: {name}')
        a = value.contiguous().reshape(-1).view(torch.uint8)
        b = other.contiguous().reshape(-1).view(torch.uint8)
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(a, b):
            raise ValueError(f'Initial model tensor differs: {name}')
        tensor_hashes[name] = dict(dtype=str(value.dtype), shape=list(value.shape),
                                  sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest())
    for name in ('initial_motion9.npy', 'P_initial_world_mm.npy'):
        exact_array(np.load(baseline / name), np.load(run / name), name)
    return dict(model_tensor_bytes_identical=True, tensor_count=len(tensor_hashes),
                initial_motion_identical=True, initial_p_identical=True,
                model_tensor_sha256=tensor_hashes)


def validate_image_loss_sources(baseline, run, config):
    """Audit the historical small-kernel validation fix without hiding the diff."""
    import torch
    import monai
    paths = [path / 'training_sources/calibration_losses.py' for path in (baseline, run)]
    sources = [path.read_text() for path in paths]
    trees = [ast.parse(source) for source in sources]
    for tree in trees:
        config_class = next((node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == 'LossConfig'), None)
        if config_class is None: raise ValueError('Archived loss module has no LossConfig class.')
        validator = next((node for node in config_class.body
                          if isinstance(node, ast.FunctionDef) and node.name == '__post_init__'), None)
        if validator is None: raise ValueError('Archived loss module has no config validator.')
        validator.body = [ast.Pass()]
    if ast.dump(trees[0], include_attributes=False) != ast.dump(trees[1], include_attributes=False):
        raise ValueError('Image-loss computation changed beyond the config validation method.')
    options = {key: config[key] for key in ('name', 'kernel_size', 'kernel_type', 'smooth_nr',
                                          'smooth_dr', 'huber_delta', 'i0', 'reduction')}
    modules, results = [], []
    try:
        for index, path in enumerate(paths):
            name = f'_sinespin_comparison_archived_loss_{index}'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # Execute the recorded source directly: avoid adding __pycache__ to
            # an immutable training-source archive during this CPU audit.
            exec(compile(sources[index], str(path), 'exec'), module.__dict__)
            modules.append(module)
        losses = [module.build_loss(**options) for module in modules]
        if losses[0].get_config() != config or losses[1].get_config() != config:
            raise ValueError('The selected loss config is not accepted identically by both archives.')
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator(device='cpu').manual_seed(7149)
            random_pred = torch.rand((2, 65, 71), dtype=dtype, generator=generator)
            random_target = torch.rand((2, 65, 71), dtype=dtype, generator=generator)
            y, x = torch.meshgrid(torch.linspace(-1, 1, 65, dtype=dtype),
                                  torch.linspace(-1, 1, 71, dtype=dtype), indexing='ij')
            blob = torch.exp(-12*(x*x+y*y))
            shifted = torch.exp(-12*((x-.06)**2+(y+.04)**2))
            cases = dict(random=(random_pred, random_target),
                         structured=(torch.stack((blob, .5*shifted)), torch.stack((shifted, .5*blob))),
                         constant=(torch.full_like(random_pred, .25), torch.full_like(random_target, .3)))
            for case, (prediction, target) in cases.items():
                evaluated = []
                for loss in losses:
                    pred = prediction.clone().requires_grad_()
                    value = loss(pred, target)
                    gradient, = torch.autograd.grad(value, pred)
                    if not torch.isfinite(value) or not torch.isfinite(gradient).all():
                        raise ValueError('Archived loss value/gradient is nonfinite during equivalence check.')
                    evaluated.append((value.detach(), gradient))
                for index in (0, 1):
                    exact_array(evaluated[0][index].numpy(), evaluated[1][index].numpy(),
                                f'Archived {case}/{dtype} loss or gradient')
                results.append(dict(case=case, dtype=str(dtype), shape=list(prediction.shape),
                                    value=float(evaluated[0][0]), value_and_gradient_bytes_identical=True))
    finally:
        for index in range(2): sys.modules.pop(f'_sinespin_comparison_archived_loss_{index}', None)
    return dict(full_module_hashes_identical=sha256(paths[0]) == sha256(paths[1]),
                computational_ast_identical_except_config_validator=True,
                selected_config_accepted_identically=True, selected_config=options,
                full_source_diff=''.join(difflib.unified_diff(sources[0].splitlines(keepends=True),
                    sources[1].splitlines(keepends=True), fromfile=str(paths[0]), tofile=str(paths[1]))),
                cpu_value_gradient_checks=results, torch_version=str(torch.__version__),
                monai_version=str(monai.__version__),
                scope='Entire module AST identical except LossConfig.__post_init__; exact diff retained; selected loss forward/backward checked on CPU only')


def validate_data(folder, acquisition):
    expected = {'target_projections.npy': 'target_sha256',
                'clean_projections.npy': 'clean_sha256',
                'photon_counts.npy': 'photon_counts_sha256',
                'P_nominal_world_mm.npy': 'nominal_pmat_sha256',
                'P_truth_world_mm.npy': 'truth_pmat_sha256',
                'landmarks.json': 'landmarks_sha256'}
    hashes = {}
    for name, key in expected.items():
        hashes[name] = sha256(folder / name)
        if hashes[name] != acquisition[key]:
            raise ValueError(f'Prepared data changed: {name}')
    for name in ('experiment.json', 'P_nominal_pixel.npy', 'P_truth_pixel.npy', 'truth_geometry.npz'):
        hashes[name] = sha256(folder / name)
    volume = Path(acquisition['volume']['path'])
    if sha256(volume) != acquisition['volume']['sha256']:
        raise ValueError('Reference attenuation volume changed.')
    hashes['reference_volume'] = acquisition['volume']['sha256']
    if acquisition['noise']['i0_photons_per_detector_pixel_per_view'] != 44000:
        raise ValueError('This comparison requires I0=44000 photons/pixel/view.')
    return hashes


def load_run(path, acquisition, input_hashes, *, required_kernel_size=31, required_epochs=100):
    import torch
    experiment, metrics = read_json(path / 'experiment.json'), read_json(path / 'metrics.json')
    recipe = experiment['recipe']
    if experiment['input'] != acquisition or metrics['recipe'] != recipe:
        raise ValueError(f'{path.name}: input or recipe provenance differs.')
    if recipe['epochs'] != required_epochs or metrics['epoch'] != required_epochs or recipe['seed'] != 1:
        raise ValueError(f'Require fixed epoch {required_epochs}, seed 1.')
    if recipe['ground_truth_geometry_used_in_optimizer'] or recipe['loss_levels'] != [1]:
        raise ValueError('Require no GT in training and single-resolution loss.')
    config = recipe['loss_config']
    if config['name'] != 'signed_lncc' or config['kernel_size'] != required_kernel_size:
        raise ValueError(f'Require signed LNCC{required_kernel_size} for this controlled study.')
    if recipe['bounds'] != dict(ts_max_mm=10., tp_max_mm=10., rot_max_deg=15.):
        raise ValueError('Require the selected 10/10/15 bounds.')
    artifact_hashes = {name: sha256(path / name) for name in (
        'experiment.json', 'metrics.json', 'initial_model.pt', 'initial_motion9.npy',
        'P_initial_world_mm.npy', 'checkpoint.pt', f'P_epoch{required_epochs:04d}.npy',
        'P_optimized_world_mm.npy', 'P_optimized_pixel.npy', 'motion9.npy', 'loss_history.csv')}
    for name, expected in metrics['artifact_sha256'].items():
        if name not in artifact_hashes: artifact_hashes[name] = sha256(path / name)
        if artifact_hashes[name] != expected:
            raise ValueError(f'{path.name}: final artifact changed: {name}')
    if metrics['landmarks_sha256'] != input_hashes['landmarks.json']:
        raise ValueError(f'{path.name}: landmark hash differs.')
    checkpoint = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if (checkpoint['epoch'] != required_epochs or checkpoint['history'][-1]['epoch'] != required_epochs
            or checkpoint['recipe'] != recipe
            or checkpoint['input_sha256'] != input_hashes['experiment.json']):
        raise ValueError(f'{path.name}: final checkpoint provenance differs.')
    del checkpoint
    final = np.load(path / 'P_optimized_world_mm.npy')
    exact_array(final, np.load(path / f'P_epoch{required_epochs:04d}.npy'), f'{path.name}: final versus epoch{required_epochs} P')
    sources = experiment['source_sha256']
    for name, expected in sources.items():
        if sha256(path / 'training_sources' / name) != expected:
            raise ValueError(f'{path.name}: archived training source changed: {name}')
    return dict(path=path, experiment=experiment, metrics=metrics, recipe=recipe,
                source_sha256=sources, artifact_sha256=artifact_hashes, p=final,
                motion=np.load(path / 'motion9.npy'))


def regularization_config(recipe):
    config = recipe.get('regularization', {})
    return dict(intrinsic_weight=float(config.get('intrinsic_weight', 0.)),
                translation_weight=float(config.get('translation_weight', 0.)),
                rotation_weight=float(config.get('rotation_weight', 0.)),
                intrinsic_scale_mm=float(config.get('intrinsic_scale_mm', recipe['bounds']['ts_max_mm'])),
                translation_scale_mm=float(config.get('translation_scale_mm', recipe['bounds']['tp_max_mm'])),
                rotation_scale_deg=float(config.get('rotation_scale_deg', recipe['bounds']['rot_max_deg'])))


def evaluate(run, truth, nominal, truth_pixel, points, visible, du, dv):
    camera = effective_parameters_from_pmat(run['p'].astype(np.float64), nominal)
    pixel = pmat_to_pixel(run['p'], du=du, dv=dv)
    np.testing.assert_allclose(pixel, np.load(run['path'] / 'P_optimized_pixel.npy'), rtol=1e-12, atol=1e-9)
    bead_error = np.linalg.norm(project_landmarks(pixel, points)-project_landmarks(truth_pixel, points), axis=-1)
    per_view = np.sqrt(np.sum(bead_error ** 2 * visible, axis=1) / visible.sum(axis=1))
    source_error = np.linalg.norm(camera['source_xyz_mm']-truth['source_xyz_mm'], axis=1)
    error = camera['parameters_9']-truth['parameters_9']
    error[:, 6:] = (error[:, 6:]+180.) % 360.-180.
    record = run['metrics']['results']['optimized']
    schema = run['metrics']['metrics_schema']
    if schema not in (2, 3):
        raise ValueError('Only metrics schemas 2 and 3 are supported.')
    image_loss = float(record['image_loss'] if schema == 3 else record['objective_loss'])
    config, group_metrics, expected_penalty = regularization_config(run['recipe']), {}, 0.
    for name, indices, unit in GROUPS:
        scale_key = name + ('_scale_deg' if name == 'rotation' else '_scale_mm')
        mean_square = float(np.mean((run['motion'][:, indices].astype(float)/config[scale_key]) ** 2))
        weighted = config[name+'_weight']*mean_square
        group_metrics[name] = dict(unit=unit, applied_parameter_component_rms=stats(run['motion'][:, indices])['rms'],
            canonical_gt_error_component_rms=stats(error[:, indices])['rms'],
            normalized_mean_square=mean_square, configured_weight=config[name+'_weight'], weighted_penalty=weighted)
        expected_penalty += weighted
    reported_penalty = float(record.get('regularization_loss', 0.))
    if not np.isclose(reported_penalty, expected_penalty, rtol=2e-6, atol=1e-9):
        raise ValueError('Saved regularization penalty disagrees with applied motion9.')
    if not np.isclose(record['objective_loss'], image_loss+reported_penalty, rtol=1e-7, atol=1e-9):
        raise ValueError('Saved objective is not image loss plus penalty.')
    for actual, saved, label in (
        (stats(bead_error[visible])['rms'], record['geometry']['ball_reprojection_error_px']['rms'], 'bead RMS'),
        (stats(source_error)['rms'], record['geometry']['source_position_error_mm']['rms'], 'source RMS')):
        if not np.isclose(actual, saved, rtol=1e-5, atol=2e-5):
            raise ValueError(f'Independent {label} differs from saved metrics.')
    summary = dict(path=str(run['path']), metrics_schema=schema, image_loss=image_loss,
        objective_loss=float(record['objective_loss']), regularization_loss=reported_penalty,
        common_image_metrics=record['common_image_metrics'], projection_metrics=record['projection']['all'],
        bead_error_px=stats(bead_error[visible]), per_view_bead_rms_px=stats(per_view),
        worst_bead_view=int(np.argmax(per_view)), views_bead_rms_below_1px=int(np.sum(per_view < 1.)),
        source_error_mm=stats(source_error), worst_source_view=int(np.argmax(source_error)),
        canonical_parameter_error={name: stats(error[:, j]) for j, name in enumerate(PARAMETER_NAMES)},
        prior_groups=group_metrics, regularization_config=config,
        shared_focal_mismatch_max_mm=float(np.abs(camera['shared_focal_mismatch_mm']).max()),
        effective_model_relative_residual_max=float(camera['effective_model_relative_residual'].max()),
        canonical_vs_applied_max_abs_by_parameter=np.abs(camera['parameters_9']-run['motion']).max(axis=0).tolist(),
        training_source_sha256=run['source_sha256'], artifact_sha256=run['artifact_sha256'])
    arrays = dict(parameters9=camera['parameters_9'], source_xyz_mm=camera['source_xyz_mm'],
                  parameter_error9=error, source_error_mm=source_error,
                  per_view_bead_rms_px=per_view, bead_error_px=bead_error)
    return summary, arrays


def make_figures(out, theta, arrays, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [r'$\Delta u$ [mm]', r'$\Delta f$ [mm]', r'$\Delta v$ [mm]',
              r'$t_x$ [mm]', r'$t_y$ [mm]', r'$t_z$ [mm]',
              r'$r_x$ [degree]', r'$r_y$ [degree]', r'$r_z$ [degree]']
    names = dict(truth='GT', baseline='Baseline: no prior', regularized='Intrinsic prior: lambda=0.01')
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, layout='constrained')
    for j, ax in enumerate(axes.flat):
        for key in ('baseline', 'regularized', 'truth'):
            ax.plot(theta, arrays[key]['parameters9'][:, j], color=COLORS[key],
                    ls='--' if key == 'truth' else '-', lw=1.3 if key == 'truth' else 1., label=names[key])
        ax.set_title(labels[j]); ax.grid(alpha=.2)
        if j >= 6: ax.set_xlabel('Scan angle [degree]')
    handles, text = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, text, loc='outside lower center', ncol=3)
    fig.suptitle('Canonical effective 9DoF from complete P; same fixed phantom / circular nominal frame\nFixed epoch 100, seed 1; signed LNCC31; no pose fit or cancellation subtraction')
    fig.savefig(out / 'canonical_parameters9_comparison.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for key in ('baseline', 'regularized'):
        axes[0, 0].plot(theta, arrays[key]['source_error_mm'], color=COLORS[key], label=names[key], lw=1)
        axes[0, 1].plot(theta, arrays[key]['per_view_bead_rms_px'], color=COLORS[key], label=names[key], lw=1)
    axes[0, 0].set(xlabel='Scan angle [degree]', ylabel='Source distance to GT [mm]', title='Physical source error')
    axes[0, 1].set(xlabel='Scan angle [degree]', ylabel='Fixed-ID bead RMS [pixel]', title='Projection geometry error')
    axes[0, 1].axhline(1., color='gray', ls=':', lw=.8, label='1 pixel')
    x = np.arange(3)
    for offset, key in ((-.18, 'baseline'), (.18, 'regularized')):
        prior = runs[key]['prior_groups']
        axes[1, 0].bar(x+offset, [prior[n]['applied_parameter_component_rms'] for n, _, _ in GROUPS],
                       width=.36, color=COLORS[key], label=names[key])
        axes[1, 1].bar(x+offset, [prior[n]['canonical_gt_error_component_rms'] for n, _, _ in GROUPS],
                       width=.36, color=COLORS[key], label=names[key])
    for ax in axes[1]:
        ax.set_xticks(x, ['Intrinsic [mm]', 'Translation [mm]', 'Rotation [degree]'])
        ax.set_ylabel('RMS over views and 3 coordinates')
    axes[1, 0].set_title('Applied correction size (prior target is zero)')
    axes[1, 1].set_title('Canonical GT error (evaluation only)')
    for ax in axes.flat: ax.grid(alpha=.2, axis='y'); ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=9); axes[0, 1].legend(fontsize=9)
    fig.suptitle('Does a smaller intrinsic correction improve complete geometry?\nGroup units differ: compare runs within each group, not bar heights across groups')
    fig.savefig(out / 'geometry_and_group_errors.png', dpi=170); plt.close(fig)


def compare(baseline, run, folder, out):
    baseline, run, folder, out = map(Path, (baseline, run, folder, out))
    acquisition = read_json(folder / 'experiment.json')
    input_hashes = validate_data(folder, acquisition)
    loaded = {name: load_run(path, acquisition, input_hashes)
              for name, path in (('baseline', baseline), ('regularized', run))}
    recipes = [{k: v for k, v in value['recipe'].items() if k != 'regularization'} for value in loaded.values()]
    if recipes[0] != recipes[1]:
        raise ValueError('Training recipes may differ only in regularization.')
    if loaded['baseline']['experiment']['train_views'] != loaded['regularized']['experiment']['train_views']:
        raise ValueError('Training view selections differ.')
    expected = dict(intrinsic_weight=.01, translation_weight=0., rotation_weight=0.,
                    intrinsic_scale_mm=10., translation_scale_mm=10., rotation_scale_deg=15.)
    config = regularization_config(loaded['regularized']['recipe'])
    if config != expected or any(regularization_config(loaded['baseline']['recipe'])[n+'_weight'] for n, _, _ in GROUPS):
        raise ValueError('Require no prior baseline versus intrinsic-only lambda=.01 and scales 10/10/15.')
    unchanged = ('calibration_geometry.py', 'fast_projectors.py', 'DoF_transform.py',
                 'models/MotionNetHash.py', 'models/hash_encoder.py')
    for name in unchanged:
        if loaded['baseline']['source_sha256'][name] != loaded['regularized']['source_sha256'][name]:
            raise ValueError(f'Core geometry/projector/model source differs: {name}')
    initialization = validate_initialization(baseline, run)
    image_loss_audit = validate_image_loss_sources(baseline, run, recipes[0]['loss_config'])
    labels = read_json(folder / 'landmarks.json')
    points = np.asarray([item['xyz_mm'] for item in labels['landmarks']], dtype=float)
    if (labels['volume_sha256'] != acquisition['volume']['sha256']
            or labels['voxel_size_mm'] != acquisition['volume']['voxel_mm']
            or labels['shape_zyx'] != acquisition['volume']['shape_zyx']):
        raise ValueError('Landmark physical reference differs from acquisition.')
    dv, du = acquisition['truth']['pixel_vu_mm']
    truth_pixel, nominal_pixel = (np.load(folder / name) for name in ('P_truth_pixel.npy', 'P_nominal_pixel.npy'))
    nominal = pixel_to_pmat(nominal_pixel, du=du, dv=dv, dtype=np.float64)
    truth_world = pixel_to_pmat(truth_pixel, du=du, dv=dv, dtype=np.float64)
    for name, exact in (('P_nominal_world_mm.npy', nominal), ('P_truth_world_mm.npy', truth_world)):
        np.testing.assert_allclose(np.load(folder / name), exact, rtol=2e-7, atol=2e-5)
    truth = effective_parameters_from_pmat(truth_world, nominal)
    with np.load(folder / 'truth_geometry.npz') as geometry:
        theta = geometry['theta_deg']
        np.testing.assert_allclose(truth['source_xyz_mm'], geometry['source_positions'], atol=1e-8, rtol=1e-11)
    true_uv = project_landmarks(truth_pixel, points)
    depth = np.einsum('vj,nj->vn', truth_pixel[:, 2], np.c_[points, np.ones(len(points))])
    rows, cols = acquisition['truth']['detector_shape_vu']
    visible = ((true_uv[..., 0] >= -.5) & (true_uv[..., 0] <= cols-.5)
               & (true_uv[..., 1] >= -.5) & (true_uv[..., 1] <= rows-.5) & (depth > 0))
    if np.any(visible.sum(axis=1) == 0): raise ValueError('A view has no visible reference beads.')
    runs, arrays = {}, {'truth': dict(parameters9=truth['parameters_9'], source_xyz_mm=truth['source_xyz_mm'])}
    for name, value in loaded.items():
        runs[name], arrays[name] = evaluate(value, truth, nominal, truth_pixel, points, visible, du, dv)
    changes = {}
    for label, getter in (
        ('image_loss', lambda d: d['image_loss']), ('bead_rms_px', lambda d: d['bead_error_px']['rms']),
        ('worst_view_bead_rms_px', lambda d: d['per_view_bead_rms_px']['maximum_absolute']),
        ('source_rms_mm', lambda d: d['source_error_mm']['rms']),
        ('source_maximum_mm', lambda d: d['source_error_mm']['maximum_absolute'])):
        a, b = (getter(runs[name]) for name in ('baseline', 'regularized'))
        changes[label] = dict(baseline=a, regularized=b, change=b-a,
                              relative_change_percent=100*(b/a-1) if a else None)
    observations = []
    for name, _, _ in GROUPS:
        a, b = [runs[key]['prior_groups'][name] for key in ('baseline', 'regularized')]
        changes[name] = {field: dict(baseline=a[field], regularized=b[field], change=b[field]-a[field])
                        for field in ('applied_parameter_component_rms', 'canonical_gt_error_component_rms')}
        for field, description in (('applied_parameter_component_rms', 'applied correction RMS'),
                                   ('canonical_gt_error_component_rms', 'canonical GT error RMS')):
            observations.append(f"{name} {description}: {a[field]:.6g} -> {b[field]:.6g} ({'increased' if b[field]>a[field] else 'decreased'}).")
    worsened = [name for name in ('bead_rms_px', 'source_rms_mm') if changes[name]['change'] > 0]
    observations.append('Complete geometry worsened in: '+', '.join(worsened)+'.' if worsened
                        else 'Both bead and source RMS decreased in this fixed seed/noise experiment.')
    source_names = set(loaded['baseline']['source_sha256']) | set(loaded['regularized']['source_sha256'])
    report = dict(scope='Fixed epoch100/seed1 comparison: intrinsic-only minimum-norm prior; original fixed phantom frame',
        comparison_variable='lambda_intrinsic=0 versus 0.01; translation/rotation weights remain zero',
        identical_recipe_except_regularization=recipes[0], initialization_checks=initialization,
        image_loss_source_audit=image_loss_audit,
        inputs_sha256=input_hashes, input_noise=acquisition['noise'], runs=runs, changes=changes,
        observations=observations, parameter_names=list(PARAMETER_NAMES),
        source_changes={name: {key: loaded[key]['source_sha256'].get(name) for key in loaded}
                        for name in sorted(source_names) if loaded['baseline']['source_sha256'].get(name)
                        != loaded['regularized']['source_sha256'].get(name)},
        report_source_sha256={name: sha256(ROOT / name) for name in
            ('compare_sinespin_regularization.py', 'calibration_gauge.py', 'calibration_geometry.py', 'denseball_landmarks.py')},
        limitations=['GT is used only for this post-hoc evaluation; no checkpoint selection or new pose fit.',
            'Intrinsic corrections, effective translation and Euler rotation are canonically recovered from complete P.',
            'The prior changes the objective and may bias geometry or redistribute error across parameters; it does not remove a gauge.',
            'Group RMS averages coordinates; translation/intrinsics use mm and rotation uses degrees.',
            'One fixed initialization and noise realization do not establish general performance.',
            'Image loss is comparable across runs; total objective includes a penalty only for the regularized run.'])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'regularization_comparison.npz', theta_deg=theta, visibility=visible,
             **{f'{name}_{key}': value for name, data in arrays.items() for key, value in data.items()})
    make_figures(out, theta, arrays, runs)
    (out / 'regularization_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', '--baseline-dir', dest='baseline', type=Path, default=RESULTS / 'loss_signed_lncc31_seed1')
    parser.add_argument('--run', '--run-dir', dest='run', type=Path, default=RESULTS / 'reg_intrinsic001_seed1')
    parser.add_argument('--input', '--input-dir', dest='input', type=Path, default=RESULTS / 'input')
    parser.add_argument('--out', '--out-dir', dest='out', type=Path, default=RESULTS / 'regularization_study')
    args = parser.parse_args()
    report = compare(args.baseline, args.run, args.input, args.out)
    print(json.dumps(report['changes'], indent=2))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
