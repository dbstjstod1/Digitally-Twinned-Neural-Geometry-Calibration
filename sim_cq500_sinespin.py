"""CQ500 head validation of nominal icono scan protocols using Joseph LS.

Three arms separate the paper's protocol comparison from a matched-view tilt
comparison. The detector visibility mask affects display and scoring only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from sim_sinespin_recon import configure_leap, geometry_record, save_json
from sinespin_geometry import build_icono_orbit


ARMS = ('circular_200', 'circular_220', 'sinespin_220')
LABELS = {'circular_200': 'Circular 200° / 496 views',
          'circular_220': 'Circular 220° / 546 views',
          'sinespin_220': 'Sine Spin 220° / 546 views'}


def build_arms(detector_bin=2):
    return {'circular_200': build_icono_orbit('circular', detector_bin=detector_bin),
            'circular_220': build_icono_orbit('circular', detector_bin=detector_bin,
                                            n_views=546, scan_angle_deg=220),
            'sinespin_220': build_icono_orbit('sinespin', detector_bin=detector_bin)}


def visibility_volumes(geometries, shape, voxel):
    z, y, x = [(np.arange(n)-(n-1)/2)*voxel for n in shape]
    X, Y = np.meshgrid(x, y, indexing='xy')
    masks = {}
    for name, geometry in geometries.items():
        limits = geometry.longitudinal_intervals(np.stack((X, Y), axis=-1))
        masks[name] = (z[:, None, None] >= limits[None, :, :, 0]) & (z[:, None, None] <= limits[None, :, :, 1])
    return masks, (z, y, x)


def measure_hu(f, truth, regions, mu_water):
    import torch
    error = (f-truth)*(1000.0/mu_water)
    metrics = {}
    for name, mask in regions.items():
        count = int(mask.sum())
        if not count:
            metrics[name] = None
            continue
        e = error[mask]
        metrics[name] = dict(voxel_count=count, rmse_hu=float(torch.sqrt(torch.mean(e*e))),
                             mean_error_hu=float(e.mean()), mae_hu=float(e.abs().mean()),
                             dark_error_fraction_below_minus50_hu=float((e < -50).float().mean()))
    return metrics


def figures(out, truth_mu, recons, masks, axes_mm, mu_water):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z, y, x = axes_mm
    xi, yi = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))
    truth_hu = truth_mu*(1000/mu_water)-1000
    images = {'truth': truth_hu}
    images.update({name: volume*(1000/mu_water)-1000 for name, volume in recons.items()})
    labels = {'truth': 'CQ500 reference', **LABELS}
    # LPS: x=left, y=posterior, z=superior. Sagittal is x=0, coronal y=0.
    planes = [('Sagittal: x=0', y, lambda a: a[:, :, xi], 'y, posterior [mm]'),
              ('Coronal: y=0', x, lambda a: a[:, yi, :], 'x, left [mm]')]
    for mode in ('raw', 'masked', 'skullbase'):
        masked = mode == 'masked'
        detail = mode == 'skullbase'
        fig, axs = plt.subplots(2, 4, figsize=(17, 6 if detail else 8), layout='constrained')
        for row, (plane, horizontal, slicer, xlabel) in enumerate(planes):
            extent = [horizontal[0]-.5, horizontal[-1]+.5, z[0]-.5, z[-1]+.5]
            for col, (name, volume) in enumerate(images.items()):
                ax = axs[row, col]
                view = slicer(volume).copy()
                if name != 'truth' and masked:
                    view[~slicer(masks[name])] = -1000
                ax.imshow(view, origin='lower', extent=extent, cmap='gray', vmin=-110, vmax=210)
                if name != 'truth' and mode == 'raw':
                    ax.contour(horizontal, z, slicer(masks[name]).astype(float), levels=[.5], colors=['#22bbdd'], linewidths=.7)
                ax.axhline(0, color='#ffdf00', ls='--', lw=.7)
                if not detail:
                    ax.axhspan(30, 70, color='#ff8800', alpha=.06)
                ax.set(title=labels[name], xlabel=xlabel, ylabel=plane+'\nz, superior [mm]',
                       ylim=(10, 90) if detail else (-30, 185))
                if detail:
                    ax.set_xlim(-90, 90)
        method = 'display-only every-view detector mask' if masked else 'raw LS + detector visibility outline'
        if detail:
            method = 'raw LS: skull-base detail, same slices as error maps'
        band = '' if detail else ' | orange band: z=30–70 mm'
        fig.suptitle(f'CQ500 head | {method} | fixed HU window [-110, 210]{band}')
        fig.savefig(out/('head_'+mode+'.png'), dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(17, 7), layout='constrained')
    for row, (plane, horizontal, slicer, xlabel) in enumerate(planes):
        extent = [horizontal[0]-.5, horizontal[-1]+.5, z[0]-.5, z[-1]+.5]
        axs[row, 0].imshow(slicer(truth_hu), origin='lower', extent=extent, cmap='gray', vmin=-110, vmax=210)
        axs[row, 0].set(title='Reference, skull-base region', xlabel=xlabel,
                       ylabel=plane+'\nz [mm]', xlim=(-90, 90), ylim=(10, 90))
        for col, name in enumerate(ARMS, 1):
            ax = axs[row, col]
            e = slicer(images[name]-truth_hu)
            im = ax.imshow(e, origin='lower', extent=extent, cmap='RdBu_r', vmin=-100, vmax=100)
            ax.set(title=LABELS[name]+' error', xlabel=xlabel, xlim=(-90, 90), ylim=(10, 90))
    fig.colorbar(im, ax=axs[:, 1:], label='Reconstruction minus reference [HU]', shrink=.7)
    fig.suptitle('Same slices and fixed error window; blue indicates negative error')
    fig.savefig(out/'head_skullbase_error.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('result_sinespin/cq500_fig9'))
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--check-every', type=int, default=40)
    parser.add_argument('--detector-bin', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue matching saved checkpoints; skip completed arms.')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.iterations < 1 or args.check_every < 1:
        parser.error('iteration counts must be positive')
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((args.input_dir/'head_metadata.json').read_text())
    voxel = inputs['recon_voxel_mm']; fine_voxel = inputs['forward_voxel_mm']; mu_water = inputs['mu_water_per_mm']
    truth_mu = np.load(args.input_dir/'reference_mu.npy')
    source_known = np.load(args.input_dir/'reference_valid.npy')
    geometries = build_arms(args.detector_bin)
    masks, axes_mm = visibility_volumes(geometries, truth_mu.shape, voxel)
    if args.plots_only:
        recons = {name: np.load(out/('recon_'+name+'.npy'), mmap_mode='r') for name in ARMS}
        figures(out, truth_mu, recons, masks, axes_mm, mu_water)
        return
    run = dict(input=inputs, geometry={name: geometry_record(g) for name,g in geometries.items()},
               reconstruction=dict(method='LEAP LS, SQS preconditioner, nonnegative, zeros initialization',
                                   iterations=args.iterations, restart_block=args.check_every,
                                   forward_and_backprojector='Joseph', reconstruction_mask_applied=False,
                                   posthoc_intensity_scaling=False, display_mask='every-view finite-detector intersection only'),
               assumptions=['SOD/SDD 750/1200 mm; zero sine phase; centered detector',
                            'Single-energy HU-to-attenuation model; no added projection noise or scatter',
                            'Finite input CT coverage; manufacturer Grangeat algorithm is not implemented'], results={})
    previous = None
    if args.resume:
        previous_path = out/'metrics.json' if (out/'metrics.json').exists() else out/'experiment.json'
        previous = json.loads(previous_path.read_text())
        for key in ('input', 'geometry'):
            if previous[key] != run[key]:
                raise ValueError(f'Resume {key} differs from the saved experiment')
        old_method = {k:v for k,v in previous['reconstruction'].items() if k != 'iterations'}
        new_method = {k:v for k,v in run['reconstruction'].items() if k != 'iterations'}
        if old_method != new_method:
            raise ValueError('Resume reconstruction settings differ from the saved experiment')
        run['results'] = previous.get('results', {})
    elif (out/'experiment.json').exists():
        raise FileExistsError('Experiment already exists; use --resume or a different output directory')
    save_json(out/'experiment.json', run)
    for name,g in geometries.items():
        np.savez(out/(name+'_geometry.npz'), source_positions=g.source_positions, module_centers=g.module_centers,
                 row_vectors=g.row_vectors, col_vectors=g.col_vectors, theta_deg=g.theta_deg, tilt_deg=g.tilt_deg,
                 P_pixel=g.projection_matrices())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    print(f'[device] physical GPU {args.gpu}: {torch.cuda.get_device_name(device)}', flush=True)
    reference = torch.from_numpy(truth_mu).to(device)
    gt_hu = reference*(1000/mu_water)-1000
    common = np.logical_and.reduce(list(masks.values())) & source_known
    common_gpu = torch.from_numpy(common).to(device)
    z, y, x = axes_mm
    X,Y = np.meshgrid(x,y,indexing='xy')
    radial = torch.from_numpy(X*X+Y*Y <= 80**2).to(device)
    soft = (gt_hu >= -100) & (gt_hu <= 150)
    skull_z = torch.from_numpy((z>=30)&(z<70)).to(device)[:,None,None]
    regions = dict(common_head=common_gpu & (gt_hu>-500), common_soft=common_gpu & soft,
                   skullbase_soft=common_gpu & soft & radial & skull_z,
                   common_bone=common_gpu & (gt_hu>=300))
    fine_np = np.load(args.input_dir/'forward_mu.npy', mmap_mode='r')
    fine = torch.from_numpy(np.array(fine_np)).to(device)
    run['runtime'] = dict(gpu=args.gpu, gpu_name=torch.cuda.get_device_name(device), torch=torch.__version__)
    for name,g in geometries.items():
        if name not in args.arms:
            continue
        ct = configure_leap(g, truth_mu.shape, voxel, device)
        library = Path(ct.libprojectors._name).resolve()
        library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
        run['runtime']['leap_sha256'] = library_hash
        if previous and previous.get('runtime', {}).get('leap_sha256', library_hash) != library_hash:
            raise ValueError('Resume LEAP library differs from the saved experiment')
        signature = {key:run[key] for key in ('input', 'geometry', 'reconstruction')}
        signature['reconstruction'] = {k:v for k,v in run['reconstruction'].items() if k != 'iterations'}
        signature['leap_sha256'] = library_hash
        signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        checkpoint = out/(name+'_checkpoint.npz')
        history = []
        f = torch.zeros_like(reference)
        if args.resume and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if str(saved['signature']) != signature:
                    raise ValueError(f'{name} checkpoint configuration differs')
                history = json.loads(str(saved['history_json']))
                f.copy_(torch.from_numpy(saved['reconstruction']).to(device))
        elif args.resume and name in run['results']:
            history = run['results'][name]['convergence']
            f.copy_(torch.from_numpy(np.load(out/('recon_'+name+'.npy'))).to(device))
        completed = history[-1]['iterations'] if history else 0
        if completed > args.iterations:
            raise ValueError('Requested iterations are fewer than the saved checkpoint')
        if completed == args.iterations and name in run['results']:
            print(f'[{name}] keep completed {completed}-iteration reconstruction', flush=True)
            del ct, f
            continue
        print(f'[{name}] {g.n_views} views / {g.scan_angle_deg:g} degrees; forward grid {tuple(fine.shape)}', flush=True)
        ct_fine = configure_leap(g, fine.shape, fine_voxel, device)
        projections = torch.zeros((g.n_views,g.detector_rows,g.detector_cols), device=device)
        start = time.perf_counter(); ct_fine.project_gpu(projections, fine); torch.cuda.synchronize()
        project_seconds = time.perf_counter()-start
        if not bool(torch.isfinite(projections).all()) or float(projections.max()) <= 0:
            raise RuntimeError('Invalid synthetic projections')
        del ct_fine
        predicted = torch.empty_like(projections)
        norm = torch.linalg.vector_norm(projections)
        prior_elapsed = history[-1]['elapsed_seconds'] if history else 0.0
        start = time.perf_counter()
        for done in range(completed, args.iterations, args.check_every):
            count = min(args.check_every, args.iterations-done)
            if ct.LS(projections,f,count,'SQS',True) is None:
                raise RuntimeError('LS reconstruction failed')
            ct.project_gpu(predicted,f)
            residual = float(torch.linalg.vector_norm(predicted-projections)/norm)
            if not bool(torch.isfinite(f).all()): raise RuntimeError('Nonfinite reconstruction')
            measurements = measure_hu(f,reference,regions,mu_water)
            torch.cuda.synchronize()
            h = dict(iterations=done+count, relative_projection_residual=residual, regions=measurements,
                     elapsed_seconds=prior_elapsed+time.perf_counter()-start)
            history.append(h)
            # A single atomic checkpoint keeps the volume and iteration count together.
            temporary = checkpoint.with_suffix('.tmp.npz')
            np.savez(temporary, reconstruction=f.cpu().numpy(), history_json=json.dumps(history), signature=signature)
            temporary.replace(checkpoint)
            save_json(out/(name+'_convergence.json'),history)
            print(f'[{name}] iter {done+count}: residual={residual:.5f}, skullbase RMSE={measurements["skullbase_soft"]["rmse_hu"]:.2f} HU',flush=True)
        rec = f.cpu().numpy(); np.save(out/('recon_'+name+'.npy'),rec)
        run['results'][name] = dict(projection_seconds=project_seconds,convergence=history)
        save_json(out/'metrics.json',run)
        checkpoint.unlink(missing_ok=True)  # Completed volume + metrics are the resume state.
        del ct,projections,predicted,f; torch.cuda.empty_cache()
    if all(name in run['results']
           and run['results'][name]['convergence'][-1]['iterations'] == args.iterations
           and (out/('recon_'+name+'.npy')).exists() for name in ARMS):
        recons = {name: np.load(out/('recon_'+name+'.npy'), mmap_mode='r') for name in ARMS}
        figures(out,truth_mu,recons,masks,axes_mm,mu_water)
    print(f'[done] {out.resolve()}',flush=True)


if __name__ == '__main__':
    main()
