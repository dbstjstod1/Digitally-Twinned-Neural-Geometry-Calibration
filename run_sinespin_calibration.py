"""Circular initialization -> Sine Spin calibration with an explicit ball volume.

The target is independent LEAP Joseph data. Training uses the existing hash MLP,
effective 9-DoF transform, Triton Joseph projector and a single-resolution image loss.
An optional group L2 prior penalizes applied nominal-relative corrections.
True poses and ball centres are evaluation labels, never an optimization loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from configs.denseball import MOTION_BOUNDS
from sinespin_geometry import build_icono_orbit
from sim_sinespin_recon import configure_leap, geometry_record, save_json
from calibration_geometry import geometry_to_pmat, pmat_to_pixel, centered_source_positions

SHAPE = (801, 929, 929)
VOXEL = .2
ROOT = Path(__file__).resolve().parent
INPUT = ROOT/'result_sinespin/ball_calibration/input'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def geometries(views=546, detector_bin=2, *, spline_config=None):
    common = dict(n_views=views, scan_angle_deg=220., detector_bin=detector_bin)
    if spline_config is not None:
        from spline_calibration_geometry import SplineGeometry
        nominal = build_icono_orbit('circular', **common)
        return nominal, SplineGeometry(nominal, spline_config)
    return build_icono_orbit('circular', **common), build_icono_orbit('sinespin', **common)


def calibration_scan_record(geometry):
    return geometry.record() if geometry.kind == 'spline9' else geometry_record(geometry)


def projector_kwargs(g, shape=SHAPE, voxel=VOXEL):
    from geometry import RT_PARAM
    return dict(nu=g.detector_cols, nv=g.detector_rows, du=g.pixel_width, dv=g.pixel_height,
                imsx=shape[2], imsy=shape[1], imsz=shape[0], dx=voxel, dy=voxel, dz=voxel,
                X0=-shape[2]*voxel/2, Y0=-shape[1]*voxel/2, Z0=-shape[0]*voxel/2,
                ureverse=-1, vreverse=-1, roi=RT_PARAM(0,0,g.detector_cols,g.detector_rows),
                recon_type=1, ori_nu=g.detector_cols, ori_nv=g.detector_rows)


def apply_motion(nominal, raw, bounds, *, shape=SHAPE, voxel=VOXEL):
    import torch
    from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
    ts, tp, rot, _ = motion9_to_ts_tp_rot(raw, **bounds)
    geo = torch.zeros((len(nominal),7), device=nominal.device)
    p, _ = apply_9DoF_transform_effective(
        nominal, geo, ts, tp, rot, nx=shape[2], ny=shape[1], nz=shape[0],
        dx=voxel, dy=voxel, dz=voxel, X0=-shape[2]*voxel/2,
        Y0=-shape[1]*voxel/2, Z0=-shape[0]*voxel/2, use_inverse_right_multiply=0)
    return p.reshape(-1,3,4), torch.cat((ts,tp,rot),dim=-1)


def project(volume, p, g, *, voxel=VOXEL):
    import torch
    from fast_projectors import sinoproj_joseph
    return sinoproj_joseph(smat=volume[None,None], Pmat=p,
                          geo_parameter=torch.zeros((len(p),7),device=p.device),
                          geo_stitch=torch.zeros((len(p),2),device=p.device),
                          **projector_kwargs(g,tuple(volume.shape),voxel))


def load_volume(path, device, shape=SHAPE):
    import torch
    if Path(path).stat().st_size != int(np.prod(shape))*4:
        raise ValueError(f'Raw file size does not match float32 shape {shape}')
    volume = np.memmap(path,dtype=np.float32,mode='r',shape=shape)
    return torch.from_numpy(np.array(volume)).to(device)


def prepare(args, device):
    import torch
    from ball_phantom_fov import require_box_fov
    from photon_noise import poisson_noisy_projections
    folder = args.input_dir
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'experiment.json').exists():
        raise FileExistsError('Prepared data already exist; choose another --input-dir')
    spline_config = (dict(seed=args.spline_seed, knots=8, amplitudes9=[2.]*9)
                     if args.trajectory == 'spline9' else None)
    circle, sine = geometries(args.views,args.detector_bin,spline_config=spline_config)
    shape,voxel=tuple(args.shape_zyx),args.voxel_mm
    try:
        fov=require_box_fov({'nominal':circle,'truth':sine},shape,voxel)
    except ValueError as error:
        if hasattr(error,'reports'):save_json(folder/'fov_audit.json',error.reports)
        raise
    save_json(folder/'fov_audit.json',fov)
    volume = load_volume(args.volume,device,shape)
    if not bool(torch.isfinite(volume).all()) or float(volume.min())<0:
        raise ValueError('Reference must contain finite nonnegative attenuation coefficients in 1/mm')
    ct = configure_leap(sine, shape, voxel, device)
    target = torch.empty((sine.n_views,sine.detector_rows,sine.detector_cols),device=device)
    start=time.perf_counter();ct.project_gpu(target,volume);torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    if not bool(torch.isfinite(target).all()) or float(target.max())<=0:
        raise RuntimeError('Invalid independently generated projections')
    clean=target.cpu().numpy()
    np.save(folder/'clean_projections.npy',clean)
    noisy,noise,counts=poisson_noisy_projections(clean,i0=args.i0,seed=args.noise_seed,return_counts=True)
    np.save(folder/'target_projections.npy',noisy)
    np.save(folder/'photon_counts.npy',counts)
    save_json(folder/'photon_noise.json',noise)
    for name,g in (('nominal',circle),('truth',sine)):
        np.save(folder/f'P_{name}_world_mm.npy',geometry_to_pmat(g))
        np.save(folder/f'P_{name}_pixel.npy',g.projection_matrices())
        np.savez(folder/f'{name}_geometry.npz',source_positions=g.source_positions,
                 module_centers=g.module_centers,row_vectors=g.row_vectors,col_vectors=g.col_vectors,
                 theta_deg=g.theta_deg,tilt_deg=g.tilt_deg)
    # An independent implementation at true geometry sets the numerical floor.
    truth_p=torch.as_tensor(geometry_to_pmat(sine),device=device)
    nominal_p=torch.as_tensor(geometry_to_pmat(circle),device=device)
    floors,baselines=[],[]
    selected=np.unique(np.r_[np.linspace(0,sine.n_views-1,min(12,sine.n_views)).round().astype(int),
                             np.argmax(sine.tilt_deg),np.argmin(sine.tilt_deg),
                             sine.n_views//2,fov['truth']['worst_margin_view'],
                             min(413,sine.n_views-1)])
    with torch.no_grad():
        for indices in np.array_split(selected,3):
            floors.append(project(volume,truth_p[indices],sine,voxel=voxel).cpu().numpy())
            baselines.append(project(volume,nominal_p[indices],sine,voxel=voxel).cpu().numpy())
    independent=np.concatenate(floors);baseline=np.concatenate(baselines)
    truth=target[selected].cpu().numpy()
    rel=lambda a:float(np.linalg.norm((a-truth).astype(np.float64))/np.linalg.norm(truth.astype(np.float64)))
    floor=rel(independent)
    if floor>.01:
        raise RuntimeError(f'Cross-implementation Joseph discrepancy {floor:g} exceeds 1%; audit coordinates before training')
    np.savez(folder/'projection_probe.npz',indices=selected,target=truth,noisy_target=noisy[selected],
             photon_counts=counts[selected],nominal=baseline,oracle=independent)
    library=Path(ct.libprojectors._name).resolve()
    metadata=dict(volume=dict(path=str(args.volume.resolve()),sha256=sha256(args.volume),
                              shape_zyx=list(shape),voxel_mm=voxel,
                              box_extent_xyz_mm=[n*voxel for n in shape[::-1]],
                              voxel_center_origin_xyz_mm=[-(n-1)*voxel/2 for n in shape[::-1]],
                              intensity='Original attenuation coefficients in 1/mm; no scale, thresholding or resampling'),
                  nominal=calibration_scan_record(circle),truth=calibration_scan_record(sine),
                  generator='Independent LEAP Joseph followed by Beer-Lambert Poisson transmission noise',
                  noise=noise,fov_audit_file='fov_audit.json',
                  no_inverse_crime_claim=False,
                  limitations=['Same sampled reference volume and Joseph discretization in generation and fitting; different CUDA implementations.',
                               'Poisson quantum noise only; no scatter, detector blur, reference-volume mismatch, or measured scanner poses.'],
                  leap_sha256=sha256(library),target_sha256=sha256(folder/'target_projections.npy'),
                  clean_sha256=sha256(folder/'clean_projections.npy'),
                  photon_counts_sha256=sha256(folder/'photon_counts.npy'),
                  nominal_pmat_sha256=sha256(folder/'P_nominal_world_mm.npy'),
                  truth_pmat_sha256=sha256(folder/'P_truth_world_mm.npy'),
                  projection_seconds=elapsed,probe_views=selected.tolist(),
                  oracle_relative_l2=floor,nominal_relative_l2=rel(baseline),
                  probe_error_reference='Clean independent Joseph target; this isolates projector discrepancy from photon noise')
    if spline_config is not None:
        metadata['spline_config'] = spline_config
        np.save(folder/'spline_motion9.npy', sine.motion9)
        metadata['spline_motion_sha256'] = sha256(folder/'spline_motion9.npy')
    archive=folder/'preparation_sources';archive.mkdir(exist_ok=True)
    sources=('run_sinespin_calibration.py','photon_noise.py','ball_phantom_fov.py',
             'sinespin_geometry.py','sim_sinespin_recon.py','calibration_geometry.py',
             'fast_projectors.py','geometry.py')
    if spline_config is not None:
        sources += ('spline_calibration_geometry.py','calibration_gauge.py','physical_camera.py')
    for name in sources:shutil.copy2(ROOT/name,archive/name)
    metadata['preparation_source_sha256']={name:sha256(archive/name) for name in sources}
    metadata['preparation_source_archive']='preparation_sources/'
    save_json(folder/'experiment.json',metadata)
    print(f'[prepare] saved {folder}; oracle relL2={floor:.6g}, nominal={rel(baseline):.6g}',flush=True)


def project_points(p, xyz):
    hom=np.column_stack((xyz,np.ones(len(xyz))))
    q=np.einsum('vij,pj->vpi',p,hom)
    return q[...,:2]/q[...,2:3]


def statistics(values):
    v=np.asarray(values,dtype=np.float64)
    return dict(rms=float(np.sqrt(np.mean(v*v))),mean=float(v.mean()),
                median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(v.max()))


def geometry_metrics(p_world, truth_pixel, sine, points):
    pixel=pmat_to_pixel(p_world,du=sine.pixel_width,dv=sine.pixel_height)
    error=project_points(pixel,points)-project_points(truth_pixel,points)
    per_point=np.linalg.norm(error,axis=-1)
    visible=sine.detector_visibility(points)
    source=centered_source_positions(p_world)
    return dict(ball_reprojection_error_px=statistics(per_point[visible]),
                visible_ball_view_pairs=int(visible.sum()),
                all_ball_view_pairs=int(visible.size),
                visibility_rule='Fixed true-geometry visibility; fixed landmark IDs; no nearest-neighbour rematching',
                all_ball_reprojection_error_px=statistics(per_point),
                source_position_error_mm=statistics(np.linalg.norm(source-sine.source_positions,axis=-1)),
                source_z_rmse_mm=float(np.sqrt(np.mean((source[:,2]-sine.source_positions[:,2])**2))),
                per_view_ball_rmse_px=np.sqrt(np.sum(per_point**2*visible,axis=1)/visible.sum(axis=1)).tolist(),
                source_xyz_mm=source.tolist())


def train(args,device):
    if args.loss_levels != [1]:
        raise ValueError('Only single-resolution losses are supported; loss_levels must be [1]')
    import torch
    import monai
    from calibration_losses import build_loss
    from calibration_regularization import AppliedMotionRegularizer, RegularizationConfig
    from models.MotionNetHash import MotionNetHash_9DoF
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    folder,out=args.input_dir,args.out_dir
    if (out/'experiment.json').exists() and not args.resume:
        raise FileExistsError('Run exists; choose another --out-dir or use --resume')
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads((folder/'experiment.json').read_text())
    args.volume=args.volume or Path(data['volume']['path'])
    shape=tuple(data['volume']['shape_zyx']);voxel=data['volume']['voxel_mm']
    if args.shape_zyx is not None and tuple(args.shape_zyx)!=shape:
        raise ValueError('Reference shape differs from prepared data')
    if args.voxel_mm is not None and args.voxel_mm!=voxel:
        raise ValueError('Reference voxel spacing differs from prepared data')
    args.shape_zyx=shape;args.voxel_mm=voxel
    input_hashes={'target_projections.npy':data['target_sha256'],
                  'P_nominal_world_mm.npy':data['nominal_pmat_sha256'],
                  'P_truth_world_mm.npy':data['truth_pmat_sha256']}
    if 'clean_sha256' in data:input_hashes['clean_projections.npy']=data['clean_sha256']
    if 'landmarks_sha256' in data:input_hashes['landmarks.json']=data['landmarks_sha256']
    if 'photon_counts_sha256' in data:input_hashes['photon_counts.npy']=data['photon_counts_sha256']
    for name,expected in input_hashes.items():
        if sha256(folder/name)!=expected: raise ValueError(f'Changed input {name}')
    # Validate evaluation files before spending time training. Their point/pose
    # values do not enter the optimizer, its loss, or checkpoint selection.
    labels=json.loads((folder/'landmarks.json').read_text())
    if (labels['volume_sha256']!=data['volume']['sha256'] or
            tuple(labels['shape_zyx'])!=shape or labels['voxel_size_mm']!=voxel):
        raise ValueError('Landmarks were extracted from a different physical reference')
    if not labels['landmarks']:raise ValueError('No evaluation landmarks')
    if sha256(args.volume)!=data['volume']['sha256']: raise ValueError('Changed reference volume')
    circle,sine=geometries(data['truth']['views'],args.detector_bin,spline_config=data.get('spline_config'))
    if calibration_scan_record(sine)!=data['truth']: raise ValueError('Detector/protocol does not match prepared inputs')
    if not np.allclose(np.load(folder/'P_truth_pixel.npy'),sine.projection_matrices(),rtol=0,atol=1e-8):
        raise ValueError('Changed truth pixel matrices')
    volume=load_volume(args.volume,device,shape)
    targets=torch.from_numpy(np.load(folder/'target_projections.npy')).to(device)
    acquisition_i0=float(data['noise']['i0_photons_per_detector_pixel_per_view'])
    loss_config=dict(name=args.loss,kernel_size=args.lncc_kernel_size,
                     kernel_type=args.lncc_kernel_type,smooth_nr=args.lncc_smooth_nr,
                     smooth_dr=args.lncc_smooth_dr,huber_delta=args.huber_delta,
                     i0=acquisition_i0)
    loss_function=build_loss(**loss_config).to(device)
    loss_config=loss_function.get_config()
    counts=None
    if args.loss=='poisson':
        if 'photon_counts_sha256' not in data:
            raise ValueError('Poisson fitting requires saved, hash-verified photon counts')
        counts=torch.from_numpy(np.load(folder/'photon_counts.npy')).to(device=device,dtype=torch.int64)
        if counts.shape!=targets.shape:raise ValueError('Photon-count shape differs from projections')
    nominal=torch.from_numpy(np.load(folder/'P_nominal_world_mm.npy')).to(device)
    model=MotionNetHash_9DoF(n_views=sine.n_views).to(device)
    # Preserve the supplied vanilla model's initialization. The nominal input P
    # is circular; that does not require replacing the learned model's initial
    # weights. Retain the earlier zero-head setting only for controlled comparison.
    initialization=getattr(args,'initialization','vanilla')
    if initialization=='zero-head':
        torch.nn.init.zeros_(model.net.model[-1].weight);torch.nn.init.zeros_(model.net.model[-1].bias)
        initialization_description='All nine outputs exactly zero by zeroing the final Linear layer'
    elif initialization=='vanilla':
        initialization_description='Original MotionNetHash_9DoF initialization; no layer reset; circular nominal input P'
    else:
        raise ValueError(f'Unknown initialization: {initialization}')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    bounds=dict(ts_max_mm=args.ts_max_mm,tp_max_mm=args.tp_max_mm,rot_max_deg=args.rot_max_deg)
    regularizer=AppliedMotionRegularizer(RegularizationConfig(
        intrinsic_weight=args.reg_intrinsic_weight,translation_weight=args.reg_translation_weight,
        rotation_weight=args.reg_rotation_weight,intrinsic_scale_mm=args.ts_max_mm,
        translation_scale_mm=args.tp_max_mm,rotation_scale_deg=args.rot_max_deg)).to(device)
    idx=torch.arange(0,sine.n_views,args.view_step,device=device)
    all_idx=torch.arange(sine.n_views,device=device)
    history=[];start_epoch=0;previous_seconds=0.
    recipe=dict(epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,seed=args.seed,
                view_step=args.view_step,loss_levels=args.loss_levels,bounds=bounds,
                initialization=initialization_description,
                ground_truth_geometry_used_in_optimizer=False,
                loss='Single-resolution full-detector image loss; no pooling',loss_config=loss_config,
                regularization=regularizer.get_config(),
                model='Existing MotionNetHash_9DoF; all nine parameters free; no sine trajectory prior',
                projector='Existing differentiable Triton Joseph',amp=False)
    if args.resume:
        checkpoint=torch.load(out/'checkpoint.pt',map_location=device,weights_only=False)
        old=checkpoint['recipe'].copy();new=recipe.copy();old.pop('epochs');new.pop('epochs')
        if 'regularization' not in old:
            raise ValueError('Legacy checkpoint predates regularization logging; use its archived runner '
                             'or start a new --out-dir. Do not mix training objectives/source archives.')
        if old!=new: raise ValueError('Resume recipe differs')
        if checkpoint['input_sha256']!=sha256(folder/'experiment.json'): raise ValueError('Changed prepared input metadata')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        history=checkpoint['history'];start_epoch=checkpoint['epoch'];previous_seconds=checkpoint['elapsed_seconds']
    source_names=('run_sinespin_calibration.py','calibration_losses.py','calibration_regularization.py','calibration_geometry.py',
             'fast_projectors.py','DoF_transform.py','models/MotionNetHash.py','models/hash_encoder.py')
    if 'spline_config' in data:
        source_names += ('spline_calibration_geometry.py','calibration_gauge.py','physical_camera.py',
                         'sinespin_geometry.py','sim_sinespin_recon.py')
    sources={name:sha256(ROOT/name) for name in source_names}
    if args.resume and json.loads((out/'experiment.json').read_text())['source_sha256']!=sources:
        raise ValueError('Resume training sources differ from the saved experiment; start a new --out-dir')
    if not args.resume:
        for name in source_names:
            destination=out/'training_sources'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination)
    save_json(out/'experiment.json',dict(input=data,recipe=recipe,source_sha256=sources,
              gpu=args.gpu,gpu_name=torch.cuda.get_device_name(device),torch=torch.__version__,
              monai=monai.__version__,
              train_views=idx.cpu().tolist(),heldout_views=np.setdiff1d(np.arange(sine.n_views),idx.cpu().numpy()).tolist()))
    if not args.resume:
        torch.save(model.state_dict(),out/'initial_model.pt')
        with torch.no_grad():
            initial_p,initial_motion=apply_motion(nominal,model(all_idx),bounds,shape=shape,voxel=voxel)
        np.save(out/'P_initial_world_mm.npy',initial_p.cpu().numpy())
        np.save(out/'initial_motion9.npy',initial_motion.cpu().numpy())
    start=time.perf_counter()
    for epoch in range(start_epoch+1,args.epochs+1):
        model.train();perm=idx[torch.randperm(len(idx),device=device)]
        total=0.;image_total=0.;reg_totals={key:0. for key in ('intrinsic','translation','rotation','total')}
        begin=time.perf_counter()
        for batch in perm.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            p,batch_motion=apply_motion(nominal[batch],model(batch),bounds,shape=shape,voxel=voxel)
            pred=project(volume,p,sine,voxel=voxel)
            image_loss=loss_function(pred,targets[batch],counts=None if counts is None else counts[batch])
            penalty=regularizer.components(batch_motion)
            loss=image_loss+penalty['total'] if regularizer.active else image_loss
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step()
            total+=float(loss.detach())*len(batch)
            image_total+=float(image_loss.detach())*len(batch)
            for key in reg_totals:reg_totals[key]+=float(penalty[key].detach())*len(batch)
        torch.cuda.synchronize(device)
        elapsed=previous_seconds+time.perf_counter()-start
        row=dict(epoch=epoch,loss=total/len(idx),image_loss=image_total/len(idx),
                 regularization_loss=reg_totals['total']/len(idx),
                 intrinsic_prior=reg_totals['intrinsic']/len(idx),
                 translation_prior=reg_totals['translation']/len(idx),
                 rotation_prior=reg_totals['rotation']/len(idx),
                 epoch_seconds=time.perf_counter()-begin,elapsed_seconds=elapsed)
        history.append(row)
        with (out/'loss_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(f'[epoch {epoch:03d}/{args.epochs}] total={row["loss"]:.7f}; image={row["image_loss"]:.7f}; '
              f'reg={row["regularization_loss"]:.7g}; {row["epoch_seconds"]:.2f}s',flush=True)
        if epoch%args.save_every==0 or epoch==args.epochs:
            temporary=out/'checkpoint.tmp'
            torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                            elapsed_seconds=elapsed,recipe=recipe,input_sha256=sha256(folder/'experiment.json')),temporary)
            temporary.replace(out/'checkpoint.pt')
            with torch.no_grad():
                p,motion=apply_motion(nominal,model(all_idx),bounds,shape=shape,voxel=voxel)
            np.save(out/f'P_epoch{epoch:04d}.npy',p.cpu().numpy())
            np.save(out/'motion9.npy',motion.cpu().numpy())
    # Final fixed-epoch model, not a checkpoint chosen by ground-truth geometry.
    model.eval()
    with torch.no_grad():
        optimized,motion=apply_motion(nominal,model(all_idx),bounds,shape=shape,voxel=voxel)
    np.save(out/'P_optimized_world_mm.npy',optimized.cpu().numpy())
    np.save(out/'P_optimized_pixel.npy',pmat_to_pixel(optimized.cpu().numpy(),du=sine.pixel_width,dv=sine.pixel_height))
    np.save(out/'motion9.npy',motion.cpu().numpy())
    evaluate(args,volume,targets,nominal,optimized,sine,history,recipe,loss_function,counts=counts,
             regularizer=regularizer,optimized_motion=motion)


def evaluate(args,volume,targets,nominal,optimized,sine,history,recipe,loss_function,counts=None,
             regularizer=None,optimized_motion=None):
    import torch
    from calibration_losses import build_loss
    # Evaluation labels first enter here, after training has finished.
    truth_pixel=np.load(args.input_dir/'P_truth_pixel.npy')
    clean_path=args.input_dir/'clean_projections.npy'
    clean=np.load(clean_path,mmap_mode='r') if clean_path.exists() else None
    count_path=args.input_dir/'photon_counts.npy'
    count_map=np.load(count_path,mmap_mode='r') if counts is None and count_path.exists() else None
    acquisition=json.loads((args.input_dir/'experiment.json').read_text())
    acquisition_i0=float(acquisition['noise']['i0_photons_per_detector_pixel_per_view'])
    reference_lncc=build_loss('lncc').to(volume.device)
    reference_poisson=build_loss('poisson',i0=acquisition_i0).to(volume.device)
    landmarks=json.loads((args.input_dir/'landmarks.json').read_text())
    points=np.asarray([item['xyz_mm'] for item in landmarks['landmarks']])
    matrices={'nominal':nominal,'optimized':optimized,
              'oracle':torch.from_numpy(np.load(args.input_dir/'P_truth_world_mm.npy')).to(volume.device)}
    summaries={};examples={};view_errors={}
    selected=np.linspace(0,sine.n_views-1,6).round().astype(int)
    with torch.no_grad():
        for name,p in matrices.items():
            sq=[];truthsq=[];clean_sq=[];clean_truthsq=[];losses=[];lncc_scores=[];deviances=[];images=[]
            for start in range(0,sine.n_views,args.batch_size):
                end=min(start+args.batch_size,sine.n_views)
                pred=project(volume,p[start:end],sine,voxel=args.voxel_mm);target=targets[start:end]
                sq.extend(((pred.double()-target.double())**2).sum((1,2)).cpu().tolist())
                truthsq.extend((target.double()**2).sum((1,2)).cpu().tolist())
                if clean is not None:
                    clean_target=torch.from_numpy(np.array(clean[start:end])).to(volume.device)
                    clean_sq.extend(((pred.double()-clean_target.double())**2).sum((1,2)).cpu().tolist())
                    clean_truthsq.extend((clean_target.double()**2).sum((1,2)).cpu().tolist())
                batch_counts=counts[start:end] if counts is not None else (
                    torch.from_numpy(np.array(count_map[start:end])).to(device=volume.device,dtype=torch.int64) if count_map is not None else None)
                losses.append((float(loss_function(pred,target,counts=batch_counts)),end-start))
                lncc_scores.append((float(reference_lncc(pred,target)),end-start))
                if batch_counts is not None:
                    deviances.extend(reference_poisson.per_view(pred.double(),counts=batch_counts).cpu().tolist())
                for j in selected[(selected>=start)&(selected<end)]:images.append(pred[j-start].cpu().numpy())
            error=np.asarray(sq);den=np.asarray(truthsq)
            splits={'all':np.arange(sine.n_views),'train':np.arange(0,sine.n_views,args.view_step)}
            splits['heldout']=np.setdiff1d(splits['all'],splits['train'])
            projection={key:dict(views=len(ids),relative_l2=float(np.sqrt(error[ids].sum()/den[ids].sum())))
                        for key,ids in splits.items() if len(ids)}
            if clean is not None:
                for key,ids in splits.items():
                    if len(ids):projection[key]['relative_l2_to_clean']=float(np.sqrt(np.asarray(clean_sq)[ids].sum()/np.asarray(clean_truthsq)[ids].sum()))
            common=dict(lncc31_loss=sum(v*n for v,n in lncc_scores)/sine.n_views,
                        postlog_mse=float(error.sum()/targets.numel()))
            if deviances:common['poisson_deviance_per_incident_photon']=float(np.mean(deviances))
            image_value=sum(v*n for v,n in losses)/sine.n_views
            reg_value=0.;reg_components={}
            if regularizer is not None:
                if name=='optimized':
                    applied=optimized_motion
                elif name=='nominal':
                    applied=torch.zeros((sine.n_views,9),device=volume.device)
                else:
                    # Oracle parameters are evaluation-only and cannot enter
                    # training or checkpoint selection through this branch.
                    from calibration_gauge import effective_parameters_from_pmat
                    applied=torch.as_tensor(effective_parameters_from_pmat(
                        p.cpu().numpy(),nominal.cpu().numpy())['parameters_9'],
                        device=volume.device,dtype=optimized_motion.dtype)
                reg_components={k:float(v) for k,v in regularizer.components(applied).items()}
                reg_value=reg_components['total']
            summaries[name]=dict(projection=projection,image_loss=image_value,
                                 regularization_loss=reg_value,regularization_components=reg_components,
                                 objective_loss=image_value+reg_value,
                                 lncc_loss=common['lncc31_loss'],common_image_metrics=common,
                                 geometry=geometry_metrics(p.cpu().numpy(),truth_pixel,sine,points))
            view_errors[name]=np.sqrt(error/den)
            examples[name]=np.stack(images)
    save_json(args.out_dir/'metrics.json',dict(metrics_schema=3,
              loss_reporting='objective_loss=image_loss+regularization_loss; lncc_loss remains reference MONAI LNCC31; compare runs using image_loss/common_image_metrics and geometry',
              recipe=recipe,landmark_count=len(points),landmarks_sha256=sha256(args.input_dir/'landmarks.json'),
              results=summaries,epoch=history[-1]['epoch'],training_seconds=history[-1]['elapsed_seconds'],
              assumptions=['No ground-truth pose, landmark correspondence, or sinusoidal fit enters training.',
                           'Same reference volume/discretization; synthetic quantum noise does not model scanner/systematic errors.'],
              artifact_sha256={name:sha256(args.out_dir/name) for name in ('P_optimized_world_mm.npy','P_optimized_pixel.npy','motion9.npy','checkpoint.pt')}))
    np.savez(args.out_dir/'projection_examples.npz',indices=selected,target=targets[selected].cpu().numpy(),**examples)
    make_figures(args.out_dir,sine,summaries,history,selected,targets[selected].cpu().numpy(),examples,view_errors)
    print('[final] '+json.dumps({name:dict(relL2=s['projection']['all']['relative_l2'],ball_px=s['geometry']['ball_reprojection_error_px']['rms'],source_mm=s['geometry']['source_position_error_mm']['rms']) for name,s in summaries.items()}),flush=True)


def make_figures(out,sine,summaries,history,selected,target,examples,view_errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    theta=sine.theta_deg
    for name in ('nominal','optimized','oracle'):
        g=summaries[name]['geometry']
        axs[0,0].plot(theta,np.asarray(g['source_xyz_mm'])[:,2],label=name)
        axs[0,1].semilogy(theta,np.maximum(g['per_view_ball_rmse_px'],1e-6),label=name)
        axs[1,0].semilogy(theta,np.maximum(view_errors[name],1e-8),label=name)
    axs[0,0].set(xlabel='Scan angle [deg]',ylabel='Source z [mm]')
    axs[0,1].set(xlabel='Scan angle [deg]',ylabel='Ball reprojection RMSE [pixel]')
    axs[1,0].set(xlabel='Scan angle [deg]',ylabel='Projection relative L2')
    axs[1,1].plot([h['epoch'] for h in history],[h['loss'] for h in history],label='Total objective')
    if any(h.get('regularization_loss',0)>0 for h in history):
        axs[1,1].plot([h['epoch'] for h in history],[h.get('image_loss',h['loss']) for h in history],label='Image loss')
        axs[1,1].legend()
    axs[1,1].set(xlabel='Epoch',ylabel='Training objective')
    for ax in axs.flat:ax.grid(alpha=.25)
    for ax in axs.flat[:3]:ax.legend()
    fig.suptitle('Ball phantom: circular initialization → Sine Spin P-matrix calibration\nExisting 9-DoF hash MLP; image loss with optional parameter prior; no true-pose supervision')
    fig.savefig(out/'geometry_recovery.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(4,3,figsize=(13,12),constrained_layout=True)
    vmax=float(np.quantile(target,.999));emax=max(float(np.quantile(np.abs(examples['nominal']-target),.995)),.1)
    for col,k in enumerate((1,2,4)):
        panels=[target[k],examples['nominal'][k],examples['optimized'][k],examples['optimized'][k]-target[k]]
        for row,im in enumerate(panels):
            axs[row,col].imshow(im,origin='lower',cmap='gray' if row<3 else 'coolwarm',
                               vmin=0 if row<3 else -emax,vmax=vmax if row<3 else emax)
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
        axs[0,col].set_title(f'View {selected[k]}, angle {theta[selected[k]]:.1f}°')
    for ax,label in zip(axs[:,0],('Observed target','Circular initialization','Optimized P-matrix','Optimized − target')):ax.set_ylabel(label)
    fig.savefig(out/'projection_comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','train'))
    p.add_argument('--input-dir',type=Path,default=INPUT)
    p.add_argument('--out-dir',type=Path,default=ROOT/'result_sinespin/ball_calibration/baseline_seed0')
    p.add_argument('--volume',type=Path,help='Little-endian float32 raw containing attenuation coefficients in 1/mm')
    p.add_argument('--shape-zyx',type=int,nargs=3,help='Explicit raw shape; required for preparation')
    p.add_argument('--voxel-mm',type=float,help='Isotropic voxel spacing in mm; required for preparation')
    p.add_argument('--i0',type=float,default=44000.,help='Preparation: incident photons per saved detector pixel/view. Training reads the stored acquisition value.')
    p.add_argument('--noise-seed',type=int,default=0,help='PCG64 Poisson seed, independent of optimization seed')
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--views',type=int,default=546)
    p.add_argument('--trajectory',choices=('sinespin','spline9'),default='sinespin',
                   help='Preparation only; training uses the recorded acquisition')
    p.add_argument('--spline-seed',type=int,default=20260923,help='Preparation-only independent GT spline seed')
    p.add_argument('--detector-bin',type=int,default=2)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--ts-max-mm',type=float,default=MOTION_BOUNDS['ts_max_mm'],
                   help='Effective intrinsic correction bound in mm, applied to all three ts components')
    p.add_argument('--tp-max-mm',type=float,default=MOTION_BOUNDS['tp_max_mm'],
                   help='Object-space translation bound in mm; not an absolute source-position bound')
    p.add_argument('--rot-max-deg',type=float,default=MOTION_BOUNDS['rot_max_deg'],
                   help='Per-axis INTERNAL Euler correction bound in degrees, before deriving the source from P')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--initialization',choices=('vanilla','zero-head'),default='vanilla',
                   help='Original model initialization, or the earlier zero-head setting for controlled comparison')
    p.add_argument('--view-step',type=int,default=1)
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--loss',choices=('lncc','signed_lncc','global_ncc','mse','huber','poisson'),default='lncc')
    p.add_argument('--lncc-kernel-size',type=int,default=31,help='One odd window size in saved detector pixels; no image pyramid')
    p.add_argument('--lncc-kernel-type',choices=('rectangular','triangular'),default='rectangular')
    p.add_argument('--lncc-smooth-nr',type=float,default=0.,help='MONAI squared-correlation numerator smoothing')
    p.add_argument('--lncc-smooth-dr',type=float,default=1e-5,help='Floor on each window variance sum, not an additive denominator epsilon')
    p.add_argument('--huber-delta',type=float,default=1.,help='Huber threshold in projection line-integral units')
    p.add_argument('--reg-intrinsic-weight',type=float,default=0.,
                   help='Weight on mean squared applied intrinsic corrections / ts-max-mm; no GT inputs')
    p.add_argument('--reg-translation-weight',type=float,default=0.,
                   help='Optional weight on mean squared applied object translations / tp-max-mm')
    p.add_argument('--reg-rotation-weight',type=float,default=0.,
                   help='Optional weight on mean squared applied Euler corrections / rot-max-deg; leave 0 to allow sineSpin tilt')
    p.add_argument('--loss-levels',type=int,nargs='+',choices=(1,),default=[1],
                   help='Compatibility option: only a single 1 is accepted; no multiscale image pooling')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.loss_levels != [1]:
        p.error('Only single-resolution losses are supported: --loss-levels 1')
    if args.mode=='prepare' and (args.volume is None or args.shape_zyx is None or args.voxel_mm is None):
        p.error('Preparation requires --volume, --shape-zyx and --voxel-mm; no implicit Denseball input')
    if args.shape_zyx is not None and min(args.shape_zyx)<=0:
        p.error('Shape dimensions must be positive')
    if args.voxel_mm is not None and (not np.isfinite(args.voxel_mm) or args.voxel_mm<=0):
        p.error('Voxel spacing must be finite and positive')
    if not np.isfinite(args.i0) or args.i0<=0 or args.noise_seed<0:
        p.error('Incident photon count must be finite/positive and noise seed nonnegative')
    if args.spline_seed < 0:
        p.error('Spline seed must be nonnegative')
    if min(args.views,args.detector_bin,args.epochs,args.batch_size,args.view_step,args.save_every,*args.loss_levels)<=0 or not np.isfinite(args.lr) or args.lr<=0:
        p.error('Counts and sampling factors must be positive; learning rate must be finite and positive')
    if not 0<=args.seed<2**32:
        p.error('Optimization seed must be in [0, 2**32)')
    if any(not np.isfinite(v) or v<=0 for v in (args.ts_max_mm,args.tp_max_mm,args.rot_max_deg)):
        p.error('Motion bounds must be finite and positive')
    if any(not np.isfinite(v) or v<0 for v in (args.reg_intrinsic_weight,args.reg_translation_weight,args.reg_rotation_weight)):
        p.error('Regularization weights must be finite and nonnegative')
    if args.mode=='train':
        from calibration_losses import LossConfig
        try:
            LossConfig(name=args.loss,kernel_size=args.lncc_kernel_size,
                       kernel_type=args.lncc_kernel_type,smooth_nr=args.lncc_smooth_nr,
                       smooth_dr=args.lncc_smooth_dr,huber_delta=args.huber_delta)
        except ValueError as error:
            p.error(str(error))
    import torch
    if not torch.cuda.is_available() or not 0<=args.gpu<torch.cuda.device_count():p.error('Requested GPU unavailable')
    torch.cuda.set_device(args.gpu);device=torch.device('cuda',args.gpu)
    (prepare if args.mode=='prepare' else train)(args,device)


if __name__=='__main__':main()
