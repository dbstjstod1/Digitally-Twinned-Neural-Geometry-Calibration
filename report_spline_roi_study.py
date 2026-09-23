"""Controlled full-image/cropped signed-LNCC window comparison, fixed epoch 100."""
from pathlib import Path
import argparse
import copy
import difflib
import json
import shutil
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from calibration_geometry import pixel_to_pmat
from calibration_gauge import effective_parameters_from_pmat, PARAMETER_NAMES
from denseball_landmarks import project_landmarks
from physical_camera import decompose_physical_camera, compose_physical_camera
from compare_sinespin_regularization import (sha256, validate_data, validate_initialization,
                                             load_run, evaluate)

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'result_spline9/ball_calibration'
COLORS=['#777777','#2469b2','#e08821','#16884a','#bb4c91']


def controlled_recipe(recipe):
    result=copy.deepcopy(recipe)
    result.pop('image_roi',None);result.pop('loss')
    config=result['loss_config']
    if config['kernel_type']!='rectangular' or config['kernel_weight_mass_2d']!=config['kernel_size']**2:
        raise ValueError('This study requires rectangular windows with mass k squared')
    result['loss_config'].pop('kernel_size')
    result['loss_config'].pop('kernel_weight_mass_2d')
    return result


def report(base,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study=base/'roi_study';folder=base/'input'
    acquisition=json.loads((folder/'experiment.json').read_text())
    hashes=validate_data(folder,acquisition)
    specs=[('Full 31',base/'signed_lncc31_seed1',31)]+[
        (f'Crop {k}',study/f'crop_lncc{k}_seed1',k) for k in (31,21,15,9)]
    runs=[load_run(path,acquisition,hashes,required_kernel_size=k) for _,path,k in specs]
    roi_path=study/'input_roi/roi.json';roi=json.loads(roi_path.read_text())
    if roi['target_sha256']!=hashes['target_projections.npy']:raise ValueError('ROI input mismatch')
    if sha256(ROOT/'prepare_projection_roi.py')!=roi['source_sha256']:
        raise ValueError('ROI detection source changed; preserve its exact implementation')
    initial=[]
    for i,run in enumerate(runs):
        if controlled_recipe(run['recipe'])!=controlled_recipe(runs[0]['recipe']):
            raise ValueError('A setting other than crop/kernel changed')
        reg=run['recipe']['regularization']
        if any(reg[key]!=0 for key in ('intrinsic_weight','translation_weight','rotation_weight')):
            raise ValueError('ROI comparison requires no regularization')
        if i:
            initial.append(validate_initialization(runs[0]['path'],run['path']))
            for name,digest in runs[0]['source_sha256'].items():
                if name!='run_sinespin_calibration.py' and run['source_sha256'].get(name)!=digest:
                    raise ValueError(f'Core training implementation changed: {name}')
            if run['source_sha256']!=runs[1]['source_sha256']:
                raise ValueError('Cropped runs must have identical complete training sources')
            if (run['recipe']['image_roi']['manifest_sha256']!=sha256(roi_path)
                    or sha256(run['path']/'loss_roi.json')!=sha256(roi_path)):
                raise ValueError('Runs used different crop manifests')
        elif 'image_roi' in run['recipe']:raise ValueError('Expected uncropped historical baseline')
    dv,du=acquisition['truth']['pixel_vu_mm']
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    truth_pixel=np.load(folder/'P_truth_pixel.npy')
    truth_p=pixel_to_pmat(truth_pixel,du=du,dv=dv,dtype=np.float64)
    truth=effective_parameters_from_pmat(truth_p,nominal)
    np.testing.assert_allclose(truth['parameters_9'],np.load(folder/'spline_motion9.npy'),atol=1e-9,rtol=0)
    labels=json.loads((folder/'landmarks.json').read_text())
    points=np.array([p['xyz_mm'] for p in labels['landmarks']]);uv=project_landmarks(truth_pixel,points)
    rows,cols=roi['detector_shape_vu'];boxes=np.array(roi['boxes_xyxy'])
    visible=(uv[:,:,0]>=0)&(uv[:,:,0]<cols)&(uv[:,:,1]>=0)&(uv[:,:,1]<rows)
    if not visible.all():raise ValueError('Expected all 35 beads on full detector')
    margin=np.stack((uv[:,:,0]-boxes[:,0,None],boxes[:,2,None]-1-uv[:,:,0],
                     uv[:,:,1]-boxes[:,1,None],boxes[:,3,None]-1-uv[:,:,1]),-1).min(-1)
    coverage=dict(evaluation_only=True,gt_used_to_define_roi=False,total_bead_view_pairs=int(margin.size),
        centers_outside_crop=int((margin<0).sum()),minimum_center_boundary_margin_px=float(margin.min()),
        minimum_centers_retained_per_view=int((margin>=0).sum(1).min()),
        full_window_at_center_pairs={str(k):int((margin>=k//2).sum()) for k in (31,21,15,9)})
    summaries=[];arrays=[];table=[]
    theta=np.load(folder/'truth_geometry.npz')['theta_deg']
    out.mkdir(parents=True,exist_ok=True)
    for (name,path,k),run in zip(specs,runs):
        summary,array=evaluate(run,truth,nominal,truth_pixel,points,visible,du,dv)
        summaries.append(summary);arrays.append(array)
        groups=summary['prior_groups']
        table.append(dict(run=name,kernel_size=k,crop=name!='Full 31',
            bead_rms_px=summary['bead_error_px']['rms'],
            median_view_bead_rms_px=float(np.median(array['per_view_bead_rms_px'])),
            p90_view_bead_rms_px=float(np.quantile(array['per_view_bead_rms_px'],.9)),
            worst_view_bead_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
            source_rms_mm=summary['source_error_mm']['rms'],
            intrinsic_component_rms_mm=groups['intrinsic']['canonical_gt_error_component_rms'],
            translation_component_rms_mm=groups['translation']['canonical_gt_error_component_rms'],
            rotation_component_rms_deg=groups['rotation']['canonical_gt_error_component_rms'],
            full_image_relative_l2_clean=summary['projection_metrics']['relative_l2_to_clean'],
            full_image_lncc31=summary['common_image_metrics']['lncc31_loss'],
            training_seconds=run['metrics']['training_seconds'],gpu=run['experiment']['gpu']))
    # Full-frame physical geometry, not crop-shifted principal points.
    physical={};closure=[]
    for name,p in [('GT',truth_p),('Nominal',nominal)]+[(s[0],r['p']) for s,r in zip(specs,runs)]:
        camera=decompose_physical_camera(p);q=camera['Q_camera_to_physical']
        with warnings.catch_warnings():
            warnings.simplefilter('error',UserWarning)
            angles=np.rad2deg(np.unwrap(np.deg2rad(Rotation.from_matrix(q).as_euler('xyz',degrees=True)),axis=0))
        np.testing.assert_allclose(Rotation.from_euler('xyz',angles,degrees=True).as_matrix(),q,atol=1e-12,rtol=0)
        restored=compose_physical_camera(camera['source_xyz_mm'],q,camera['K_mm'],projective_scale=camera['projective_scale'])
        np.testing.assert_allclose(restored,p,atol=1e-8,rtol=1e-11)
        physical[name]=np.column_stack((camera['intrinsics_f_cu_cv_mm'],camera['source_xyz_mm'],angles))
        closure.append(dict(run=name,matrix_roundtrip_max_abs=float(abs(restored-p).max()),
                            focal_anisotropy_max_mm=float(abs(camera['focal_anisotropy_mm']).max()),
                            skew_max_mm=float(abs(camera['skew_mm']).max())))
    labels9=['f [mm]','cu [mm]','cv [mm]','Source x [mm]','Source y [mm]','Source z [mm]',
             'Camera x [degree]','Camera y [degree]','Camera z [degree]']
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
    for j,ax in enumerate(axes.flat):
        ax.plot(theta,physical['Nominal'][:,j],':',color='#db7b20',label='Nominal')
        for (name,_,_),color in zip(specs,COLORS):ax.plot(theta,physical[name][:,j],color=color,lw=.9,label=name)
        ax.plot(theta,physical['GT'][:,j],'--',color='black',lw=1.2,label='GT')
        ax.set_title(labels9[j]);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',useOffset=False,style='plain')
        if j>=6:ax.set_xlabel('Nominal angle [degree]')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=4)
    fig.suptitle('Actual nine physical components from complete P\nOriginal detector coordinates; same fixed phantom frame; no pose fit')
    fig.savefig(out/'geometry_components9.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    for (name,_,_),color,array in zip(specs,COLORS,arrays):
        axes[0].plot(theta,array['per_view_bead_rms_px'],color=color,label=name,lw=1)
        axes[1].plot(theta,array['source_error_mm'],color=color,label=name,lw=1)
    axes[0].set_ylabel('All 35 beads: RMS [pixel]');axes[1].set_ylabel('Source distance to GT [mm]')
    axes[1].set_xlabel('Nominal scan angle [degree]');axes[0].legend(ncol=5)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Full-detector geometry evaluation at epoch 100\nSame noisy data, seed 1 and initial weights; no regularization')
    fig.savefig(out/'geometry_errors.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
    for j,ax in enumerate(axes.flat):
        for (name,_,_),color,array in zip(specs,COLORS,arrays):ax.plot(theta,array['parameters9'][:,j],color=color,lw=.9,label=name)
        ax.plot(theta,truth['parameters_9'][:,j],'--',color='black',lw=1.2,label='GT spline')
        ax.set_title(PARAMETER_NAMES[j]);ax.set_ylabel('mm' if j<6 else 'degree');ax.grid(alpha=.2)
        if j<6:ax.set_ylim(-10,10)
        if j>=6:ax.set_xlabel('Nominal scan angle [degree]')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Canonical effective parameters from complete P\nGT spline values shown, not zero error lines; mm axes fixed to +/-10')
    fig.savefig(out/'canonical_parameters9.png',dpi=170);plt.close(fig)
    # Six preset views with identical display scales; show crop boundary on residuals.
    from matplotlib.patches import Rectangle
    examples=[dict(np.load(r['path']/'projection_examples.npz')) for r in runs]
    for e in examples[1:]:
        np.testing.assert_array_equal(e['indices'],examples[0]['indices'])
        np.testing.assert_array_equal(e['target'],examples[0]['target'])
    target=examples[0]['target'];selected=examples[0]['indices']
    residuals=[e['optimized']-target for e in examples]
    scale=max(float(np.quantile(np.abs(np.concatenate([r.ravel() for r in residuals])),.995)),1e-5)
    fig,axes=plt.subplots(6,len(specs)+1,figsize=(19,17),layout='constrained')
    vmax=float(np.quantile(target,.999))
    for i,v in enumerate(selected):
        axes[i,0].imshow(target[i],origin='lower',cmap='gray',vmin=0,vmax=vmax)
        axes[i,0].set_ylabel(f'View {v}')
        for j,(name,_,_) in enumerate(specs,1):
            im=axes[i,j].imshow(residuals[j-1][i],origin='lower',cmap='RdBu_r',vmin=-scale,vmax=scale)
            x0,y0,x1,y1=boxes[v]
            axes[i,j].add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#ee9222',lw=.5))
            if i==0:axes[i,j].set_title(name+' minus target')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    axes[0,0].set_title('Poisson target')
    fig.colorbar(im,ax=axes[:,1:],shrink=.45,label='Line integral residual')
    fig.suptitle('Actual full-detector residuals; shared scales; no alignment or image normalization\nOrange outline: frozen crop; bottom plate remains visible for evaluation')
    fig.savefig(out/'projection_residuals.png',dpi=130);plt.close(fig)
    saved=dict(theta_deg=theta,truth_parameters9=truth['parameters_9'],roi_boxes_xyxy=boxes)
    for i,array in enumerate(arrays):saved.update({f'run{i}_{key}':value for key,value in array.items()})
    saved.update({f'physical_{key.replace(" ","_")}':value for key,value in physical.items()})
    np.savez(out/'comparison.npz',**saved)
    for name in ('roi_preview.png','kernel_windows.png','roi.json'):shutil.copy2(study/'input_roi'/name,out/name)
    old=(runs[0]['path']/'training_sources/run_sinespin_calibration.py').read_text()
    new=(runs[1]['path']/'training_sources/run_sinespin_calibration.py').read_text()
    result=dict(table=table,runs=summaries,roi=roi,posthoc_roi_coverage=coverage,
        experiment_sequence='Preplanned full31 versus crop31/15/9; crop21 added after inspecting final crop31/15 geometry to explore the intermediate window. No ROI/checkpoint adjustment.',
        initialization_checks=initial,physical_decomposition_checks=closure,input_sha256=hashes,
        baseline_runner_diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='historical',tofile='roi-enabled')),
        controlled_conditions=controlled_recipe(runs[0]['recipe']),
        limitations=['One GT realization, noise seed and optimization seed; fixed epoch 100, not proof of full convergence.',
            'No crop parameters or checkpoints selected using GT; all beads and full images remain in evaluation.',
            'Different cropped/window image-loss values are not comparable; use common full-image metrics and geometry.',
            'Crop affects the loss only; full-detector Joseph projection is still computed.',
            'The original LNCC zero padding is retained at crop edges. 31-pixel windows touch the boundary near some beads.',
            'Rectangle excludes the bright bottom plate band but is not a material segmentation.',
            'The crop jointly removes surrounding air and the bottom plate; their separate effects are not isolated.',
            'Two identical-model A6000 GPUs were used; initial model, motion and P are byte-verified. GPU timing is indicative.'],
        source_sha256={name:sha256(ROOT/name) for name in ('report_spline_roi_study.py','compare_sinespin_regularization.py','calibration_roi.py','prepare_projection_roi.py')},
        artifact_sha256={path.name:sha256(path) for path in sorted(out.iterdir()) if path.suffix in ('.png','.npz')})
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(table,indent=2));print(json.dumps(coverage,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=BASE)
    parser.add_argument('--out-dir',type=Path,default=BASE/'roi_study/comparison')
    args=parser.parse_args();report(args.base_dir,args.out_dir)
