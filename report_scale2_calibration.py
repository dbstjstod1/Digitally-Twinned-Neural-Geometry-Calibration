"""Compare the enlarged-phantom run with both original LNCC31 baselines."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from calibration_gauge import effective_parameters_from_pmat, PARAMETER_NAMES
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from denseball_landmarks import project_landmarks
from physical_camera import decompose_physical_camera, compose_physical_camera
from compare_sinespin_regularization import validate_data, load_run, evaluate, sha256

ROOT=Path(__file__).resolve().parent


def report():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    old=ROOT/'result_spline9/ball_calibration';new=ROOT/'result_spline9_scale2/ball_calibration'
    out=new/'comparison';out.mkdir(parents=True,exist_ok=True)
    specs=[('Original full31',old,old/'signed_lncc31_seed1','#888888'),
           ('Original crop31',old,old/'roi_study/crop_lncc31_seed1','#2469b2'),
           ('Scale2 crop31',new,new/'crop_lncc31_seed1','#159052')]
    common_points=np.array([p['xyz_mm'] for p in json.loads((old/'input/landmarks.json').read_text())['landmarks']])
    common_truth_uv=project_landmarks(np.load(old/'input/P_truth_pixel.npy'),common_points)
    runs=[];summaries=[];arrays=[];cameras=[];table=[];cache={};checks=[]
    for name,base,path,color in specs:
        folder=base/'input'
        if folder not in cache:
            meta=json.loads((folder/'experiment.json').read_text());hashes=validate_data(folder,meta)
            cache[folder]=(meta,hashes)
        meta,hashes=cache[folder];run=load_run(path,meta,hashes);runs.append(run)
        dv,du=meta['truth']['pixel_vu_mm'];v,u=meta.get('detector_padding_vu',[0,0])
        nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
        truth_pixel=np.load(folder/'P_truth_pixel.npy');truth_p=pixel_to_pmat(truth_pixel,du=du,dv=dv,dtype=np.float64)
        truth=effective_parameters_from_pmat(truth_p,nominal)
        np.testing.assert_allclose(truth['parameters_9'],np.load(old/'input/spline_motion9.npy'),atol=1e-9,rtol=0)
        labels=json.loads((folder/'landmarks.json').read_text());points=np.array([p['xyz_mm'] for p in labels['landmarks']])
        uv=project_landmarks(truth_pixel,points);rows,cols=meta['truth']['detector_shape_vu']
        visible=(uv[:,:,0]>=-.5)&(uv[:,:,0]<cols-.5)&(uv[:,:,1]>=-.5)&(uv[:,:,1]<rows-.5)
        if not visible.all():raise ValueError('A bead lies outside acquisition detector')
        summary,array=evaluate(run,truth,nominal,truth_pixel,points,visible,du,dv)
        summary['label']=name;summaries.append(summary);arrays.append(array)
        groups=summary['prior_groups']
        table.append(dict(run=name,bead_rms_px=summary['bead_error_px']['rms'],
            source_rms_mm=summary['source_error_mm']['rms'],worst_bead_view_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
            intrinsic_rms_mm=groups['intrinsic']['canonical_gt_error_component_rms'],
            translation_rms_mm=groups['translation']['canonical_gt_error_component_rms'],
            rotation_rms_deg=groups['rotation']['canonical_gt_error_component_rms'],
            training_seconds=run['metrics']['training_seconds'],image_loss=summary['image_loss']))
        # Common ORIGINAL detector-edge coordinate origin, without a pose fit.
        shift=np.array([[1.,0.,-u*du],[0.,1.,-v*dv],[0.,0.,1.]])
        p=shift@run['p'];camera=decompose_physical_camera(p);q=camera['Q_camera_to_physical']
        common_uv=project_landmarks(pmat_to_pixel(p,du=du,dv=dv),common_points)
        table[-1]['common_original_physical_points_rms_px']=float(np.sqrt(np.mean(np.sum((common_uv-common_truth_uv)**2,axis=-1))))
        restored=compose_physical_camera(camera['source_xyz_mm'],q,camera['K_mm'],projective_scale=camera['projective_scale'])
        np.testing.assert_allclose(restored,p,rtol=1e-11,atol=1e-8)
        angle=np.rad2deg(np.unwrap(np.deg2rad(Rotation.from_matrix(q).as_euler('xyz',degrees=True)),axis=0))
        cameras.append(np.c_[camera['intrinsics_f_cu_cv_mm'],camera['source_xyz_mm'],angle])
        if not len(checks):
            reference_state=torch.load(path/'initial_model.pt',map_location='cpu',weights_only=True)
            reference_motion=np.load(path/'initial_motion9.npy')
            reference_initial=pmat_to_pixel(np.load(path/'P_initial_world_mm.npy'),du=du,dv=dv)
            reference_points=points.copy();reference_nominal=shift@nominal;reference_truth=shift@truth_p
            reference_recipe=run['recipe'].copy();reference_recipe.pop('image_roi',None);reference_recipe.pop('loss')
        state=torch.load(path/'initial_model.pt',map_location='cpu',weights_only=True)
        if state.keys()!=reference_state.keys():raise ValueError('Initial model keys differ')
        for key in state:
            if not torch.equal(state[key],reference_state[key]):raise ValueError(f'Initial model differs: {key}')
        np.testing.assert_array_equal(np.load(path/'initial_motion9.npy'),reference_motion)
        initial=pmat_to_pixel(shift@np.load(path/'P_initial_world_mm.npy'),du=du,dv=dv)
        discrepancy=np.max(abs(project_landmarks(initial,reference_points)-project_landmarks(reference_initial,reference_points)))
        if discrepancy>1e-3:raise ValueError('Initial physical rays changed beyond float32 tolerance')
        np.testing.assert_allclose(shift@nominal,reference_nominal,atol=1e-9,rtol=0)
        np.testing.assert_allclose(shift@truth_p,reference_truth,atol=1e-9,rtol=0)
        recipe=run['recipe'].copy();recipe.pop('image_roi',None);recipe.pop('loss')
        if recipe!=reference_recipe:raise ValueError('Optimization recipe differs beyond ROI')
        for key,value in runs[0]['source_sha256'].items():
            if key!='run_sinespin_calibration.py' and run['source_sha256'].get(key)!=value:
                raise ValueError(f'Core training code changed: {key}')
        if 'image_roi' in run['recipe']:
            if sha256(path/'loss_roi.json')!=run['recipe']['image_roi']['manifest_sha256']:raise ValueError('ROI archive changed')
        checks.append(dict(run=name,initial_model_tensors_identical=True,initial_motion_identical=True,
            initial_ray_max_error_px=float(discrepancy),physical_nominal_and_GT_unchanged=True,
            parameter_r2=(1-np.sum(array['parameter_error9']**2,axis=0)/np.sum((truth['parameters_9']-truth['parameters_9'].mean(0))**2,axis=0)).tolist(),
            matrix_roundtrip_max_abs=float(abs(restored-p).max())))
    theta=np.load(old/'input/truth_geometry.npz')['theta_deg'];motion=np.load(old/'input/spline_motion9.npy')
    ref=decompose_physical_camera(reference_truth);nom=decompose_physical_camera(reference_nominal)
    physical=lambda c:np.c_[c['intrinsics_f_cu_cv_mm'],c['source_xyz_mm'],
        np.rad2deg(np.unwrap(np.deg2rad(Rotation.from_matrix(c['Q_camera_to_physical']).as_euler('xyz',degrees=True)),axis=0))]
    for kind,truth_values,values,titles in [('canonical_parameters9',motion,[a['parameters9'] for a in arrays],PARAMETER_NAMES),
            ('geometry_components9',physical(ref),cameras,['f [mm]','cu [mm]','cv [mm]','Source x [mm]','Source y [mm]','Source z [mm]',
                'Camera x [degree]','Camera y [degree]','Camera z [degree]'])]:
        fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
        for j,ax in enumerate(axes.flat):
            if kind=='geometry_components9':ax.plot(theta,physical(nom)[:,j],':',color='#db7b20',label='Nominal')
            for spec,value in zip(specs,values):ax.plot(theta,value[:,j],color=spec[3],label=spec[0],lw=1)
            ax.plot(theta,truth_values[:,j],'--',color='black',label='GT',lw=1.2)
            ax.set_title(titles[j]);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if kind=='canonical_parameters9' and j<6:ax.set_ylim(-10,10)
            if j>=6:ax.set_xlabel('Scan angle [degree]')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=5)
        fig.suptitle('Scale-2 phantom, all-bead ROI, signed LNCC31, no prior, epoch 100\n'
            +('Canonical effective parameters; fixed physical frame' if kind.startswith('canonical') else
              'Actual physical components; detector origins converted to the original detector; no pose alignment'))
        fig.savefig(out/(kind+'.png'),dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    for spec,array in zip(specs,arrays):
        axes[0].plot(theta,array['per_view_bead_rms_px'],color=spec[3],label=spec[0])
        axes[1].plot(theta,array['source_error_mm'],color=spec[3],label=spec[0])
    axes[0].set_ylabel('35-bead RMS [pixel]');axes[1].set_ylabel('Source error [mm]');axes[1].set_xlabel('Scan angle [degree]')
    axes[0].legend();[ax.grid(alpha=.2) for ax in axes]
    fig.suptitle('Independent full-camera geometry evaluation; no bead rematching or pose fit')
    fig.savefig(out/'geometry_errors.png',dpi=170);plt.close(fig)
    examples=np.load(specs[-1][2]/'projection_examples.npz');roi=json.loads((new/'input/loss_roi.json').read_text())
    box=np.array(roi['boxes_xyxy'])[0];x0,y0,x1,y1=box
    target=examples['target'][:,y0:y1,x0:x1];pred=examples['optimized'][:,y0:y1,x0:x1];residual=pred-target
    vmax=float(np.quantile(target,.999));rmax=float(np.quantile(abs(residual),.995))
    fig,axes=plt.subplots(6,3,figsize=(11,19),layout='constrained')
    for i,view in enumerate(examples['indices']):
        for j,array in enumerate((target[i],pred[i],residual[i])):
            im=axes[i,j].imshow(array,origin='lower',cmap='RdBu_r' if j==2 else 'gray',
                vmin=-rmax if j==2 else 0,vmax=rmax if j==2 else vmax)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Poisson target','Estimated projection','Estimate minus target'][j])
        axes[i,0].set_ylabel(f'View {view}')
    fig.colorbar(im,ax=axes[:,2],shrink=.5,label='Line integral residual')
    fig.suptitle('Actual 2x phantom fits in the frozen all-bead ROI\nShared display scales; no alignment, intensity fit or resizing of data')
    fig.savefig(out/'projection_fits.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,k in zip(axes,[0,3,5]):
        ax.imshow(examples['target'][k],origin='lower',cmap='gray',vmin=0,vmax=vmax)
        ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#35d4ef',lw=1.5))
        ax.set_title(f'View {examples["indices"][k]}: frozen {y1-y0} x {x1-x0} ROI')
    fig.suptitle('Actual full-acquisition noisy projections and training crop; virtual extended detector')
    fig.savefig(out/'training_roi.png',dpi=150);plt.close(fig)
    saved=dict(theta_deg=theta,truth_parameters9=motion)
    for i,array in enumerate(arrays):saved.update({f'run{i}_{k}':v for k,v in array.items()})
    np.savez(out/'comparison.npz',**saved)
    report=dict(table=table,runs=summaries,checks=checks,
        roi_audit=json.loads((new/'input/roi_audit.json').read_text()),
        acquisition=cache[new/'input'][0],coupling=json.loads((ROOT/'result_spline9_scale2/coupling/coupling.json').read_text()),
        limitations=['Whole phantom scaled; bead size, spacing, material path length, ROI and virtual detector extent all change; not a bead-radius-only ablation.',
            'Same I0 and noise seed, but different physical projections and detector extent produce different noise samples.',
            'ROI is fixed union of user-approved preview boxes; preview design used known simulated bead extent. Not a blind observation-only crop.',
            'One noise/initialization/GT realization and fixed 100 epochs; no GT checkpoint selection.',
            'Local point-Jacobian diagnostic is not the LNCC Hessian; no diagnostic GT labels enter model optimization.'],
        source_sha256={p:sha256(ROOT/p) for p in ['report_scale2_calibration.py','setup_scale2_calibration.py','extended_detector.py','audit_intrinsic_rigid_coupling.py']},
        artifact_sha256={p.name:sha256(p) for p in out.iterdir() if p.suffix in ('.png','.npz')})
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(table,indent=2))


if __name__=='__main__':report()
