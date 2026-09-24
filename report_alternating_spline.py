"""Final-epoch, equal-forward/backward-budget comparison of spline block updates."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from alternating_spline import SplitBSplineMotion9
from spline_motion_model import BSplineMotion9
from calibration_gauge import PARAMETER_NAMES, effective_parameters_from_pmat
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from denseball_landmarks import project_landmarks
from physical_camera import decompose_physical_camera
from compare_sinespin_regularization import validate_data, load_run, evaluate
from run_sinespin_calibration import apply_motion, sha256

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'result_spline9_scale2/ball_calibration'
OUT=BASE/'alternating_comparison'
SPECS=[
    ('Joint 100',BASE/'bspline20_lncc31_seed1',100,'#888888'),
    ('Joint 200',BASE/'bspline20_joint200_seed1',200,'#2469b2'),
    ('Rigid then K 100',BASE/'bspline20_alternating_rk_seed1',100,'#ca582c')]


def checkpoint_audit(run,meta,points,reference_initial):
    config=run['recipe'];alternate='update_scheme' in config
    model_type=SplitBSplineMotion9 if alternate else BSplineMotion9
    views=meta['truth']['views'];dv,du=meta['truth']['pixel_vu_mm']
    model=model_type(views,config['motion_model_config']['control_points'],**config['bounds'])
    initial=torch.load(run['path']/'initial_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(initial)
    assert torch.count_nonzero(model.raw_coefficients).item()==0
    np.testing.assert_array_equal(np.load(run['path']/'initial_motion9.npy'),0*np.load(run['path']/'motion9.npy'))
    for name in ('basis','knots','scales'):
        np.testing.assert_array_equal(initial[name],reference_initial[name])
    # Restore the full final spline curve and its camera chain on CPU.
    checkpoint=torch.load(run['path']/'checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    archive=np.load(run['path']/'spline_coefficients.npz')
    np.testing.assert_array_equal(archive['raw_coefficients'],model.raw_coefficients.detach().numpy())
    np.testing.assert_array_equal(archive['basis'],model.basis.numpy())
    with torch.no_grad():
        p,m=apply_motion(torch.from_numpy(np.load(BASE/'input/P_nominal_world_mm.npy')),
            model(torch.arange(views)),config['bounds'],physical=True,
            shape=tuple(meta['volume']['shape_zyx']),voxel=meta['volume']['voxel_mm'])
    np.testing.assert_allclose(m.numpy(),run['motion'],rtol=1e-6,atol=2e-6)
    pixel=pmat_to_pixel(p.numpy(),du=du,dv=dv)
    ray_difference=float(abs(project_landmarks(pixel,points)-
        project_landmarks(np.load(run['path']/'P_optimized_pixel.npy'),points)).max())
    assert ray_difference<1e-3,ray_difference
    batches=(len(run['experiment']['train_views'])+config['batch_size']-1)//config['batch_size']
    block_updates=config['epochs']*batches
    audit=dict(initial_coefficients_zero=True,initial_basis_and_scales_identical=True,
        CPU_checkpoint_ray_max_abs_difference_px=ray_difference,
        coefficient_archive_sha256=sha256(run['path']/'spline_coefficients.npz'),
        training_batches_per_epoch=batches,forward_backward_batches=block_updates*(2 if alternate else 1),
        updates_per_component=block_updates)
    if alternate:
        optimizer=checkpoint['optimizer']
        assert optimizer['updates']==dict(rigid=block_updates,intrinsic=block_updates)
        assert optimizer['inactive_checks']==2*block_updates
        assert checkpoint['history'][-1]['inactive_block_checks']==2*block_updates
        for name,width in (('intrinsic',3),('rigid',6)):
            state=optimizer['optimizers'][name]['state']
            assert len(state)==1
            state=next(iter(state.values()))
            assert int(state['step'])==block_updates
            assert tuple(state['exp_avg'].shape)==(20,width)
        audit['verified_inactive_parameter_freeze_checks']=optimizer['inactive_checks']
        audit['separate_adam_step_counts']=optimizer['updates']
    events=[]
    for event_path in sorted(run['path'].glob('resume_epoch*.json')):
        event=json.loads(event_path.read_text())
        assert event['status']=='complete'
        assert event['sampler']['reference_generator_permutation_and_state_identical']
        assert event['resumed_epoch_shuffles']==config['epochs']-event['from_epoch']
        assert event['adapter_source_sha256']==sha256(ROOT/'resume_bspline_shuffle.py')
        previous=run['path']/f"experiment_before_resume_epoch{event['from_epoch']:04d}.json"
        assert sha256(previous)==event['previous_experiment_sha256']
        events.append(dict(event=event,sha256=sha256(event_path)))
    if events:audit['resume_events']=events
    migration_path=run['path']/'gpu_migration.json'
    if migration_path.exists():
        migration=json.loads(migration_path.read_text())
        audit['migration']=migration
        audit['discarded_forward_backward_batches_min']=migration['discarded_completed_epochs']*batches
        audit['discarded_forward_backward_batches_upper_bound']=(migration['discarded_completed_epochs']+1)*batches
    return audit


def convergence_report(nominal,truth,truth_pixel,points):
    """Post-hoc evaluation of all saved checkpoints; no checkpoint selection."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    records={}
    meta=json.loads((BASE/'input/experiment.json').read_text());dv,du=meta['truth']['pixel_vu_mm']
    gt_uv=project_landmarks(truth_pixel,points)
    for label,path,epochs,color in SPECS:
        series=[]
        for saved in sorted(path.glob('P_epoch*.npy')):
            epoch=int(saved.stem.removeprefix('P_epoch'))
            if epoch>epochs:continue
            p=np.load(saved);camera=effective_parameters_from_pmat(p,nominal)
            error=camera['parameters_9']-truth['parameters_9']
            source=camera['source_xyz_mm']-truth['source_xyz_mm']
            uv=project_landmarks(pmat_to_pixel(p,du=du,dv=dv),points)-gt_uv
            series.append(dict(epoch=epoch,forward_backward_view_passes=epoch*(2 if 'Rigid' in label else 1),
                intrinsic_rms_mm=float(np.sqrt(np.mean(error[:,:3]**2))),
                translation_rms_mm=float(np.sqrt(np.mean(error[:,3:6]**2))),
                source_rms_mm=float(np.sqrt(np.mean(np.sum(source**2,axis=1)))),
                bead_rms_px=float(np.sqrt(np.mean(np.sum(uv**2,axis=-1))))))
        records[label]=series
        for ax,key,title in zip(axes.flat,
            ('intrinsic_rms_mm','translation_rms_mm','source_rms_mm','bead_rms_px'),
            ('K components RMS [mm]','Translation components RMS [mm]','Source RMS [mm]','Bead RMS [pixel]')):
            ax.plot([s['forward_backward_view_passes'] for s in series],[s[key] for s in series],
                    label=label,color=color,marker='.',lw=1.2)
            ax.set(xlabel='Retained forward/backward view passes',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Post-hoc GT audit of every saved checkpoint; final epochs were fixed before training\n'
                 'No selection by GT; alternating100 and joint200 both end at 200 view passes')
    fig.savefig(OUT/'convergence.png',dpi=160);plt.close(fig)
    return records


def block_coupling_diagnostic():
    """Evaluation-only GT-local point tangent, not the LNCC Hessian."""
    from calibration_gauge import compose_effective_parameters
    from spline_motion_model import cubic_bspline_basis
    folder=BASE/'input';meta=json.loads((folder/'experiment.json').read_text())
    dv,du=meta['truth']['pixel_vu_mm'];motion=np.load(folder/'spline_motion9.npy')
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    points=np.array([p['xyz_mm'] for p in json.loads((folder/'landmarks.json').read_text())['landmarks']])
    basis,_=cubic_bspline_basis(len(motion),20)
    columns=[]
    for k,h in enumerate([.0005]*6+[.000005]*3):
        plus=motion.copy();minus=motion.copy();plus[:,k]+=h;minus[:,k]-=h
        def project(values):
            p=pmat_to_pixel(compose_effective_parameters(nominal,values),du=du,dv=dv)
            return project_landmarks(p,points).reshape(len(motion),-1)
        columns.append((project(plus)-project(minus))/(2*h))
    jac=np.stack(columns,axis=-1)
    joint=np.einsum('vpi,vc->vpci',jac,basis).reshape(-1,180)
    ki=np.array([9*c+j for c in range(20) for j in range(3)])
    ri=np.array([9*c+j for c in range(20) for j in range(3,9)])
    qk=np.linalg.qr(joint[:,ki],mode='reduced')[0]
    qr=np.linalg.qr(joint[:,ri],mode='reduced')[0]
    cosines=np.linalg.svd(qk.T@qr,compute_uv=False)
    if cosines[0]>=1:raise ValueError('Point subspaces unexpectedly intersect numerically')
    rho=float(cosines[0]**2)
    return dict(model='GT-local known-point tangent dtheta = B20 dC; never used for optimization',
        maximum_subspace_correlation=float(cosines[0]),
        smallest_principal_angle_deg=float(np.rad2deg(np.arccos(cosines[0]))),
        ideal_exact_two_block_LS_spectral_radius=rho,
        worst_linear_mode_cycles_for_10x_reduction=float(np.log(.1)/np.log(rho)),
        derivation='Exact alternating linear least squares has K-error map inv(A.T A) A.T D inv(D.T D) D.T A; spectral radius equals sigma_max(QA.T QD)^2, A=JK*B, D=JR*B.',
        limitations='Hypothetical exact block solves for a noiseless local point quadratic; not the signed-LNCC Hessian or a prediction of Adam iteration counts. It demonstrates remaining smooth K-rigid coupling, not an exact gauge.')


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    folder=BASE/'input';meta=json.loads((folder/'experiment.json').read_text())
    hashes=validate_data(folder,meta);dv,du=meta['truth']['pixel_vu_mm']
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    truth_pixel=np.load(folder/'P_truth_pixel.npy')
    truth_p=pixel_to_pmat(truth_pixel,du=du,dv=dv,dtype=np.float64)
    truth=effective_parameters_from_pmat(truth_p,nominal)
    gt=np.load(folder/'spline_motion9.npy')
    np.testing.assert_allclose(truth['parameters_9'],gt,rtol=0,atol=1e-9)
    points=np.array([p['xyz_mm'] for p in json.loads((folder/'landmarks.json').read_text())['landmarks']])
    uv=project_landmarks(truth_pixel,points);rows,cols=meta['truth']['detector_shape_vu']
    visible=(uv[:,:,0]>=-.5)&(uv[:,:,0]<cols-.5)&(uv[:,:,1]>=-.5)&(uv[:,:,1]<rows-.5)
    assert visible.all()
    theta=np.load(folder/'truth_geometry.npz')['theta_deg']
    runs=[];summaries=[];arrays=[];checks=[];table=[];recipes=[]
    reference_initial=torch.load(SPECS[0][1]/'initial_model.pt',map_location='cpu',weights_only=True)
    for label,path,epochs,color in SPECS:
        run=load_run(path,meta,hashes,required_epochs=epochs);runs.append(run)
        summary,array=evaluate(run,truth,nominal,truth_pixel,points,visible,du,dv)
        summaries.append(summary);arrays.append(array)
        audit=checkpoint_audit(run,meta,points,reference_initial);audit['label']=label;checks.append(audit)
        recipe=run['recipe'].copy()
        for key in ('epochs','update_scheme'):recipe.pop(key,None)
        recipes.append(recipe)
        table.append(dict(label=label,epochs=epochs,
            forward_backward_batches=audit['forward_backward_batches'],
            intrinsic_rms_mm=summary['prior_groups']['intrinsic']['canonical_gt_error_component_rms'],
            translation_rms_mm=summary['prior_groups']['translation']['canonical_gt_error_component_rms'],
            rotation_rms_deg=summary['prior_groups']['rotation']['canonical_gt_error_component_rms'],
            source_rms_mm=summary['source_error_mm']['rms'],
            bead_rms_px=summary['bead_error_px']['rms'],
            worst_view_bead_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
            final_image_loss=summary['image_loss'],training_seconds=run['metrics']['training_seconds'],
            parameter_rms9=np.sqrt(np.mean(array['parameter_error9']**2,axis=0)).tolist(),
            parameter_r2=(1-np.sum(array['parameter_error9']**2,axis=0)/np.sum((gt-gt.mean(0))**2,axis=0)).tolist()))
    assert recipes[0]==recipes[1]==recipes[2],'Recipe changed beyond update scheme and epoch budget'
    assert checks[1]['forward_backward_batches']==checks[2]['forward_backward_batches']
    for run in runs:
        assert sha256(run['path']/'loss_roi.json')==run['recipe']['image_roi']['manifest_sha256']
        for name,digest in runs[0]['source_sha256'].items():
            if name!='run_sinespin_calibration.py':
                assert run['source_sha256'][name]==digest, name
    # No pose fit: only convert the known virtual-detector coordinate origin.
    pad_v,pad_u=meta['detector_padding_vu']
    shift=np.array([[1.,0.,-pad_u*du],[0.,1.,-pad_v*dv],[0.,0.,1.]])
    def physical(p):
        camera=decompose_physical_camera(shift@p)
        angles=Rotation.from_matrix(camera['Q_camera_to_physical']).as_euler('xyz',degrees=True)
        return np.c_[camera['intrinsics_f_cu_cv_mm'],camera['source_xyz_mm'],
                     np.rad2deg(np.unwrap(np.deg2rad(angles),axis=0))]
    for kind,truth_values,nominal_values,values,names in [
        ('canonical_parameters9',gt,np.zeros_like(gt),[a['parameters9'] for a in arrays],PARAMETER_NAMES),
        ('geometry_components9',physical(truth_p),physical(nominal),[physical(r['p']) for r in runs],
         ['f [mm]','cu [mm]','cv [mm]','Source x [mm]','Source y [mm]','Source z [mm]',
          'Camera x [degree]','Camera y [degree]','Camera z [degree]'])]:
        fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
        for j,ax in enumerate(axes.flat):
            ax.plot(theta,nominal_values[:,j],':',color='#b38b45',lw=.8,label='Nominal')
            for spec,value in zip(SPECS,values):ax.plot(theta,value[:,j],color=spec[3],lw=1.1,label=spec[0])
            ax.plot(theta,truth_values[:,j],'--',color='black',lw=1.2,label='GT')
            ax.set_title(names[j]);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if kind=='canonical_parameters9' and j<6:ax.set_ylim(-10,10)
            if j>=6:ax.set_xlabel('Scan angle [degree]')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=5)
        fig.suptitle('B20: simultaneous vs rigid-first / K-second updates; same zero initialization and data\n'
            'Alternating 100 and Joint 200 use equal retained forward/backward counts; no pose alignment')
        fig.savefig(OUT/(kind+'.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for spec,array,run in zip(SPECS,arrays,runs):
        axes[0,0].plot(theta,array['per_view_bead_rms_px'],label=spec[0],color=spec[3])
        axes[0,1].plot(theta,array['source_error_mm'],label=spec[0],color=spec[3])
        h=np.genfromtxt(run['path']/'loss_history.csv',delimiter=',',names=True)
        passes=h['epoch']*(2 if 'update_scheme' in run['recipe'] else 1)
        axes[1,0].plot(passes,h['image_loss'],color=spec[3],label=spec[0])
    h=np.genfromtxt(runs[-1]['path']/'loss_history.csv',delimiter=',',names=True)
    axes[1,1].plot(h['epoch'],h['rigid_pre_step_image_loss'],label='Before rigid step')
    axes[1,1].plot(h['epoch'],h['intrinsic_pre_step_image_loss'],label='After rigid / before K step')
    axes[0,0].set(xlabel='Scan angle [degree]',ylabel='Bead reprojection RMS [pixel]')
    axes[0,1].set(xlabel='Scan angle [degree]',ylabel='Source error [mm]')
    axes[1,0].set(xlabel='Retained forward/backward view passes',ylabel='Mean training signed LNCC31')
    axes[1,1].set(xlabel='Alternating epoch',ylabel='Mean block input loss')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend();axes[1,1].legend()
    fig.suptitle('Independent geometry errors and training objectives\n'
                 'Alternating history averages the two pre-step losses; final full-data losses are in the report table')
    fig.savefig(OUT/'geometry_errors.png',dpi=160);plt.close(fig)
    samples=[np.load(r['path']/'projection_examples.npz') for r in runs]
    for sample in samples[1:]:
        np.testing.assert_array_equal(sample['indices'],samples[0]['indices'])
        np.testing.assert_array_equal(sample['target'],samples[0]['target'])
    box=json.loads((runs[-1]['path']/'loss_roi.json').read_text())['boxes_xyxy'][0]
    x0,y0,x1,y1=box
    target=samples[-1]['target'][:,y0:y1,x0:x1]
    joint=samples[1]['optimized'][:,y0:y1,x0:x1]
    alt=samples[2]['optimized'][:,y0:y1,x0:x1]
    residual=alt-target;vmax=float(np.quantile(target,.999));rmax=float(np.quantile(abs(residual),.995))
    fig,axes=plt.subplots(len(target),4,figsize=(14,19),layout='constrained')
    for i,view in enumerate(samples[-1]['indices']):
        for j,picture in enumerate((target[i],joint[i],alt[i],residual[i])):
            im=axes[i,j].imshow(picture,origin='lower',cmap='RdBu_r' if j==3 else 'gray',
                vmin=-rmax if j==3 else 0,vmax=rmax if j==3 else vmax)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Poisson target','Joint 200','Rigid then K 100','Alternating minus target'][j])
        axes[i,0].set_ylabel(f'View {view}')
    fig.colorbar(im,ax=axes[:,3],shrink=.5,label='Line integral residual',extend='both')
    fig.suptitle('Same fixed all-bead ROI and display scales; actual predictions, no image registration')
    fig.savefig(OUT/'projection_fits.png',dpi=130);plt.close(fig)
    convergence=convergence_report(nominal,truth,truth_pixel,points)
    record=dict(table=table,runs=summaries,checks=checks,convergence=convergence,block_coupling=block_coupling_diagnostic(),identical_recipe_except_scheme_and_epochs=recipes[0],
        input_sha256=hashes,comparison='Joint100 reference; Joint200 vs alternating100 have 27400 retained forward/backward batches; GPU migration overhead is recorded separately',
        limitations=['One seed and fixed final epochs, not a learning-rate or block-step-count sweep.',
            'Each block takes one Adam step per existing 4-view batch; not a converged block solve.',
            'B20 curves are shared across views, not 546 independently fitted cameras.',
            'Alternating does not add observations or guarantee removal of K-rigid compensation.',
            'Same retained forward/backward count does not mean same per-parameter Adam step count or identical wall time. Joint200 GPU migration repeated 5 completed epochs and at most one partial epoch; it is logged separately.',
            'Original all-bead preview ROI used simulated bead boxes; no labels enter the training loss.'],
        report_source_sha256=sha256(Path(__file__)),resume_adapter_source_sha256=sha256(ROOT/'resume_bspline_shuffle.py'),loader_source_sha256=sha256(ROOT/'compare_sinespin_regularization.py'))
    (OUT/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez(OUT/'comparison.npz',theta_deg=theta,truth=gt,
        **{f'run{i}_{k}':v for i,a in enumerate(arrays) for k,v in a.items()})
    for name in ('canonical_parameters9','geometry_components9','geometry_errors','projection_fits','convergence'):
        shutil.copy2(OUT/(name+'.png'),ROOT/'docs'/('spline_alternating_'+name+'.png'))
    shutil.copy2(OUT/'comparison.json',ROOT/'docs/spline_alternating_comparison.json')
    print(json.dumps(dict(table=table,checks=checks),indent=2))


if __name__=='__main__':report()
