"""SineSpin nominal geocal: free K splines vs shared K and LR decay.

All GT metrics are post-hoc. Configuration ranking uses only held-out
observed-image signed LNCC at the fixed final epoch.
"""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from calibration_gauge import effective_parameters_from_pmat, PARAMETER_NAMES
from physical_camera import decompose_physical_camera
from denseball_landmarks import project_landmarks
from compare_sinespin_regularization import validate_data, load_run, evaluate
from spline_motion_model import BSplineMotion9, SharedIntrinsicBSplineMotion9
from run_sinespin_calibration import sha256, apply_motion

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'result_sinespin_geocal'
OUT=BASE/'comparison'
SPECS=[('Per-view K / fixed LR','per_view_fixed','#777777'),
       ('Shared K / fixed LR','shared_fixed','#dd8230'),
       ('Shared K / cosine LR','shared_cosine','#126bbb')]
BOUNDS=dict(ts_max_mm=3.,tp_max_mm=3.,rot_max_deg=1.)


def audit(run,meta,points):
    path=run['path'];shared=run['recipe']['motion_model_config']['kind']=='shared_intrinsic_cubic_bspline'
    model=(SharedIntrinsicBSplineMotion9 if shared else BSplineMotion9)(meta['truth']['views'],20,**BOUNDS)
    initial=torch.load(path/'initial_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(initial)
    assert torch.count_nonzero(model.raw_coefficients).item()==0
    assert np.count_nonzero(np.load(path/'initial_motion9.npy'))==0
    checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    archive=np.load(path/'spline_coefficients.npz')
    np.testing.assert_array_equal(archive['raw_coefficients'],model.raw_coefficients.detach().numpy())
    np.testing.assert_array_equal(archive['basis'],model.basis.numpy())
    np.testing.assert_array_equal(archive['scales'],model.scales.numpy())
    np.testing.assert_allclose(archive['physical_coefficients'],model.physical_coefficients().detach().numpy(),atol=2e-6,rtol=1e-6)
    with torch.no_grad():
        p,m=apply_motion(torch.from_numpy(np.load(BASE/'input/P_nominal_world_mm.npy')),
            model(torch.arange(meta['truth']['views'])),BOUNDS,physical=True,
            shape=tuple(meta['volume']['shape_zyx']),voxel=meta['volume']['voxel_mm'])
    np.testing.assert_allclose(m.numpy(),run['motion'],atol=2e-6,rtol=1e-6)
    assert np.all(abs(run['motion'])<=model.scales.numpy()+1e-6)
    if shared:np.testing.assert_array_equal(run['motion'][:,:3],np.tile(run['motion'][0,:3],(len(m),1)))
    dv,du=meta['truth']['pixel_vu_mm']
    delta=project_landmarks(pmat_to_pixel(p.numpy(),du=du,dv=dv),points)-project_landmarks(np.load(path/'P_optimized_pixel.npy'),points)
    ray=float(abs(delta).max());assert ray<1e-3
    h=np.genfromtxt(path/'loss_history.csv',delimiter=',',names=True)
    if 'lr_schedule' in run['recipe']:
        expected=.001*(.05+.95*(1+np.cos(np.pi*(h['epoch']-1)/99))/2)
        np.testing.assert_allclose(h['learning_rate'],expected,rtol=1e-12,atol=1e-16)
    return dict(zero_nominal_initialization=True,parameter_count=sum(p.numel() for p in model.parameters()),
        shared_K=shared,CPU_checkpoint_ray_max_abs_difference_px=ray,
        motion_max_abs9=abs(run['motion']).max(0).tolist(),
        coefficient_fraction_above_95pct_bound9=np.mean(abs(archive['physical_coefficients'])>=.95*archive['scales'],axis=0).tolist(),
        full_epoch_history_verified=bool(np.array_equal(h['epoch'],np.arange(1,101))),
        coefficient_archive_sha256=sha256(path/'spline_coefficients.npz'))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    folder=BASE/'input';meta=json.loads((folder/'experiment.json').read_text())
    hashes=validate_data(folder,meta)
    assert meta['nominal']['kind']=='sinespin'
    assert meta['nominal']['tilt_min_max_deg'][1]>9.9
    for name,digest in meta['preparation_source_sha256'].items():
        assert sha256(folder/'preparation_sources'/name)==digest
    dv,du=meta['truth']['pixel_vu_mm']
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    truth_pixel=np.load(folder/'P_truth_pixel.npy')
    truth_p=pixel_to_pmat(truth_pixel,du=du,dv=dv,dtype=np.float64)
    gt=effective_parameters_from_pmat(truth_p,nominal)
    np.testing.assert_allclose(gt['parameters_9'],np.load(folder/'spline_motion9.npy'),atol=1e-9)
    points=np.array([p['xyz_mm'] for p in json.loads((folder/'landmarks.json').read_text())['landmarks']])
    uv=project_landmarks(truth_pixel,points);rows,cols=meta['truth']['detector_shape_vu']
    visible=(uv[:,:,0]>=-.5)&(uv[:,:,0]<cols-.5)&(uv[:,:,1]>=-.5)&(uv[:,:,1]<rows-.5)
    assert visible.all()
    theta=np.load(folder/'nominal_geometry.npz')['theta_deg']
    runs=[];summaries=[];arrays=[];checks=[];table=[];recipes=[]
    for label,name,color in SPECS:
        r=load_run(BASE/name,meta,hashes,required_epochs=100,required_bounds=BOUNDS)
        assert not r['recipe']['regularization']['active']
        summary,array=evaluate(r,gt,nominal,truth_pixel,points,visible,du,dv)
        check=audit(r,meta,points)
        runs.append(r);summaries.append(summary);arrays.append(array);checks.append(check)
        recipe=copy.deepcopy(r['recipe'])
        for key in ('model','motion_model_config','lr_schedule'):recipe.pop(key,None)
        recipes.append(recipe)
        result=r['metrics']['results']['optimized']
        split=result['image_loss_by_split']
        train=r['experiment']['train_views'];held=r['experiment']['heldout_views']
        assert train==list(range(0,len(theta),2)) and held==list(range(1,len(theta),2))
        np.testing.assert_allclose(split['all'],sum(split[k]*len(ids) for k,ids in [('train',train),('heldout',held)])/len(theta),atol=1e-12)
        np.testing.assert_allclose(split['all'],summary['image_loss'],atol=1e-7)
        assert sha256(r['path']/'loss_roi.json')==r['recipe']['image_roi']['manifest_sha256']
        table.append(dict(label=label,run=name,train_image_loss=split['train'],heldout_image_loss=split['heldout'],
            all_image_loss=summary['image_loss'],
            intrinsic_rms_mm=summary['prior_groups']['intrinsic']['canonical_gt_error_component_rms'],
            translation_rms_mm=summary['prior_groups']['translation']['canonical_gt_error_component_rms'],
            rotation_rms_deg=summary['prior_groups']['rotation']['canonical_gt_error_component_rms'],
            source_rms_mm=summary['source_error_mm']['rms'],bead_rms_px=summary['bead_error_px']['rms'],
            heldout_bead_rms_px=float(np.sqrt(np.mean(array['bead_error_px'][held]**2))),
            worst_view_bead_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
            parameter_rms9=np.sqrt(np.mean(array['parameter_error9']**2,axis=0)).tolist(),
            parameter_mean9=array['parameters9'].mean(0).tolist()))
    assert recipes[0]==recipes[1]==recipes[2]
    for r in runs[1:]:
        assert r['source_sha256']==runs[0]['source_sha256']
        for name in ('P_initial_world_mm.npy','initial_motion9.npy'):
            np.testing.assert_array_equal(np.load(r['path']/name),np.load(runs[0]['path']/name))
    selected=int(np.argmin([t['heldout_image_loss'] for t in table]))
    pv,pu=meta['detector_padding_vu']
    shift=np.array([[1.,0.,-pu*du],[0.,1.,-pv*dv],[0.,0.,1.]])
    def physical(p):
        c=decompose_physical_camera(shift@p)
        rot=Rotation.from_matrix(c['Q_camera_to_physical']).as_euler('xyz',degrees=True)
        return np.c_[c['intrinsics_f_cu_cv_mm'],c['source_xyz_mm'],np.rad2deg(np.unwrap(np.deg2rad(rot),axis=0))]
    pn,pt=physical(nominal),physical(truth_p);values=[physical(r['p']) for r in runs]
    physical_names=['f [mm]','cu [mm]','cv [mm]','Source x [mm]','Source y [mm]','Source z [mm]',
                    'Camera x [degree]','Camera y [degree]','Camera z [degree]']
    for kind,truth_value,nominal_value,estimated,names in [
        ('geometry_components9',pt,pn,values,physical_names),
        ('corrections9',gt['parameters_9'],np.zeros_like(gt['parameters_9']),[a['parameters9'] for a in arrays],PARAMETER_NAMES)]:
        fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
        for j,ax in enumerate(axes.flat):
            ax.plot(theta,nominal_value[:,j],':',color='#ab873e',lw=.8,label='Nominal sineSpin')
            for spec,value in zip(SPECS,estimated):ax.plot(theta,value[:,j],color=spec[2],lw=1.2,label=spec[0])
            ax.plot(theta,truth_value[:,j],'--',color='black',lw=1.3,label='GT')
            ax.set_title(names[j]);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if j>=6:ax.set_xlabel('Scan angle [degree]')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,fontsize=9)
        fig.suptitle('SineSpin nominal + small calibration errors; fixed final epoch 100\n'
                     '273 training / 273 held-out views; single-scale signed LNCC31; no pose alignment')
        fig.savefig(OUT/(kind+'.png'),dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(15,6))
    fig.subplots_adjust(left=.03,right=.98,bottom=.12,top=.95,wspace=.35)
    ax=fig.add_subplot(121,projection='3d');err=fig.add_subplot(122)
    for label,v,color,ls in [('Nominal sineSpin',pn,'#ab873e',':'),('GT',pt,'black','--')]+[(s[0],v,s[2],'-') for s,v in zip(SPECS,values)]:
        ax.plot(v[:,3],v[:,4],v[:,5],label=label,color=color,ls=ls,lw=1.2)
    ax.set(xlabel='Source x [mm]',ylabel='Source y [mm]',zlabel='Source z [mm]')
    ax.set_box_aspect(np.ptp(pt[:,3:6],axis=0));ax.legend(fontsize=7)
    nominal_error=np.linalg.norm(pn[:,3:6]-pt[:,3:6],axis=1)
    err.plot(theta,nominal_error,':',color='#ab873e',label='Nominal sineSpin')
    for s,a in zip(SPECS,arrays):err.plot(theta,a['source_error_mm'],color=s[2],label=s[0])
    err.set(xlabel='Scan angle [degree]',ylabel='Source distance to GT [mm]');err.grid(alpha=.2);err.legend(fontsize=8)
    fig.savefig(OUT/'source_trajectory.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');convergence={}
    for spec,r,a in zip(SPECS,runs,arrays):
        h=np.genfromtxt(r['path']/'loss_history.csv',delimiter=',',names=True)
        axes[0,0].plot(h['epoch'],h['image_loss'],color=spec[2],label=spec[0])
        axes[0,1].plot(theta,a['per_view_bead_rms_px'],color=spec[2],label=spec[0])
        series=[]
        for path in sorted(r['path'].glob('P_epoch*.npy')):
            epoch=int(path.stem.removeprefix('P_epoch'))
            c=effective_parameters_from_pmat(np.load(path),nominal)
            e=c['parameters_9']-gt['parameters_9']
            series.append(dict(epoch=epoch,K_rms_mm=float(np.sqrt(np.mean(e[:,:3]**2))),
                source_rms_mm=float(np.sqrt(np.mean(np.sum((c['source_xyz_mm']-gt['source_xyz_mm'])**2,axis=1))))))
        convergence[spec[1]]=series
        axes[1,0].plot([x['epoch'] for x in series],[x['K_rms_mm'] for x in series],color=spec[2],label=spec[0])
        axes[1,1].plot([x['epoch'] for x in series],[x['source_rms_mm'] for x in series],color=spec[2],label=spec[0])
    for ax,x,y in zip(axes.flat,['Epoch','Scan angle [degree]','Epoch','Epoch'],
        ['Training signed LNCC31','Bead reprojection RMS [pixel]','K component RMS [mm]','Source RMS [mm]']):
        ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('GT checkpoint curves are post-hoc; model selection uses held-out images at epoch 100')
    fig.savefig(OUT/'convergence.png',dpi=160);plt.close(fig)
    sample=np.load(runs[selected]['path']/'projection_examples.npz')
    boxes=np.array(json.loads((folder/'loss_roi.json').read_text())['boxes_xyxy'])
    cropped={k:np.stack([sample[k][i,boxes[v,1]:boxes[v,3],boxes[v,0]:boxes[v,2]]
                         for i,v in enumerate(sample['indices'])]) for k in ['target','nominal','optimized']}
    residual=cropped['optimized']-cropped['target'];vmax=float(np.quantile(cropped['target'],.999));rmax=float(np.quantile(abs(residual),.995))
    fig,axes=plt.subplots(len(sample['indices']),4,figsize=(14,18),layout='constrained')
    for i,view in enumerate(sample['indices']):
        for j,key in enumerate(['target','nominal','optimized','residual']):
            im=axes[i,j].imshow(residual[i] if j==3 else cropped[key][i],origin='lower',
                cmap='RdBu_r' if j==3 else 'gray',vmin=-rmax if j==3 else 0,vmax=rmax if j==3 else vmax)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Poisson observation','Nominal sineSpin',SPECS[selected][0],'Estimated minus observed'][j])
        axes[i,0].set_ylabel(f'View {view}')
    fig.colorbar(im,ax=axes[:,3],shrink=.5,label='Line integral residual',extend='both')
    fig.suptitle('Selected by held-out image loss; fixed ROI; no image registration for display')
    fig.savefig(OUT/'projection_fits.png',dpi=130);plt.close(fig)
    record=dict(table=table,runs=summaries,checks=checks,convergence=convergence,
        selection=dict(rule='Minimum heldout observed signed LNCC31 at fixed epoch 100; GT never used for ranking',
            selected=SPECS[selected][1],heldout_is_validation_not_external_test=True),
        nominal_metrics=runs[0]['metrics']['results']['nominal'],oracle_metrics=runs[0]['metrics']['results']['oracle'],
        nominal_source_rms_mm=float(np.sqrt(np.mean(nominal_error**2))),input_sha256=hashes,
        common_recipe_except_model_and_schedule=recipes[0],source_sha256=runs[0]['source_sha256'],
        all_35_beads_visible_all_views=True,
        limitations=['Assumed small residual errors, not measured ARTIS icono tolerances. SOD/SDD 750/1200 mm are assumptions.',
            'Primary simulation K is constant; shared K is a matched structural prior, not a demonstration of arbitrary K drift recovery.',
            'Enlarged .4mm phantom uses virtual detector 1084x1038, not actual device footprint.',
            'Same voxel object and Joseph family in independent generation and fitting; noise is Poisson only.',
            'Odd views are validation for configuration ranking, not an independent external final test.',
            'Single geometry/noise/optimization seed; no claim of universal optimizer improvement.'],
        report_source_sha256=sha256(Path(__file__)),loader_source_sha256=sha256(ROOT/'compare_sinespin_regularization.py'))
    (OUT/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez(OUT/'comparison.npz',theta_deg=theta,truth=gt['parameters_9'],nominal_physical=pn,truth_physical=pt,
             **{f'run{i}_{k}':v for i,a in enumerate(arrays) for k,v in a.items()})
    for kind in ['geometry_components9','corrections9','source_trajectory','convergence','projection_fits']:
        shutil.copy2(OUT/(kind+'.png'),ROOT/'docs'/('sinespin_geocal_'+kind+'.png'))
    shutil.copy2(OUT/'comparison.json',ROOT/'docs/sinespin_geocal_comparison.json')
    print(json.dumps(dict(table=table,selection=record['selection'],checks=checks),indent=2))


if __name__=='__main__':report()
