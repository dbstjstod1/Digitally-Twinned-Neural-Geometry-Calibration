"""Audit and compare the 20-control-point B-spline estimator with scale2 MLP.

GT fitting is an evaluation-only representation diagnostic, never initialization
or a training target. Both final estimators are evaluated at fixed epoch 100.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_gauge import PARAMETER_NAMES, compose_effective_parameters, effective_parameters_from_pmat
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from compare_sinespin_regularization import validate_data, load_run, evaluate
from denseball_landmarks import project_landmarks
from physical_camera import decompose_physical_camera
from run_sinespin_calibration import sha256
from spline_motion_model import cubic_bspline_basis, BSplineMotion9

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'result_spline9_scale2/ball_calibration'
OUT = BASE/'spline_basis_comparison'


def representation_diagnostic():
    folder = BASE/'input'
    meta = json.loads((folder/'experiment.json').read_text())
    gt = np.load(folder/'spline_motion9.npy')
    b, knots = cubic_bspline_basis(len(gt), 20)
    coefficients = np.linalg.lstsq(b, gt, rcond=None)[0]
    approx = b @ coefficients
    bounds = np.array([10.]*6+[15.]*3)
    if np.any(abs(coefficients) >= bounds):
        raise ValueError('Unbounded representation fit exceeds estimator coefficient bounds')
    dv, du = meta['truth']['pixel_vu_mm']
    nominal = pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    labels = json.loads((folder/'landmarks.json').read_text())
    xyz = np.array([p['xyz_mm'] for p in labels['landmarks']])
    p = compose_effective_parameters(nominal, approx)
    truth = np.load(folder/'P_truth_pixel.npy')
    errors = project_landmarks(pmat_to_pixel(p,du=du,dv=dv),xyz)-project_landmarks(truth,xyz)
    record = dict(
        description='Evaluation-only least-squares fit of GT parameter curves to the preselected B20 space; never fed to training',
        not_a_projection_error_lower_bound=True,
        gt_generation='8-knot natural cubic spline; estimator uses 20-control-point open-uniform clamped B-spline',
        knots_progress=knots.tolist(), parameter_rms_by_component=np.sqrt(np.mean((approx-gt)**2,axis=0)).tolist(),
        parameter_max_abs_by_component=np.max(abs(approx-gt),axis=0).tolist(),
        coefficient_max_abs_by_component=np.max(abs(coefficients),axis=0).tolist(),
        representation_bead_rms_px=float(np.sqrt(np.mean(np.sum(errors**2,axis=-1)))),
        coefficient_rank=int(np.linalg.matrix_rank(b)), basis_condition=float(np.linalg.cond(b)),
        source_sha256=sha256(Path(__file__)), estimator_source_sha256=sha256(ROOT/'spline_motion_model.py'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'representation_diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez(OUT/'representation_diagnostic.npz',basis=b,knots=knots,gt=gt,
             posthoc_gt_fit=approx,posthoc_gt_coefficients=coefficients)
    return record


def local_spline_uncertainty():
    """GT-local point model with perturbations restricted to B20, evaluation only.

    Linearizes at GT with theta = theta_GT + B dC, so this measures variance
    reduction from the subspace, not approximation bias or LNCC uncertainty.
    """
    folder=BASE/'input'
    meta=json.loads((folder/'experiment.json').read_text());dv,du=meta['truth']['pixel_vu_mm']
    gt=np.load(folder/'spline_motion9.npy');b,_=cubic_bspline_basis(len(gt),20)
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    points=np.array([p['xyz_mm'] for p in json.loads((folder/'landmarks.json').read_text())['landmarks']])
    columns=[]
    for k,h in enumerate([.0005]*6+[.000005]*3):
        plus=gt.copy();minus=gt.copy();plus[:,k]+=h;minus[:,k]-=h
        def project(m):
            return project_landmarks(pmat_to_pixel(compose_effective_parameters(nominal,m),du=du,dv=dv),points).reshape(len(gt),-1)
        columns.append((project(plus)-project(minus))/(2*h))
    jac=np.stack(columns,-1)
    joint=np.einsum('vpi,vc->vpci',jac,b).reshape(-1,180)
    # Marginalize all other coefficient components; physical mm/degree units.
    _,s,vh=np.linalg.svd(joint,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10))
    if rank!=180: raise ValueError('Spline point-Jacobian is rank deficient')
    cov=(vh.T/s**2)@vh*.1**2
    per_view_cov=np.einsum('vc,cidj,vd->vij',b,cov.reshape(20,9,20,9),b)
    sigma=np.sqrt(np.diagonal(per_view_cov,axis1=1,axis2=2))
    free_sigma=[]
    for j in jac:
        _,s,vh=np.linalg.svd(j,full_matrices=False)
        free_sigma.append(np.sqrt(np.diag((vh.T/s**2)@vh*.1**2)))
    free_sigma=np.array(free_sigma)
    record=dict(rank=rank,coefficient_count=180,
        point_coordinate_noise_std_px=.1,
        free_per_view_parameter_std_median=np.median(free_sigma,axis=0).tolist(),
        spline_parameter_std_median=np.median(sigma,axis=0).tolist(),
        tangent_model='theta = theta_GT + B20 dC; all 180 physical coefficients estimated jointly',
        limitations='Evaluation-only independent 0.1px Gaussian known-point noise. Ignores representation bias, nonlinear image loss, spatially correlated photon noise and optimizer error. Not LNCC error bars or a comparison against the MLP temporal prior.')
    (OUT/'local_uncertainty.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def report():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_sinespin_calibration import apply_motion

    representation = representation_diagnostic()
    uncertainty=local_spline_uncertainty()
    folder = BASE/'input'
    meta = json.loads((folder/'experiment.json').read_text())
    input_hashes = validate_data(folder,meta)
    dv,du = meta['truth']['pixel_vu_mm']
    nominal = pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    truth_pixel = np.load(folder/'P_truth_pixel.npy')
    truth_p = pixel_to_pmat(truth_pixel,du=du,dv=dv,dtype=np.float64)
    truth = effective_parameters_from_pmat(truth_p,nominal)
    gt = np.load(folder/'spline_motion9.npy')
    np.testing.assert_allclose(truth['parameters_9'],gt,atol=1e-9,rtol=0)
    points = np.array([p['xyz_mm'] for p in json.loads((folder/'landmarks.json').read_text())['landmarks']])
    uv = project_landmarks(truth_pixel,points)
    rows,cols = meta['truth']['detector_shape_vu']
    visible = (uv[:,:,0]>=-.5)&(uv[:,:,0]<cols-.5)&(uv[:,:,1]>=-.5)&(uv[:,:,1]<rows-.5)
    if not visible.all(): raise ValueError('All beads must be visible')
    theta = np.load(folder/'truth_geometry.npz')['theta_deg']
    specs = [('Hash MLP', BASE/'crop_lncc31_seed1','#777777'),
             ('B-spline 20',BASE/'bspline20_lncc31_seed1','#126bbb')]
    runs=[];arrays=[];summaries=[];table=[]
    for label,path,color in specs:
        run=load_run(path,meta,input_hashes)
        summary,array=evaluate(run,truth,nominal,truth_pixel,points,visible,du,dv)
        runs.append(run);summaries.append(summary);arrays.append(array)
        table.append(dict(run=label,
            intrinsic_rms_mm=summary['prior_groups']['intrinsic']['canonical_gt_error_component_rms'],
            translation_rms_mm=summary['prior_groups']['translation']['canonical_gt_error_component_rms'],
            rotation_rms_deg=summary['prior_groups']['rotation']['canonical_gt_error_component_rms'],
            source_rms_mm=summary['source_error_mm']['rms'],bead_rms_px=summary['bead_error_px']['rms'],
            worst_view_bead_rms_px=summary['per_view_bead_rms_px']['maximum_absolute'],
            final_image_loss=summary['image_loss'],training_seconds=run['metrics']['training_seconds'],
            parameter_rms9=np.sqrt(np.mean(array['parameter_error9']**2,axis=0)).tolist(),
            parameter_r2=(1-np.sum(array['parameter_error9']**2,axis=0)/np.sum((gt-gt.mean(0))**2,axis=0)).tolist()))
    # This comparison changes representation and initialization, not the data/loss.
    recipes=[]
    for run in runs:
        r=run['recipe'].copy()
        for k in ('model','motion_model_config','initialization'):r.pop(k,None)
        recipes.append(r)
    if recipes[0]!=recipes[1]:raise ValueError('Unexpected change beyond model/initialization')
    for name,digest in runs[0]['source_sha256'].items():
        if name!='run_sinespin_calibration.py' and runs[1]['source_sha256'].get(name)!=digest:
            raise ValueError(f'Core training code changed: {name}')
    for run in runs:
        if sha256(run['path']/'loss_roi.json')!=run['recipe']['image_roi']['manifest_sha256']:
            raise ValueError('Changed archived ROI')
    run=runs[1];bounds=run['recipe']['bounds']
    model=BSplineMotion9(len(gt),20,**bounds)
    checkpoint=torch.load(run['path']/'checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    initial=torch.load(run['path']/'initial_model.pt',map_location='cpu',weights_only=True)
    if torch.count_nonzero(initial['raw_coefficients']).item()!=0:raise ValueError('Expected nominal initialization')
    if np.count_nonzero(np.load(run['path']/'initial_motion9.npy')):raise ValueError('Initial motion is not zero')
    archive=np.load(run['path']/'spline_coefficients.npz')
    np.testing.assert_array_equal(archive['raw_coefficients'],model.raw_coefficients.detach().numpy())
    np.testing.assert_array_equal(archive['basis'],model.basis.numpy())
    np.testing.assert_allclose(archive['basis']@archive['physical_coefficients'],run['motion'],atol=1e-6,rtol=1e-6)
    with torch.no_grad():
        recovered_p,recovered_motion=apply_motion(torch.from_numpy(np.load(folder/'P_nominal_world_mm.npy')),
            model(torch.arange(len(gt))),bounds,physical=True,
            shape=tuple(meta['volume']['shape_zyx']),voxel=meta['volume']['voxel_mm'])
    np.testing.assert_allclose(recovered_motion.numpy(),run['motion'],atol=2e-6,rtol=1e-6)
    pixel=pmat_to_pixel(recovered_p.numpy(),du=du,dv=dv)
    delta=project_landmarks(pixel,points)-project_landmarks(np.load(run['path']/'P_optimized_pixel.npy'),points)
    ray_restore=float(np.max(abs(delta)))
    if ray_restore>1e-3:raise ValueError('Restored CPU checkpoint rays differ from saved GPU rays')
    # Convert detector origin only; never align poses to GT.
    pad_v,pad_u=meta.get('detector_padding_vu',[0,0])
    shift=np.array([[1.,0.,-pad_u*du],[0.,1.,-pad_v*dv],[0.,0.,1.]])
    def physical(p):
        c=decompose_physical_camera(shift@p)
        angles=Rotation.from_matrix(c['Q_camera_to_physical']).as_euler('xyz',degrees=True)
        return np.c_[c['intrinsics_f_cu_cv_mm'],c['source_xyz_mm'],np.rad2deg(np.unwrap(np.deg2rad(angles),axis=0))]
    for name,truth_values,nom_values,values,titles in (
        ('canonical_parameters9',gt,np.zeros_like(gt),[a['parameters9'] for a in arrays],PARAMETER_NAMES),
        ('geometry_components9',physical(truth_p),physical(nominal),[physical(r['p']) for r in runs],
         ['f [mm]','cu [mm]','cv [mm]','Source x [mm]','Source y [mm]','Source z [mm]',
          'Camera x [degree]','Camera y [degree]','Camera z [degree]'])):
        fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
        for j,ax in enumerate(axes.flat):
            ax.plot(theta,nom_values[:,j],':',color='#bf7b22',label='Nominal')
            for spec,value in zip(specs,values):ax.plot(theta,value[:,j],color=spec[2],lw=1.2,label=spec[0])
            ax.plot(theta,truth_values[:,j],'--',color='black',lw=1.2,label='GT')
            ax.set_title(titles[j]);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if name=='canonical_parameters9' and j<6:ax.set_ylim(-10,10)
            if j>=6:ax.set_xlabel('Scan angle [degree]')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=4)
        fig.suptitle('B-spline coefficient estimation vs hash MLP: same scale2 projections / ROI / signed LNCC31\n'
                     'Fixed epoch 100; B20 starts at nominal, MLP uses vanilla initialization; no GT alignment')
        fig.savefig(OUT/(name+'.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for spec,array,run in zip(specs,arrays,runs):
        axes[0,0].plot(theta,array['per_view_bead_rms_px'],label=spec[0],color=spec[2])
        axes[0,1].plot(theta,array['source_error_mm'],label=spec[0],color=spec[2])
        history=np.genfromtxt(run['path']/'loss_history.csv',delimiter=',',names=True)
        axes[1,0].plot(history['epoch'],history['image_loss'],label=spec[0],color=spec[2])
    axes[0,0].set(xlabel='Scan angle [degree]',ylabel='Bead reprojection RMS [pixel]')
    axes[0,1].set(xlabel='Scan angle [degree]',ylabel='Source error [mm]')
    axes[1,0].set(xlabel='Epoch',ylabel='Training image loss: signed LNCC31')
    axes[1,1].plot(theta,archive['basis'],lw=.8)
    axes[1,1].set(xlabel='Scan angle [degree]',ylabel='Basis weight',title='20 fixed cubic basis functions')
    [a.grid(alpha=.2) for a in axes.flat];axes[0,0].legend()
    fig.suptitle('Same enlarged phantom; only model and initialization differ')
    fig.savefig(OUT/'geometry_errors.png',dpi=160);plt.close(fig)
    examples=np.load(run['path']/'projection_examples.npz')
    x0,y0,x1,y1=np.array(json.loads((run['path']/'loss_roi.json').read_text())['boxes_xyxy'])[0]
    target=examples['target'][:,y0:y1,x0:x1];pred=examples['optimized'][:,y0:y1,x0:x1]
    residual=pred-target;vmax=float(np.quantile(target,.999));rmax=float(np.quantile(abs(residual),.995))
    fig,axes=plt.subplots(len(target),3,figsize=(11,19),layout='constrained')
    for i,view in enumerate(examples['indices']):
        for j,im in enumerate((target[i],pred[i],residual[i])):
            image=axes[i,j].imshow(im,origin='lower',cmap='RdBu_r' if j==2 else 'gray',
                vmin=-rmax if j==2 else 0,vmax=rmax if j==2 else vmax)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Poisson target','B20 estimate','Estimate minus target'][j])
        axes[i,0].set_ylabel(f'View {view}')
    fig.colorbar(image,ax=axes[:,2],shrink=.5,label='Line integral residual')
    fig.suptitle('B-spline 20: actual fits in fixed all-bead ROI; no image alignment or intensity rescaling')
    fig.savefig(OUT/'projection_fits.png',dpi=130);plt.close(fig)
    data=dict(table=table,runs=summaries,representation=representation,local_uncertainty=uncertainty,
        checks=dict(same_input=True,same_recipe_except_model_initialization=True,
                    same_projector_transform_loss=True,checkpoint_CPU_ray_max_abs_difference_px=ray_restore,
                    initial_spline_coefficients_zero=True,initialization_same_as_MLP=False,
                    checkpoint_coefficient_and_motion_consistency=True),
        spline_config=run['recipe']['motion_model_config'],input_sha256=input_hashes,
        coefficient_archive_sha256=sha256(run['path']/'spline_coefficients.npz'),
        report_source_sha256=sha256(Path(__file__)),
        limitations=['One seed and fixed epoch 100; not an optimizer or control-count sweep.',
            'MLP uses vanilla random initialization, B-spline starts at nominal; not a pure representation ablation.',
            'Smooth K and rigid curves may still compensate; no guarantee of identifiability from a spline prior.',
            'Both models use exactly the same Poisson projections and frozen all-bead ROI; original ROI preview used simulated bead boxes.',
            'Representation diagnostic uses GT separately from training; it is not an initializer, training loss or choice of knots.'])
    (OUT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    np.savez(OUT/'comparison.npz',theta_deg=theta,truth=gt,
             **{f'run{i}_{k}':v for i,a in enumerate(arrays) for k,v in a.items()})
    for source,target in [('comparison.json','spline_basis20_comparison.json'),
                          ('canonical_parameters9.png','spline_basis20_canonical_parameters9.png'),
                          ('geometry_components9.png','spline_basis20_geometry_components9.png'),
                          ('geometry_errors.png','spline_basis20_geometry_errors.png'),
                          ('projection_fits.png','spline_basis20_projection_fits.png')]:
        shutil.copy2(OUT/source,ROOT/'docs'/target)
    print(json.dumps(dict(table=table,representation=representation,checks=data['checks']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-only',action='store_true')
    args=parser.parse_args()
    if args.diagnostic_only:
        print(json.dumps(dict(representation=representation_diagnostic(),uncertainty=local_spline_uncertainty()),indent=2))
    else: report()
