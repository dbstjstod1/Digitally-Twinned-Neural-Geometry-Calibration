"""Audit and plot recovery of nine independently prescribed spline GT curves."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from calibration_gauge import PARAMETER_NAMES, effective_parameters_from_pmat, compose_effective_parameters
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from denseball_landmarks import project_landmarks
from compare_sinespin_regularization import sha256, validate_data
from compare_sinespin_regularization_sweep import compare_sweep

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'result_spline9/ball_calibration'


def verify_truth(folder):
    acquisition=json.loads((folder/'experiment.json').read_text())
    if acquisition['truth']['kind']!='spline9':
        raise ValueError('This report requires spline9 acquisition labels')
    validate_data(folder, acquisition)
    for name, expected in acquisition['preparation_source_sha256'].items():
        if sha256(folder/'preparation_sources'/name)!=expected:
            raise ValueError(f'Preparation source changed: {name}')
    if sha256(folder/'spline_motion9.npy')!=acquisition['spline_motion_sha256']:
        raise ValueError('Spline labels changed')
    motion=np.load(folder/'spline_motion9.npy')
    dv,du=acquisition['truth']['pixel_vu_mm']
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    truth=pixel_to_pmat(np.load(folder/'P_truth_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    decomposed=effective_parameters_from_pmat(truth,nominal)['parameters_9']
    np.testing.assert_allclose(motion,decomposed,atol=1e-9,rtol=0)
    return acquisition,motion,np.load(folder/'truth_geometry.npz')['theta_deg']


def preview(folder,out,acquisition,motion,theta):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,3,figsize=(14,9),sharex=True,layout='constrained')
    spline=acquisition['truth']['spline']
    knots=theta[0]+np.asarray(spline['knot_progress'])*(theta[-1]-theta[0])
    values=np.asarray(spline['knot_values_applied'])
    for j,ax in enumerate(axs.flat):
        ax.plot(theta,motion[:,j],color='black',label='GT natural cubic spline')
        ax.scatter(knots,values[:,j],s=16,color='#db7b20',label='Prescribed knots')
        ax.axhline(0,lw=.6,color='gray');ax.grid(alpha=.2)
        ax.set_title(PARAMETER_NAMES[j]);ax.set_ylabel('mm' if j<6 else 'degree')
        if j>=6:ax.set_xlabel('Nominal scan angle [degree]')
    axs.flat[0].legend(fontsize=8)
    fig.suptitle('Prescribed independent GT in all nine effective components\nCircular nominal; knots/GT are never inputs to the estimator')
    fig.savefig(out/'gt_spline9.png',dpi=160);plt.close(fig)
    with np.load(folder/'projection_probe.npz') as data:
        chosen=np.linspace(0,len(data['indices'])-1,4).round().astype(int)
        clean=data['target'][chosen];noisy=data['noisy_target'][chosen];nominal=data['nominal'][chosen]
        indices=data['indices'][chosen]
    vmax=float(np.quantile(clean,.999));resmax=max(float(np.quantile(abs(nominal-clean),.995)),1e-5)
    fig,axs=plt.subplots(4,4,figsize=(15,12),layout='constrained')
    for i,index in enumerate(indices):
        for j,array in enumerate((clean[i],noisy[i],nominal[i],nominal[i]-clean[i])):
            axs[i,j].imshow(array,origin='lower',cmap='RdBu_r' if j==3 else 'gray',
                            vmin=-resmax if j==3 else 0,vmax=resmax if j==3 else vmax)
            axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            if i==0:axs[i,j].set_title(['GT clean','Poisson target (I0=44,000)','Circular nominal','Nominal minus GT clean'][j])
        axs[i,0].set_ylabel(f'View {index}\n{theta[index]:.1f} degrees')
    fig.suptitle('Actual full-detector simulated bead projections; shared image/residual scales')
    fig.savefig(out/'projection_inputs.png',dpi=140);plt.close(fig)


def recovery_report(folder,baseline,regularized,out):
    import matplotlib.pyplot as plt
    acquisition,motion,theta=verify_truth(folder)
    report=compare_sweep(baseline,[regularized],folder,out)
    if report['runs'][0]['training_source_sha256']!=report['runs'][1]['training_source_sha256']:
        raise ValueError('Paired runs must use identical complete archived training sources')
    preview(folder,out,acquisition,motion,theta)
    arrays=np.load(out/'regularization_sweep.npz')
    np.testing.assert_allclose(arrays['truth_parameters9'],motion,atol=1e-9,rtol=0)
    # Evaluation-only counterfactuals expose cancellation between parameter groups.
    dv,du=acquisition['truth']['pixel_vu_mm']
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    labels=json.loads((folder/'landmarks.json').read_text())
    points=np.array([bead['xyz_mm'] for bead in labels['landmarks']])
    truth_uv=project_landmarks(np.load(folder/'P_truth_pixel.npy'),points)
    cancellation=[]
    for i in range(2):
        errors={}
        for name,group in (('intrinsic_only',slice(0,3)),('translation_only',slice(3,6)),
                           ('rotation_only',slice(6,9)),('all_together',slice(0,9))):
            hybrid=motion.copy();hybrid[:,group]=arrays[f'run{i}_parameters9'][:,group]
            pixel=pmat_to_pixel(compose_effective_parameters(nominal,hybrid),du=du,dv=dv)
            errors[name]=float(np.sqrt(np.mean(np.sum((project_landmarks(pixel,points)-truth_uv)**2,axis=-1))))
        np.testing.assert_allclose(errors['all_together'],report['table'][i]['bead_rms_px'],atol=1e-4,rtol=0)
        cancellation.append(errors)
    parameter_rows=[]
    for j,name in enumerate(PARAMETER_NAMES):
        row=dict(parameter=name,unit='mm' if j<6 else 'degree',gt_rms=float(np.sqrt(np.mean(motion[:,j]**2))))
        variance=np.sum((motion[:,j]-motion[:,j].mean())**2)
        for i,label in enumerate(('unregularized','intrinsic001')):
            prediction=arrays[f'run{i}_parameters9'][:,j]
            error=arrays[f'run{i}_parameter_error9'][:,j]
            row[label+'_rmse']=float(np.sqrt(np.mean(error**2)))
            row[label+'_r2']=float(1-np.sum(error**2)/variance)
            row[label+'_pearson_r']=float(np.corrcoef(motion[:,j],prediction)[0,1])
            row[label+'_max_abs_error']=float(np.max(np.abs(error)))
        parameter_rows.append(row)
    with (out/'parameter_recovery.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(parameter_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(parameter_rows)
    # Show saved final predictions, without another fit or GT-based view selection.
    examples=[dict(np.load(path/'projection_examples.npz')) for path in (baseline,regularized)]
    np.testing.assert_array_equal(examples[0]['indices'],examples[1]['indices'])
    np.testing.assert_array_equal(examples[0]['target'],examples[1]['target'])
    target=examples[0]['target'];pred=[e['optimized'] for e in examples]
    vmax=float(np.quantile(target,.999))
    resmax=max(float(np.quantile(np.concatenate([abs(p-target).ravel() for p in pred]),.995)),1e-5)
    fig,axs=plt.subplots(len(target),5,figsize=(17,3*len(target)),layout='constrained')
    for i,index in enumerate(examples[0]['indices']):
        for j,array in enumerate((target[i],pred[0][i],pred[1][i],pred[0][i]-target[i],pred[1][i]-target[i])):
            im=axs[i,j].imshow(array,origin='lower',cmap='RdBu_r' if j>=3 else 'gray',
                              vmin=-resmax if j>=3 else 0,vmax=resmax if j>=3 else vmax)
            axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            if i==0:axs[i,j].set_title(['Poisson target','No prior','Intrinsic lambda=0.01','No prior minus target','Prior minus target'][j])
        axs[i,0].set_ylabel(f'View {index}\n{theta[index]:.1f} degrees')
    fig.colorbar(im,ax=axs[:,-2:],shrink=.5,label='Line integral residual')
    fig.suptitle('Final-epoch fits, full detector, shared intensity and residual scales\nSix evenly spaced views; no image alignment, masking, or contrast refitting')
    fig.savefig(out/'projection_fits.png',dpi=140);plt.close(fig)
    # Physical source trajectory is derived from the complete fitted matrix.
    from calibration_geometry import centered_source_positions
    truth_source=centered_source_positions(np.load(folder/'P_truth_world_mm.npy'))
    fig=plt.figure(figsize=(12,8),layout='constrained');ax=fig.add_subplot(projection='3d')
    for values,color,label in ((truth_source,'black','GT spline'),
                              (arrays['run0_source_xyz_mm'],'#2469b2','No prior'),
                              (arrays['run1_source_xyz_mm'],'#8a45b7','Intrinsic lambda=0.01')):
        ax.plot(*values.T,color=color,lw=1.1,label=label)
    ax.set(xlabel='Physical x [mm]',ylabel='Physical y [mm]',zlabel='Physical z [mm]')
    ax.set_box_aspect((1.,1.,.6))
    ax.legend();fig.suptitle('Physical source trajectories from complete P; no pose alignment\nUnequal visual axis scales to show the small longitudinal motion; all coordinates in mm')
    fig.savefig(out/'source_trajectory3d.png',dpi=160);plt.close(fig)
    report['scope']='Recovery of independent spline GT in every effective parameter; circular nominal; fixed seed1/epoch100'
    report['acquisition']=acquisition
    report['parameter_recovery']=parameter_rows
    report['group_cancellation_diagnostic']=dict(
        definition='Replace only the named GT group with its fitted values, keep other groups at GT, and measure fixed-ID bead RMS in pixels. Evaluation-only hybrid cameras; not additional fitted results.',
        runs=cancellation)
    report['spline_label_decomposition_max_abs']=float(np.max(abs(arrays['truth_parameters9']-motion)))
    report['limitations'] += ['Natural cubic splines generate GT only; fitted model is the unchanged vanilla nine-output MLP.',
        'The zero-centered intrinsic prior is deliberately mismatched to the nonzero true intrinsic corrections.',
        'R2 and Pearson correlation are diagnostics; no affine fit or smoothing is applied to plotted estimates.',
        'Known attenuation volume is held fixed; this does not test joint volume/pose reconstruction.',
        'Uniform 2 mm / 2 degree peak amplitudes and one GT spline realization are a first stress test, not measured scanner motion.']
    report['source_sha256']['report_spline_calibration.py']=sha256(Path(__file__))
    report['artifact_sha256'].update({name:sha256(out/name) for name in (
        'parameter_recovery.csv','gt_spline9.png','projection_inputs.png','projection_fits.png','source_trajectory3d.png')})
    (out/'spline_recovery.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['table'],indent=2))
    print(json.dumps(parameter_rows,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=BASE/'input')
    parser.add_argument('--baseline',type=Path,default=BASE/'signed_lncc31_seed1')
    parser.add_argument('--regularized',type=Path,default=BASE/'reg_intrinsic001_seed1')
    parser.add_argument('--out-dir',type=Path,default=BASE/'comparison')
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.preview_only:
        acquisition,motion,theta=verify_truth(args.input_dir)
        preview(args.input_dir,args.out_dir,acquisition,motion,theta)
    else:
        recovery_report(args.input_dir,args.baseline,args.regularized,args.out_dir)


if __name__=='__main__':main()
