"""Local bead-coordinate identifiability diagnostic; never a training loss.

Project K sensitivities orthogonally to the rigid-motion tangent space. This
measures weak information, not an exact gauge or an actual image-loss Hessian.
"""
import json
from pathlib import Path
import numpy as np
from calibration_gauge import compose_effective_parameters
from calibration_geometry import pixel_to_pmat, pmat_to_pixel
from denseball_landmarks import project_landmarks
from run_sinespin_calibration import sha256

ROOT=Path(__file__).resolve().parent


def diagnostic(folder):
    meta=json.loads((folder/'experiment.json').read_text());dv,du=meta['truth']['pixel_vu_mm']
    nominal=pixel_to_pmat(np.load(folder/'P_nominal_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    motion=np.load(folder/'spline_motion9.npy')
    labels=json.loads((folder/'landmarks.json').read_text());points=np.array([b['xyz_mm'] for b in labels['landmarks']])
    steps=np.array([.001]*6+[.00001]*3)
    def jacobian(multiplier):
        columns=[]
        for k,h in enumerate(steps*multiplier):
            plus=motion.copy();minus=motion.copy();plus[:,k]+=h;minus[:,k]-=h
            images=[]
            for pars in (plus,minus):
                p=pmat_to_pixel(compose_effective_parameters(nominal,pars),du=du,dv=dv)
                images.append(project_landmarks(p,points).reshape(len(p),-1))
            columns.append((images[0]-images[1])/(2*h))
        return np.stack(columns,-1)
    jac=jacobian(1.);half=jacobian(.5)
    discrepancy=float(np.linalg.norm(jac-half)/np.linalg.norm(half))
    if discrepancy>1e-6:raise ValueError('Finite-difference derivative did not converge')
    conditions=[];remaining=[];sigmas=[];angles=[];ranks=[];sensitivities=[]
    for j in half:
        norm=np.linalg.norm(j,axis=0);unit=j/norm
        sv=np.linalg.svd(unit,compute_uv=False)
        ranks.append(int(np.sum(sv>sv[0]*1e-10)));conditions.append(float(sv[0]/sv[-1]))
        jk,jr=j[:,:3],j[:,3:]
        qr=np.linalg.qr(jr,mode='reduced')[0]
        residual=jk-qr@(qr.T@jk)
        remaining.append(np.linalg.norm(residual,axis=0)/np.linalg.norm(jk,axis=0))
        sensitivities.append(np.linalg.norm(residual,axis=0))
        qk=np.linalg.qr(jk,mode='reduced')[0]
        cosines=np.linalg.svd(qk.T@qr,compute_uv=False)
        angles.append(np.rad2deg(np.arccos(np.clip(cosines,0,1))))
        # Conditional-on-fixed-K and marginalized-K uncertainty differ. This
        # Schur complement marginalizes all six rigid parameters, not K peers.
        _,s,vh=np.linalg.svd(residual,full_matrices=False)
        covariance=(vh.T/s**2)@vh*.1**2
        _,full_s,full_vh=np.linalg.svd(j,full_matrices=False)
        full_covariance=(full_vh.T/full_s**2)@full_vh*.1**2
        np.testing.assert_allclose(covariance,full_covariance[:3,:3],rtol=1e-6,atol=1e-8)
        sigmas.append(np.sqrt(np.diag(covariance)))
    remaining=np.array(remaining);sigmas=np.array(sigmas);sensitivities=np.array(sensitivities)
    summary=dict(acquisition_sha256=sha256(folder/'experiment.json'),
        minimum_numerical_rank=min(ranks),maximum_numerical_rank=max(ranks),
        normalized_jacobian_condition_median=float(np.median(conditions)),
        normalized_jacobian_condition_max=float(np.max(conditions)),
        intrinsic_order=['delta_u_mm','delta_f_mm','delta_v_mm'],
        unabsorbed_sensitivity_fraction_median=np.median(remaining,axis=0).tolist(),
        unabsorbed_sensitivity_norm_px_per_mm_median=np.median(sensitivities,axis=0).tolist(),
        principal_angles_deg_median=np.median(angles,axis=0).tolist(),
        ideal_K_std_mm_at_point_noise_0_1px_median=np.median(sigmas,axis=0).tolist(),
        finite_difference_half_step_relative_difference=discrepancy)
    return summary,dict(remaining=remaining,ideal_std=sigmas,condition=np.array(conditions),
                        sensitivity=sensitivities,principal_angles=np.array(angles))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'result_spline9_scale2/coupling';out.mkdir(parents=True,exist_ok=True)
    summaries={};arrays={}
    for label,folder in [('Original',ROOT/'result_spline9/ball_calibration/input'),
                         ('Scale2',ROOT/'result_spline9_scale2/ball_calibration/input')]:
        summaries[label],arrays[label]=diagnostic(folder)
    theta=np.load(ROOT/'result_spline9/ball_calibration/input/truth_geometry.npz')['theta_deg']
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,layout='constrained')
    for label,color in [('Original','#777777'),('Scale2','#2469b2')]:
        for k,name in enumerate(['delta u','delta f','delta v']):
            axes[0,k].plot(theta,100*arrays[label]['remaining'][:,k],color=color,label=label)
            axes[1,k].plot(theta,arrays[label]['ideal_std'][:,k],color=color,label=label)
            axes[0,k].set_title(name);axes[0,k].set_ylabel('K sensitivity NOT absorbed by rigid [%]')
            axes[1,k].set_ylabel('Ideal K standard deviation [mm]');axes[1,k].set_xlabel('Scan angle [degree]')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Evaluation-only local identifiability at GT, all 35 known bead correspondences\n'
        'Bottom: hypothetical independent 0.1 px coordinate noise; not an uncertainty estimate for LNCC training')
    fig.savefig(out/'intrinsic_rigid_coupling.png',dpi=160);plt.close(fig)
    report=dict(results=summaries,
        method='Central finite differences at GT; J=[JK,JR]; project JK to orthogonal complement of column space of JR; SVD and Schur-complement covariance',
        limitations=['GT correspondences and true motion used only in this diagnostic, never in calibration training.',
            'Full numerical rank excludes an exact local gauge in this point model at tested poses, not all nonlinear ambiguities.',
            'Covariance assumes independent isotropic 0.1px point noise, known exact 3D reference and per-view parameters; no temporal prior.',
            'This is neither the signed-LNCC image Hessian nor an empirical covariance of trained estimates.',
            'QR/SVD decorrelates update coordinates locally; it does not create new observations or guarantee correct physical K.'],
        references=['https://ceres-solver.readthedocs.io/latest/nnls_covariance.html',
                    'https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/'],
        source_sha256=sha256(Path(__file__)))
    (out/'coupling.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(out/'coupling_arrays.npz',theta_deg=theta,**{f'{name}_{k}':v for name,a in arrays.items() for k,v in a.items()})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
