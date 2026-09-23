"""Compare the full 3x4 GT and fitted P entries with one projective scale rule.

No nominal subtraction, camera refitting, pose alignment, or elementwise scaling.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from calibration_geometry import pixel_to_pmat
from compare_sinespin_regularization import sha256

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'result_spline9/ball_calibration'


def normalize_pmat(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (3,4) or not np.isfinite(p).all():
        raise ValueError('Expected finite [views,3,4] matrices')
    sign, _ = np.linalg.slogdet(p[:,:,:3])
    scale = sign*np.linalg.norm(p[:,2,:3],axis=1)
    if np.any(scale == 0):
        raise ValueError('Require nonsingular finite camera matrices')
    return p/scale[:,None,None], scale


def compare(folder, report_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(report_path.read_text())
    acquisition = json.loads((folder/'experiment.json').read_text())
    if sha256(folder/'experiment.json') != report['input_sha256']['experiment.json']:
        raise ValueError('Acquisition metadata changed since recovery report')
    for name in ('P_truth_pixel.npy','P_truth_world_mm.npy'):
        if sha256(folder/name) != report['input_sha256'][name]:
            raise ValueError(f'Changed input: {name}')
    dv,du = acquisition['truth']['pixel_vu_mm']
    # Float64 generation labels avoid mixing GT storage roundoff with fit error.
    gt = pixel_to_pmat(np.load(folder/'P_truth_pixel.npy'),du=du,dv=dv,dtype=np.float64)
    matrices, paths = [gt], [str(folder/'P_truth_pixel.npy')]
    for run in report['runs']:
        path = Path(run['path'])/'P_optimized_world_mm.npy'
        if sha256(path) != run['artifact_sha256'][path.name]:
            raise ValueError(f'Changed fitted P: {path}')
        matrices.append(np.load(path));paths.append(str(path))
    if len(matrices) != 3:
        raise ValueError('Expected GT and two fixed-epoch fitted runs')
    theta_path = folder/'truth_geometry.npz'
    if sha256(theta_path) != report['input_sha256'][theta_path.name]:
        raise ValueError('Changed view angles')
    theta = np.load(theta_path)['theta_deg']
    normalized, scales = zip(*(normalize_pmat(p) for p in matrices))
    # Verify normalization preserves projection and arbitrary signed scale.
    points = np.c_[np.random.default_rng(914).uniform(-50,50,(37,3)),np.ones(37)]
    for raw,p in zip(matrices, normalized):
        a,b = np.asarray(raw,dtype=float)@points.T,p@points.T
        np.testing.assert_allclose(a[:,:2]/a[:,2:],b[:,:2]/b[:,2:],atol=1e-10,rtol=1e-12)
        multiplier = np.random.default_rng(122).uniform(.2,5,len(p))*(-1.)**np.arange(len(p))
        np.testing.assert_allclose(normalize_pmat(raw*multiplier[:,None,None])[0],p,atol=1e-9,rtol=1e-12)
    names = ['GT spline','Estimated: no prior','Estimated: intrinsic lambda=0.01']
    colors = ['black','#2469b2','#8a45b7']
    out.mkdir(parents=True,exist_ok=True)
    for magnitude in (False,True):
        fig,axes = plt.subplots(3,4,figsize=(17,10),sharex=True,layout='constrained')
        for row in range(3):
            for col in range(4):
                ax=axes[row,col]
                for index in (1,2,0):
                    values=normalized[index][:,row,col]
                    ax.plot(theta,np.abs(values) if magnitude else values,
                            color=colors[index],ls='--' if index==0 else '-',
                            lw=1.2 if index==0 else 1.,label=names[index])
                unit = ('mm^2' if col==3 else 'mm') if row<2 else ('mm' if col==3 else 'dimensionless')
                ax.set_title(('|' if magnitude else '')+f'P[{row+1},{col+1}]'+('|' if magnitude else '')+f' [{unit}]')
                ax.grid(alpha=.2)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useOffset=False)
                if row==2:ax.set_xlabel('Nominal scan angle [degree]')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend([handles[i] for i in (2,0,1)],[labels[i] for i in (2,0,1)],loc='outside lower center',ncol=3)
        fig.suptitle(('Magnitudes of normalized full P entries' if magnitude else 'Full P entries: actual signed values, no nominal subtraction')+
                     '\nWORLD=(physical x,z,y) mm to detector mm; third-row spatial norm = 1, positive left-block determinant')
        fig.savefig(out/('pmat_components_magnitude.png' if magnitude else 'pmat_components.png'),dpi=170)
        plt.close(fig)
    np.savez(out/'pmat_components.npz',theta_deg=theta,gt=normalized[0],
             unregularized=normalized[1],intrinsic001=normalized[2],normalization_scales=np.array(scales))
    with (out/'pmat_components.csv').open('w',newline='') as stream:
        fields=['view','theta_deg']+[f'{label}_P{i+1}{j+1}' for label in ('gt','unregularized','intrinsic001') for i in range(3) for j in range(4)]
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(fields)
        for v,t in enumerate(theta):
            writer.writerow([v,t]+[float(x) for p in normalized for x in p[v].ravel()])
    metrics=[]
    for k in (1,2):
        error=normalized[k]-normalized[0]
        metrics.append(dict(label=names[k],entry_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                            entry_max_abs_error=np.max(np.abs(error),axis=0).tolist()))
    result=dict(scope='Absolute matrix-entry values; same coordinate system and projective scale; no refit',
        normalization='P / (sign(det(P[:,:3])) * norm(P[2,:3])); independently applied per view and per run; no GT matching',
        coordinates='WORLD=(physical x,z,y) in mm; detector u/v in mm measured from pixel edges',
        units='Rows 1-2 columns 1-3: mm; rows 1-2 column 4: mm^2; row 3 columns 1-3: dimensionless; P34: mm',
        interpretation='Signed plot is the full coefficient value, not delta from nominal or error. Magnitude plot additionally applies abs. Similar curves do not prove equal P or equal intrinsic/extrinsic parameters.',
        no_global_matrix_error='Different entries have different units and scales; a single raw Frobenius percentage would depend on coordinate units and origin.',
        normalization_ray_invariance_verified=True,signed_scale_invariance_verified=True,
        inputs=paths,recovery_report_sha256=sha256(report_path),source_sha256=sha256(Path(__file__)),
        normalized_entry_metrics=metrics,
        artifacts={name:sha256(out/name) for name in ('pmat_components.png','pmat_components_magnitude.png','pmat_components.csv','pmat_components.npz')})
    (out/'pmat_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Saved 12-entry GT/estimate comparisons in {out}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=BASE/'input')
    parser.add_argument('--report',type=Path,default=BASE/'comparison/spline_recovery.json')
    parser.add_argument('--out-dir',type=Path,default=BASE/'pmat_comparison')
    args=parser.parse_args()
    compare(args.input_dir,args.report,args.out_dir)


if __name__=='__main__':main()
