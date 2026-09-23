"""Plot actual physical K, source coordinates and orientation from complete P."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from physical_camera import decompose_physical_camera, compose_physical_camera
from compare_sinespin_regularization import sha256

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'result_spline9/ball_calibration'
NAMES=('f_mm','cu_mm','cv_mm','source_x_mm','source_y_mm','source_z_mm',
       'camera_euler_x_deg','camera_euler_y_deg','camera_euler_z_deg')
RUNS=('gt','unregularized','intrinsic001','nominal')


def plot(folder,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    provenance=json.loads((folder/'pmat_comparison.json').read_text())
    data_path=folder/'pmat_components.npz'
    if sha256(data_path)!=provenance['artifacts'][data_path.name]:
        raise ValueError('Normalized matrices changed since the validated P comparison')
    arrays=np.load(data_path);theta=arrays['theta_deg']
    acquisition_dir=Path(provenance['inputs'][0]).parent
    acquisition=json.loads((acquisition_dir/'experiment.json').read_text())
    nominal_path=acquisition_dir/'P_nominal_world_mm.npy'
    if sha256(nominal_path)!=acquisition['nominal_pmat_sha256']:
        raise ValueError('Nominal P differs from the recorded training acquisition')
    matrices={name:arrays[name] for name in RUNS[:3]}
    matrices['nominal']=np.load(nominal_path)
    if matrices['nominal'].shape!=matrices['gt'].shape:
        raise ValueError('Nominal P view count differs from GT')
    values=[];checks=[]
    for name in RUNS:
        p=matrices[name];camera=decompose_physical_camera(p)
        q=camera['Q_camera_to_physical']
        with warnings.catch_warnings():
            warnings.simplefilter('error',UserWarning)
            angles=Rotation.from_matrix(q).as_euler('xyz',degrees=True)
        # Only multiples of 360 are changed, independently for each scan.
        angles=np.rad2deg(np.unwrap(np.deg2rad(angles),axis=0))
        np.testing.assert_allclose(Rotation.from_euler('xyz',angles,degrees=True).as_matrix(),q,atol=1e-12,rtol=0)
        reconstructed=compose_physical_camera(camera['source_xyz_mm'],q,camera['K_mm'],
                                               projective_scale=camera['projective_scale'])
        np.testing.assert_allclose(reconstructed,p,atol=1e-8,rtol=1e-11)
        values.append(np.column_stack((camera['intrinsics_f_cu_cv_mm'],camera['source_xyz_mm'],angles)))
        checks.append(dict(run=name,max_matrix_roundtrip_abs=float(abs(reconstructed-p).max()),
            max_focal_anisotropy_mm=float(abs(camera['focal_anisotropy_mm']).max()),
            max_skew_mm=float(abs(camera['skew_mm']).max()),
            min_abs_cos_middle_euler=float(np.min(abs(np.cos(np.deg2rad(angles[:,1])))))))
    labels=[r'$f$ [mm]',r'$c_u$ [mm]',r'$c_v$ [mm]',r'$C_x$ [mm]',r'$C_y$ [mm]',r'$C_z$ [mm]',
            r'Camera $\alpha_x$ [degree]',r'Camera $\alpha_y$ [degree]',r'Camera $\alpha_z$ [degree]']
    legends=['GT spline','Estimated: no prior','Estimated: intrinsic lambda=0.01','Nominal circular']
    colors=['black','#2469b2','#8a45b7','#db7b20']
    styles=['--','-','-',':']
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,layout='constrained')
    for j,ax in enumerate(axes.flat):
        for i in (3,1,2,0):
            ax.plot(theta,values[i][:,j],color=colors[i],ls=styles[i],
                    lw=1.6 if i==3 else (1.3 if i==0 else 1.),label=legends[i])
        ax.set_title(labels[j]);ax.grid(alpha=.2)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        if j>=6:ax.set_xlabel('Nominal scan angle [degree]')
    handles,text=axes.flat[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in (3,0,1,2)],[text[i] for i in (3,0,1,2)],loc='outside lower center',ncol=4)
    fig.suptitle('Nine physical geometry components from complete P: actual values\n'
                 'Intrinsics / physical source xyz / camera-to-physical xyz Euler angles; no nominal subtraction or pose fit')
    fig.savefig(out/'geometry_components9.png',dpi=170);plt.close(fig)
    np.savez(out/'geometry_components9.npz',theta_deg=theta,**dict(zip(RUNS,values)))
    with (out/'geometry_components9.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['view','theta_deg']+[f'{run}_{p}' for run in RUNS for p in NAMES])
        for v,t in enumerate(theta):writer.writerow([v,t]+[float(x) for value in values for x in value[v]])
    report=dict(parameter_names=NAMES,
        scope='Actual signed physical geometric quantities, not nominal-relative motion or absolute error',
        intrinsic_convention='f=(K00+K11)/2; cu/cv in detector mm from pixel edges; skew/focal anisotropy are roundoff residuals reported separately',
        source_convention='Camera centre C in centred physical xyz mm; not object-motion translation or extrinsic t=-R C',
        angle_convention='Q camera-to-physical has columns (detector col, negative detector row, source-to-plane normal); Q=Rz(az) Ry(ay) Rx(ax). Independent 360-degree unwrapping only; no reference camera subtraction.',
        decomposition_checks=checks,input_sha256=sha256(data_path),parent_report_sha256=sha256(folder/'pmat_comparison.json'),
        nominal_input=dict(path=str(nominal_path),sha256=sha256(nominal_path),
                           acquisition_sha256=sha256(acquisition_dir/'experiment.json')),
        source_sha256={p:sha256(ROOT/p) for p in ('plot_spline_geometry_components.py','physical_camera.py','calibration_gauge.py')},
        artifacts={p:sha256(out/p) for p in ('geometry_components9.png','geometry_components9.csv','geometry_components9.npz')})
    (out/'geometry_components9.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    print(f'Saved {out}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=BASE/'pmat_comparison')
    parser.add_argument('--out-dir',type=Path,default=BASE/'geometry_components')
    args=parser.parse_args();plot(args.input_dir,args.out_dir)


if __name__=='__main__':main()
