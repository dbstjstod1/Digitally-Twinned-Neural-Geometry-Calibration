"""Preview doubled physical voxel spacing with the current detector unchanged."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run_sinespin_calibration import geometries, load_volume, sha256
from sim_sinespin_recon import configure_leap
from photon_noise import poisson_noisy_projections
from denseball_landmarks import project_landmarks

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'result_spline9/ball_calibration/input')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'result_spline9_scale2/preview')
    args=parser.parse_args()
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    torch.cuda.set_device(args.gpu);device=torch.device('cuda',args.gpu)
    folder=args.input_dir;meta=json.loads((folder/'experiment.json').read_text())
    if sha256(meta['volume']['path'])!=meta['volume']['sha256']:raise ValueError('Reference volume changed')
    if sha256(folder/'target_projections.npy')!=meta['target_sha256']:raise ValueError('Original observations changed')
    _,geo=geometries(meta['truth']['views'],2,spline_config=meta['spline_config'])
    selected=np.array([0,geo.n_views//2,geo.n_views-1])
    arrays=[np.ascontiguousarray(a[selected]) for a in geo.modular_arrays()]
    # Extend only the diagnostic canvas. Plane centre, axes, pixel pitch and
    # source remain fixed. The central rectangle is exactly the real detector.
    rows,cols=geo.detector_rows,geo.detector_cols;extra=160
    extended=SimpleNamespace(n_views=len(selected),detector_rows=rows+2*extra,
        detector_cols=cols+2*extra,pixel_height=geo.pixel_height,pixel_width=geo.pixel_width,
        modular_arrays=lambda:arrays)
    shape=tuple(meta['volume']['shape_zyx']);voxel=meta['volume']['voxel_mm']*2
    volume=load_volume(meta['volume']['path'],device,shape)
    ct=configure_leap(extended,shape,voxel,device)
    if sha256(Path(ct.libprojectors._name).resolve())!=meta['leap_sha256']:
        raise ValueError('Preview requires the same pinned Joseph library')
    target=torch.empty((len(selected),extended.detector_rows,extended.detector_cols),device=device)
    ct.project_gpu(target,volume);torch.cuda.synchronize(device)
    clean=target.cpu().numpy()
    noisy,noise,_=poisson_noisy_projections(clean,i0=44000,seed=0)
    cropped=noisy[:,extra:extra+rows,extra:extra+cols]
    # Check the diagnostic central rays against an independently rendered
    # current-size detector, rather than relying on array indexing alone.
    current=SimpleNamespace(n_views=len(selected),detector_rows=rows,detector_cols=cols,
        pixel_height=geo.pixel_height,pixel_width=geo.pixel_width,modular_arrays=lambda:arrays)
    original_ct=configure_leap(current,shape,voxel,device)
    direct=torch.empty((len(selected),rows,cols),device=device)
    original_ct.project_gpu(direct,volume);torch.cuda.synchronize(device)
    centre=clean[:,extra:extra+rows,extra:extra+cols]
    delta=direct.cpu().numpy()-centre
    relative=float(np.linalg.norm(delta.astype(float))/np.linalg.norm(centre.astype(float)))
    if relative>1e-4:raise ValueError(f'Diagnostic canvas shifted detector rays: {relative}')
    original=np.array(np.load(folder/'target_projections.npy',mmap_mode='r')[selected])
    labels=json.loads((folder/'landmarks.json').read_text())
    points=2*np.array([b['xyz_mm'] for b in labels['landmarks']])
    uv=project_landmarks(geo.projection_matrices()[selected],points)
    inside=(uv[:,:,0]>=-.5)&(uv[:,:,0]<cols-.5)&(uv[:,:,1]>=-.5)&(uv[:,:,1]<rows-.5)
    vmax=float(np.quantile(np.concatenate((original.ravel(),noisy.ravel())),.999))
    fig,axes=plt.subplots(3,3,figsize=(15,13),layout='constrained')
    extents=[[-.5,cols-.5,-.5,rows-.5]]*2+[[-extra-.5,cols+extra-.5,-extra-.5,rows+extra-.5]]
    titles=['Original: 0.2 mm voxels','Doubled: 0.4 mm, CURRENT detector',
            'Doubled: diagnostic extended canvas']
    for j,v in enumerate(selected):
        for i,images in enumerate((original,cropped,noisy)):
            ax=axes[i,j];im=ax.imshow(images[j],origin='lower',extent=extents[i],cmap='gray',vmin=0,vmax=vmax)
            ax.set_title(f'{titles[i]}\nView {v}, angle {geo.theta_deg[v]:.1f} degrees',fontsize=10)
            ax.set_xlabel('Original detector column');ax.set_ylabel('Original detector row')
            if i==1:
                ax.text(.03,.97,f'{inside[j].sum()}/35 bead centers inside',transform=ax.transAxes,
                        color='#ffb44c',va='top',fontsize=9)
            if i==2:
                ax.add_patch(Rectangle((-.5,-.5),cols,rows,fill=False,edgecolor='#ffb44c',lw=1.6))
                outside=uv[j,~inside[j]]
                ax.scatter(outside[:,0],outside[:,1],s=45,facecolors='none',edgecolors='#ff6363',lw=1)
    fig.colorbar(im,ax=axes,shrink=.5,label='Noisy line integral (shared display scale)')
    fig.suptitle('Same raw array; whole phantom scaled by voxel spacing only\n'
        'Same spline GT/source/detector plane; Poisson I0=44,000; independent LEAP Joseph\n'
        'Orange box = CURRENT detector; red circles = bead centers outside it. Bottom row is diagnostic only.',fontsize=12)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out_dir/'scale2_projection_preview.png',dpi=160);plt.close(fig)
    np.savez(args.out_dir/'preview_arrays.npz',indices=selected,original_noisy=original,
             doubled_current_noisy=cropped,doubled_extended_noisy=noisy,doubled_extended_clean=clean,
             doubled_gt_bead_uv=uv,inside_current_detector=inside)
    record=dict(indices=selected.tolist(),voxel_mm_original=meta['volume']['voxel_mm'],voxel_mm_doubled=voxel,
        extent_xyz_mm=(np.array(shape[::-1])*voxel).tolist(),original_detector_vu=[rows,cols],
        diagnostic_extra_pixels_each_edge=extra,source_geometry_unchanged=True,
        diagnostic_canvas_is_not_an_acquisition_choice=True,
        inside_centers_per_view=inside.sum(1).tolist(),diagnostic_vs_direct_relative_l2=relative,
        noise=noise,shared_display_vmax=vmax,volume_sha256=meta['volume']['sha256'],
        input_metadata_sha256=sha256(folder/'experiment.json'),source_sha256=sha256(Path(__file__)))
    (args.out_dir/'preview.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
