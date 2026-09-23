"""Crop the extended physical projection to retain all beads, excluding the base."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
from run_sinespin_calibration import geometries, sha256
from denseball_landmarks import project_landmarks


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    root=Path(__file__).resolve().parent;folder=root/'result_spline9_scale2/preview'
    data=np.load(folder/'preview_arrays.npz');meta=json.loads((folder/'preview.json').read_text())
    images=data['doubled_extended_noisy'];indices=data['indices'];extra=meta['diagnostic_extra_pixels_each_edge']
    rows,cols=meta['original_detector_vu'];boxes=[];details=[]
    for image in images:
        sm=gaussian_filter(image,2);lab,_=label(sm>.04);counts=np.bincount(lab.ravel());counts[0]=0
        ys,xs=np.where(lab==counts.argmax());x0,x1,y0,y1=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
        lo=int(x0+.15*(x1-x0));hi=int(x1-.15*(x1-x0))
        profile=gaussian_filter1d(np.median(sm[:,lo:hi],axis=1),1)
        peak=y0+int(np.argmax(profile[y0:y0+(y1-y0)//4]))
        bg=float(np.median(profile[peak+60:y1-60]));threshold=bg+.15*(float(profile[peak])-bg)
        crossings=np.flatnonzero(profile[peak:y0+(y1-y0)//2]<threshold)
        if not len(crossings):raise ValueError('No observed plate edge')
        cut=peak+int(crossings[0])+3
        boxes.append([x0-8,cut,x1+8,y1+8])
        details.append(dict(cylinder_xyxy=[x0,y0,x1,y1],plate_peak_row=peak,plate_cut_row=cut))
    boxes=np.array(boxes);width=int(np.max(boxes[:,2]-boxes[:,0]));height=int(np.max(boxes[:,3]-boxes[:,1]))
    boxes[:,2]=boxes[:,0]+width;boxes[:,3]=boxes[:,1]+height
    if np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>images.shape[2]) or np.any(boxes[:,3]>images.shape[1]):
        raise ValueError('Proposed crop exceeds the extended projection')
    # This requested all-bead visualization uses the known simulated phantom
    # to protect complete bead boxes plus a 31-window half-width. It is not
    # an observation-only training ROI or a calibration experiment.
    inputs=root/'result_spline9/ball_calibration/input'
    acquisition=json.loads((inputs/'experiment.json').read_text())
    labels=json.loads((inputs/'landmarks.json').read_text())
    _,geo=geometries(acquisition['truth']['views'],2,spline_config=acquisition['spline_config'])
    corners=[]
    for bead in labels['landmarks']:
        bounds=2*np.array(bead['threshold_bbox_edge_xyz_mm'])
        bounds[0]-=.8;bounds[1]+=.8
        corners.extend(product(*zip(bounds[0],bounds[1])))
    uv=project_landmarks(geo.projection_matrices()[indices],np.array(corners))+extra
    boxes[:,:2]=np.minimum(boxes[:,:2],np.floor(uv.min(1)-16).astype(int))
    boxes[:,2:]=np.maximum(boxes[:,2:],np.ceil(uv.max(1)+17).astype(int))
    width=int(np.max(boxes[:,2]-boxes[:,0]));height=int(np.max(boxes[:,3]-boxes[:,1]))
    boxes[:,2]=boxes[:,0]+width;boxes[:,3]=boxes[:,1]+height
    if np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>images.shape[2]) or np.any(boxes[:,3]>images.shape[1]):
        raise ValueError('All-bead crop exceeds diagnostic projection')
    margin=np.stack((uv[:,:,0]-boxes[:,0,None]+.5,boxes[:,2,None]-.5-uv[:,:,0],
                     uv[:,:,1]-boxes[:,1,None]+.5,boxes[:,3,None]-.5-uv[:,:,1]),-1).min((1,2))
    if np.any(margin<0):raise ValueError(f'A bead bounding box was cut: {margin}')
    old=json.loads((folder/'roi_lncc31_preview.json').read_text())
    vmax=meta['shared_display_vmax'];fig,axes=plt.subplots(3,3,figsize=(15,14),layout='constrained')
    for j,(image,view,box) in enumerate(zip(images,indices,boxes)):
        x0,y0,x1,y1=map(int,box);sx=old['views'][j]['spot']['x'];sy=old['views'][j]['spot']['y']
        ex,ey=sx+extra,sy+extra
        displays=[(image,[-extra-.5,cols+extra-.5,-extra-.5,rows+extra-.5]),
            (image[y0:y1,x0:x1],[x0-extra-.5,x1-extra-.5,y0-extra-.5,y1-extra-.5]),
            (image[ey-30:ey+31,ex-30:ex+31],[-30.5,30.5,-30.5,30.5])]
        for i,(array,extent) in enumerate(displays):
            ax=axes[i,j];im=ax.imshow(array,origin='lower',cmap='gray',vmin=0,vmax=vmax,extent=extent,interpolation='nearest')
            cx,cy=(sx,sy) if i<2 else (0,0)
            ax.add_patch(Rectangle((cx-15.5,cy-15.5),31,31,fill=False,edgecolor='#ffab32',lw=1.4))
            ax.set_xlabel('Original detector column' if i<2 else 'Pixel offset')
            ax.set_ylabel('Original detector row' if i<2 else 'Pixel offset')
        axes[0,j].add_patch(Rectangle((-.5,-.5),cols,rows,fill=False,edgecolor='#ff7272',ls='--',lw=1.3))
        axes[0,j].add_patch(Rectangle((x0-extra-.5,y0-extra-.5),width,height,fill=False,edgecolor='#35d4ef',lw=1.6))
        axes[0,j].set_title(f'View {view}: extended simulation\nCyan = new ROI; dashed red = old detector',fontsize=10)
        axes[1,j].set_title(f'NEW crop: {height} x {width} pixels\n35/35 complete bead boxes retained',fontsize=10)
        axes[2,j].set_title('Same bead, magnified for display\nOrange = 31 x 31 pixel LNCC window',fontsize=10)
        axes[2,j].set_xticks([-30,-15,0,15,30]);axes[2,j].set_yticks([-30,-15,0,15,30])
    fig.colorbar(im,ax=axes,shrink=.4,label='Noisy line integral (shared scale)')
    fig.suptitle('2x whole phantom: crop expanded to include ALL beads\n'
        'Most of bottom plate excluded; original pixel pitch and magnification retained; Poisson I0=44,000\n'
        'Uses simulated detector area beyond the original physical detector. Three-view preview only.',fontsize=12)
    fig.savefig(folder/'scale2_allballs_roi_lncc31_preview.png',dpi=160);plt.close(fig)
    report=dict(scope='Three displayed views only; virtual detector extension required; no training performed',
        crop_shape_vu=[height,width],views=indices.tolist(),boxes_extended_xyxy=boxes.tolist(),
        boxes_original_detector_xyxy=(boxes-extra).tolist(),algorithm='Observed cylinder/plate crop expanded to include known simulated bead boxes plus 16 pixels of context; shared max height/width',
        observed_plate_detection=details,ground_truth_used_to_define_crop=True,
        ground_truth_scope='Visualization-only ROI adjustment requested to retain all simulated beads; not an observation-only training manifest',
        posthoc_check='Every corner of all 35 bead threshold bounding boxes, enlarged by two voxels on every face, is inside the crop',
        minimum_complete_bead_box_margin_px_per_view=margin.tolist(),original_detector_vu=[rows,cols],
        source_sha256=sha256(Path(__file__)),input_sha256=sha256(folder/'preview_arrays.npz'))
    (folder/'allballs_roi_preview.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
