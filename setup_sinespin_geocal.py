"""Freeze a known-reference / nominal-only bead ROI for the sineSpin geocal study."""
import json
from pathlib import Path
from itertools import product
import shutil
import numpy as np
from run_sinespin_calibration import sha256
from denseball_landmarks import project_landmarks

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'result_sinespin_geocal/input'


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    meta=json.loads((FOLDER/'experiment.json').read_text())
    old=ROOT/'result_spline9_scale2/ball_calibration/input/landmarks.json'
    labels=json.loads(old.read_text())
    assert labels['volume_sha256']==meta['volume']['sha256']
    assert labels['voxel_size_mm']==meta['volume']['voxel_mm']
    assert labels['shape_zyx']==meta['volume']['shape_zyx']
    shutil.copy2(old,FOLDER/'landmarks.json')
    meta['landmarks_sha256']=sha256(FOLDER/'landmarks.json')
    corners=[]
    for bead in labels['landmarks']:
        bounds=np.array(bead['threshold_bbox_edge_xyz_mm'])
        bounds[0]-=.8;bounds[1]+=.8
        corners.extend(product(*zip(bounds[0],bounds[1])))
    corners=np.array(corners)
    projected=project_landmarks(np.load(FOLDER/'P_nominal_pixel.npy'),corners)
    # Geometry/3-D reference are available inputs. Never consult GT to select ROI.
    lower=np.floor(projected.min(1)-20).astype(int)
    upper=np.ceil(projected.max(1)+20).astype(int)+1
    size=(upper-lower).max(0)
    centers=(lower+upper)/2
    lo=np.floor(centers-size/2).astype(int)
    boxes=np.c_[lo,lo+size]
    rows,cols=meta['truth']['detector_shape_vu']
    assert np.all(boxes[:,:2]>=0) and np.all(boxes[:,2]<=cols) and np.all(boxes[:,3]<=rows)
    roi=dict(definition='Nominal-only per-view projection of known reference bead bounding boxes; 20px margin; common crop shape; frozen before fitting, no GT-dependent adaptation',
        target_sha256=meta['target_sha256'],detector_shape_vu=[rows,cols],
        boxes_xyxy=boxes.tolist(),crop_shape_vu=size[::-1].tolist(),
        design_used_reference_bead_boxes=True,design_used_gt_geometry=False,
        training_landmark_correspondences=False,
        limitation='Rectangle retains every bead; any bottom plate overlapping bead rows remains. No intensity mask or resampling.',
        source_sha256=sha256(Path(__file__)))
    (FOLDER/'loss_roi.json').write_text(json.dumps(roi,indent=2)+'\n')
    truth=project_landmarks(np.load(FOLDER/'P_truth_pixel.npy'),corners)
    margin=np.stack((truth[:,:,0]-boxes[:,None,0]+.5,boxes[:,None,2]-.5-truth[:,:,0],
                     truth[:,:,1]-boxes[:,None,1]+.5,boxes[:,None,3]-.5-truth[:,:,1]),-1)
    assert margin.min()>0,'Frozen nominal ROI clips truth beads'
    audit=dict(all_bead_boxes_visible=True,minimum_margin_px=float(margin.min()),
               per_view_min_margin_px=margin.min((1,2)).tolist(),roi_sha256=sha256(FOLDER/'loss_roi.json'),
               verification_only=True)
    (FOLDER/'roi_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    meta['roi_setup_source_sha256']=sha256(Path(__file__))
    (FOLDER/'experiment.json').write_text(json.dumps(meta,indent=2)+'\n')
    images=np.load(FOLDER/'target_projections.npy',mmap_mode='r')
    indices=np.linspace(0,len(images)-1,6).round().astype(int)
    tilts=np.load(FOLDER/'nominal_geometry.npz')['tilt_deg']
    xyz=np.array([b['xyz_mm'] for b in labels['landmarks']])
    centers=project_landmarks(np.load(FOLDER/'P_nominal_pixel.npy'),xyz)
    largest=int(np.argmax([np.prod(b['threshold_bbox_size_xyz_mm']) for b in labels['landmarks']]))
    fig,axes=plt.subplots(2,3,figsize=(15,11),layout='constrained')
    vmax=float(np.quantile(np.asarray(images[indices]),.999))
    for ax,i in zip(axes.flat,indices):
        ax.imshow(images[i],origin='lower',cmap='gray',vmin=0,vmax=vmax)
        x0,y0,x1,y1=boxes[i]
        ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,ec='#f09b20',lw=1.2))
        u,v=centers[i,largest]
        ax.add_patch(Rectangle((u-15.5,v-15.5),31,31,fill=False,ec='#29c7de',lw=1.2))
        ax.set_title(f'View {i}; nominal tilt {tilts[i]:.1f} deg')
        ax.set_xlabel('Detector u [pixel]');ax.set_ylabel('Detector v [pixel]')
    fig.suptitle('SineSpin nominal + small calibration errors; Poisson I0=44,000\nOrange: fixed nominal/reference ROI; cyan: LNCC31 window; enlarged phantom / virtual detector')
    out=ROOT/'docs/sinespin_geocal_input_preview.png'
    fig.savefig(out,dpi=140);plt.close(fig)
    print(json.dumps(dict(crop_shape_vu=roi['crop_shape_vu'],minimum_margin_px=audit['minimum_margin_px']),indent=2))


if __name__=='__main__':setup()
