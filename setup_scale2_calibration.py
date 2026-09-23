"""Scale fixed reference labels and freeze the union of the approved preview ROIs."""
import json
from pathlib import Path
from itertools import product
import numpy as np
from run_sinespin_calibration import sha256
from denseball_landmarks import project_landmarks

ROOT=Path(__file__).resolve().parent


def main():
    old=ROOT/'result_spline9/ball_calibration/input'
    folder=ROOT/'result_spline9_scale2/ball_calibration/input'
    preview=ROOT/'result_spline9_scale2/preview/allballs_roi_preview.json'
    meta=json.loads((folder/'experiment.json').read_text())
    labels=json.loads((old/'landmarks.json').read_text())
    if labels['volume_sha256']!=meta['volume']['sha256']:raise ValueError('Different raw volume')
    factor=meta['volume']['voxel_mm']/labels['voxel_size_mm']
    if factor!=2:raise ValueError('This protocol requires exactly double physical spacing')
    scaled=[]
    for bead in labels['landmarks']:
        item=bead.copy()
        for key in ('xyz_mm','threshold_bbox_edge_xyz_mm','threshold_bbox_size_xyz_mm'):
            item[key]=(factor*np.asarray(item[key])).tolist()
        item['integrated_threshold_contrast_mm2']*=factor**3
        scaled.append(item)
    shape=np.array(meta['volume']['shape_zyx']);origin=-(shape[::-1]-1)*meta['volume']['voxel_mm']/2
    points=np.array([b['xyz_mm'] for b in scaled])
    indices=np.array([b['index_zyx'] for b in scaled])
    np.testing.assert_allclose(points,origin+meta['volume']['voxel_mm']*indices[:,::-1],atol=1e-12,rtol=0)
    record=dict(volume_filename=labels['volume_filename'],volume_sha256=labels['volume_sha256'],
        shape_zyx=shape.tolist(),voxel_size_mm=meta['volume']['voxel_mm'],first_voxel_xyz_mm=origin.tolist(),
        landmark_count=len(scaled),landmarks=scaled,source_landmarks_sha256=sha256(old/'landmarks.json'),
        derivation='Same raw array and fixed connected-component IDs; physical coordinates x2, volume integrals x8; no spatial rematching',
        use='Geometric evaluation only; no per-view landmark supervision in optimization')
    (folder/'landmarks.json').write_text(json.dumps(record,indent=2)+'\n')
    meta['landmarks_sha256']=sha256(folder/'landmarks.json')
    meta['physical_scale_experiment']=dict(factor=factor,source_acquisition_sha256=sha256(old/'experiment.json'),
        attenuation_values_changed=False,voxel_counts_changed=False,
        same_gt_motion_sha256=sha256(folder/'spline_motion9.npy'))
    np.testing.assert_array_equal(np.load(old/'spline_motion9.npy'),np.load(folder/'spline_motion9.npy'))
    accepted=json.loads(preview.read_text());boxes=np.array(accepted['boxes_original_detector_xyxy'])
    union=np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)]
    v,u=meta['detector_padding_vu'];box=union+np.array([u,v,u,v])
    height,width=int(box[3]-box[1]),int(box[2]-box[0]);views=meta['truth']['views']
    roi=dict(definition='Fixed union of the three user-approved all-bead preview crops; same rectangle in every view; no resampling or per-view GT ROI adaptation',
        target_sha256=meta['target_sha256'],detector_shape_vu=meta['truth']['detector_shape_vu'],
        boxes_xyxy=np.tile(box,(views,1)).tolist(),crop_shape_vu=[height,width],
        preview_design_used_known_simulated_bead_boxes=True,preview_sha256=sha256(preview),
        original_detector_box_xyxy=union.tolist(),source_sha256=sha256(Path(__file__)))
    (folder/'loss_roi.json').write_text(json.dumps(roi,indent=2)+'\n')
    # Posthoc verification only: never adjust the frozen union based on this.
    corners=[]
    for bead in scaled:
        bound=np.array(bead['threshold_bbox_edge_xyz_mm']);bound[0]-=.8;bound[1]+=.8
        corners.extend(product(*zip(bound[0],bound[1])))
    uv=project_landmarks(np.load(folder/'P_truth_pixel.npy'),np.array(corners))
    margins=np.stack((uv[:,:,0]-box[0]+.5,box[2]-.5-uv[:,:,0],uv[:,:,1]-box[1]+.5,box[3]-.5-uv[:,:,1]),-1)
    if margins.min()<0:raise ValueError('Frozen preview-union ROI cuts a bead; inspect before training')
    audit=dict(crop_shape_vu=[height,width],box_original_detector_xyxy=union.tolist(),
        all_35_bead_boxes_in_all_views=True,views=views,minimum_bead_box_margin_px=float(margins.min()),
        complete_box_definition='Threshold component bbox plus two 0.4mm voxels on every face',
        per_view_min_margin_px=margins.min((1,2)).tolist(),roi_sha256=sha256(folder/'loss_roi.json'))
    (folder/'roi_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (folder/'experiment.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='per_view_min_margin_px'},indent=2))


if __name__=='__main__':main()
