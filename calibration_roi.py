"""Fixed observation-defined detector crops; no image resampling or P changes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


class ProjectionROI:
    def __init__(self, manifest, *, views, rows, cols, target_sha256, device):
        self.manifest=json.loads(Path(manifest).read_text())
        record=self.manifest
        if record['target_sha256']!=target_sha256 or record['detector_shape_vu']!=[rows,cols]:
            raise ValueError('ROI manifest belongs to another projection dataset')
        boxes=np.asarray(record['boxes_xyxy'])
        if boxes.shape!=(views,4) or not np.issubdtype(boxes.dtype,np.integer):
            raise ValueError('Require one integer xyxy box per acquired view')
        widths=boxes[:,2]-boxes[:,0];heights=boxes[:,3]-boxes[:,1]
        if (np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>cols) or np.any(boxes[:,3]>rows)
                or widths.min()<=0 or heights.min()<=0 or np.ptp(widths) or np.ptp(heights)):
            raise ValueError('Crops must have common positive size and fit inside the detector')
        self.height,self.width=int(heights[0]),int(widths[0])
        self.boxes=torch.as_tensor(boxes,dtype=torch.long,device=device)
        self.ys=torch.arange(self.height,device=device)[None,:,None]
        self.xs=torch.arange(self.width,device=device)[None,None,:]
        self.shape=(rows,cols)
        self.record=dict(manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                         crop_shape_vu=[self.height,self.width],definition=record['definition'],
                         target_sha256=target_sha256,manifest_archive='loss_roi.json')

    def crop(self, images, views):
        if tuple(images.shape[1:])!=self.shape or len(images)!=len(views):
            raise ValueError('ROI image batch shape differs from its detector/view indices')
        boxes=self.boxes[views]
        batch=torch.arange(len(views),device=images.device)[:,None,None]
        return images[batch,boxes[:,1,None,None]+self.ys,boxes[:,0,None,None]+self.xs]

    def loss(self, function, pred, target, views, counts=None):
        return function(self.crop(pred,views),self.crop(target,views),
                        counts=None if counts is None else self.crop(counts,views))
