"""Show observation-defined crops and a literal 31-pixel window on scale-2 data.

This is a three-view preview, not a complete acquisition/training ROI manifest.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter
from run_sinespin_calibration import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path(__file__).resolve().parent/'result_spline9_scale2/preview')
    args=parser.parse_args();folder=args.folder
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    data=np.load(folder/'preview_arrays.npz')
    images=data['doubled_current_noisy'];indices=data['indices'];rows,cols=images.shape[1:]
    records=[]
    for index,image in zip(indices,images):
        smooth=gaussian_filter(image,2)
        labels,_=label(smooth>.04);counts=np.bincount(labels.ravel());counts[0]=0
        if not counts.max():raise ValueError('No cylinder component detected')
        yy,xx=np.where(labels==counts.argmax())
        # The cylinder is already truncated vertically; keep all acquired rows.
        x0=max(0,int(xx.min())-8);x1=min(cols,int(xx.max())+9)
        hp=gaussian_filter(image,.6)-gaussian_filter(image,8)
        peaks=np.argwhere((hp==maximum_filter(hp,21))&(hp>.6))
        candidates=[]
        for y,x in peaks:
            if not(35<=y<rows-35 and x0+35<=x<x1-35):continue
            patch=hp[y-12:y+13,x-12:x+13]
            component,_=label(patch>.5*hp[y,x]);c=component[12,12]
            if not c:continue
            ys,xs=np.where(component==c);h,w=int(np.ptp(ys)+1),int(np.ptp(xs)+1)
            if 4<=min(h,w) and max(h,w)<=18 and max(h,w)/min(h,w)<1.35:
                candidates.append(dict(x=int(x),y=int(y),fwhm_vu=[h,w],peak=float(hp[y,x])))
        if not candidates:raise ValueError('No isolated round bright spot for window illustration')
        spot=max(candidates,key=lambda s:(min(s['fwhm_vu']),s['peak']))
        records.append(dict(view=int(index),box_xyxy=[x0,0,x1,rows],spot=spot,
                            detector_row_edges_touch_cylinder=bool(yy.min()==0 and yy.max()==rows-1)))
    # A shared width works in a batch without image resizing; left edges remain
    # observation-defined. These are only the three displayed preview views.
    width=max(r['box_xyxy'][2]-r['box_xyxy'][0] for r in records)
    for r in records:
        r['box_xyxy'][0]=min(r['box_xyxy'][0],cols-width)
        r['box_xyxy'][2]=r['box_xyxy'][0]+width
    vmax=json.loads((folder/'preview.json').read_text())['shared_display_vmax']
    fig,axes=plt.subplots(3,len(indices),figsize=(15,13),layout='constrained')
    for j,(image,record) in enumerate(zip(images,records)):
        x0,y0,x1,y1=record['box_xyxy'];x,y=record['spot']['x'],record['spot']['y']
        crop=image[y0:y1,x0:x1];patch=image[y-30:y+31,x-30:x+31]
        displays=[(image,[-.5,cols-.5,-.5,rows-.5]),
                  (crop,[x0-.5,x1-.5,y0-.5,y1-.5]),
                  (patch,[-30.5,30.5,-30.5,30.5])]
        for i,(array,extent) in enumerate(displays):
            ax=axes[i,j];im=ax.imshow(array,origin='lower',cmap='gray',vmin=0,vmax=vmax,extent=extent,interpolation='nearest')
            cx,cy=(x,y) if i<2 else (0,0)
            ax.add_patch(Rectangle((cx-15.5,cy-15.5),31,31,fill=False,edgecolor='#ffab32',lw=1.5))
            ax.set_xlabel('Detector column' if i<2 else 'Pixel offset')
            ax.set_ylabel('Detector row' if i<2 else 'Pixel offset')
        axes[0,j].add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#35d4ef',lw=1.6))
        axes[0,j].set_title(f'View {record["view"]}: current 476 x 646 detector\nCyan = proposed cylinder ROI')
        axes[1,j].set_title(f'Actual crop: {rows} x {width} pixels\nx={x0}:{x1}; all acquired rows retained')
        axes[2,j].set_title('Same observed spot, enlarged for viewing\nOrange window = exactly 31 x 31 pixels')
        axes[2,j].set_xticks([-30,-15,0,15,30]);axes[2,j].set_yticks([-30,-15,0,15,30])
        axes[2,j].grid(alpha=.15)
    fig.colorbar(im,ax=axes,shrink=.45,label='Noisy line integral (same display scale)')
    fig.suptitle('2x whole phantom (voxel 0.4 mm), current detector, Poisson I0=44,000\n'
        'Observation-only bounding boxes; no image resampling. Orange squares are example LNCC windows.\n'
        'Cylinder top/bottom are outside the acquired field; cropping removes side air only.',fontsize=12)
    fig.savefig(folder/'scale2_roi_lncc31_preview.png',dpi=160);plt.close(fig)
    output=dict(scope='Three-view ROI/window preview only; not a complete training manifest',
        roi_definition='Largest component of Gaussian sigma2 image above 0.04, x extent plus 8 pixels, all acquired rows, common width over three preview views',
        training_intensity='Original noisy arrays, unfiltered; no resampling',kernel_size=31,
        kernel_definition='31 x 31 original detector pixels, not millimetres or resized display pixels',
        crop_shape_vu=[rows,width],retained_pixel_fraction=width/cols,views=records,
        input_sha256=sha256(folder/'preview_arrays.npz'),source_sha256=sha256(Path(__file__)))
    (folder/'roi_lncc31_preview.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
