"""Detect a cylinder bounding box and remove its bottom plate from noisy data."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d,label,maximum_filter
from compare_sinespin_regularization import sha256
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'result_spline9/ball_calibration'


def prepare(folder,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    meta=json.loads((folder/'experiment.json').read_text())
    path=folder/'target_projections.npy';digest=sha256(path)
    if digest!=meta['target_sha256']:raise ValueError('Changed noisy observations')
    images=np.load(path,mmap_mode='r');n,rows,cols=images.shape
    boxes=[];diagnostics=[];spots=[]
    selected=np.linspace(0,n-1,6).round().astype(int)
    for index in range(n):
        image=np.array(images[index]);sm=gaussian_filter(image,2)
        components,_=label(sm>.04);counts=np.bincount(components.ravel());counts[0]=0
        if counts.max()<100:raise ValueError(f'No cylinder in view {index}')
        ys,xs=np.where(components==np.argmax(counts))
        x0,x1,y0,y1=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
        lo=int(x0+.15*(x1-x0));hi=int(x1-.15*(x1-x0))
        profile=gaussian_filter1d(np.median(sm[:,lo:hi],axis=1),1)
        peak=y0+int(np.argmax(profile[y0:y0+(y1-y0)//4]))
        background=float(np.median(profile[peak+30:y1-30]))
        threshold=background+.15*(float(profile[peak])-background)
        below=np.flatnonzero(profile[peak:y0+(y1-y0)//2]<threshold)
        if not len(below):raise ValueError(f'No plate upper edge in view {index}')
        cut=int(peak+below[0]+3)
        boxes.append([x0-8,cut,x1+8,y1+8])
        diagnostics.append(dict(view=index,cylinder_xyxy=[x0,y0,x1,y1],plate_peak_row=peak,
                                plate_cut_row=cut,plate_profile_threshold=threshold))
        if index in selected:
            hp=gaussian_filter(image,.6)-gaussian_filter(image,5)
            peaks=np.argwhere((hp==maximum_filter(hp,13))&(hp>.4))
            for y,x in peaks:
                if not(cut+8<y<y1-12 and x0+12<x<x1-12):continue
                patch=hp[y-8:y+9,x-8:x+9];labels,_=label(patch>.5*hp[y,x]);component=labels[8,8]
                if not component:continue
                yy,xx=np.where(labels==component);h,w=int(np.ptp(yy)+1),int(np.ptp(xx)+1)
                if 2<=h<=12 and 2<=w<=12 and max(w,h)/min(w,h)<1.4:
                    spots.append(dict(view=int(index),x=int(x),y=int(y),width_fwhm_px=w,height_fwhm_px=h,peak=float(hp[y,x])))
    boxes=np.asarray(boxes);width=int(np.max(boxes[:,2]-boxes[:,0]));height=int(np.max(boxes[:,3]-boxes[:,1]))
    boxes[:,2]=boxes[:,0]+width;boxes[:,3]=boxes[:,1]+height
    if np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>cols) or np.any(boxes[:,3]>rows):raise ValueError('Detected crop exceeds detector')
    out.mkdir(parents=True,exist_ok=True)
    record=dict(definition='Observation-only cylinder bounding box with bottom plate excluded; fixed per-view rectangles; no resizing, P shift, or predicted-image-dependent updates',
                target_sha256=digest,detector_shape_vu=[rows,cols],boxes_xyxy=boxes.tolist(),
                crop_shape_vu=[height,width],pixel_fraction=height*width/(rows*cols),
                algorithm=dict(detection_only_gaussian_sigma_px=2,silhouette_threshold=.04,component='largest 4-connected',
                    plate_profile='Median of central 70% cylinder width, Gaussian sigma=1 along rows',
                    plate_peak_search='Bottom quarter of detected cylinder',plate_upper_edge='First postpeak crossing below background + 0.15*(peak-background), plus 3 rows',
                    background='Median profile from peak+30 to cylinder_top-30',side_top_margin_px=8,
                    common_shape='Maximum detected width and height over acquisition; extend right/top only',
                    intensity_for_training='Original unfiltered noisy line integrals'),
                per_view=diagnostics,spot_measurements=spots,
                spot_measurement_definition='Observation-only high-pass bright round candidates in six preset views; FWHM of connected peak; not calibrated bead diameters or GT landmarks',
                source_sha256=sha256(Path(__file__)))
    (out/'roi.json').write_text(json.dumps(record,indent=2)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for ax,index in zip(axs.flat,selected):
        image=np.array(images[index]);x0,y0,x1,y1=boxes[index]
        ax.imshow(image,origin='lower',cmap='gray',vmin=0,vmax=np.quantile(image,.999))
        ax.add_patch(Rectangle((x0-.5,y0-.5),width,height,fill=False,edgecolor='#ee9222',lw=1.5))
        ax.axhline(y0-.5,color='#ee9222',ls=':',lw=.8)
        ax.set_title(f'View {index}; ROI x={x0}:{x1}, y={y0}:{y1}');ax.set_xlabel('Detector column');ax.set_ylabel('Detector row')
    fig.suptitle('Frozen training ROI: cylinder bounding box, bottom plate excluded\nDetected only from Poisson observations; original full detector shown')
    fig.savefig(out/'roi_preview.png',dpi=150);plt.close(fig)
    spot=max(spots,key=lambda s:min(s['width_fwhm_px'],s['height_fwhm_px']))
    x,y=spot['x'],spot['y'];patch=images[spot['view'],y-22:y+23,x-22:x+23]
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,k in zip(axs,[31,15,9]):
        ax.imshow(patch,origin='lower',cmap='gray',extent=[-22.5,22.5,-22.5,22.5])
        ax.add_patch(Rectangle((-k/2,-k/2),k,k,fill=False,edgecolor='#ee9222',lw=1.5))
        ax.set_title(f'Signed LNCC {k} x {k}');ax.set_xlabel('Pixel offset');ax.set_ylabel('Pixel offset')
    fig.suptitle(f'Observed bright spot in view {spot["view"]}; same unfiltered patch in every panel')
    fig.savefig(out/'kernel_windows.png',dpi=170);plt.close(fig)
    sizes=np.array([[s['width_fwhm_px'],s['height_fwhm_px']] for s in spots])
    print(json.dumps(dict(crop_shape=[height,width],pixel_fraction=record['pixel_fraction'],
                         candidates=len(spots),fwhm_quantiles=np.quantile(sizes,[0,.5,.95,1],axis=0).tolist()),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=BASE/'input')
    parser.add_argument('--out-dir',type=Path,default=BASE/'roi_study/input_roi')
    args=parser.parse_args();prepare(args.input_dir,args.out_dir)
