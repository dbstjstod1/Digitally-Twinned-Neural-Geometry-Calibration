"""Audit saved detector dimensions and head placement without reconstructing data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sinespin_geometry import SineSpinGeometry, build_icono_orbit
from sim_sinespin_recon import save_json


def saved_geometry(out, name, record):
    with np.load(out/(name+'_geometry.npz')) as pose:
        geometry = SineSpinGeometry(
            kind=record['kind'], source_positions=pose['source_positions'],
            module_centers=pose['module_centers'], row_vectors=pose['row_vectors'],
            col_vectors=pose['col_vectors'], theta_deg=pose['theta_deg'], tilt_deg=pose['tilt_deg'],
            detector_rows=record['detector_shape_vu'][0], detector_cols=record['detector_shape_vu'][1],
            pixel_height=record['pixel_vu_mm'][0], pixel_width=record['pixel_vu_mm'][1],
            sod_mm=record['sod_mm'], sdd_mm=record['sdd_mm'], scan_angle_deg=record['scan_angle_deg'])
        np.testing.assert_allclose(geometry.projection_matrices(), pose['P_pixel'], atol=1e-9)
    return geometry


def translate_z(array, offset_voxels):
    """Translate along z with zero padding; never wrap source anatomy."""
    result = np.zeros_like(array)
    source_start, target_start = max(0, -offset_voxels), max(0, offset_voxels)
    count = min(len(array)-source_start, len(array)-target_start)
    if count > 0:
        result[target_start:target_start+count] = array[source_start:source_start+count]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('result_sinespin/cq500_fig9'))
    parser.add_argument('--input-dir', type=Path)
    args = parser.parse_args()
    out = args.out_dir
    input_dir = args.input_dir or out/'input'
    run = json.loads((out/'metrics.json').read_text())
    meta = json.loads((input_dir/'head_metadata.json').read_text())
    if meta != run['input']:
        raise ValueError('Input metadata differs from the completed reconstruction')
    truth = np.load(input_dir/'reference_mu.npy')
    known = np.load(input_dir/'reference_valid.npy')
    voxel, mu_water = meta['voxel_mm'], meta['mu_water_per_mm']
    z, y, x = [(np.arange(n)-(n-1)/2)*voxel for n in truth.shape]
    shift = meta['requested_center_shift_xyz_mm'][2]
    offset = int(round(-shift/voxel))
    if not np.isclose(offset*voxel, -shift):
        raise ValueError('This reference-only audit requires an integer-voxel z translation')
    head = known & (truth > .5*mu_water)  # Acquired HU > -500; not anatomical segmentation.
    centred_head = translate_z(head, offset)
    if int(centred_head.sum()) != int(head.sum()):
        raise ValueError('Undoing placement clips measured non-air voxels')
    centred_truth = translate_z(truth, offset)
    occupied_zyx = [np.flatnonzero(head.any(axis=other))
                    for other in ((1,2),(0,2),(0,1))]
    lower_xyz = [axis[index[0]] for axis,index in zip((z,y,x),occupied_zyx)][::-1]
    upper_xyz = [axis[index[-1]] for axis,index in zip((z,y,x),occupied_zyx)][::-1]
    X, Y = np.meshgrid(x, y, indexing='xy')
    xy = np.stack((X, Y), axis=-1)
    geometries = {n: saved_geometry(out, n, r) for n, r in run['geometry'].items()}
    audit = dict(
        paper_url='https://doi.org/10.1117/1.JMI.11.4.043503',
        paper_table1_reconstruction_xyz_mm={'circular': [248,248,182], 'sinespin': [249,249,181]},
        computational_box_xyz_mm=(np.array(truth.shape[::-1])*voxel).tolist(),
        numerical_voxel_mm=voxel, paper_voxel_mm=.485,
        assumptions=['SOD/SDD and sine phase are nominal, not measured manufacturer poses',
                     'Detector footprint/arc/tilt follow the paper; full-FOV scale is consistent with Table 1',
                     'All-view visibility is not manufacturer reconstruction support or exact completeness'],
        display=dict(skullbase_zoom_horizontal_mm=[-90,90], skullbase_zoom_z_mm=[10,90],
                     skullbase_zoom_extent_mm=[180,80], previous_raw_z_mm=[-30,185],
                     full_scan_overview_z_mm=[-100,100]),
        placement=dict(saved_head_shift_z_mm=shift, reference_only_comparison_head_shift_z_mm=0,
                       mask='acquired CT voxels with HU > -500; includes any support material',
                       voxel_count=int(head.sum()), nonair_bounds_xyz_mm=[lower_xyz,upper_xyz],
                       new_reconstruction_performed=False), geometry={})
    for name, g in geometries.items():
        bounds = g.longitudinal_intervals(xy)
        visible = (z[:,None,None]>=bounds[None,:,:,0]) & (z[:,None,None]<=bounds[None,:,:,1])
        binned = [build_icono_orbit(g.kind, detector_bin=b, n_views=g.n_views,
                                  scan_angle_deg=g.scan_angle_deg, sod_mm=g.sod_mm, sdd_mm=g.sdd_mm)
                  for b in (1,2,4)]
        central = g.longitudinal_intervals([[0,0]])[0]
        audit['geometry'][name] = dict(
            detector_uv_mm=[g.detector_width_mm,g.detector_height_mm],
            pixel_uv_mm=[g.pixel_width,g.pixel_height], samples_uv=[g.detector_cols,g.detector_rows],
            measured_source_radius_min_max_mm=[float(np.linalg.norm(g.source_positions,axis=1).min()),
                                               float(np.linalg.norm(g.source_positions,axis=1).max())],
            measured_sdd_min_max_mm=[float(np.linalg.norm(g.module_centers-g.source_positions,axis=1).min()),
                                    float(np.linalg.norm(g.module_centers-g.source_positions,axis=1).max())],
            untilted_isocenter_footprint_uv_mm=[g.detector_width_mm*g.sod_mm/g.sdd_mm,
                                               g.detector_height_mm*g.sod_mm/g.sdd_mm],
            every_view_isocenter_z_mm=central.tolist(), every_view_isocenter_height_mm=float(np.diff(central)[0]),
            bin_factors_1_2_4_footprint_uv_mm=[[a.detector_width_mm,a.detector_height_mm] for a in binned],
            saved_placement_visible_fraction=float(np.count_nonzero(head&visible)/head.sum()),
            centred_reference_visible_fraction=float(np.count_nonzero(centred_head&visible)/head.sum()))
    save_json(out/'fov_audit.json',audit)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(10,10),layout='constrained')
    xi, yi = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))
    planes = [('Sagittal x≈0',y,lambda a:a[:,:,xi]), ('Coronal y≈0',x,lambda a:a[:,yi,:])]
    sine = geometries['sinespin_220']
    for row,(label,horizontal,slicer) in enumerate(planes):
        points = np.column_stack((np.full_like(y,x[xi]),y) if row==0 else (x,np.full_like(x,y[yi])))
        limits = sine.longitudinal_intervals(points)
        for col,(volume,description) in enumerate(((truth,f'Current head shift +{shift:g} mm'),
                                                  (centred_truth,'Reference at original centre: shift 0 mm'))):
            ax=axes[row,col]
            ax.imshow(slicer(volume)*(1000/mu_water)-1000,origin='lower',cmap='gray',vmin=-110,vmax=210,
                      extent=[horizontal[0]-voxel/2,horizontal[-1]+voxel/2,z[0]-voxel/2,z[-1]+voxel/2])
            ax.plot(horizontal,limits[:,0],color='cyan',lw=1)
            ax.plot(horizontal,limits[:,1],color='cyan',lw=1)
            ax.axhline(0,color='yellow',ls='--',lw=.8)
            ax.set(title=description,xlabel='posterior y [mm]' if row==0 else 'left x [mm]',
                   ylabel=label+'\nsuperior z [mm]',ylim=(-120,195))
    fig.suptitle('REFERENCE-ONLY placement audit: unchanged Sine detector/orbit\n'
                 'Cyan: every-view visibility; yellow: circular source plane; no new reconstruction')
    fig.savefig(out/'head_placement_fov.png',dpi=160)
    plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
