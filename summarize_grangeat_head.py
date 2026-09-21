"""Reproduce matched-ROI CQ500 FDK/Grangeat measurements from saved arrays.

CPU only; reads numerical outputs and writes one JSON summary. The final nominal
and enlarged-panel runs must use 128x512 target-plane quadrature; the nominal
quadrature_64x256 archive supplies an optional convergence comparison. Source/runtime
hashes are preserved, with any later figure-generation provenance kept separate.
All methods are measured on the same prescribed nominal masks, then on a second
common subset where the enlarged-panel strict result is finite.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np

from sinespin_geometry import SineSpinGeometry
from run_grangeat_head import visibility_masks, score_regions

def read_json(path):
    return json.loads(path.read_text())

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            value.update(chunk)
    return value.hexdigest()

def load_geometry(folder, run):
    geometries = {}
    for name, rec in run['geometry'].items():
        with np.load(folder/(name+'_geometry.npz')) as a:
            geometries[name] = SineSpinGeometry(
                kind=rec['kind'], source_positions=a['source_positions'], module_centers=a['module_centers'],
                row_vectors=a['row_vectors'], col_vectors=a['col_vectors'], theta_deg=a['theta_deg'], tilt_deg=a['tilt_deg'],
                detector_rows=rec['detector_shape_vu'][0], detector_cols=rec['detector_shape_vu'][1],
                pixel_height=rec['pixel_vu_mm'][0], pixel_width=rec['pixel_vu_mm'][1],
                sod_mm=rec['sod_mm'], sdd_mm=rec['sdd_mm'], scan_angle_deg=rec['scan_angle_deg'])
    return geometries

def main(argv=None):
    repository_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-dir', type=Path,
                        default=repository_root/'result_sinespin/cq500_grangeat')
    parser.add_argument('--control-dir', type=Path,
                        default=repository_root/'result_sinespin/cq500_grangeat_untruncated')
    parser.add_argument('--summary-json', type=Path,
                        default=repository_root/'docs/cq500_grangeat_summary.json')
    parser.add_argument('--skip-control', action='store_true',
                        help='Summarize the nominal run without an enlarged-detector control.')
    args = parser.parse_args(argv)
    nominal_dir, control_dir = args.nominal_dir.resolve(), args.control_dir.resolve()
    coarse_dir = nominal_dir/'quadrature_64x256'
    output = args.summary_json.resolve()

    def display_path(path):
        try:
            return str(path.resolve().relative_to(repository_root))
        except ValueError:
            return str(path.resolve())

    nominal = read_json(nominal_dir/'metrics.json')
    coarse = read_json(coarse_dir/'metrics.json') if (coarse_dir/'metrics.json').exists() else None
    control = None if args.skip_control else read_json(control_dir/'metrics.json')
    run_specs = [('nominal', nominal, (128,512))]
    if coarse is not None: run_specs.append(('coarse', coarse, (64,256)))
    if control is not None: run_specs.append(('control', control, (128,512)))
    for name, run, angles in run_specs:
        if not ((run['methods']['n_polar'],run['methods']['n_azimuth']) == angles):
            raise ValueError(f'{name}: stale quadrature')
        d = run['results']['sinespin_220']['diagnostics']
        if not (d.get('rebinning', '').startswith('Target plane/source intersection first')):
            raise ValueError(f'{name}: stale rebinning')
        if not (run['methods']['radon_step_mm'] == 1.):
            raise ValueError("Summary validation failed: run['methods']['radon_step_mm'] == 1.")
        if not (run['reconstruction_grid']['voxel_mm'] == 2.):
            raise ValueError("Summary validation failed: run['reconstruction_grid']['voxel_mm'] == 2.")
        if not (d['reconstruction_iterations'] == 0 and not d['data_fitted_scale'] and not d['positivity_clipping']):
            raise ValueError("Summary validation failed: d['reconstruction_iterations'] == 0 and not d['data_fitted_scale'] and not d['positivity_clipping']")
        if not (run['input_sha256'] == nominal['input_sha256']):
            raise ValueError(f'{name}: different input')
        if not (run['source_sha256'] == nominal['source_sha256']):
            raise ValueError(f'{name}: different implementation')
        if not (run['runtime']['leap_sha256'] == nominal['runtime']['leap_sha256']):
            raise ValueError("Summary validation failed: run['runtime']['leap_sha256'] == nominal['runtime']['leap_sha256']")
    if coarse is not None and nominal['geometry'] != coarse['geometry']:
        raise ValueError("Summary validation failed: nominal['geometry'] == coarse['geometry']")
    if nominal['detector']['padding_factor'] != 1 or (control is not None and control['detector']['padding_factor'] != 2):
        raise ValueError("Summary validation failed: nominal['detector']['padding_factor'] == 1 and control['detector']['padding_factor'] == 2")
    if not (nominal['input']['requested_center_shift_xyz_mm'] == [0.,0.,0.]):
        raise ValueError("Summary validation failed: nominal['input']['requested_center_shift_xyz_mm'] == [0.,0.,0.]")

    geometries = load_geometry(nominal_dir, nominal)
    coordinate_error = []
    if control is not None:
        control_geometries = load_geometry(control_dir, control)
        normal_g, control_g = geometries['sinespin_220'], control_geometries['sinespin_220']
        for x,y in zip(normal_g.modular_arrays(dtype=np.float64),control_g.modular_arrays(dtype=np.float64)):
            np.testing.assert_array_equal(x,y)
        if not (normal_g.pixel_height == control_g.pixel_height and normal_g.pixel_width == control_g.pixel_width):
            raise ValueError('Summary validation failed: normal_g.pixel_height == control_g.pixel_height and normal_g.pixel_width == control_g.pixel_width')
        if not (control_g.detector_rows == 2*normal_g.detector_rows and control_g.detector_cols == 2*normal_g.detector_cols):
            raise ValueError('Summary validation failed: control_g.detector_rows == 2*normal_g.detector_rows and control_g.detector_cols == 2*normal_g.detector_cols')
        coordinate_error = []
        for count, padded_count, pitch in [(normal_g.detector_rows,control_g.detector_rows,normal_g.pixel_height),
                                           (normal_g.detector_cols,control_g.detector_cols,normal_g.pixel_width)]:
            a = (np.arange(count)-(count-1)/2)*pitch
            b = (np.arange(count)+(padded_count-count)//2-(padded_count-1)/2)*pitch
            coordinate_error.append(float(np.max(np.abs(a-b))))

    reference = np.load(nominal_dir/'reference_mu.npy',mmap_mode='r')
    known = np.load(nominal_dir/'reference_valid.npy')
    hull = np.load(nominal_dir/'source_hull_mask_sinespin_220.npy')
    if control is not None:
        for name in ('reference_mu.npy','reference_valid.npy','source_hull_mask_sinespin_220.npy'):
            if not (digest(nominal_dir/name) == digest(control_dir/name)):
                raise ValueError(name)
    masks, axes = visibility_masks(geometries,reference.shape,2.)
    base = known & hull & np.logical_and.reduce(list(masks.values()))
    z,y,x = axes; X,Y = np.meshgrid(x,y,indexing='xy')
    mu_water = float(nominal['input']['mu_water_per_mm'])
    hu = reference*(1000./mu_water)-1000.
    soft = (hu >= -100.) & (hu <= 150.)
    regions = {'head':base & (hu > -500.), 'soft_tissue':base & soft, 'bone':base & (hu >= 300.),
               'skullbase_soft':base & soft & ((z>=-70.) & (z< -30.))[:,None,None] & ((X*X+Y*Y)<=80.**2)[None]}
    arrays = {
        'circular_200_fdk': nominal_dir/'recon_circular_200.npy',
        'circular_220_fdk': nominal_dir/'recon_circular_220.npy',
        'nominal_sinespin_estimate': nominal_dir/'finite_detector_estimate_sinespin_220.npy',
        'nominal_sinespin_strict': nominal_dir/'recon_sinespin_220.npy',
    }
    if coarse is not None:
        arrays['coarse_sinespin_estimate'] = coarse_dir/'finite_detector_estimate_sinespin_220.npy'
    if control is not None:
        arrays['enlarged_sinespin_estimate'] = control_dir/'finite_detector_estimate_sinespin_220.npy'
        arrays['enlarged_sinespin_strict'] = control_dir/'recon_sinespin_220.npy'
    measurements = {name:score_regions(np.load(path,mmap_mode='r'),reference,regions,mu_water) for name,path in arrays.items()}
    strict_nominal = int(np.isfinite(np.load(arrays['nominal_sinespin_strict'],mmap_mode='r')).sum())
    strict_control, strict_control_regions, strict_control_matched = None, {}, {}
    if control is not None:
        strict_control_mask = np.isfinite(np.load(arrays['enlarged_sinespin_strict'],mmap_mode='r'))
        strict_control = int(strict_control_mask.sum())
        strict_control_regions = {name:mask & strict_control_mask for name,mask in regions.items()}
        strict_control_matched = {name:score_regions(np.load(path,mmap_mode='r'),reference,strict_control_regions,mu_water)
                                 for name,path in arrays.items() if name != 'nominal_sinespin_strict'}
    delta_metrics = None
    if coarse is not None:
        delta = np.load(arrays['nominal_sinespin_estimate'],mmap_mode='r') - np.load(arrays['coarse_sinespin_estimate'],mmap_mode='r')
        delta_metrics = score_regions(delta,np.zeros_like(delta),regions,mu_water)

    def run_record(folder, run):
        return {
            'directory':display_path(folder), 'reconstruction_grid':run['reconstruction_grid'],
            'quadrature':{k:run['methods'][k] for k in ('n_polar','n_azimuth','radon_step_mm')},
            'effective_line_step_mm':run['results']['sinespin_220']['diagnostics']['line_step_mm'],
            'rebinning':run['results']['sinespin_220']['diagnostics']['rebinning'],
            'runtime':{k:run['runtime'][k] for k in ('gpu','gpu_name','torch','leap_sha256')},
            'source_sha256':run['source_sha256'],
            'figure_generation':run.get('figure_generation'),
            'timings_seconds':{name:{k:arm[k] for k in ('projection_seconds','reconstruction_seconds')}
                               for name,arm in run['results'].items()},
            'projection_reuse':{name:arm['reused_projections'] for name,arm in run['results'].items()},
            'sinespin_diagnostics':{k:run['results']['sinespin_220']['diagnostics'][k] for k in
                ('hemisphere_normals','angular_weight_sum','line_seconds','rebin_seconds','total_seconds',
                 'finite_rebinned_fraction','complete_voxel_fraction','source_hull_voxel_fraction')},
        }

    crop_path = control_dir/'projection_crop_check.json'
    crop = None
    if control is not None:
        crop = read_json(crop_path) if crop_path.exists() else {
            'projection_values_compared':False,
            'note':'Optional projection_crop_check.json is absent; pixel-coordinate alignment is checked here.'}
        crop.update(bit_identical=(crop.get('max_absolute_difference') == 0.) if crop_path.exists() else None,
                    source_and_detector_poses_identical=True, pixel_pitch_identical=True,
                    central_crop_pixel_coordinate_max_difference_vu_mm=coordinate_error,
                    interpretation='Sampling coordinates align exactly; any recorded projection differences remain unchanged.')
    runs = {'nominal_final':run_record(nominal_dir,nominal)}
    if coarse is not None: runs['nominal_coarse'] = run_record(coarse_dir,coarse)
    if control is not None: runs['enlarged_control'] = run_record(control_dir,control)
    input_keys = ('patient','requested_center_shift_xyz_mm','native_index_xyz_to_lps_mm','isocenter_lps_mm',
                  'mu_water_per_mm','native_shape_zyx','dicom_slice_thickness_mm','slice_normal_spacing_mm','head_coverage_limitation')
    summary = {
        'schema_version':1, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'description':'CPU measurements of actual saved CQ500 reconstructions on identical reference-defined masks.',
        'input':{**{k:nominal['input'][k] for k in input_keys}, 'sha256':nominal['input_sha256'],
                 'forward_shape_zyx':nominal['input']['forward_shape_zyx'], 'forward_voxel_mm':nominal['input']['forward_voxel_mm']},
        'limitations':[
            'The source convex hull is a necessary support condition, not an exact reconstruction certification.',
            f'Nominal-panel strict Grangeat output has {strict_nominal} finite voxels. Its displayed finite-detector estimate is approximate under data truncation.',
            'Enlarged-detector control is a diagnostic with twice as many rows and columns, not the detector in the paper.',
            'These are idealized simulations with assumed 750/1200 mm SOD/SDD, no added noise or scatter, and no manufacturer reconstruction implementation.',
            ('Only two angular quadrature levels were compared; their nonzero difference does not establish convergence.' if coarse is not None else 'No coarse quadrature archive is available; quadrature convergence was not assessed.'),
        ],
        'geometry':{'nominal':nominal['geometry'],'enlarged_control':control['geometry'] if control is not None else None},
        'methods':{k:nominal['methods'][k] for k in ('forward','circular','sinespin','strict','partial','finite_detector_estimate')},
        'runs':runs,
        'identical_roi_definition':{
            'base':'Acquired source CT AND every-view detector visibility in all three nominal protocols AND saved Sine Spin source convex hull.',
            'coordinates':'Isocentre-centred LPS; voxel centres (index-(size-1)/2)*2 mm.',
            'head':'Reference HU > -500', 'soft_tissue':'-100 <= reference HU <= 150', 'bone':'Reference HU >= 300',
            'skullbase_soft':'Base AND -70 <= z < -30 mm AND x^2+y^2 <= 80^2 mm^2 AND -100 <= reference HU <= 150.',
            'voxel_counts':{name:int(mask.sum()) for name,mask in regions.items()},
            'error_hu':'(reconstruction_mu-reference_mu)*1000/mu_water; float64 CPU accumulation; no fitted scale or clipping.',
            'mask_rule':'Every reported reconstruction below uses exactly these masks; finite requested/evaluated counts are retained.'},
        'identical_roi_measurements':measurements,
        'strict_control_matched_subset':{
            'definition':'Additional comparison: each prescribed nominal ROI intersected with the finite strict enlarged-control output. All methods below use the same restricted voxels; full original ROI measurements above are unchanged.',
            'voxel_counts':{name:int(mask.sum()) for name,mask in strict_control_regions.items()},
            'measurements':strict_control_matched} if control is not None else None,
        'quadrature_change':{'coarse':[64,256],'fine':[128,512], 'same_source_hashes_inputs_and_geometry':True,
                             'fine_minus_coarse_error_hu':delta_metrics,
                             'skullbase_rmse_change_hu':measurements['nominal_sinespin_estimate']['skullbase_soft']['rmse_hu']-measurements['coarse_sinespin_estimate']['skullbase_soft']['rmse_hu']} if coarse is not None else None,
        'strict_finite_voxel_counts':{'nominal':strict_nominal,'enlarged_control':strict_control},
        'projection_crop_check':crop,
    }
    artifacts = list(arrays.values()) + [nominal_dir/'reference_mu.npy',nominal_dir/'reference_valid.npy',
                                         nominal_dir/'source_hull_mask_sinespin_220.npy',nominal_dir/'metrics.json']
    if coarse is not None: artifacts.append(coarse_dir/'metrics.json')
    if control is not None:
        artifacts.append(control_dir/'metrics.json')
        if crop_path.exists(): artifacts.append(crop_path)
    summary['artifact_sha256'] = {display_path(p):digest(p) for p in artifacts}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    reloaded = read_json(output)
    if reloaded['strict_finite_voxel_counts']['nominal'] != strict_nominal:
        raise RuntimeError('Written strict count differs from measured count')
    print(f'Wrote {output} ({output.stat().st_size:,} bytes)')
    print('Skull-base HU RMSE on prescribed nominal ROI:')
    for name, values in measurements.items():
        row = values['skullbase_soft']
        rmse = 'unavailable' if row['rmse_hu'] is None else f"{row['rmse_hu']:.6f}"
        print(f"  {name}: {rmse}; finite {row['evaluated_voxels']}/{row['requested_voxels']}")
    if control is not None: print('Same comparison restricted to finite strict-control voxels:')
    for name, values in strict_control_matched.items():
        row = values['skullbase_soft']
        print(f"  {name}: {row['rmse_hu']:.6f}; voxels {row['evaluated_voxels']}")
    return summary


if __name__ == '__main__':
    main()
