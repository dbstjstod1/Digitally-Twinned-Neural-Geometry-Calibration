"""Extract fixed connected-component ball landmarks from an attenuation raw.

CPU only. Dimensions and voxel spacing must be supplied explicitly because a
headerless raw cannot establish its physical units. These landmarks are for
post-hoc geometry evaluation, never calibration initialization or supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def weighted_component_center(section, component, threshold, start_zyx):
    """Use only this fixed component, with weights max(mu-threshold, 0)."""
    weights = np.where(component, np.maximum(section.astype(np.float64)-threshold, 0.), 0.)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return None, 0.
    coordinates = np.indices(weights.shape, dtype=np.float64)
    center = np.array([(coordinates[i]*weights).sum()/weight_sum for i in range(3)])
    return np.asarray(start_zyx, dtype=np.float64)+center, weight_sum


def extract(volume_path, *, shape_zyx, voxel_mm, threshold=.05,
            sensitivity_threshold=.1, expected_components=None,
            known_sha256=None, metadata_source=None):
    path = Path(volume_path)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape <= 0) or not np.isfinite(voxel_mm) or voxel_mm <= 0:
        raise ValueError('Supply three positive dimensions and a positive finite voxel size.')
    if not (np.isfinite(threshold) and np.isfinite(sensitivity_threshold)
            and 0 < threshold < sensitivity_threshold):
        raise ValueError('Thresholds must be finite and 0 < threshold < sensitivity threshold.')
    if known_sha256 is not None:
        if len(known_sha256) != 64 or any(c not in '0123456789abcdef' for c in known_sha256.lower()):
            raise ValueError('Supplied SHA256 must contain 64 hexadecimal characters.')
        known_sha256 = known_sha256.lower()
    expected_bytes = int(np.prod(shape))*4
    if path.stat().st_size != expected_bytes:
        raise ValueError(f'Raw byte count {path.stat().st_size} differs from {expected_bytes}.')
    origin_xyz = -.5*(shape[::-1]-1)*voxel_mm
    volume = np.memmap(path, dtype='<f4', mode='r', shape=tuple(shape))
    high = np.empty(tuple(shape), dtype=bool)
    nonzero_low = shape.copy()
    nonzero_high = np.full(3, -1, dtype=np.int64)
    histogram = Counter()
    finite_count = nonzero_count = 0
    hasher = hashlib.sha256() if known_sha256 is None else None
    for start in range(0, int(shape[0]), 16):
        section = volume[start:start+16]
        if hasher is not None:
            hasher.update(memoryview(section).cast('B'))
        finite_count += int(np.isfinite(section).sum())
        values, counts = np.unique(section, return_counts=True)
        histogram.update(dict(zip(values.tolist(), counts.tolist())))
        high[start:start+16] = section > threshold
        nonzero = section != 0
        nonzero_count += int(nonzero.sum())
        zz, yy, xx = np.where(nonzero)
        if len(zz):
            nonzero_low = np.minimum(nonzero_low, [start+zz.min(), yy.min(), xx.min()])
            nonzero_high = np.maximum(nonzero_high, [start+zz.max(), yy.max(), xx.max()])
    if finite_count != int(np.prod(shape)) or min(histogram) < 0:
        raise ValueError('Expected finite nonnegative linear attenuation values.')
    if nonzero_count == 0:
        raise ValueError('Raw contains no object.')
    source_hash = known_sha256 if known_sha256 is not None else hasher.hexdigest()
    connectivity = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(high, structure=connectivity, output=np.uint32)
    del high
    if expected_components is not None and count != expected_components:
        raise ValueError(f'Found {count} components, expected {expected_components}; no filtering or rematching is applied.')
    if count == 0:
        raise ValueError('No component exceeds the supplied threshold.')
    landmarks, sensitivity = [], []
    for component_label, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            raise ValueError('Unexpected empty component label.')
        start = np.array([item.start for item in box], dtype=np.int64)
        stop = np.array([item.stop for item in box], dtype=np.int64)
        section = volume[box]
        component = labels[box] == component_label
        index, mass = weighted_component_center(section, component, threshold, start)
        xyz = origin_xyz+voxel_mm*index[::-1]
        identifier = component_label-1
        landmark = dict(
            landmark_id=identifier, component_label=component_label,
            index_zyx=index.tolist(), xyz_mm=xyz.tolist(),
            threshold_voxel_count=int(component.sum()),
            threshold_bbox_index_zyx_start_inclusive=start.tolist(),
            threshold_bbox_index_zyx_stop_exclusive=stop.tolist(),
            threshold_bbox_edge_xyz_mm=[
                (origin_xyz+voxel_mm*(start[::-1]-.5)).tolist(),
                (origin_xyz+voxel_mm*(stop[::-1]-.5)).tolist()],
            threshold_bbox_size_xyz_mm=((stop-start)[::-1]*voxel_mm).tolist(),
            component_maximum_attenuation_mm_inverse=float(section[component].max()),
            integrated_threshold_contrast_mm2=mass*voxel_mm**3,
        )
        landmarks.append(landmark)
        # A higher threshold is a subset of the primary component. Membership
        # supplies the correspondence, without nearest-neighbour rematching.
        alternate_mask = component & (section > sensitivity_threshold)
        _, alternate_count = ndimage.label(alternate_mask, structure=connectivity)
        alternate_index, _ = weighted_component_center(section, alternate_mask, sensitivity_threshold, start)
        alternate_xyz = None if alternate_index is None else origin_xyz+voxel_mm*alternate_index[::-1]
        sensitivity.append(dict(
            landmark_id=identifier, connected_components_at_higher_threshold=int(alternate_count),
            threshold_voxel_count=int(alternate_mask.sum()),
            index_zyx=None if alternate_index is None else alternate_index.tolist(),
            xyz_mm=None if alternate_xyz is None else alternate_xyz.tolist(),
            centroid_shift_mm=None if alternate_xyz is None else float(np.linalg.norm(alternate_xyz-xyz)),
        ))
    centers = np.array([item['xyz_mm'] for item in landmarks])
    shifts = np.array([item['centroid_shift_mm'] for item in sensitivity if item['centroid_shift_mm'] is not None])
    sensitivity_record = dict(
        primary_threshold_mm_inverse=threshold, sensitivity_threshold_mm_inverse=sensitivity_threshold,
        correspondence='Original primary-component membership and fixed landmark_id; no spatial rematching.',
        topology_preserved_for_all_components=all(item['connected_components_at_higher_threshold'] == 1 for item in sensitivity),
        surviving_primary_components=len(shifts),
        maximum_centroid_shift_mm=None if not len(shifts) else float(shifts.max()),
        rms_centroid_shift_mm=None if not len(shifts) else float(np.sqrt(np.mean(shifts**2))),
        components=sensitivity,
    )
    common = dict(
        volume_filename=path.name, volume_sha256=source_hash, volume_bytes=expected_bytes,
        volume_sha256_source='Supplied precomputed hash' if known_sha256 else 'Computed while scanning all raw bytes',
        shape_zyx=shape.tolist(), raw_dtype='little-endian float32; C-order z,y,x',
        voxel_size_mm=voxel_mm, first_voxel_xyz_mm=origin_xyz.tolist(),
        physical_metadata_source=metadata_source or 'Explicit caller-supplied isotropic voxel size and linear attenuation units; raw contains no physical header.',
        coordinate_rule='xyz_mm = first_voxel_xyz_mm + voxel_size_mm * index_zyx[::-1]',
        coordinate_permutation_applied=False,
        threshold_mm_inverse=threshold, connectivity='6-connected faces in the 3-D index grid',
        centroid_method='Within each fixed mu > threshold component, weighted mean of voxel-centre indices with weights max(mu-threshold,0). No fitted sphere, local background subtraction or neighbouring-ring assumption.',
        landmark_count=count,
        landmark_id_rule='Zero-based original scipy.ndimage.label component label in C-order z,y,x scanning; IDs remain fixed under camera optimization.',
        use='Geometric evaluation only; not initialization, optimization supervision or checkpoint selection.',
    )
    landmark_record = dict(**common,
        attenuation_min_max_mm_inverse=[min(histogram), max(histogram)], nonzero_voxel_count=nonzero_count,
        landmark_bounds_xyz_mm=[centers.min(0).tolist(), centers.max(0).tolist()],
        threshold_sensitivity=sensitivity_record, landmarks=landmarks)
    audit = dict(**common,
        finite_voxel_count=finite_count, total_voxel_count=int(np.prod(shape)),
        attenuation_min_max_mm_inverse=[min(histogram), max(histogram)],
        distinct_attenuation_values=len(histogram), nonzero_voxel_count=nonzero_count,
        dominant_values=[dict(value_mm_inverse=value, voxel_count=n) for value, n in histogram.most_common(20)],
        nonzero_bbox_index_zyx_start_inclusive=nonzero_low.tolist(),
        nonzero_bbox_index_zyx_stop_exclusive=(nonzero_high+1).tolist(),
        nonzero_bbox_voxel_centre_xyz_mm=[(origin_xyz+voxel_mm*nonzero_low[::-1]).tolist(),
                                         (origin_xyz+voxel_mm*nonzero_high[::-1]).tolist()],
        nonzero_bbox_voxel_edge_xyz_mm=[(origin_xyz+voxel_mm*(nonzero_low[::-1]-.5)).tolist(),
                                       (origin_xyz+voxel_mm*(nonzero_high[::-1]+.5)).tolist()],
        component_voxel_counts=[item['threshold_voxel_count'] for item in landmarks],
        threshold_sensitivity=sensitivity_record,
        limits='Thresholded component extents and weighted centroids describe the supplied sampled volume. They are not claims of exact manufactured sphere dimensions or centres.')
    return landmark_record, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume', type=Path, required=True)
    parser.add_argument('--shape-zyx', type=int, nargs=3, required=True)
    parser.add_argument('--voxel-mm', type=float, required=True)
    parser.add_argument('--threshold', type=float, default=.05)
    parser.add_argument('--sensitivity-threshold', type=float, default=.1)
    parser.add_argument('--expect-components', type=int)
    parser.add_argument('--volume-sha256')
    parser.add_argument('--metadata-source')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    landmarks, audit = extract(args.volume, shape_zyx=args.shape_zyx, voxel_mm=args.voxel_mm,
        threshold=args.threshold, sensitivity_threshold=args.sensitivity_threshold,
        expected_components=args.expect_components, known_sha256=args.volume_sha256,
        metadata_source=args.metadata_source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, result in [('landmarks.json', landmarks), ('phantom_audit.json', audit)]:
        destination = args.out_dir/filename
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('Wrote', destination)
    print(json.dumps(dict(landmark_count=landmarks['landmark_count'], volume_sha256=landmarks['volume_sha256'],
        threshold_sensitivity={key: value for key, value in landmarks['threshold_sensitivity'].items() if key != 'components'}), indent=2))


if __name__ == '__main__':
    main()
