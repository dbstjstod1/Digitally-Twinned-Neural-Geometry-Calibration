"""Prepare the predetermined CQ500 case on independent forward/reconstruction grids."""
import argparse
from pathlib import Path
import numpy as np
from cq500_head import read_cq500_head, resample_cq500_head, source_coverage_mask
from sim_sinespin_recon import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/mirlab/Desktop/Flow_matching_motion_3D/data/CQ500'))
    parser.add_argument('--out-dir', type=Path, default=Path('result_sinespin/cq500_fig9/input'))
    parser.add_argument('--test-index', type=int, default=0)
    parser.add_argument('--head-shift-z-mm', type=float, default=100.0,
                        help='Chosen from the reference anatomy before reconstruction: skull base near z=+50 mm.')
    args = parser.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    head = read_cq500_head(args.data_root, args.test_index)
    print(f"[input] CQ500CT{head.metadata['patient']}, native {head.native_hu.shape}", flush=True)
    shift = (0, 0, args.head_shift_z_mm)
    coarse, metadata = resample_cq500_head(head, (384,256,256), 1.0, center_shift_xyz_mm=shift)
    valid = source_coverage_mask(metadata)
    np.save(out/'reference_mu.npy', coarse)
    np.save(out/'reference_valid.npy', valid)
    print(f'[reference] {coarse.shape}, acquired CT coverage {valid.mean():.3f}', flush=True)
    # Validate that padding/cropping has not removed any acquired non-air tissue.
    affine = head.native_index_xyz_to_lps_mm
    iso = np.asarray(metadata['isocenter_lps_mm'])
    xx, yy = np.meshgrid(np.arange(head.native_hu.shape[2]), np.arange(head.native_hu.shape[1]))
    lateral = xx[...,None]*affine[:3,0] + yy[...,None]*affine[:3,1] + affine[:3,3]-iso
    half_extent = np.array([128,128,192])
    cropped = 0
    for i in range(head.native_hu.shape[0]):
        world = lateral + i*affine[:3,2]
        outside = (np.abs(world)>half_extent).any(-1)
        cropped += int(np.count_nonzero(outside & (head.native_hu[i]>-500)))
    metadata['nonair_native_centres_outside_target_box'] = cropped
    if cropped:
        raise RuntimeError(f'Target box clips {cropped} native non-air voxel centres; enlarge it before reconstruction')
    fine, fine_metadata = resample_cq500_head(head,(768,512,512),.5,center_shift_xyz_mm=shift)
    np.save(out/'forward_mu.npy',fine)
    metadata.update(recon_voxel_mm=1.0, forward_voxel_mm=.5, forward_shape_zyx=list(fine.shape),
                    placement_rationale='Reference-only anatomical inspection before reconstruction: native skull base approximately -50 mm relative to series centre; +100 mm translation puts it near +50 mm, away from the circular source plane.',
                    fixed_skullbase_region=dict(z_mm=[30,70],radius_mm=80,soft_tissue_hu=[-100,150]),
                    head_coverage_limitation='Selected head CT excludes much of the jaw and neck shown in paper Fig. 9.',
                    fine_sampling=fine_metadata)
    save_json(out/'head_metadata.json',metadata)
    print(f'[forward] {fine.shape} @0.5mm; preparation complete: {out}',flush=True)


if __name__ == '__main__':
    main()
