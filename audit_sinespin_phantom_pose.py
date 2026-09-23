"""Estimate one phantom-frame rigid transform from observed bead projections.

This is a sparse 3-D bead reconstruction, not an attenuation-volume reconstruction.
Frozen fitted cameras supply patch associations and triangulation. Ground-truth
cameras are read only after the transform and all sensitivity estimates are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration_gauge import fit_rigid_landmark_pose, transform_points as transform


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def project(pmat, points):
    homogeneous = np.einsum('vij,bj->vbi', pmat, np.c_[points, np.ones(len(points))])
    return homogeneous[..., :2] / homogeneous[..., 2:]


def rigid_registration(source, target):
    """Return H mapping source column coordinates to target, no scale/reflection."""
    return fit_rigid_landmark_pose(source, target)['H']


def pose_description(matrix):
    return dict(H_estimated_frame_to_reference=matrix.tolist(),
                translation_mm=matrix[:3, 3].tolist(),
                rotation_xyz_extrinsic_deg=Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True).tolist(),
                rotation_angle_deg=float(np.rad2deg(Rotation.from_matrix(matrix[:3, :3]).magnitude())))


def difference(a, b):
    relative = a @ np.linalg.inv(b)
    return dict(rotation_angle_deg=float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude())),
                translation_norm_mm=float(np.linalg.norm(relative[:3, 3])))


def detect(target, projected, *, inner_radius=4.5, annulus_inner=5.,
           annulus_outer=7., seed_offset=(0., 0.)):
    """Signed contrast moments: no positivity clipping or centre regularizer.

    A full-support aperture surrounds the bead. An affine background is fit only
    in its annulus. Keeping signed background-subtracted moments avoids the
    noise-dependent attraction to the aperture centre caused by positive clipping.
    """
    nviews, nbeads = projected.shape[:2]
    uv = np.full((nviews, nbeads, 2), np.nan)
    diagnostics = np.full((nviews, nbeads, 3), np.nan)
    rejected = Counter()
    radius = int(np.ceil(annulus_outer)) + 1
    height, width = target.shape[1:]
    yy0, xx0 = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    for view in range(nviews):
        distances = np.linalg.norm(projected[view, :, None] - projected[view, None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        image = target[view]
        for bead in range(nbeads):
            if distances[bead].min() < 2 * annulus_outer + 2:
                rejected['another_bead_within_background_support'] += 1
                continue
            u, v = projected[view, bead] + np.asarray(seed_offset)
            c, r = int(round(u)), int(round(v))
            if c < radius or r < radius or c + radius >= width or r + radius >= height:
                rejected['detector_boundary'] += 1
                continue
            patch = image[r-radius:r+radius+1, c-radius:c+radius+1].astype(np.float64)
            xx, yy = xx0+c, yy0+r
            radial = np.hypot(xx-u, yy-v)
            annulus = (radial >= annulus_inner) & (radial <= annulus_outer)
            design = np.c_[np.ones(annulus.sum()), xx[annulus]-u, yy[annulus]-v]
            coefficients = np.linalg.lstsq(design, patch[annulus], rcond=None)[0]
            contrast = patch - (coefficients[0] + coefficients[1]*(xx-u) + coefficients[2]*(yy-v))
            support = radial <= inner_radius
            peak = float(contrast[support].max())
            background_rmse = float(np.sqrt(np.mean(contrast[annulus]**2)))
            # Reject body boundaries/poor local backgrounds before triangulation.
            if peak <= 0 or background_rmse > .15 * peak:
                rejected['nonaffine_background_or_low_contrast'] += 1
                continue
            weights = contrast * support  # signed: deliberately not max(contrast,0)
            mass = weights.sum()
            if mass < 1.:
                rejected['insufficient_contrast_integral'] += 1
                continue
            center = np.array([(xx*weights).sum(), (yy*weights).sum()]) / mass
            if np.linalg.norm(center - projected[view, bead]) > 2.:
                rejected['association_distance_above_2px'] += 1
                continue
            uv[view, bead] = center
            diagnostics[view, bead] = mass, peak, background_rmse
    accepted = np.isfinite(uv[..., 0])
    if int(accepted.sum()) + sum(rejected.values()) != nviews*nbeads:
        raise AssertionError('Incomplete detection accounting')
    return uv, diagnostics, dict(rejected)


def triangulate(pmat, uv, views=None):
    """Fixed cameras; only each bead's three coordinates are optimized."""
    points, records = [], []
    permitted = np.ones(len(pmat), dtype=bool)
    if views is not None:
        permitted[:] = False
        permitted[views] = True
    for bead in range(uv.shape[1]):
        indices = np.flatnonzero(np.isfinite(uv[:, bead, 0]) & permitted)
        if len(indices) < 20:
            raise ValueError(f'Bead {bead} has only {len(indices)} independent view observations')
        matrix, measured = pmat[indices], uv[indices, bead]
        design = np.concatenate((measured[:, 0, None]*matrix[:, 2] - matrix[:, 0],
                                 measured[:, 1, None]*matrix[:, 2] - matrix[:, 1]))
        _, _, vt = np.linalg.svd(design, full_matrices=False)
        initial = vt[-1, :3]/vt[-1, 3]

        def residual(point):
            q = matrix @ np.r_[point, 1.]
            return (q[:, :2]/q[:, 2:] - measured).ravel()

        fit = least_squares(residual, initial, loss='soft_l1', f_scale=.05,
                            max_nfev=100, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if not fit.success:
            raise RuntimeError(f'Bead {bead} refinement did not converge: {fit.message}')
        points.append(fit.x)
        errors = residual(fit.x).reshape(-1, 2)
        records.append(dict(landmark_id=bead, observation_count=len(indices),
                            detector_component_rmse_px=float(np.sqrt(np.mean(errors**2))),
                            detector_radial_rmse_px=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
                            initial_to_refined_mm=float(np.linalg.norm(fit.x-initial)),
                            reprojection_jacobian_singular_values=np.linalg.svd(fit.jac, compute_uv=False).tolist()))
    return np.asarray(points), records


def rms_points(a, b):
    return float(np.sqrt(np.mean(np.sum((a-b)**2, axis=-1))))


def run(args):
    folder, run_dir, out = args.input_dir, args.run_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((folder/'experiment.json').read_text())
    label_file = folder/'landmarks.json'
    metrics = json.loads((run_dir/'metrics.json').read_text())
    if metrics['epoch'] != 100 or metrics['recipe']['epochs'] != 100:
        raise ValueError('This final-run audit requires the completed 100-epoch checkpoint')
    label_hash = sha256(label_file)
    if label_hash != metrics['landmarks_sha256'] or label_hash != metadata['landmarks_sha256']:
        raise ValueError('Reference landmarks differ from the frozen run/acquisition metadata')
    landmark_record = json.loads(label_file.read_text())
    landmarks = landmark_record['landmarks']
    ids = [entry['landmark_id'] for entry in landmarks]
    if ids != list(range(len(ids))):
        raise ValueError('This audit requires unchanged ordered landmark IDs')
    reference = np.array([entry['xyz_mm'] for entry in landmarks])
    if len(reference) != metrics['landmark_count']:
        raise ValueError('Reference landmark count differs from the frozen run')
    p_file, target_file = run_dir/'P_optimized_pixel.npy', folder/'target_projections.npy'
    pmat = np.load(p_file).astype(np.float64)
    target = np.load(target_file, mmap_mode='r')
    target_hash = sha256(target_file)
    if target_hash != metadata['target_sha256']:
        raise ValueError('Observed projection SHA256 differs from acquisition metadata')
    if sha256(p_file) != metrics['artifact_sha256']['P_optimized_pixel.npy']:
        raise ValueError('Estimated P artifact differs from the frozen run')
    projected = project(pmat, reference)
    observed, quality, rejected = detect(target, projected)
    points, per_bead = triangulate(pmat, observed)
    pose = rigid_registration(points, reference)
    registered = transform(points, pose)
    np.save(out/'H_estimated_frame_to_reference.npy', pose)
    np.savez_compressed(out/'sparse_phantom_reconstruction.npz', observed_uv_px=observed,
                        detection_quality=quality, estimated_pixel_pmat=pmat,
                        reconstructed_xyz_mm=points, registered_xyz_mm=registered,
                        reference_xyz_mm=reference, H_estimated_frame_to_reference=pose)
    splits = {}
    split_matrices = []
    for name, first in [('even_views', 0), ('odd_views', 1)]:
        split_points, split_records = triangulate(pmat, observed, np.arange(first, len(pmat), 2))
        matrix = rigid_registration(split_points, reference)
        split_matrices.append(matrix)
        splits[name] = dict(**pose_description(matrix),
                           min_observations_per_bead=min(item['observation_count'] for item in split_records),
                           registered_bead_rms_mm=rms_points(transform(split_points, matrix), reference),
                           pose_difference_from_all_views=difference(matrix, pose))
    splits['odd_even_pose_disagreement'] = difference(*split_matrices)
    sensitivity = {}
    configurations = {
        'aperture_radius_4p25': dict(inner_radius=4.25),
        'aperture_radius_4p75_wider_background': dict(inner_radius=4.75, annulus_inner=5.25, annulus_outer=7.5),
        'association_seed_u_plus_quarter_pixel': dict(seed_offset=(.25, 0)),
        'association_seed_u_minus_quarter_pixel': dict(seed_offset=(-.25, 0)),
        'association_seed_v_plus_quarter_pixel': dict(seed_offset=(0, .25)),
        'association_seed_v_minus_quarter_pixel': dict(seed_offset=(0, -.25)),
    }
    for name, configuration in configurations.items():
        uv, _, rejection = detect(target, projected, **configuration)
        variant_points, variant_records = triangulate(pmat, uv)
        variant_pose = rigid_registration(variant_points, reference)
        common = np.isfinite(uv[..., 0]) & np.isfinite(observed[..., 0])
        sensitivity[name] = dict(parameters=configuration, **pose_description(variant_pose),
            pose_difference_from_primary=difference(variant_pose, pose),
            min_observations_per_bead=min(item['observation_count'] for item in variant_records),
            common_detection_center_shift_rms_px=rms_points(uv[common], observed[common]),
            reconstructed_point_shift_rms_mm=rms_points(variant_points, points),
            registered_bead_rms_mm=rms_points(transform(variant_points, variant_pose), reference),
            rejected=rejection)
    # No true camera is opened above this line. This block evaluates the frozen
    # transform; it must never choose its sign, magnitude, or detector settings.
    truth_pmat = np.load(folder/'P_truth_pixel.npy').astype(np.float64)
    true_sources = -np.linalg.solve(truth_pmat[:, :, :3], truth_pmat[:, :, 3, None])[..., 0]
    estimated_sources = -np.linalg.solve(pmat[:, :, :3], pmat[:, :, 3, None])[..., 0]
    true_projection = project(truth_pmat, reference)
    oracle_points, _ = triangulate(truth_pmat, observed)
    oracle_bias_pose = rigid_registration(oracle_points, reference)
    valid = np.isfinite(observed[..., 0])
    detection_error = observed[valid] - true_projection[valid]
    corrected_pmat = pmat @ np.linalg.inv(pose)
    np.save(out/'P_pose_aligned_pixel.npy', corrected_pmat)
    evaluation = dict(
        source_rms_before_mm=rms_points(estimated_sources, true_sources),
        source_rms_after_mm=rms_points(transform(estimated_sources, pose), true_sources),
        fixed_id_bead_reprojection_rms_before_px=rms_points(project(pmat, reference), true_projection),
        fixed_id_bead_reprojection_rms_after_px=rms_points(project(corrected_pmat, reference), true_projection),
        observed_centroid_vs_gt_projected_volume_centroid_radial_rms_px=float(np.sqrt(np.mean(np.sum(detection_error**2, axis=1)))),
        observed_centroid_vs_gt_projected_volume_centroid_mean_uv_px=detection_error.mean(0).tolist(),
        frozen_centroids_oracle_camera_diagnostic=dict(
            sparse_bead_rms_mm=rms_points(oracle_points, reference),
            registered_sparse_bead_rms_mm=rms_points(transform(oracle_points, oracle_bias_pose), reference),
            apparent_pose_bias=pose_description(oracle_bias_pose),
            used_to_correct_primary_pose=False,
            interpretation='With perfect cameras and the identical already frozen detections, remaining 3-D centre errors estimate combined detection/noise/voxel-centroid mismatch for this realization; this oracle is never used in the actual pose estimate.'),
        note='Evaluation only after every pose was frozen. Projected attenuation-volume centroids are not exact image-contrast centroids; this residual includes perspective, voxelization, noise and local-background detection bias.')
    for index, record in enumerate(per_bead):
        record['reconstructed_xyz_mm'] = points[index].tolist()
        record['registered_xyz_mm'] = registered[index].tolist()
        record['reference_xyz_mm'] = reference[index].tolist()
        record['error_before_mm'] = float(np.linalg.norm(points[index]-reference[index]))
        record['error_after_mm'] = float(np.linalg.norm(registered[index]-reference[index]))
    report = dict(
        schema_version=1, run=run_dir.name, method='Observed-image sparse 3-D phantom reconstruction and one rigid pose registration',
        H_reconstruction_to_reference=pose.tolist(),
        coordinate_system='Centred physical xyz in mm; detector coordinates are zero-based pixel centres',
        inputs=dict(target_sha256=target_hash, estimated_pixel_pmat_sha256=sha256(p_file),
                    landmarks_sha256=label_hash, script_sha256=sha256(__file__),
                    calibration_gauge_sha256=sha256(Path(__file__).with_name('calibration_gauge.py'))),
        invariants=dict(optimized_camera_matrices_frozen=True, true_cameras_used_for_pose_estimation=False,
                        true_projected_centers_used_for_detection=False, fixed_landmark_ids=True,
                        rigid_transform_count=1, scale_fitted=False, per_view_transforms=False,
                        reconstructed_attenuation_volume=False, model_retrained=False),
        detection=dict(association='Estimated final P projects known reference bead IDs only to select isolated patches. No nearest-neighbour ID reassignment.',
            centroid='Signed first moments of affine-background-subtracted observed postlog intensity inside radius4.5px; no positivity clipping, fitted-centre prior, or blending with projected centres.',
            background_annulus_radius_px=[5., 7.], other_bead_minimum_separation_px=16.,
            background_rmse_maximum_fraction_of_peak=.15, minimum_contrast_integral=1.,
            maximum_association_distance_px=2.,
            total_candidates=int(valid.size), accepted=int(valid.sum()), rejected=rejected),
        triangulation=dict(initialization='Homogeneous DLT using frozen estimated cameras',
                           refinement='Only bead xyz, scipy least_squares soft_l1 f_scale0.05px',
                           reference_points_enter_refinement=False),
        global_pose=dict(**pose_description(pose),
                         reconstructed_bead_rms_before_mm=rms_points(points, reference),
                         reconstructed_bead_rms_after_mm=rms_points(registered, reference),
                         reference_point_cloud_singular_values_mm=np.linalg.svd(reference-reference.mean(0), compute_uv=False).tolist()),
        split_view_sensitivity=splits, detection_sensitivity=sensitivity, evaluation=evaluation,
        per_bead=per_bead,
        limits=[
            'The original calibration fits a fixed reference volume, so its physical frame is already anchored. This post-hoc registration measures a residual common pose; it is not evidence that a global gauge remained free during training.',
            'Sparse triangulation reconstructs 35 bead centres, not the complete attenuation volume.',
            'The fitted P and known reference positions seed associations; detection is therefore locally initialized, although measured centres and triangulated xyz are not supervised by true cameras.',
            'A single rigid transform cannot remove view-dependent geometry errors or nine-parameter cancellation. Before/after errors are both retained, even when registration worsens source error.',
            'Odd/even and aperture/seed perturbations are sensitivity diagnostics, not confidence intervals or an absolute bias correction.',
            'Ring/cylinder symmetry alone would leave ambiguity; fixed IDs include the inner helical beads and unequal bead sizes. No symmetry rematching is permitted.',
            'Reference landmarks are thresholded attenuation-weighted voxel centroids; image contrast centroids can differ systematically at subpixel scale.'])
    (out/'phantom_pose.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    make_plot(out, points, registered, reference, report)
    print(json.dumps(dict(global_pose=report['global_pose'], evaluation=evaluation,
                          split_view_sensitivity=splits, accepted=int(valid.sum())), indent=2))


def make_plot(out, points, registered, reference, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(13, 5), layout='constrained')
    axes = figure.add_subplot(121, projection='3d')
    axes.scatter(*reference.T, s=40, facecolors='none', edgecolors='#222222', label='Reference bead centres')
    axes.scatter(*points.T, s=14, c='#c76a16', label='Observed-image triangulation')
    axes.set(xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]', title='Sparse 3-D phantom reconstruction')
    axes.set_box_aspect(np.ptp(reference, axis=0))
    axes.legend(fontsize=8)
    axes = figure.add_subplot(122)
    axes.plot(np.linalg.norm(points-reference, axis=1), 'o-', ms=3, label='Before one global rigid pose')
    axes.plot(np.linalg.norm(registered-reference, axis=1), 'o-', ms=3, label='After one global rigid pose')
    axes.set(xlabel='Fixed bead ID', ylabel='3-D centre error [mm]', title='All 35 identities retained; no fitted scale')
    axes.legend(fontsize=8)
    axes.grid(alpha=.2)
    pose = report['global_pose']
    figure.suptitle(f"Signed LNCC31 seed1 | bead RMS {pose['reconstructed_bead_rms_before_mm']:.4f} → {pose['reconstructed_bead_rms_after_mm']:.4f} mm\nObserved noisy projections; estimated cameras frozen; no GT camera in pose estimation", fontsize=12)
    figure.savefig(out/'sparse_phantom_pose.png', dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('result_sinespin/ball_calibration')
    parser.add_argument('--input-dir', type=Path, default=base/'input')
    parser.add_argument('--run-dir', type=Path, default=base/'loss_signed_lncc31_seed1')
    parser.add_argument('--out-dir', type=Path, default=base/'loss_signed_lncc31_seed1/pose_audit')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
