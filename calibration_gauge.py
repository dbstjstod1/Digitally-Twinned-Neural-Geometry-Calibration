"""Canonical camera parameters and a single rigid change of phantom frame.

These are reporting utilities: no parameters are fitted to source trajectories,
and no per-view alignment or scale fit is provided. The fixed known phantom
already supplies a metric coordinate frame in the calibration experiment.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import rq
from scipy.spatial.transform import Rotation

from calibration_geometry import centered_source_positions


_SWAP_YZ = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
PARAMETER_NAMES = ("delta_u_mm", "delta_f_mm", "delta_v_mm",
                   "translation_x_mm", "translation_y_mm", "translation_z_mm",
                   "rotation_x_deg", "rotation_y_deg", "rotation_z_deg")


def _matrices(pmat):
    matrix = np.asarray(pmat, dtype=np.float64)
    if matrix.ndim == 2 and matrix.shape == (3, 4):
        matrix = matrix[None]
    elif matrix.ndim == 2 and matrix.shape[1] == 12:
        matrix = matrix.reshape(-1, 3, 4)
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite projection matrices [views,3,4].")
    return matrix


def _center(center_internal_mm):
    center = np.asarray(center_internal_mm, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Expected a finite three-dimensional volume-box center.")
    return center


def decompose_world_pmat(pmat):
    """Return the unique positive-focal, proper-R decomposition in WORLD/mm.

    ``P = projective_scale * K @ [R | t]``, with K[2,2]=1 and det(R)=+1.
    A negative homogeneous multiplier has no effect on K, R, t or source.
    This function expects calibration WORLD=(physical x,z,y), not centred
    physical pixel P: the latter has a reflected detector coordinate frame.
    """
    matrix = _matrices(pmat)
    intrinsic, rotation, translation, scale = [], [], [], []
    for item in matrix:
        sign, log_abs_det = np.linalg.slogdet(item[:, :3])
        if sign == 0 or not np.isfinite(log_abs_det):
            raise ValueError("A finite camera centre requires a nonsingular left P block.")
        oriented = sign * item
        k, r = rq(oriented[:, :3])
        signs = np.where(np.diag(k) < 0., -1., 1.)
        k = k * signs[None, :]
        r = signs[:, None] * r
        divisor = k[2, 2]
        if divisor <= 0 or np.linalg.det(r) < 0:
            raise ValueError("Failed to obtain the required proper camera frame.")
        k /= divisor
        t = np.linalg.solve(k, oriented[:, 3] / divisor)
        intrinsic.append(k)
        rotation.append(r)
        translation.append(t)
        scale.append(sign * divisor)
    intrinsic, rotation, translation = map(np.asarray, (intrinsic, rotation, translation))
    source = -np.einsum("vji,vj->vi", rotation, translation)
    return dict(K=intrinsic, R=rotation, t=translation,
                source_world_mm=source, projective_scale=np.asarray(scale))


def compose_effective_parameters(nominal_pmat, parameters_9, *, center_internal_mm=(0., 0., 0.)):
    """Float64 counterpart of the forward effective 9DoF model (not tanh/raw).

    Input order is delta_u, shared delta_f, delta_v, physical/internal xyz
    translation, and xyz Euler rotation in degrees (Rz @ Ry @ Rx).
    The forward right-multiplication convention is used, without inversion.
    """
    nominal = decompose_world_pmat(nominal_pmat)
    parameters = np.asarray(parameters_9, dtype=np.float64)
    if parameters.shape != (len(nominal["K"]), 9) or not np.isfinite(parameters).all():
        raise ValueError("Expected finite applied parameters [views,9].")
    intrinsic = nominal["K"].copy()
    intrinsic[:, 0, 0] += parameters[:, 1]
    intrinsic[:, 1, 1] += parameters[:, 1]
    intrinsic[:, 0, 2] += parameters[:, 0]
    intrinsic[:, 1, 2] += parameters[:, 2]
    intrinsic[:, 0, 1] = 0.  # Matches the effective model, which has no skew DoF.
    r_internal = Rotation.from_euler("xyz", parameters[:, 6:], degrees=True).as_matrix()
    r_world = _SWAP_YZ @ r_internal @ _SWAP_YZ
    center_world = _SWAP_YZ @ _center(center_internal_mm)
    t_world = parameters[:, 3:6] @ _SWAP_YZ + center_world
    t_world -= np.einsum("vij,j->vi", r_world, center_world)
    r = nominal["R"] @ r_world
    t = nominal["t"] + np.einsum("vij,vj->vi", nominal["R"], t_world)
    return intrinsic @ np.concatenate((r, t[:, :, None]), axis=2)


def effective_parameters_from_pmat(pmat, nominal_pmat, *, center_internal_mm=(0., 0., 0.)):
    """Extract all nine effective parameters, removing only P's arbitrary scale.

    K/R/t are canonicalized independently before computing E_nominal^-1 E.
    There is no source/GT fit, cancellation deletion or inferred object pose.
    Any mismatch to the shared-focal/zero-skew family is explicitly reported.
    """
    matrix, nominal_matrix = _matrices(pmat), _matrices(nominal_pmat)
    if matrix.shape != nominal_matrix.shape:
        raise ValueError("Estimate and nominal camera batches must have identical shapes.")
    camera, nominal = decompose_world_pmat(matrix), decompose_world_pmat(nominal_matrix)
    relative_r_world = nominal["R"].transpose(0, 2, 1) @ camera["R"]
    relative_t_world = np.einsum("vji,vj->vi", nominal["R"], camera["t"] - nominal["t"])
    center_world = _SWAP_YZ @ _center(center_internal_mm)
    translation_world = relative_t_world - center_world
    translation_world += np.einsum("vij,j->vi", relative_r_world, center_world)
    relative_r_internal = _SWAP_YZ @ relative_r_world @ _SWAP_YZ
    delta_k = camera["K"] - nominal["K"]
    parameters = np.column_stack((delta_k[:, 0, 2],
                                   .5 * (delta_k[:, 0, 0] + delta_k[:, 1, 1]),
                                   delta_k[:, 1, 2],
                                   translation_world @ _SWAP_YZ,
                                   Rotation.from_matrix(relative_r_internal).as_euler("xyz", degrees=True)))
    reconstructed = compose_effective_parameters(nominal_matrix, parameters,
                                                  center_internal_mm=center_internal_mm)
    normalized = matrix / camera["projective_scale"][:, None, None]
    error = np.linalg.norm(reconstructed - normalized, axis=(1, 2))
    error /= np.linalg.norm(normalized, axis=(1, 2))
    # WORLD camera rows are proper; convert physical coordinates and reverse
    # image v to obtain the proper physical basis (col, -row, beam normal).
    image_reflection = np.diag([1., -1., 1.])
    rotation_physical = image_reflection @ camera["R"] @ _SWAP_YZ
    return dict(parameters_9=parameters, parameter_names=PARAMETER_NAMES,
                source_xyz_mm=centered_source_positions(matrix, center_internal_mm=center_internal_mm),
                K_mm=camera["K"], R_world=camera["R"], t_world=camera["t"],
                R_physical=rotation_physical,
                relative_rotation_internal=relative_r_internal,
                projective_scale=camera["projective_scale"],
                shared_focal_mismatch_mm=delta_k[:, 0, 0] - delta_k[:, 1, 1],
                skew_mm=camera["K"][:, 0, 1],
                effective_model_relative_residual=error)


def _rigid_matrix(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4x4 rigid transform.")
    if not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-10, rtol=0):
        raise ValueError("Transform must have homogeneous last row [0,0,0,1].")
    r = transform[:3, :3]
    if (not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-8, rtol=0)):
        raise ValueError("Only proper rigid transforms are allowed; no scale or reflection.")
    return transform


def transform_points(points, transform):
    """Change frame using column-vector convention X_new = H @ X_old."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite 3-D points [points,3].")
    transform = _rigid_matrix(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def apply_rigid_frame(pmat, transform):
    """Return P_new=P_old@inv(H), preserving projections of X_new=H@X_old.

    P and H must use the SAME object coordinate system. For a physical-xyz
    phantom pose, pass centred-physical pixel P, not calibration WORLD P.
    One shared H is required for the entire view batch.
    """
    return _matrices(pmat) @ np.linalg.inv(_rigid_matrix(transform))


def fit_rigid_landmark_pose(source_points, target_points):
    """Rigid Kabsch registration of already paired phantom landmarks only.

    Correspondences must be established independently; this function does not
    rematch beads, reject outliers, use camera/source GT, or estimate scale.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Expected at least three finite paired 3-D landmarks.")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    a, b = source - source_center, target - target_center
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        raise ValueError("Collinear/coincident landmarks cannot determine a rigid pose.")
    u, singular_values, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = 1. if np.linalg.det(vt.T @ u.T) > 0 else -1.
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    aligned = transform_points(source, transform)
    return dict(H=transform, rotation=rotation, translation=translation,
                singular_values=singular_values,
                rms_before_mm=float(np.sqrt(np.mean(np.sum((source-target)**2, axis=1)))),
                rms_after_mm=float(np.sqrt(np.mean(np.sum((aligned-target)**2, axis=1)))),
                aligned_points=aligned)
