"""Physical source/camera/intrinsic representation of calibrated P matrices.

The representation is independent of the nominal orbit and an isocentre. It
does not refit cameras or delete parameter discrepancies. Arrays use float64.
WORLD P maps WORLD=(internal x,z,y) to detector millimetres from pixel edges.
Physical xyz is centred on the volume box. The proper camera frame has axes
``(detector column, -detector row, source-to-detector normal)``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_gauge import decompose_world_pmat


_SWAP_YZ = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
_FLIP_CAMERA_V = np.diag([1., -1., 1.])
PHYSICAL_PARAMETER_NAMES = (
    "source_x_mm", "source_y_mm", "source_z_mm",
    "orientation_x_deg", "orientation_y_deg", "orientation_z_deg",
    "f_mm", "c_u_mm", "c_v_mm",
)


def _center(center_internal_mm):
    center = np.asarray(center_internal_mm, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Expected three finite volume-box centre coordinates.")
    return center


def _source_array(source_xyz_mm):
    source = np.asarray(source_xyz_mm, dtype=np.float64)
    if source.shape == (3,):
        source = source[None]
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("Expected finite physical source coordinates [views,3].")
    return source


def _matrix_batch(value, views, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 3):
        matrix = np.broadcast_to(matrix, (views, 3, 3))
    if matrix.shape != (views, 3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite [views,3,3] or one common [3,3] matrix.")
    return matrix


def _proper_rotations(value, views):
    rotation = _matrix_batch(value, views, "Camera orientation")
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)):
        raise ValueError("Camera orientation must be a proper rotation; no reflections or scale.")
    return rotation


def decompose_physical_camera(pmat, *, center_internal_mm=(0., 0., 0.)):
    """Return physical camera arrays from WORLD/mm P, without nominal geometry.

    Returns a dict containing ``source_xyz_mm`` [V,3],
    ``Q_camera_to_physical`` [V,3,3], full ``K_mm`` [V,3,3],
    ``intrinsics_f_cu_cv_mm`` [V,3], and ``projective_scale`` [V].
    The remaining arrays report detector axes and non-isotropic/nonzero-skew
    residuals. Q columns are (col, -row, normal); det(Q)=+1.

    f=(K00+K11)/2 and cu=K02, cv=K12, with principal points measured from the
    original detector pixel edges. K is preserved exactly. Averaging f for the
    nine-component summary does not silently erase ``focal_anisotropy_mm``.
    Positive or negative homogeneous scaling of P does not change C, Q, or K.
    """
    camera = decompose_world_pmat(pmat)
    source = camera["source_world_mm"] @ _SWAP_YZ - _center(center_internal_mm)
    r_physical = _FLIP_CAMERA_V @ camera["R"] @ _SWAP_YZ
    q = r_physical.transpose(0, 2, 1)
    k = camera["K"]
    intrinsics = np.column_stack((.5 * (k[:, 0, 0] + k[:, 1, 1]), k[:, 0, 2], k[:, 1, 2]))
    return dict(source_xyz_mm=source, Q_camera_to_physical=q, K_mm=k,
                intrinsics_f_cu_cv_mm=intrinsics,
                projective_scale=camera["projective_scale"],
                focal_anisotropy_mm=k[:, 0, 0] - k[:, 1, 1],
                skew_mm=k[:, 0, 1],
                col_xyz=q[:, :, 0], row_xyz=-q[:, :, 1], normal_xyz=q[:, :, 2])


def compose_physical_camera(source_xyz_mm, Q_camera_to_physical, K_mm, *,
                            center_internal_mm=(0., 0., 0.), projective_scale=None):
    """Exactly construct WORLD/mm P from physical C, proper Q, and full K.

    Returns [V,3,4]. C has shape [V,3]; Q and K have shape [V,3,3] or [3,3].
    The optional scalar or [V] projective scale restores P's saved numerical
    coefficients as well as its rays. K must be normalized upper triangular,
    with positive focal lengths. Nonzero skew and unequal focal entries are
    retained. No detector normal is inferred from the source or an isocentre.
    """
    source = _source_array(source_xyz_mm)
    views = len(source)
    q = _proper_rotations(Q_camera_to_physical, views)
    k = _matrix_batch(K_mm, views, "Intrinsic K")
    if (not np.allclose(k[:, 2, :], [0., 0., 1.], atol=1e-10, rtol=0)
            or not np.allclose(k[:, 1, 0], 0., atol=1e-10, rtol=0)
            or np.any(k[:, 0, 0] <= 0.) or np.any(k[:, 1, 1] <= 0.)):
        raise ValueError("K must be upper triangular with K22=1 and positive focal lengths.")
    source_world = (source + _center(center_internal_mm)) @ _SWAP_YZ
    r_world = _FLIP_CAMERA_V @ q.transpose(0, 2, 1) @ _SWAP_YZ
    translation = -np.einsum("vij,vj->vi", r_world, source_world)
    matrix = k @ np.concatenate((r_world, translation[:, :, None]), axis=2)
    if projective_scale is not None:
        scale = np.asarray(projective_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(views, scale)
        if scale.shape != (views,) or not np.isfinite(scale).all() or np.any(scale == 0.):
            raise ValueError("Projective scale must be finite and nonzero, scalar or [views].")
        matrix = matrix * scale[:, None, None]
    return matrix


def physical_camera_residuals(estimated, truth):
    """Compare two decomposed camera dicts in one fixed physical coordinate frame.

    ``residuals9`` [V,9] is (C_est-C_gt, omega_xyz_degrees, f/cu/cv differences).
    omega=Log(Q_est Q_gt^T) is the shortest active rotation from GT orientation
    to estimated orientation, expressed in fixed PHYSICAL axes. It is neither
    an Euler-angle subtraction nor a rotation in the moving camera frame.
    ``orientation_error_deg`` is its norm, equal to the SO(3) geodesic angle.
    At exactly 180 degrees a rotation-vector axis has the usual sign ambiguity.
    """
    source_est = _source_array(estimated["source_xyz_mm"])
    source_gt = _source_array(truth["source_xyz_mm"])
    if source_est.shape != source_gt.shape:
        raise ValueError("Estimated and reference source batches must match.")
    views = len(source_est)
    q_est = _proper_rotations(estimated["Q_camera_to_physical"], views)
    q_gt = _proper_rotations(truth["Q_camera_to_physical"], views)
    rotation_delta = q_est @ q_gt.transpose(0, 2, 1)
    omega_deg = Rotation.from_matrix(rotation_delta).as_rotvec(degrees=True)
    intrinsic_est = np.asarray(estimated["intrinsics_f_cu_cv_mm"], dtype=np.float64)
    intrinsic_gt = np.asarray(truth["intrinsics_f_cu_cv_mm"], dtype=np.float64)
    if (intrinsic_est.shape != (views, 3) or intrinsic_gt.shape != (views, 3)
            or not np.isfinite(intrinsic_est).all() or not np.isfinite(intrinsic_gt).all()):
        raise ValueError("Expected matching finite intrinsic summary arrays [views,3].")
    source_difference = source_est - source_gt
    return dict(residuals9=np.column_stack((source_difference, omega_deg, intrinsic_est-intrinsic_gt)),
                source_error_mm=np.linalg.norm(source_difference, axis=1),
                orientation_error_deg=np.linalg.norm(omega_deg, axis=1),
                orientation_rotvec_physical_deg=omega_deg,
                parameter_names=PHYSICAL_PARAMETER_NAMES)


def compose_physical_nine(parameters9, *, reference_Q=None, center_internal_mm=(0., 0., 0.)):
    """Construct P from Cxyz, a physical-axis rotation vector, and f/cu/cv.

    ``parameters9`` [V,9] holds absolute source xyz in mm, rotation-vector xyz
    in degrees, and absolute f,cu,cv in detector mm. Q=Exp(omega)@reference_Q;
    omitted reference_Q is identity. This is a local orientation chart when a
    reference Q is supplied, not an extra phantom registration or optimization.

    The nine-parameter family uses ONE focal length and zero skew. Use the full
    ``compose_physical_camera`` for exact reconstruction of saved float32 P,
    whose decomposed K can have tiny focal anisotropy/skew. Those discrepancies
    are reported by ``decompose_physical_camera`` and must not be hidden.
    """
    parameters = np.asarray(parameters9, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 9 or not np.isfinite(parameters).all():
        raise ValueError("Expected finite physical parameters [views,9].")
    views = len(parameters)
    reference = np.eye(3) if reference_Q is None else reference_Q
    q = Rotation.from_rotvec(parameters[:, 3:6], degrees=True).as_matrix()
    q = q @ _proper_rotations(reference, views)
    k = np.zeros((views, 3, 3), dtype=np.float64)
    k[:, 0, 0] = k[:, 1, 1] = parameters[:, 6]
    k[:, 0, 2], k[:, 1, 2] = parameters[:, 7], parameters[:, 8]
    k[:, 2, 2] = 1.
    return compose_physical_camera(parameters[:, :3], q, k, center_internal_mm=center_internal_mm)
