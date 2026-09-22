"""Bridge centred physical scan geometry to the calibration projector's P matrices.

The calibration projector uses WORLD=(INTERNAL x, INTERNAL z, INTERNAL y),
detector millimetres measured from pixel edges, and voxel-box edge origins.
``SineSpinGeometry`` instead uses centred physical xyz and zero-based pixel
centre coordinates. These helpers use a full detector, no pixel reversal, and
zero stitch offsets. They do not change the acquisition or fit any geometry.
"""
from __future__ import annotations

import numpy as np


_SWAP_YZ = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def _coordinate_transforms(du, dv, center_internal_mm):
    if not (np.isfinite(du) and np.isfinite(dv) and du > 0 and dv > 0):
        raise ValueError("Detector pitches du and dv must be finite and positive")
    center = np.asarray(center_internal_mm, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_internal_mm must contain three finite coordinates")
    # Pixel centre index j maps to the calibration detector coordinate (j+1/2)*du.
    detector = np.array([[du, 0., .5*du], [0., dv, .5*dv], [0., 0., 1.]])
    # Centred physical xyz = swap_yz(WORLD xyz) - INTERNAL volume-box centre.
    volume = np.eye(4)
    volume[:3, :3] = _SWAP_YZ
    volume[:3, 3] = -center
    return detector, volume


def _matrix_array(pmat):
    matrix = np.asarray(pmat, dtype=np.float64)
    if matrix.ndim == 2 and matrix.shape[-1] == 12:
        matrix = matrix.reshape(-1, 3, 4)
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 4):
        raise ValueError("P matrices must have shape (views, 3, 4) or (views, 12)")
    if not np.isfinite(matrix).all():
        raise ValueError("P matrices must be finite")
    return matrix


def pixel_to_pmat(pixel_pmat, *, du, dv, center_internal_mm=(0., 0., 0.),
                  dtype=np.float32):
    """Convert centred physical xyz -> pixel-centre P into calibration WORLD/mm P.

    Set ``center_internal_mm`` to ``origin_edge_xyz + shape_xyz*spacing_xyz/2``.
    The default is appropriate for an acquisition centred on a symmetric box.
    """
    detector, volume = _coordinate_transforms(du, dv, center_internal_mm)
    return np.ascontiguousarray(detector @ _matrix_array(pixel_pmat) @ volume,
                                dtype=dtype)


def geometry_to_pmat(geometry, *, center_internal_mm=(0., 0., 0.), dtype=np.float32):
    """Return ``[views,3,4]`` calibration P matrices from physical scan geometry."""
    return pixel_to_pmat(geometry.projection_matrices(), du=geometry.pixel_width,
                         dv=geometry.pixel_height, center_internal_mm=center_internal_mm,
                         dtype=dtype)


def pmat_to_pixel(pmat, *, du, dv, center_internal_mm=(0., 0., 0.)):
    """Return float64 centred-physical xyz -> zero-based pixel-centre P matrices.

    Homogeneous scale is preserved. Use divided projected coordinates or source
    positions for comparisons; unnormalised P coefficient errors are ambiguous.
    """
    detector, volume = _coordinate_transforms(du, dv, center_internal_mm)
    return np.linalg.inv(detector) @ _matrix_array(pmat) @ np.linalg.inv(volume)


def centered_source_positions(pmat, *, center_internal_mm=(0., 0., 0.)):
    """Recover source positions in centred physical xyz, invariant to P scale."""
    matrix = _matrix_array(pmat)
    _, volume = _coordinate_transforms(1., 1., center_internal_mm)
    source_world = np.linalg.solve(matrix[:, :, :3], -matrix[:, :, 3, None])[..., 0]
    return source_world @ _SWAP_YZ + volume[:3, 3]
