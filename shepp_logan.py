"""Deterministic 3-D Shepp--Logan attenuation phantom and exact ray integrals.

Ellipsoid geometry is the corrected Table 3.2 in Kak and Slaney, *Principles
of Computerized Tomographic Imaging* (1988), author's errata (2001):
https://slaney.org/pct/pct-errata.html

``modified=True`` explicitly replaces that table's small contrast increments
with [1, -.8, -.2, -.2, .1, .1, .05, .05, .1, -.1]. It is a contrast-enhanced
adaptation of the cited 3-D geometry, not a claim to a unique standard 3-D
modified phantom. ``modified=False`` uses the published gray levels.

World coordinates and ellipsoid parameters use x/y/z; arrays use z/y/x.
The origin is the centre of the volume, and samples are at voxel centres:
(index - (count - 1)/2) * spacing. Physical dimensions are fixed independently
of the sampling grid. Default outer diameters are 200 x 200 x 260 mm, to put
phantom material both inside and outside a roughly 180-mm longitudinal FOV.
This is a numerical test object, not the paper's physical quality phantom.
Attenuation is in inverse millimetres; ray integrals are dimensionless.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


SOURCE_URL = "https://slaney.org/pct/pct-errata.html"
DEFAULT_OUTER_DIAMETERS_XYZ_MM = (200.0, 200.0, 260.0)

# x, y, z, a, b, c, rotation about z (degrees), published additive gray level.
# Coordinates/semiaxes are dimensionless, as in the corrected source table.
_ELLIPSOIDS = np.array([
    [0.00,  0.000,  0.000, .6900, .920, .900,   0.,  2.00],
    [0.00,  0.000,  0.000, .6624, .874, .880,   0., -.98],
    [-.22,  0.000, -.250, .4100, .160, .210, 108., -.02],
    [0.22,  0.000, -.250, .3100, .110, .220,  72., -.02],
    [0.00,  0.350, -.250, .2100, .250, .500,   0.,  .02],
    [0.00,  0.100, -.250, .0460, .046, .046,   0.,  .02],
    [-.08, -.650, -.250, .0460, .023, .020,   0.,  .01],
    [0.06, -.650, -.250, .0460, .023, .020,  90.,  .01],
    [0.06, -.105,  .625, .0560, .040, .100,  90.,  .02],
    [0.00,  .100,  .625, .0560, .056, .100,   0., -.02],
], dtype=np.float64)
_MODIFIED_LEVELS = np.array([1., -.8, -.2, -.2, .1, .1, .05, .05, .1, -.1])


def _triple(value: float | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain three finite positive values")
    return array


def ellipsoid_parameters(
    outer_diameters_xyz_mm: Sequence[float] = DEFAULT_OUTER_DIAMETERS_XYZ_MM,
    attenuation_scale: float = 0.02,
    modified: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical centres, local-to-world axis matrices, and amplitudes.

    Shapes are (10, 3), (10, 3, 3), and (10,). An ellipsoid is the set
    ``centre + axes @ q`` for ``dot(q, q) <= 1``. Affine scaling of the
    dimensionless source phantom preserves its nesting when the outer x/y
    diameters are changed independently.
    """
    diameters = _triple(outer_diameters_xyz_mm, "outer_diameters_xyz_mm")
    if not np.isfinite(attenuation_scale) or attenuation_scale <= 0:
        raise ValueError("attenuation_scale must be finite and positive")
    scale = diameters / (2.0 * _ELLIPSOIDS[0, 3:6])
    centres = _ELLIPSOIDS[:, :3] * scale
    axes = np.zeros((len(_ELLIPSOIDS), 3, 3), dtype=np.float64)
    for i, row in enumerate(_ELLIPSOIDS):
        phi = np.deg2rad(row[6])
        c, s = np.cos(phi), np.sin(phi)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        axes[i] = scale[:, None] * rotation * row[None, 3:6]
    levels = _MODIFIED_LEVELS if modified else _ELLIPSOIDS[:, 7]
    return centres, axes, levels.copy() * attenuation_scale


def phantom_metadata(
    outer_diameters_xyz_mm: Sequence[float] = DEFAULT_OUTER_DIAMETERS_XYZ_MM,
    attenuation_scale: float = 0.02,
    modified: bool = True,
) -> dict:
    """JSON-serializable complete phantom specification."""
    centres, axes, amplitudes = ellipsoid_parameters(
        outer_diameters_xyz_mm, attenuation_scale, modified)
    return {
        "name": "3D contrast-enhanced Shepp-Logan" if modified else "3D Shepp-Logan",
        "geometry_source": SOURCE_URL,
        "contrast_modified": bool(modified),
        "outer_diameters_xyz_mm": _triple(outer_diameters_xyz_mm, "diameters").tolist(),
        "attenuation_scale_per_mm": float(attenuation_scale),
        "array_order": "zyx",
        "voxel_coordinate_rule": "(index - (count - 1) / 2) * voxel_mm",
        "ellipsoids": [
            {"centre_xyz_mm": centre.tolist(), "local_to_world_axes_mm": matrix.tolist(),
             "additive_attenuation_per_mm": float(amplitude)}
            for centre, matrix, amplitude in zip(centres, axes, amplitudes)
        ],
    }


def create_shepp_logan(
    shape_zyx: Sequence[int],
    voxel_mm: float | Sequence[float],
    *,
    outer_diameters_xyz_mm: Sequence[float] = DEFAULT_OUTER_DIAMETERS_XYZ_MM,
    attenuation_scale: float = 0.02,
    modified: bool = True,
) -> np.ndarray:
    """Sample ellipsoid sums at voxel centres into a float32 z/y/x volume.

    A three-element ``voxel_mm`` is interpreted in z/y/x order. The grid may
    intentionally crop the phantom; its physical dimensions never rescale to
    fit the grid. No smoothing, random numbers, or clipping are applied.
    """
    raw_shape = np.asarray(shape_zyx)
    if (raw_shape.shape != (3,) or not np.all(np.isfinite(raw_shape))
            or np.any(raw_shape <= 0) or np.any(raw_shape != raw_shape.astype(np.int64))):
        raise ValueError("shape_zyx must contain three positive integers")
    shape = tuple(raw_shape.astype(int))
    spacing = _triple(voxel_mm, "voxel_mm")
    z, y, x = [
        ((np.arange(n, dtype=np.float64) - (n - 1) / 2) * step)
        for n, step in zip(shape, spacing)
    ]
    centres, axes, amplitudes = ellipsoid_parameters(
        outer_diameters_xyz_mm, attenuation_scale, modified)
    # Both published and modified amplitudes are integer hundredths. Sum those
    # exactly before converting to attenuation, so 1 - .8 - .2 produces zero
    # rather than a tiny negative floating-point background.
    levels = _MODIFIED_LEVELS if modified else _ELLIPSOIDS[:, 7]
    integer_levels = np.rint(100.0 * levels).astype(np.int16)
    level_volume = np.zeros(shape, dtype=np.int16)
    # All rotations are around z. Only a 2-D scratch array plus one axial
    # boolean slab is needed; large physical volumes need not allocate xyz grids.
    for centre, matrix, level in zip(centres, axes, integer_levels):
        inverse = np.linalg.inv(matrix)
        dx, dy, dz = x - centre[0], y - centre[1], z - centre[2]
        qx = inverse[0, 0] * dx[None, :] + inverse[0, 1] * dy[:, None]
        qy = inverse[1, 0] * dx[None, :] + inverse[1, 1] * dy[:, None]
        xy_squared = qx * qx + qy * qy
        z_squared = (inverse[2, 2] * dz) ** 2
        for iz in np.flatnonzero(z_squared <= 1.0):
            level_volume[iz] += level * (xy_squared <= 1.0 - z_squared[iz])
    volume = level_volume.astype(np.float32)
    volume *= np.float32(attenuation_scale / 100.0)
    return volume


def analytic_line_integrals(
    source_xyz_mm: np.ndarray,
    detector_xyz_mm: np.ndarray,
    *,
    outer_diameters_xyz_mm: Sequence[float] = DEFAULT_OUTER_DIAMETERS_XYZ_MM,
    attenuation_scale: float = 0.02,
    modified: bool = True,
    chunk_size: int = 262144,
) -> np.ndarray:
    """Exact attenuation integrals along finite source-to-detector segments.

    Inputs broadcast to (..., 3); output is float32 with shape (...,). Units
    are millimetres and 1/mm. Intersecting analytical ellipsoids avoids sharing
    a voxel discretization with a numerical reconstruction projector. Detector
    pixels are point sampled; pixel-area integration is not simulated.
    """
    source, detector = np.broadcast_arrays(
        np.asarray(source_xyz_mm, dtype=np.float64),
        np.asarray(detector_xyz_mm, dtype=np.float64))
    if source.ndim < 1 or source.shape[-1] != 3:
        raise ValueError("source and detector arrays must end in an xyz axis of length 3")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(detector)):
        raise ValueError("ray coordinates must be finite")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    output_shape = source.shape[:-1]
    source = source.reshape(-1, 3)
    detector = detector.reshape(-1, 3)
    centres, axes, amplitudes = ellipsoid_parameters(
        outer_diameters_xyz_mm, attenuation_scale, modified)
    inverses = np.linalg.inv(axes)
    output = np.zeros(len(source), dtype=np.float32)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        origins = source[start:stop]
        direction = detector[start:stop] - origins
        length = np.linalg.norm(direction, axis=-1)
        result = np.zeros(stop - start, dtype=np.float64)
        for centre, inverse, amplitude in zip(centres, inverses, amplitudes):
            p = (origins - centre) @ inverse.T
            d = direction @ inverse.T
            a = np.einsum("ij,ij->i", d, d)
            b = np.einsum("ij,ij->i", p, d)
            c = np.einsum("ij,ij->i", p, p) - 1.0
            discriminant = b * b - a * c
            valid = (a > 0.0) & (discriminant > 0.0)
            safe_a = np.where(valid, a, 1.0)
            root = np.sqrt(np.maximum(discriminant, 0.0))
            enter = np.maximum(0.0, (-b - root) / safe_a)
            leave = np.minimum(1.0, (-b + root) / safe_a)
            result += amplitude * length * np.where(valid, np.maximum(leave - enter, 0.0), 0.0)
        output[start:stop] = result
    return output.reshape(output_shape)
