"""Circular cone-beam FDK with physical units and Parker short-scan weighting.

Input is line-integral data in (view, row, column) order, with the centered
geometry from ``sinespin_geometry``. The output is a centered (z, y, x) grid.
FDK is an approximate 3D reconstruction for circular orbits; this module does
not implement Grangeat inversion or iterative reconstruction.

Parker's smooth redundancy weights follow DOI:10.1118/1.595078, with the
corrected last-ramp denominator also given in equation (14) of
https://pmc.ncbi.nlm.nih.gov/articles/PMC5496733/ . Here short-scan weights sum
to one across conjugate rays, and a complete 360-degree scan uses weight 1/2.
No empirical amplitude correction is used.
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sinespin_geometry import SineSpinGeometry


def angular_weights(angles_rad: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights for strictly increasing view angles."""
    angles = np.asarray(angles_rad, dtype=np.float64)
    if angles.ndim != 1 or len(angles) < 2 or not np.isfinite(angles).all():
        raise ValueError("At least two finite one-dimensional view angles are required")
    gaps = np.diff(angles)
    if np.any(gaps <= 0):
        raise ValueError("View angles must be strictly increasing")
    weights = np.empty_like(angles)
    weights[0], weights[-1] = gaps[0] / 2, gaps[-1] / 2
    weights[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return weights


def _parker_profile(beta, gamma, span):
    """Parker profile for beta in [0, span], gamma opposite detector-u fan angle."""
    beta, gamma = np.broadcast_arrays(np.asarray(beta), np.asarray(gamma))
    delta = (span - np.pi) / 2
    first_end = 2 * (delta - gamma)
    last_start = np.pi - 2 * gamma
    first_phase = np.divide(beta, first_end, out=np.zeros_like(beta, dtype=float), where=first_end > 0)
    last_width = 2 * (delta + gamma)
    last_phase = np.divide(span - beta, last_width, out=np.zeros_like(beta, dtype=float), where=last_width > 0)
    result = np.ones(beta.shape, dtype=np.float64)
    result = np.where(beta < first_end, np.sin(0.5 * np.pi * first_phase) ** 2, result)
    result = np.where(beta > last_start, np.sin(0.5 * np.pi * last_phase) ** 2, result)
    return np.where((beta < 0) | (beta > span), 0.0, result)


def parker_weights(angles_rad: np.ndarray, detector_u_mm: np.ndarray, sdd_mm: float) -> np.ndarray:
    """Return (view, column) redundancy weights for this geometry's u direction.

    The source moves in the positive detector-column direction, so the Parker
    fan angle is -atan(u/SDD). For a ray at (beta, gamma), its conjugate is at
    (beta + pi + 2*gamma, -gamma). Endpoint-inclusive 360-degree scans use 1/2.
    """
    angular_weights(angles_rad)
    angles = np.asarray(angles_rad, dtype=np.float64)
    u = np.asarray(detector_u_mm, dtype=np.float64)
    if u.ndim != 1 or not np.isfinite(u).all() or not np.isfinite(sdd_mm) or sdd_mm <= 0:
        raise ValueError("Finite detector coordinates and positive SDD are required")
    beta = angles - angles[0]
    span = float(beta[-1])
    if np.isclose(span, 2 * np.pi, rtol=0, atol=1e-7):
        return np.full((len(beta), len(u)), 0.5, dtype=np.float32)
    gamma = -np.arctan(u / sdd_mm)
    if span > 2 * np.pi or span < np.pi + 2 * np.max(np.abs(gamma)) - 1e-10:
        raise ValueError("Parker weighting requires pi + full fan angle <= scan span <= 2*pi")
    return _parker_profile(beta[:, None], gamma[None, :], span).astype(np.float32)


def ramp_filter(projections: np.ndarray, pixel_width_mm: float) -> np.ndarray:
    """Zero-padded Ram-Lak convolution along detector columns, in physical mm.

    The band-limited ramp's sampled convolution coefficients, including the
    integration step, are h[0]=1/(4*du), h[odd]=-1/(pi^2*du*k^2), h[even]=0.
    FFT padding prevents opposite detector edges from wrapping into each other.
    """
    data = np.asarray(projections, dtype=np.float32)
    if data.ndim < 1 or data.shape[-1] < 2:
        raise ValueError("At least two detector columns are required")
    if not np.isfinite(pixel_width_mm) or pixel_width_mm <= 0:
        raise ValueError("Detector pixel width must be positive")
    columns = data.shape[-1]
    size = 1 << (2 * columns - 1).bit_length()
    offsets = np.arange(size, dtype=np.int64)
    offsets[offsets > size // 2] -= size
    kernel = np.zeros(size, dtype=np.float64)
    kernel[0] = 1 / (4 * pixel_width_mm)
    odd = (offsets % 2) != 0
    kernel[odd] = -1 / (np.pi ** 2 * pixel_width_mm * offsets[odd] ** 2)
    transformed = np.fft.rfft(data, n=size, axis=-1)
    transformed *= np.fft.rfft(kernel)
    return np.fft.irfft(transformed, n=size, axis=-1)[..., :columns].astype(np.float32)


_BACKPROJECT_KERNEL = None


def _backproject_kernel():
    """Create the CUDA kernel lazily so CPU filtering/tests do not require CUDA."""
    global _BACKPROJECT_KERNEL
    if _BACKPROJECT_KERNEL is not None:
        return _BACKPROJECT_KERNEL
    from numba import cuda, float32

    @cuda.jit
    def backproject(projections, source, normal, columns, rows, integration, volume,
                    nx, ny, nz, voxel, du, dv, sod, sdd):
        index = cuda.grid(1)
        if index >= nx * ny * nz:
            return
        ix = index % nx
        iy = (index // nx) % ny
        iz = index // (nx * ny)
        half, one = float32(0.5), float32(1.0)
        x = (float32(ix) - half * float32(nx - 1)) * voxel
        y = (float32(iy) - half * float32(ny - 1)) * voxel
        z = (float32(iz) - half * float32(nz - 1)) * voxel
        nv, nu = projections.shape[1], projections.shape[2]
        value = float32(0.0)
        for view in range(projections.shape[0]):
            rx, ry, rz = x - source[view, 0], y - source[view, 1], z - source[view, 2]
            depth = rx * normal[view, 0] + ry * normal[view, 1] + rz * normal[view, 2]
            if depth <= 0:
                continue
            u = sdd / depth * (rx * columns[view, 0] + ry * columns[view, 1] + rz * columns[view, 2]) / du + half * float32(nu - 1)
            v = sdd / depth * (rx * rows[view, 0] + ry * rows[view, 1] + rz * rows[view, 2]) / dv + half * float32(nv - 1)
            if u < 0 or v < 0 or u > nu - 1 or v > nv - 1:
                continue
            u0, v0 = int(math.floor(u)), int(math.floor(v))
            u1, v1 = min(u0 + 1, nu - 1), min(v0 + 1, nv - 1)
            fu, fv = u - float32(u0), v - float32(v0)
            q = ((one - fu) * (one - fv) * projections[view, v0, u0]
                 + fu * (one - fv) * projections[view, v0, u1]
                 + (one - fu) * fv * projections[view, v1, u0]
                 + fu * fv * projections[view, v1, u1])
            # Filtering used detector mm. Its magnification conversion D/R
            # multiplies the usual circular FDK divergence weight (R/depth)^2.
            value += q * integration[view] * (sdd * sod / (depth * depth))
        volume[index] += value

    _BACKPROJECT_KERNEL = backproject
    return backproject


def _logical_gpu(gpu: int) -> int:
    """Map a physical GPU index through a numeric CUDA_VISIBLE_DEVICES mask."""
    if int(gpu) != gpu or gpu < 0:
        raise ValueError("gpu must be a nonnegative physical GPU index")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return int(gpu)
    devices = [part.strip() for part in visible.split(",")]
    if str(gpu) not in devices:
        raise ValueError(f"Physical GPU {gpu} is not in CUDA_VISIBLE_DEVICES={visible!r}; use a numeric mask")
    return devices.index(str(gpu))


def circular_fdk(projections: np.ndarray, geometry: SineSpinGeometry,
                 shape_zyx, voxel_mm: float, gpu: int = 1) -> np.ndarray:
    """Reconstruct a centered float32 volume from a zero-tilt circular scan.

    Angles must increase and include both endpoints of a 200/220-degree short
    scan or a full 360-degree scan. The volume uses isotropic ``voxel_mm``.
    ``gpu`` is a physical index; ``CUDA_VISIBLE_DEVICES=1`` maps gpu=1 to CUDA 0.
    Work is processed in 16-view batches to bound FFT and GPU memory usage.
    """
    g = geometry
    data = np.asarray(projections)
    if data.shape != (g.n_views, g.detector_rows, g.detector_cols):
        raise ValueError("Projection shape must match geometry (views, detector rows, detector columns)")
    if len(shape_zyx) != 3 or any(int(n) != n or n < 1 for n in shape_zyx):
        raise ValueError("shape_zyx must contain three positive integers")
    if not np.isfinite(voxel_mm) or voxel_mm <= 0:
        raise ValueError("voxel_mm must be positive")
    if not np.allclose(g.tilt_deg, 0, rtol=0, atol=1e-7):
        raise ValueError("Circular FDK requires zero tilt; use Grangeat for the sine-on-sphere orbit")
    angles = np.deg2rad(np.asarray(g.theta_deg, dtype=np.float64))
    integration = angular_weights(angles).astype(np.float32)
    if not np.isclose(np.rad2deg(angles[-1] - angles[0]), g.scan_angle_deg, rtol=0, atol=1e-6):
        raise ValueError("View endpoints must match geometry.scan_angle_deg")
    if not (np.isfinite(g.sod_mm) and np.isfinite(g.sdd_mm) and 0 < g.sod_mm < g.sdd_mm):
        raise ValueError("Geometry requires 0 < SOD < SDD")
    radial = np.stack((np.sin(angles), -np.cos(angles), np.zeros_like(angles)), axis=-1)
    col = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1)
    expected = (g.sod_mm * radial, (g.sod_mm - g.sdd_mm) * radial,
                np.tile([0., 0., 1.], (g.n_views, 1)), col)
    actual = (g.source_positions, g.module_centers, g.row_vectors, g.col_vectors)
    if any(np.shape(a) != e.shape or not np.allclose(a, e, rtol=0, atol=1e-4) for a, e in zip(actual, expected)):
        raise ValueError("FDK requires the centered circular geometry and detector axes from build_icono_orbit")
    if min(g.pixel_width, g.pixel_height) <= 0 or not np.isfinite([g.pixel_width, g.pixel_height]).all():
        raise ValueError("Detector pixel spacing must be positive")
    u = (np.arange(g.detector_cols) - (g.detector_cols - 1) / 2) * g.pixel_width
    v = (np.arange(g.detector_rows) - (g.detector_rows - 1) / 2) * g.pixel_height
    redundancy = parker_weights(angles, u, g.sdd_mm)
    cosine = (g.sdd_mm / np.sqrt(g.sdd_mm ** 2 + v[:, None] ** 2 + u[None, :] ** 2)).astype(np.float32)
    from numba import cuda
    cuda.select_device(_logical_gpu(gpu))
    kernel = _backproject_kernel()
    nz, ny, nx = map(int, shape_zyx)
    output = cuda.to_device(np.zeros(nx * ny * nz, dtype=np.float32))
    block = 256
    for start in range(0, g.n_views, 16):
        stop = min(start + 16, g.n_views)
        batch = np.array(data[start:stop], dtype=np.float32, order="C", copy=True)
        if not np.isfinite(batch).all():
            raise ValueError("Projection line integrals must be finite")
        batch *= cosine[None, :, :]
        batch *= redundancy[start:stop, None, :]
        filtered = ramp_filter(batch, g.pixel_width)
        device_inputs = [cuda.to_device(np.ascontiguousarray(a, dtype=np.float32)) for a in (
            filtered, g.source_positions[start:stop], g.normals[start:stop],
            g.col_vectors[start:stop], g.row_vectors[start:stop], integration[start:stop]
        )]
        kernel[(output.size + block - 1) // block, block](
            *device_inputs, output, nx, ny, nz, np.float32(voxel_mm),
            np.float32(g.pixel_width), np.float32(g.pixel_height),
            np.float32(g.sod_mm), np.float32(g.sdd_mm))
        cuda.synchronize()
        del device_inputs
    return output.copy_to_host().reshape(nz, ny, nx)
