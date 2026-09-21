"""Centered, physical-coordinate model of the ARTIS icono sine-on-sphere orbit.

The paper specifies the detector, scan arcs, view counts, and sinusoidal tilt.
It does not publish calibrated source/detector poses or SOD/SDD. The default
750/1200 mm distances are explicit simulation assumptions, not fitted values.

Coordinates are millimetres: x/y transverse, z longitudinal, isocentre (0,0,0).
Detector rows increase along ``row_vectors`` and columns along ``col_vectors``;
pixel centres use (N-1)/2. Arrays are directly compatible with LEAP modularbeam.
There is no coordinate swap, projection reversal, or inherited 4T half-cone shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SineSpinGeometry:
    kind: str
    source_positions: np.ndarray
    module_centers: np.ndarray
    row_vectors: np.ndarray
    col_vectors: np.ndarray
    detector_rows: int
    detector_cols: int
    pixel_height: float
    pixel_width: float
    theta_deg: np.ndarray
    tilt_deg: np.ndarray
    sod_mm: float
    sdd_mm: float
    scan_angle_deg: float

    @property
    def n_views(self) -> int:
        return len(self.theta_deg)

    @property
    def detector_height_mm(self) -> float:
        return self.detector_rows * self.pixel_height

    @property
    def detector_width_mm(self) -> float:
        return self.detector_cols * self.pixel_width

    @property
    def normals(self) -> np.ndarray:
        """Unit vectors from source to detector (central-ray direction)."""
        return (self.module_centers - self.source_positions) / self.sdd_mm

    def modular_arrays(self, dtype=np.float32) -> tuple[np.ndarray, ...]:
        """LEAP order: sourcePositions, moduleCenters, rowVectors, colVectors."""
        return tuple(np.ascontiguousarray(a, dtype=dtype) for a in (
            self.source_positions, self.module_centers, self.row_vectors, self.col_vectors
        ))

    def projection_matrices(self) -> np.ndarray:
        """Return [views,3,4] matrices mapping physical xyz to pixel-centre uv.

        Homogeneous division produces col index u and row index v. In particular
        the isocentre projects exactly to ((cols-1)/2, (rows-1)/2).
        """
        n = self.normals
        u = self.sdd_mm / self.pixel_width * self.col_vectors
        u += 0.5 * (self.detector_cols - 1) * n
        v = self.sdd_mm / self.pixel_height * self.row_vectors
        v += 0.5 * (self.detector_rows - 1) * n
        linear = np.stack((u, v, n), axis=1)
        offset = -np.einsum("vij,vj->vi", linear, self.source_positions)
        return np.concatenate((linear, offset[:, :, None]), axis=2)

    def detector_coordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical detector u,v and central-ray depth, each [views,points]."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rel = points[None, :, :] - self.source_positions[:, None, :]
        depth = np.einsum("vpi,vi->vp", rel, self.normals)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.sdd_mm * np.einsum("vpi,vi->vp", rel, self.col_vectors) / depth
            v = self.sdd_mm * np.einsum("vpi,vi->vp", rel, self.row_vectors) / depth
        return u, v, depth

    def detector_visibility(self, points: np.ndarray, *, pixel_edges: bool = True) -> np.ndarray:
        """Boolean [views,points] finite-detector visibility, not Tuy completeness.

        By default the active footprint extends half a pixel beyond the outer
        pixel centres. ``pixel_edges=False`` uses the centre-to-centre footprint.
        """
        u, v, depth = self.detector_coordinates(points)
        hu = 0.5 * (self.detector_cols - (not pixel_edges)) * self.pixel_width
        hv = 0.5 * (self.detector_rows - (not pixel_edges)) * self.pixel_height
        return (depth > 0) & (np.abs(u) <= hu + 1e-10) & (np.abs(v) <= hv + 1e-10)

    def visibility_fraction(self, points: np.ndarray, *, chunk_size: int = 16384) -> np.ndarray:
        """Fraction of views seeing each xyz point, in memory-bounded chunks."""
        points = np.asarray(points, dtype=np.float64)
        shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        result = np.empty(len(points), dtype=np.float32)
        for start in range(0, len(points), chunk_size):
            result[start:start + chunk_size] = self.detector_visibility(
                points[start:start + chunk_size]
            ).mean(axis=0)
        return result.reshape(shape)

    def longitudinal_intervals(self, xy: np.ndarray, *, pixel_edges: bool = True, chunk_size: int = 4096) -> np.ndarray:
        """Exact z interval seen by every view for each transverse xy point.

        Returns [...,2] lower/upper z limits (mm), or NaNs for an empty interval.
        Intersects the four cone half-spaces of every detector and positive
        source-depth half-spaces. This is purely finite-detector support and
        makes no claim about exact noncircular reconstruction support.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape[-1] != 2:
            raise ValueError("xy must have last dimension 2")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        shape = xy.shape[:-1]
        points = xy.reshape(-1, 2)
        n = self.normals
        hu = 0.5 * (self.detector_cols - (not pixel_edges)) * self.pixel_width
        hv = 0.5 * (self.detector_rows - (not pixel_edges)) * self.pixel_height
        normals = np.concatenate((
            hu * n + self.sdd_mm * self.col_vectors,
            hu * n - self.sdd_mm * self.col_vectors,
            hv * n + self.sdd_mm * self.row_vectors,
            hv * n - self.sdd_mm * self.row_vectors,
            n,
        ))
        sources = np.tile(self.source_positions, (5, 1))
        # Each half-space is a*z+b >= 0.
        a = normals[:, 2]
        offsets = np.einsum("ij,ij->i", sources, normals)
        positive, negative, zero = a > 1e-10, a < -1e-10, np.abs(a) <= 1e-10
        intervals = np.empty((len(points), 2), dtype=np.float64)
        for start in range(0, len(points), chunk_size):
            part = points[start:start + chunk_size]
            b = part @ normals[:, :2].T - offsets
            lower = np.max(-b[:, positive] / a[positive], axis=1) if positive.any() else np.full(len(part), -np.inf)
            upper = np.min(-b[:, negative] / a[negative], axis=1) if negative.any() else np.full(len(part), np.inf)
            empty = (lower > upper) | np.any(b[:, zero] < -1e-9, axis=1)
            section = np.stack((lower, upper), axis=-1)
            section[empty] = np.nan
            intervals[start:start + chunk_size] = section
        return intervals.reshape(*shape, 2)

    def fov_at_radius(self, radius_mm: float = 65.0, *, samples: int = 360) -> dict:
        """Azimuth-resolved all-view longitudinal visibility at a fixed radius."""
        if radius_mm < 0 or samples < 4:
            raise ValueError("radius_mm must be nonnegative and samples must be >=4")
        angles = np.arange(samples) * (360.0 / samples)
        xy = radius_mm * np.stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))), axis=-1)
        intervals = self.longitudinal_intervals(xy)
        height = intervals[:, 1] - intervals[:, 0]
        return {
            "definition": "all-view finite-detector visibility; not reconstruction completeness",
            "radius_mm": float(radius_mm),
            "azimuth_deg": angles.tolist(),
            "z_lower_mm": intervals[:, 0].tolist(),
            "z_upper_mm": intervals[:, 1].tolist(),
            "height_mm": height.tolist(),
            "height_min_mm": float(np.nanmin(height)),
            "height_max_mm": float(np.nanmax(height)),
            "height_mean_mm": float(np.nanmean(height)),
        }


def build_icono_orbit(
    kind: Literal["circular", "sinespin"] = "sinespin",
    *,
    detector_bin: int = 1,
    sod_mm: float = 750.0,
    sdd_mm: float = 1200.0,
    n_views: int | None = None,
    scan_angle_deg: float | None = None,
    tilt_amplitude_deg: float | None = None,
    tilt_phase_deg: float = 0.0,
    start_angle_deg: float | None = None,
) -> SineSpinGeometry:
    """Build the paper's nominal circular or one-cycle sine-on-sphere scan.

    Table 1 gives 200 degrees/496 views and 220 degrees/546 views, both rounded
    to 0.4 degree/view. We preserve total arc and count, including both endpoints
    (actual steps 0.404040 and 0.403670 degrees). This convention is explicit.

    The acquisition's 2x2 hardware binning gives approximately 1292x951 pixels
    at 0.308 mm. ``detector_bin`` is additional simulation downsampling. Ceil
    pixel counts and independently adjusted row/column pitches preserve exactly
    the same 397.936x292.908 mm active footprint at every simulation resolution.
    """
    if kind not in {"circular", "sinespin"}:
        raise ValueError(f"Unsupported orbit: {kind}")
    if int(detector_bin) != detector_bin or detector_bin < 1:
        raise ValueError("detector_bin must be a positive integer")
    if not (0 < sod_mm < sdd_mm):
        raise ValueError("Require 0 < sod_mm < sdd_mm")
    n_views = (496 if kind == "circular" else 546) if n_views is None else int(n_views)
    scan_angle_deg = (200.0 if kind == "circular" else 220.0) if scan_angle_deg is None else float(scan_angle_deg)
    amplitude = (0.0 if kind == "circular" else 10.0) if tilt_amplitude_deg is None else float(tilt_amplitude_deg)
    if n_views < 2 or not 0 < scan_angle_deg <= 360 or abs(amplitude) >= 90:
        raise ValueError("Require >=2 views, 0<scan<=360 degrees, and abs(tilt)<90 degrees")
    if start_angle_deg is None:
        start_angle_deg = -0.5 * scan_angle_deg
    progress = np.linspace(0.0, 1.0, n_views)
    theta_deg = float(start_angle_deg) + scan_angle_deg * progress
    tilt_deg = amplitude * np.sin(2 * np.pi * progress + np.deg2rad(tilt_phase_deg))
    theta, tilt = np.deg2rad(theta_deg), np.deg2rad(tilt_deg)
    radial = np.stack((np.cos(tilt) * np.sin(theta), -np.cos(tilt) * np.cos(theta), np.sin(tilt)), axis=-1)
    col = np.stack((np.cos(theta), np.sin(theta), np.zeros(n_views)), axis=-1)
    row = np.cross(radial, col)
    cols = int(np.ceil(1292 / detector_bin))
    rows = int(np.ceil(951 / detector_bin))
    return SineSpinGeometry(
        kind=kind,
        source_positions=float(sod_mm) * radial,
        module_centers=(float(sod_mm) - float(sdd_mm)) * radial,
        row_vectors=row,
        col_vectors=col,
        detector_rows=rows,
        detector_cols=cols,
        pixel_height=951 * 0.308 / rows,
        pixel_width=1292 * 0.308 / cols,
        theta_deg=theta_deg,
        tilt_deg=tilt_deg,
        sod_mm=float(sod_mm),
        sdd_mm=float(sdd_mm),
        scan_angle_deg=scan_angle_deg,
    )
