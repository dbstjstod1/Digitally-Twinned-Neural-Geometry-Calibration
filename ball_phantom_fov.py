"""Conservative full-volume detector containment before ball-phantom simulation.

The entire centred voxel BOX is checked, including half a voxel beyond the
outer voxel centres. With positive camera depth, a perspective projection of a
convex box is contained in a rectangular detector exactly when all eight box
corners satisfy its four detector half-spaces. This is a finite-detector test;
it is not a Tuy completeness or reconstruction-quality test.
"""
from __future__ import annotations

from itertools import product

import numpy as np


def _shape_spacing(shape_zyx, voxel_mm):
    shape = np.asarray(shape_zyx, dtype=np.float64)
    spacing = np.asarray(voxel_mm, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)
    if shape.shape != (3,) or not np.isfinite(shape).all() or np.any(shape <= 0) or np.any(shape != np.floor(shape)):
        raise ValueError("shape_zyx must contain three positive integer dimensions")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("voxel_mm must be positive finite scalar or three spacings in z,y,x order")
    return shape.astype(np.int64), spacing


def centered_box_corners(shape_zyx, voxel_mm):
    """Return [8,3] physical xyz corners of the full centred voxel box in mm.

    ``voxel_mm`` is an isotropic scalar or a three-vector in z,y,x order. No
    thresholded object support, cropping, recentering, or rescaling is applied.
    """
    shape, spacing = _shape_spacing(shape_zyx, voxel_mm)
    half_extent_xyz = (shape * spacing)[::-1] / 2
    return np.asarray(list(product((-1., 1.), repeat=3))) * half_extent_xyz


def _finite_list(array):
    return [float(value) if np.isfinite(value) else None for value in np.asarray(array).ravel()]


def audit_box_fov(geometry, shape_zyx, voxel_mm, *, margin_px=0.):
    """Report every-view full-box containment without modifying any input.

    ``geometry.projection_matrices()`` must map physical xyz in millimetres to
    pixel-centre (column,row) coordinates, with row 3 expressing positive camera
    depth in millimetres, as in :class:`sinespin_geometry.SineSpinGeometry`.
    ``margin_px`` optionally requires a nonnegative clearance from every edge.
    The detector's active pixel-edge bounds are [-.5,N-.5].
    """
    shape, spacing = _shape_spacing(shape_zyx, voxel_mm)
    margin_px = float(margin_px)
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("margin_px must be finite and nonnegative")
    pmat = np.asarray(geometry.projection_matrices(), dtype=np.float64)
    if pmat.ndim != 3 or pmat.shape[1:] != (3, 4) or not len(pmat) or not np.isfinite(pmat).all():
        raise ValueError("Geometry must provide finite [views,3,4] physical-to-pixel matrices")
    rows, cols = int(geometry.detector_rows), int(geometry.detector_cols)
    dv, du = float(geometry.pixel_height), float(geometry.pixel_width)
    if rows <= 0 or cols <= 0 or not np.isfinite([dv, du]).all() or min(dv, du) <= 0:
        raise ValueError("Detector dimensions and pitches must be positive and finite")
    corners = centered_box_corners(shape, spacing)
    homogeneous = np.c_[corners, np.ones(8)]
    projected = np.einsum("vij,nj->vni", pmat, homogeneous)
    depths = projected[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[..., :2] / depths[..., None]
    # Edge order is stable and recorded explicitly, including unequal u/v pitch.
    corner_margins = np.stack((uv[..., 0]+.5, cols-.5-uv[..., 0],
                               uv[..., 1]+.5, rows-.5-uv[..., 1]), axis=-1)
    edge_margins = np.min(corner_margins, axis=1)
    minimum_margin = np.min(edge_margins, axis=1)
    min_depth = np.min(depths, axis=1)
    finite = np.isfinite(uv).all(axis=(1, 2))
    positive = (depths > 0).all(axis=1)
    # Tolerance only handles round-off at the exact pixel edge; it is recorded.
    tolerance_px = 1e-9
    detector_inside = finite & (minimum_margin >= margin_px-tolerance_px)
    passed = positive & detector_inside
    theta = getattr(geometry, "theta_deg", np.arange(len(pmat)))
    tilt = getattr(geometry, "tilt_deg", np.zeros(len(pmat)))
    per_view = []
    for view in range(len(pmat)):
        per_view.append(dict(view=view, theta_deg=float(theta[view]), tilt_deg=float(tilt[view]),
                             passed=bool(passed[view]), all_corners_positive_depth=bool(positive[view]),
                             min_camera_depth_mm=float(min_depth[view]),
                             min_detector_margin_px=_finite_list([minimum_margin[view]])[0],
                             detector_edge_margins_px=_finite_list(edge_margins[view]),
                             detector_edge_margins_mm=_finite_list(edge_margins[view]*[du, du, dv, dv]),
                             projected_u_min_max_px=_finite_list([uv[view, :, 0].min(), uv[view, :, 0].max()]),
                             projected_v_min_max_px=_finite_list([uv[view, :, 1].min(), uv[view, :, 1].max()])))
    finite_margin = np.where(np.isfinite(minimum_margin), minimum_margin, -np.inf)
    worst_view = int(np.argmin(finite_margin))
    failures = np.flatnonzero(~passed).tolist()
    return dict(passed=bool(passed.all()), geometry_kind=str(getattr(geometry, "kind", "unspecified")),
                definition="Every full voxel-box corner is inside the active detector edges with positive camera depth in every acquired view; no rescale, crop, support threshold or shift",
                guarantee="For positive-depth convex boxes, checking eight corners bounds the entire box under perspective projection; not a Tuy completeness claim",
                shape_zyx=shape.tolist(), voxel_spacing_zyx_mm=spacing.tolist(),
                full_box_extent_xyz_mm=(shape*spacing)[::-1].tolist(),
                box_corners_xyz_mm=corners.tolist(), detector_shape_vu=[rows, cols],
                detector_pixel_vu_mm=[dv, du], detector_u_bounds_px=[-.5, cols-.5],
                detector_v_bounds_px=[-.5, rows-.5], required_margin_px=margin_px,
                numeric_edge_tolerance_px=tolerance_px,
                edge_order=["u_lower", "u_upper", "v_lower", "v_upper"],
                n_views=len(pmat), failed_view_count=len(failures), failed_views=failures,
                nonpositive_depth_views=np.flatnonzero(~positive).tolist(),
                minimum_camera_depth_mm=float(min_depth.min()), worst_margin_view=worst_view,
                minimum_detector_margin_px=per_view[worst_view]["min_detector_margin_px"],
                per_view=per_view)


def require_box_fov(geometries, shape_zyx, voxel_mm, *, margin_px=0.):
    """Return reports for a named geometry mapping, or reject any failing scan.

    Example: ``require_box_fov({'nominal': circle, 'truth': sine}, shape, .2)``.
    The exception has a ``reports`` attribute, so a runner can preserve the
    rejected audit before stopping without generating/training cropped data.
    """
    if not geometries:
        raise ValueError("At least one named geometry is required")
    reports = {name: audit_box_fov(geometry, shape_zyx, voxel_mm, margin_px=margin_px)
               for name, geometry in geometries.items()}
    failures = []
    for name, report in reports.items():
        if not report["passed"]:
            failures.append(f"{name}: {report['failed_view_count']}/{report['n_views']} views fail; "
                            f"worst detector margin {report['minimum_detector_margin_px']} px "
                            f"at view {report['worst_margin_view']}; "
                            f"minimum camera depth {report['minimum_camera_depth_mm']:.6g} mm")
    if failures:
        error = ValueError("Full phantom voxel box does not fit every detector view. " + "; ".join(failures))
        error.reports = reports
        raise error
    return reports
