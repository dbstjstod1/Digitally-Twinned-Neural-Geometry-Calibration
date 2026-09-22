import json
import unittest
from types import SimpleNamespace

import numpy as np

from ball_phantom_fov import audit_box_fov, centered_box_corners, require_box_fov
from sinespin_geometry import build_icono_orbit


class BallPhantomFovTests(unittest.TestCase):
    def test_full_voxel_edges_and_anisotropic_zyx_order(self):
        corners = centered_box_corners((3, 4, 5), (.2, .3, .4))
        np.testing.assert_allclose(corners.min(axis=0), [-1., -.6, -.3])
        np.testing.assert_allclose(corners.max(axis=0), [1., .6, .3])
        self.assertEqual(len(np.unique(corners, axis=0)), 8)

    def test_small_box_fits_both_orbits_at_every_view(self):
        geos = {kind: build_icono_orbit(kind, detector_bin=2, n_views=546, scan_angle_deg=220)
                for kind in ("circular", "sinespin")}
        reports = require_box_fov(geos, (651, 643, 643), .2)
        for report in reports.values():
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["per_view"]), 546)
            self.assertGreater(report["minimum_detector_margin_px"], 0)
            self.assertEqual(report["failed_views"], [])
            json.dumps(report, allow_nan=False)

    def test_corners_agree_with_independent_ray_detector_coordinates(self):
        geometry = build_icono_orbit("sinespin", detector_bin=3, n_views=41)
        corners = centered_box_corners((20, 30, 40), (.2, .3, .4))
        report = audit_box_fov(geometry, (20, 30, 40), (.2, .3, .4))
        u, v, depth = geometry.detector_coordinates(corners)
        up = u/geometry.pixel_width+(geometry.detector_cols-1)/2
        vp = v/geometry.pixel_height+(geometry.detector_rows-1)/2
        for index, view in enumerate(report["per_view"]):
            np.testing.assert_allclose(view["projected_u_min_max_px"], [up[index].min(), up[index].max()])
            np.testing.assert_allclose(view["projected_v_min_max_px"], [vp[index].min(), vp[index].max()])
            self.assertAlmostEqual(view["min_camera_depth_mm"], depth[index].min())

    def test_voxel_centres_fitting_does_not_allow_box_edges_to_be_cut(self):
        # All centres x,y=+/-.5,z=0 fit, but near-face full voxel edges do not.
        matrix = np.array([[[10., 0., .5, 5.], [0., 10., .5, 5.], [0., 0., 1., 10.]]])
        geometry = SimpleNamespace(projection_matrices=lambda: matrix, detector_rows=2,
                                   detector_cols=2, pixel_height=1., pixel_width=1.)
        centres = np.array([[x, y, 0., 1.] for x in (-.5, .5) for y in (-.5, .5)])
        projected = centres @ matrix[0].T
        self.assertTrue(((projected[:, :2]/projected[:, 2:] >= -.5)
                         & (projected[:, :2]/projected[:, 2:] <= 1.5)).all())
        report = audit_box_fov(geometry, (1, 2, 2), 1.)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["minimum_detector_margin_px"], -1/19)
        with self.assertRaisesRegex(ValueError, "does not fit") as raised:
            require_box_fov({"synthetic": geometry}, (1, 2, 2), 1.)
        self.assertEqual(raised.exception.reports["synthetic"]["failed_views"], [0])

    def test_behind_source_and_zero_depth_are_rejected_even_if_uv_fits(self):
        for offset in (-10., 0.):
            # Tiny lateral magnification keeps nonzero-depth pixels on detector.
            matrix = np.array([[[.001, 0., .5, .5*offset], [0., .001, .5, .5*offset], [0., 0., 1., offset]]])
            geometry = SimpleNamespace(projection_matrices=lambda: matrix, detector_rows=2,
                                       detector_cols=2, pixel_height=1., pixel_width=1.)
            report = audit_box_fov(geometry, (2, 2, 2), 1.)
            self.assertFalse(report["passed"])
            self.assertEqual(report["nonpositive_depth_views"], [0])
            json.dumps(report, allow_nan=False)
        zero_face = matrix.copy()
        zero_face[0, 2, 3] = 1.
        geometry.projection_matrices = lambda: zero_face
        report = audit_box_fov(geometry, (2, 2, 2), 1.)
        self.assertFalse(report["passed"])
        json.dumps(report, allow_nan=False)

    def test_requested_margin_and_binning_invariant_physical_clearance(self):
        geometry = build_icono_orbit("sinespin", detector_bin=2, n_views=19)
        report = audit_box_fov(geometry, (30, 40, 50), .5)
        self.assertTrue(report["passed"])
        rejected = audit_box_fov(geometry, (30, 40, 50), .5,
                                 margin_px=report["minimum_detector_margin_px"]+1)
        self.assertFalse(rejected["passed"])
        binned = build_icono_orbit("sinespin", detector_bin=5, n_views=19)
        other = audit_box_fov(binned, (30, 40, 50), .5)
        np.testing.assert_allclose([v["detector_edge_margins_mm"] for v in report["per_view"]],
                                   [v["detector_edge_margins_mm"] for v in other["per_view"]], atol=1e-12)

    def test_invalid_dimensions_and_spacing_are_not_silently_changed(self):
        for shape, spacing in [((1, 2, 2.5), 1), ((1, 0, 2), 1), ((1, 2), 1),
                               ((1, 2, 3), 0), ((1, 2, 3), [1, 2]), ((1, 2, 3), np.nan)]:
            with self.assertRaises(ValueError):
                centered_box_corners(shape, spacing)


if __name__ == "__main__":
    unittest.main()
