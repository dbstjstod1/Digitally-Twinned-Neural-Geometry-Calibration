"""CPU regressions: python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from shepp_logan import analytic_line_integrals, create_shepp_logan
from sinespin_geometry import build_icono_orbit


class SineSpinGeometryTests(unittest.TestCase):
    def test_common_isocenter_and_leap_xyz_frame(self):
        for kind in ("circular", "sinespin"):
            g = build_icono_orbit(kind, n_views=5, detector_bin=2)
            s, d, row, col = g.modular_arrays()
            np.testing.assert_allclose(s + (d-s) * (750/1200), 0, atol=7e-5)
            np.testing.assert_allclose(np.linalg.norm(s, axis=1), 750, atol=1e-4)
            np.testing.assert_allclose(np.linalg.norm(d-s, axis=1), 1200, atol=1e-4)
            np.testing.assert_allclose(s[2], [0, -750, 0], atol=1e-5)
            np.testing.assert_allclose(d[2], [0, 450, 0], atol=1e-5)
            np.testing.assert_allclose(row[2], [0, 0, 1], atol=1e-6)
            np.testing.assert_allclose(col[2], [1, 0, 0], atol=1e-6)
            np.testing.assert_allclose(np.sum(row*col, axis=1), 0, atol=1e-6)
            for a in (s, d, row, col):
                self.assertEqual(a.dtype, np.float32)
                self.assertTrue(a.flags.c_contiguous)
            p = g.projection_matrices()
            principal = p[:, :2, 3] / p[:, 2, 3, None]
            expected = [(g.detector_cols-1)/2, (g.detector_rows-1)/2]
            np.testing.assert_allclose(principal, np.tile(expected, (5, 1)), atol=1e-10)

    def test_pixel_matrix_against_independent_ray_plane_intersections(self):
        g = build_icono_orbit(detector_bin=4)
        points = np.array([[0, 0, 0], [31, -47, 63], [-72, 18, -52]])
        for i in (0, 136, 272, 409, 545):
            s, d = g.source_positions[i], g.module_centers[i]
            row, col = g.row_vectors[i], g.col_vectors[i]
            plane_normal = np.cross(row, col)
            for point in points:
                ray = point-s
                t = np.dot(d-s, plane_normal) / np.dot(ray, plane_normal)
                intersection = s+t*ray
                offset = intersection-d
                expected = [np.dot(offset, col)/g.pixel_width+(g.detector_cols-1)/2,
                            np.dot(offset, row)/g.pixel_height+(g.detector_rows-1)/2]
                projected = g.projection_matrices()[i] @ np.r_[point, 1.0]
                np.testing.assert_allclose(projected[:2]/projected[2], expected, atol=1e-10)

    def test_longitudinal_interval_is_actual_detector_boundary(self):
        xy = np.random.default_rng(19).uniform(-65, 65, (17, 2))
        for kind in ("circular", "sinespin"):
            g = build_icono_orbit(kind, detector_bin=2)
            limits = g.longitudinal_intervals(xy, chunk_size=3)
            for end, outward in ((0, -1), (1, 1)):
                boundary = np.c_[xy, limits[:, end]]
                self.assertTrue(g.detector_visibility(boundary).all())
                outside = boundary + [0, 0, outward*.001]
                self.assertTrue((~g.detector_visibility(outside).all(axis=0)).all())
                inside = boundary - [0, 0, outward*.001]
                self.assertTrue(g.detector_visibility(inside).all())
            self.assertTrue(np.isnan(g.longitudinal_intervals([[2000., 0.]])).all())

    def test_binning_preserves_physical_detector_and_fov(self):
        xy = np.array([[0., 0.], [65, 0], [-65, 0], [0, 65], [0, -65]])
        for kind in ("circular", "sinespin"):
            reference = build_icono_orbit(kind).longitudinal_intervals(xy)
            for factor in (1, 2, 4):
                g = build_icono_orbit(kind, detector_bin=factor)
                self.assertAlmostEqual(g.detector_width_mm, 397.936, places=10)
                self.assertAlmostEqual(g.detector_height_mm, 292.908, places=10)
                np.testing.assert_allclose(g.longitudinal_intervals(xy), reference, atol=1e-10)

    def test_zero_tilt_matches_circular_on_same_arc(self):
        options = dict(n_views=37, scan_angle_deg=220., detector_bin=2)
        circular = build_icono_orbit("circular", **options)
        zero = build_icono_orbit("sinespin", tilt_amplitude_deg=0, **options)
        for expected, actual in zip(circular.modular_arrays(), zero.modular_arrays()):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(zero.projection_matrices(), circular.projection_matrices())

    def test_paper_arcs_and_sinespin_is_not_a_tilted_circle(self):
        circular, sine = build_icono_orbit("circular"), build_icono_orbit("sinespin")
        self.assertEqual((circular.n_views, sine.n_views), (496, 546))
        self.assertEqual((np.ptp(circular.theta_deg), np.ptp(sine.theta_deg)), (200, 220))
        self.assertAlmostEqual(sine.tilt_deg.max(), 10, places=4)
        self.assertAlmostEqual(sine.tilt_deg.min(), -10, places=4)
        centered = sine.source_positions-sine.source_positions.mean(axis=0)
        _, _, directions = np.linalg.svd(centered, full_matrices=False)
        self.assertGreater(np.sqrt(np.mean((centered @ directions[-1])**2)), 40.)


class SheppLoganTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_does_not_resize_phantom(self):
        small = create_shepp_logan((71, 61, 61), 4.)
        large = create_shepp_logan((81, 71, 71), 4.)
        np.testing.assert_array_equal(small, create_shepp_logan(small.shape, 4.))
        np.testing.assert_array_equal(small, large[5:-5, 5:-5, 5:-5])
        occupied = np.argwhere(small > 0)
        span_mm = (occupied.max(axis=0)-occupied.min(axis=0))*4.
        np.testing.assert_allclose(span_mm, [260, 200, 200], atol=4.)
        self.assertGreaterEqual(float(small.min()), 0.)

    def test_analytic_equatorial_ray_has_known_chord_sum(self):
        # At y=z=0 only the two concentric outer ellipsoids intersect the x ray.
        expected = 200*.02 - 192*.016
        rays = analytic_line_integrals([[-750, 0, 0], [-750, 200, 0]],
                                      [[450, 0, 0], [450, 200, 0]])
        self.assertAlmostEqual(float(rays[0]), expected, places=6)
        self.assertEqual(float(rays[1]), 0.)


if __name__ == "__main__":
    unittest.main()
