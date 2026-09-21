"""CPU checks for FDK integration, redundancy normalization, and physical units."""
import unittest
from unittest.mock import patch

import numpy as np

from circular_fdk import _logical_gpu, _parker_profile, angular_weights, parker_weights, ramp_filter


class CircularFDKTests(unittest.TestCase):
    def test_nonuniform_trapezoid_integrates_linear_function(self):
        angles = np.array([-1.7, -1.6, -1.1, -.2, .4, 1.8])
        weights = angular_weights(angles)
        self.assertAlmostEqual(weights.sum(), angles[-1] - angles[0])
        self.assertAlmostEqual(weights @ angles, .5 * (angles[-1] ** 2 - angles[0] ** 2))
        with self.assertRaises(ValueError):
            angular_weights([0, 1, 1])

    def test_parker_conjugate_rays_sum_to_one(self):
        for span in np.deg2rad([200., 220., 300.]):
            for gamma in np.deg2rad([-8., -3., 0., 5., 8.]):
                beta = np.linspace(0, span - np.pi - 2 * gamma, 67)
                paired = beta + np.pi + 2 * gamma
                total = _parker_profile(beta, gamma, span) + _parker_profile(paired, -gamma, span)
                np.testing.assert_allclose(total, 1., atol=3e-14)

    def test_central_ray_short_and_full_scans_integrate_to_pi(self):
        for span in (200., 220., 360.):
            angles = np.linspace(-1.3, -1.3 + np.deg2rad(span), 4001)
            weights = parker_weights(angles, np.array([0.]), 1200.)[:, 0]
            self.assertAlmostEqual(angular_weights(angles) @ weights, np.pi, places=7)
            self.assertTrue(np.all((weights >= 0) & (weights <= 1)))
            if span == 360.:
                np.testing.assert_array_equal(weights, .5)
            else:
                np.testing.assert_allclose(weights[[0, -1]], 0., atol=1e-25)

    def test_fan_direction_matches_orbit_and_incomplete_scan_rejected(self):
        # u increases with source motion: positive u conjugates at pi-2*atan(u/D).
        span = np.deg2rad(200.)
        angles = np.linspace(0, span, 901)
        u = np.array([-180., 0., 180.])
        actual = parker_weights(angles, u, 1200.)
        expected = _parker_profile(angles[:, None], -np.arctan(u[None, :] / 1200.), span)
        np.testing.assert_allclose(actual, expected, atol=3e-8)
        self.assertLess(actual[50, 2], actual[50, 0])
        with self.assertRaises(ValueError):
            parker_weights(np.linspace(0, np.deg2rad(190.), 20), u, 1200.)

    def test_ramlak_impulse_has_physical_scale_and_no_edge_wrap(self):
        impulse = np.zeros((2, 37), dtype=np.float32)
        impulse[0, 0] = 1
        impulse[1, 18] = 1
        spacing = .7
        filtered = ramp_filter(impulse, spacing)
        for row, center in ((0, 0), (1, 18)):
            for j in range(37):
                offset = j - center
                expected = (1 / (4 * spacing) if offset == 0 else
                            -1 / (np.pi ** 2 * spacing * offset ** 2) if offset % 2 else 0.)
                self.assertAlmostEqual(float(filtered[row, j]), expected, places=7)
        np.testing.assert_allclose(ramp_filter(impulse, 2 * spacing), filtered / 2, atol=1e-8)

    def test_gpu_mask_maps_physical_one_to_logical_zero(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertEqual(_logical_gpu(1), 0)
            with self.assertRaises(ValueError):
                _logical_gpu(0)


if __name__ == "__main__":
    unittest.main()
