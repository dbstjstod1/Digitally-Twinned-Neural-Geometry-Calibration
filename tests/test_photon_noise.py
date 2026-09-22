"""CPU tests of the physical count model and reproducible post-log simulation."""
import unittest

import numpy as np

from photon_noise import (DEFAULT_I0, UINT32_MAX, counts_to_line_integrals,
                          poisson_noisy_projections)


class PhotonNoiseTests(unittest.TestCase):
    def test_default_incident_fluence_matches_user_value(self):
        self.assertEqual(DEFAULT_I0, 44000.0)

    def test_count_moments_match_poisson_law(self):
        # For a Poisson variable, mean=variance=lambda. These independent
        # moment identities distinguish photon counting from post-log Gaussian
        # perturbations. Six standard errors avoid a brittle sampling test.
        shape = (30, 100, 100)
        n = np.prod(shape)
        for rate in (1.0, 50.0, 44000.0):
            with self.subTest(expected_count=rate):
                clean = np.full(shape, np.log(DEFAULT_I0 / rate), dtype=np.float64)
                _, _, counts = poisson_noisy_projections(clean, seed=823, return_counts=True)
                observed_mean = float(counts.mean())
                observed_variance = float(counts.var(ddof=1))
                mean_tolerance = 6 * np.sqrt(rate / n)
                variance_tolerance = 6 * np.sqrt((rate + 2 * rate ** 2) / (n - 1))
                self.assertLess(abs(observed_mean - rate), mean_tolerance)
                self.assertLess(abs(observed_variance - rate), variance_tolerance)

    def test_zero_policy_changes_only_zero_and_preserves_negative_logs(self):
        counts = np.array([0, 1, 2, 44000, 88000], dtype=np.uint32)
        actual = counts_to_line_integrals(counts)
        expected = np.array([np.log(88000), np.log(44000), np.log(22000), 0, -np.log(2)])
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
        self.assertEqual(actual.dtype, np.float32)
        self.assertLess(actual[-1], 0)
        self.assertTrue(np.isfinite(counts_to_line_integrals(counts, i0=1e-300)).all())

    def test_air_generates_negative_postlog_without_clipping(self):
        noisy, stats, counts = poisson_noisy_projections(
            np.zeros((10, 100, 100)), seed=4, return_counts=True)
        negative = counts > 44000
        np.testing.assert_array_equal(noisy < 0, negative)
        self.assertGreater(negative.mean(), .48)
        self.assertLess(negative.mean(), .52)
        self.assertEqual(stats["negative_postlog_count"], int(negative.sum()))

    def test_zero_transmission_remains_finite(self):
        noisy, stats, counts = poisson_noisy_projections(
            np.full((3, 7, 11), 1000.0), return_counts=True)
        np.testing.assert_array_equal(counts, 0)
        np.testing.assert_allclose(noisy, np.log(88000), rtol=1e-7)
        self.assertTrue(np.isfinite(noisy).all())
        self.assertEqual(stats["zero_count"], noisy.size)

    def test_seed_and_c_order_are_invariant_to_chunks_and_memory_layout(self):
        clean = np.linspace(-.2, 13, 13 * 7 * 11).reshape(13, 7, 11)
        original = clean.copy()
        a, _, counts_a = poisson_noisy_projections(clean, seed=99, chunk_views=1, return_counts=True)
        b, _, counts_b = poisson_noisy_projections(clean, seed=99, chunk_views=3, return_counts=True)
        c, _, counts_c = poisson_noisy_projections(
            np.asfortranarray(clean), seed=99, chunk_views=13, return_counts=True)
        np.testing.assert_array_equal(counts_a, counts_b)
        np.testing.assert_array_equal(counts_a, counts_c)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(clean, original)
        self.assertEqual(counts_a.dtype, np.uint32)
        different, _, _ = poisson_noisy_projections(clean, seed=100)
        self.assertFalse(np.array_equal(a, different))

    def test_counts_are_optional_without_changing_noisy_measurements(self):
        clean = np.ones((4, 5, 6))
        a, _, counts = poisson_noisy_projections(clean, seed=12, return_counts=True)
        b, _, omitted = poisson_noisy_projections(clean, seed=12, return_counts=False)
        np.testing.assert_array_equal(a, b)
        self.assertIsNone(omitted)
        self.assertEqual(counts.dtype, np.uint32)

    def test_invalid_rates_and_storage_overflow_are_rejected(self):
        for i0 in (0, -1, np.inf, np.nan, UINT32_MAX + 1):
            with self.subTest(i0=i0), self.assertRaises(ValueError):
                poisson_noisy_projections(np.zeros((1, 2, 3)), i0=i0)
        for clean in (np.array([np.nan]), np.array([np.inf]), np.array([-1000.0]),
                      np.empty((0, 2, 3)), np.array([1 + 2j])):
            with self.subTest(clean=clean), self.assertRaises(ValueError):
                poisson_noisy_projections(clean)
        for kwargs in ({"seed": -1}, {"seed": .5}, {"chunk_views": 0}, {"chunk_views": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                poisson_noisy_projections(np.zeros((1, 2, 3)), **kwargs)
        for counts in (np.array([-1]), np.array([UINT32_MAX + 1]), np.array([.5])):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                counts_to_line_integrals(counts)


if __name__ == "__main__":
    unittest.main()
