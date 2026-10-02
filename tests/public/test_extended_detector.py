import unittest
import numpy as np
from geocal.detector import ExtendedDetector
from geocal.sinespin import build_icono_orbit
from geocal.pipeline import project_points


class ExtendedDetectorTests(unittest.TestCase):
    def test_padding_changes_pixel_origin_without_changing_physical_rays(self):
        a = build_icono_orbit("sinespin", n_views=31, detector_bin=2)
        b = ExtendedDetector(a, (240, 180))
        for left, right in zip(a.modular_arrays(), b.modular_arrays()):
            np.testing.assert_array_equal(left, right)
        points = np.random.default_rng(4).uniform(-100, 100, (35, 3))
        np.testing.assert_allclose(
            project_points(b.projection_matrices(), points),
            project_points(a.projection_matrices(), points) + [180, 240],
            atol=1e-10,
        )
