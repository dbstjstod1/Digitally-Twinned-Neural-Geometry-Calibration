"""Independent image-centroid and fixed-camera 3-D reconstruction checks."""
import unittest

import numpy as np

from audit_sinespin_phantom_pose import detect, project, triangulate
from sinespin_geometry import build_icono_orbit


class PhantomPoseAuditTests(unittest.TestCase):
    def test_centroid_follows_image_not_association_seed(self):
        yy, xx = np.mgrid[:61, :61]
        for center in (np.array([30.35, 29.68]), np.array([31., 30.1])):
            image = .2 + .001*xx + .002*yy
            image += np.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2*.8**2))
            measured = []
            for offset in ((0., 0.), (.25, 0.), (-.25, 0.), (0., .25)):
                uv, _, _ = detect(image[None], np.array([[[30., 30.]]]), seed_offset=offset)
                self.assertTrue(np.isfinite(uv).all())
                np.testing.assert_allclose(uv[0, 0], center, atol=.003, rtol=0)
                measured.append(uv[0, 0])
            self.assertLess(float(np.ptp(measured, axis=0).max()), .003)

    def test_fixed_camera_triangulation_recovers_nonplanar_points(self):
        geometry = build_icono_orbit('sinespin', n_views=40, detector_bin=2,
                                    scan_angle_deg=220.)
        camera = geometry.projection_matrices()
        points = np.array([[5., -8., 4.], [-20., 11., -30.], [10., 20., 24.],
                           [0., -15., 10.]])
        observations = project(camera, points)
        observations[::5, 0] = np.nan  # Independent missing views remain valid.
        reconstructed, records = triangulate(camera, observations)
        np.testing.assert_allclose(reconstructed, points, atol=1e-7, rtol=0)
        self.assertEqual(records[0]['observation_count'], 32)
        self.assertLess(max(r['detector_radial_rmse_px'] for r in records), 1e-8)


if __name__ == '__main__':
    unittest.main()
