"""CPU-only physical-coordinate checks; no CQ500 data or DICOM decoder needed."""
import unittest

import numpy as np

from cq500_head import CQ500Head, resample_cq500_head, source_coverage_mask


def make_head(hu, affine):
    size = np.array(hu.shape[::-1])
    corners = np.array([[x, y, z] for x in (-.5, size[0] - .5)
                        for y in (-.5, size[1] - .5) for z in (-.5, size[2] - .5)])
    points = corners @ affine[:3, :3].T + affine[:3, 3]
    return CQ500Head(np.asarray(hu, dtype=np.float32), affine, {
        "native_shape_zyx": list(hu.shape),
        "native_lps_bounds_mm": [points.min(0).tolist(), points.max(0).tolist()],
    })


class CQ500HeadTests(unittest.TestCase):
    def test_sheared_lps_linear_hu_field_survives_resampling(self):
        affine = np.array([[.8, .12, .14, 5.], [.1, .9, .2, -3.],
                           [0., -.24, 1.1, 2.], [0., 0., 0., 1.]])
        z, y, x = np.indices((13, 15, 17), dtype=float)
        source_world = np.stack([x, y, z], -1) @ affine[:3, :3].T + affine[:3, 3]
        hu = 20. + source_world @ np.array([3., -2., 1.])
        head = make_head(hu, affine)
        mu, metadata = resample_cq500_head(head, (5, 7, 9), .5)
        self.assertTrue(source_coverage_mask(metadata).all())
        z, y, x = np.indices(mu.shape, dtype=float)
        target_world = (np.stack([x, y, z], -1) - [4., 3., 2.]) * .5 + head.centre_lps_mm
        expected_hu = 20. + target_world @ np.array([3., -2., 1.])
        np.testing.assert_allclose(mu, (expected_hu + 1000) * .02 / 1000, rtol=2e-7)

    def test_known_dicom_world_landmark_maps_to_isocenter(self):
        # Tilted planes translated along LPS z: slice step is not plane normal.
        affine = np.array([[1., 0., 0., -5.], [0., .96, 0., 7.],
                           [0., -.28, .65, 11.], [0., 0., 0., 1.]])
        hu = np.full((11, 9, 7), -1000., dtype=np.float32)
        hu[5, 4, 3] = 2000.
        head = make_head(hu, affine)
        landmark_lps = np.array([-2., 10.84, 13.13])
        mu, metadata = resample_cq500_head(head, (1, 1, 1), 1., isocenter_lps_mm=landmark_lps)
        self.assertAlmostEqual(float(mu[0, 0, 0]), .06, places=7)
        self.assertTrue(source_coverage_mask(metadata)[0, 0, 0])

    def test_positive_head_shift_moves_landmark_superiorly(self):
        affine = np.eye(4)
        affine[:3, 3] = [-3., -4., -5.]
        hu = np.full((11, 9, 7), -1000., dtype=np.float32)
        hu[5, 4, 3] = 2000.
        head = make_head(hu, affine)
        mu, metadata = resample_cq500_head(head, (9, 3, 3), 1.,
                                         isocenter_lps_mm=(0, 0, 0), center_shift_xyz_mm=(0, 0, 2))
        # z index 4 is simulated zero, so +2 mm must land at index 6.
        self.assertEqual(np.unravel_index(mu.argmax(), mu.shape), (6, 1, 1))
        self.assertAlmostEqual(float(mu[6, 1, 1]), .06, places=7)
        np.testing.assert_allclose(metadata["isocenter_lps_mm"], [0, 0, -2])

    def test_source_coverage_excludes_sheared_box_corners_and_air_padding(self):
        affine = np.array([[1., 0., 0., 0.], [0., 1., 1., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
        head = make_head(np.zeros((7, 7, 7), dtype=np.float32), affine)
        mu, metadata = resample_cq500_head(head, (9, 13, 9), 1., isocenter_lps_mm=(3, 6, 3))
        valid = source_coverage_mask(metadata)
        z, y, x = np.indices(mu.shape)
        wx, wy, wz = x - 1, y, z - 1
        # The affine is x=i, y=j+k, z=k; solve directly, without loader matrices.
        expected = ((wx >= 0) & (wx <= 6) & (wz >= 0) & (wz <= 6)
                    & (wy - wz >= 0) & (wy - wz <= 6))
        np.testing.assert_array_equal(valid, expected)
        np.testing.assert_allclose(mu[valid], .02, atol=1e-8)
        np.testing.assert_array_equal(mu[~valid], 0.)
        self.assertFalse(valid[7, 0, 4])  # LPS (3,0,6): in AABB, outside acquired planes.

    def test_native_hu_is_preserved_and_only_subair_attenuation_is_clipped(self):
        hu = np.array([[[-3024., -1000., 0., 3071.]]], dtype=np.float32)
        head = make_head(hu.copy(), np.eye(4))
        mu, metadata = resample_cq500_head(head, hu.shape, 1.)
        np.testing.assert_allclose(mu[0, 0], [0., 0., .02, .08142], rtol=1e-7)
        np.testing.assert_array_equal(head.native_hu, hu)
        self.assertEqual(mu.dtype, np.float32)
        self.assertTrue(source_coverage_mask(metadata).all())


if __name__ == "__main__":
    unittest.main()
