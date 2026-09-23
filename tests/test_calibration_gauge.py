"""Canonical P decomposition and fixed-label, single-frame pose audit checks."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_gauge import (apply_rigid_frame, compose_effective_parameters,
                               decompose_world_pmat, effective_parameters_from_pmat,
                               fit_rigid_landmark_pose, transform_points)
from calibration_geometry import geometry_to_pmat, pmat_to_pixel
from sinespin_geometry import build_icono_orbit


def project(matrix, points):
    values = matrix @ np.c_[points, np.ones(len(points))].T
    return (values[:, :2] / values[:, 2, None]).transpose(0, 2, 1)


class CalibrationGaugeTests(unittest.TestCase):
    def setUp(self):
        self.options = dict(n_views=17, scan_angle_deg=220., detector_bin=2)
        self.nominal = build_icono_orbit("circular", **self.options)
        self.truth = build_icono_orbit("sinespin", **self.options)
        self.p0 = geometry_to_pmat(self.nominal, dtype=np.float64)

    def test_arbitrary_positive_and_negative_scale_give_same_nine_parameters(self):
        rng = np.random.default_rng(281)
        p9 = rng.uniform(-1., 1., (17, 9)) * np.array([5.]*6 + [12.]*3)
        matrix = compose_effective_parameters(self.p0, p9)
        scale = rng.uniform(.2, 4., 17) * rng.choice([-1., 1.], 17)
        scaled_nominal = self.p0 * (-2.3)
        result = effective_parameters_from_pmat(matrix * scale[:, None, None], scaled_nominal)
        np.testing.assert_allclose(result["parameters_9"], p9, atol=8e-12)
        np.testing.assert_allclose(np.linalg.det(result["R_physical"]), 1., atol=1e-14)
        self.assertLess(result["effective_model_relative_residual"].max(), 2e-14)
        decomposition = decompose_world_pmat(matrix * scale[:, None, None])
        rebuilt = decomposition["K"] @ np.concatenate((decomposition["R"], decomposition["t"][:, :, None]), axis=2)
        rebuilt *= decomposition["projective_scale"][:, None, None]
        np.testing.assert_allclose(rebuilt, matrix * scale[:, None, None], atol=2e-9)

    def test_nonzero_pivot_and_original_float32_forward_agree(self):
        import torch
        from DoF_transform import apply_9DoF_transform_effective

        center = np.array([12., -5., 42.])
        nominal = geometry_to_pmat(self.nominal, center_internal_mm=center, dtype=np.float64)
        rng = np.random.default_rng(482)
        p9 = rng.uniform(-1., 1., (17, 9)) * np.array([5.]*6 + [12.]*3)
        matrix = compose_effective_parameters(nominal, p9, center_internal_mm=center)
        recovered = effective_parameters_from_pmat(matrix, nominal, center_internal_mm=center)
        np.testing.assert_allclose(recovered["parameters_9"], p9, atol=8e-12)
        origin = center - np.array([643, 643, 651]) * .2 / 2
        original, _ = apply_9DoF_transform_effective(
            torch.tensor(nominal, dtype=torch.float32), torch.zeros((17, 7)),
            torch.tensor(p9[:, :3], dtype=torch.float32), torch.tensor(p9[:, 3:6], dtype=torch.float32),
            torch.tensor(p9[:, 6:], dtype=torch.float32),
            nx=643, ny=643, nz=651, dx=.2, dy=.2, dz=.2,
            X0=origin[0], Y0=origin[1], Z0=origin[2])
        points = rng.uniform(-60., 60., (80, 3))
        conversion = dict(du=self.truth.pixel_width, dv=self.truth.pixel_height, center_internal_mm=center)
        error = project(pmat_to_pixel(original.numpy(), **conversion), points)
        error -= project(pmat_to_pixel(matrix, **conversion), points)
        self.assertLess(np.abs(error).max(), 3e-4)

    def test_all_546_truth_views_independently_recover_analytic_tilt_and_camera_basis(self):
        options = dict(n_views=546, scan_angle_deg=220., detector_bin=2)
        nominal = build_icono_orbit("circular", **options)
        truth = build_icono_orbit("sinespin", **options)
        p0 = geometry_to_pmat(nominal, dtype=np.float64)
        p = geometry_to_pmat(truth, dtype=np.float64)
        result = effective_parameters_from_pmat(p, p0)
        expected = Rotation.from_rotvec(np.deg2rad(truth.tilt_deg)[:, None] * truth.col_vectors)
        np.testing.assert_allclose(result["parameters_9"][:, :6], 0., atol=2e-12)
        np.testing.assert_allclose(result["relative_rotation_internal"], expected.as_matrix(), atol=2e-15)
        np.testing.assert_allclose(result["source_xyz_mm"], truth.source_positions, atol=7e-13)
        expected_basis = np.stack((truth.col_vectors, -truth.row_vectors, truth.normals), axis=1)
        np.testing.assert_allclose(result["R_physical"], expected_basis, atol=2e-15)
        reconstructed = compose_effective_parameters(p0, result["parameters_9"])
        np.testing.assert_allclose(reconstructed, p, atol=7e-10)

    def test_one_landmark_pose_recovers_global_transform_and_preserves_every_projection(self):
        rng = np.random.default_rng(491)
        source = rng.uniform(-50., 50., (35, 3))
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [7., -3., 4.], degrees=True).as_matrix()
        transform[:3, 3] = [2., -1., 3.]
        target = transform_points(source, transform)
        fit = fit_rigid_landmark_pose(source, target)
        np.testing.assert_allclose(fit["H"], transform, atol=2e-14)
        self.assertLess(fit["rms_after_mm"], 5e-14)
        original_p = self.truth.projection_matrices()
        changed_p = apply_rigid_frame(original_p, fit["H"])
        np.testing.assert_allclose(project(changed_p, target), project(original_p, source), atol=2e-13)
        old_source = np.linalg.solve(original_p[:, :, :3], -original_p[:, :, 3, None])[:, :, 0]
        new_source = np.linalg.solve(changed_p[:, :, :3], -changed_p[:, :, 3, None])[:, :, 0]
        np.testing.assert_allclose(new_source, transform_points(old_source, transform), atol=5e-13)

    def test_pose_does_not_fit_scale_reflection_or_view_specific_warps(self):
        rng = np.random.default_rng(783)
        source = rng.uniform(-50., 50., (35, 3))
        target = source * 1.1
        fit = fit_rigid_landmark_pose(source, target)
        np.testing.assert_allclose(fit["rotation"].T @ fit["rotation"], np.eye(3), atol=2e-15)
        self.assertGreater(fit["rms_after_mm"], 1.)
        reflected = source.copy()
        reflected[:, 0] *= -1
        fit = fit_rigid_landmark_pose(source, reflected)
        self.assertAlmostEqual(np.linalg.det(fit["rotation"]), 1.)
        self.assertGreater(fit["rms_after_mm"], 10.)
        with self.assertRaises(ValueError):
            apply_rigid_frame(self.p0, np.diag([1.1, 1.1, 1.1, 1.]))
        with self.assertRaises(ValueError):
            apply_rigid_frame(self.p0, np.tile(np.eye(4), (17, 1, 1)))
        with self.assertRaises(ValueError):
            fit_rigid_landmark_pose(np.arange(9).reshape(3, 3), np.arange(9).reshape(3, 3))


if __name__ == "__main__":
    unittest.main()
