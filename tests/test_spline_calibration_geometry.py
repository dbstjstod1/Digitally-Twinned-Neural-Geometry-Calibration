"""Check independently composed spline cameras against the fitted model and rays."""
import unittest
import numpy as np
import torch
from spline_calibration_geometry import SplineGeometry, spline_motion9
from sinespin_geometry import build_icono_orbit
from calibration_geometry import geometry_to_pmat, pmat_to_pixel
from calibration_gauge import effective_parameters_from_pmat, compose_effective_parameters
from run_sinespin_calibration import apply_motion
from ball_phantom_fov import require_box_fov


class SplineGeometryTests(unittest.TestCase):
    def setUp(self):
        self.nominal = build_icono_orbit('circular', n_views=31, scan_angle_deg=220, detector_bin=2)
        self.config = dict(seed=20260923, knots=8, amplitudes9=[2.]*9)
        self.truth = SplineGeometry(self.nominal, self.config)

    def test_spline_labels_and_recovery(self):
        motion = self.truth.motion9
        np.testing.assert_allclose(abs(motion).max(axis=0), 2., atol=1e-12)
        np.testing.assert_allclose(motion.mean(axis=0), 0., atol=1e-12)
        np.testing.assert_array_equal(motion, spline_motion9(31, self.config)[0])
        recovered = effective_parameters_from_pmat(geometry_to_pmat(self.truth, dtype=np.float64),
            geometry_to_pmat(self.nominal, dtype=np.float64))['parameters_9']
        np.testing.assert_allclose(recovered, motion, atol=2e-10)

    def test_modular_ray_closure(self):
        analytic = pmat_to_pixel(self.truth.world_pmat, du=self.truth.pixel_width, dv=self.truth.pixel_height)
        actual = self.truth.projection_matrices()
        np.testing.assert_allclose(actual, analytic, atol=1e-9, rtol=1e-11)
        points = np.random.default_rng(7).uniform(-60, 60, (23, 3))
        # Physical plane intersection gives pixel coordinates independently.
        direction = points[None]-self.truth.source_positions[:, None]
        scale = self.truth.focal_mm[:, None]/np.einsum('vpi,vi->vp', direction, self.truth.normals)
        impact = self.truth.source_positions[:, None]+scale[..., None]*direction-self.truth.module_centers[:, None]
        uv = np.stack((np.einsum('vpi,vi->vp', impact, self.truth.col_vectors)/self.truth.pixel_width+(self.truth.detector_cols-1)/2,
                       np.einsum('vpi,vi->vp', impact, self.truth.row_vectors)/self.truth.pixel_height+(self.truth.detector_rows-1)/2), axis=-1)
        q = actual @ np.c_[points, np.ones(len(points))].T
        np.testing.assert_allclose(q[:, :2]/q[:, 2:], uv.transpose(0, 2, 1), atol=1e-9)
        self.assertTrue(self.truth.detector_visibility(points).all())
        require_box_fov({'spline': self.truth}, (651, 643, 643), .2)

    def test_fitted_transform_matches_independent_labels_and_has_gradients(self):
        bounds = dict(ts_max_mm=10., tp_max_mm=10., rot_max_deg=15.)
        raw = torch.tensor(np.arctanh(self.truth.motion9/np.array([10.]*6+[15.]*3)), dtype=torch.float32, requires_grad=True)
        p, applied = apply_motion(torch.tensor(geometry_to_pmat(self.nominal, dtype=np.float64)), raw, bounds,
                                  shape=(651, 643, 643), voxel=.2)
        # The production transform intentionally computes float32. Compare rays.
        self.assertEqual(p.dtype, torch.float32)
        np.testing.assert_allclose(applied.detach().numpy(), self.truth.motion9, atol=1e-6)
        points = torch.tensor(np.c_[np.random.default_rng(9).uniform(-50, 50, (19, 3)), np.ones(19)], dtype=p.dtype)
        projected = p @ points.T
        oracle = self.truth.world_pmat @ points.numpy().T
        np.testing.assert_allclose((projected[:, :2]/projected[:, 2:]).detach().numpy(),
            oracle[:, :2]/oracle[:, 2:], atol=2e-4, rtol=0)
        (projected[:, :2]/projected[:, 2:]).square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad.abs().sum(0) > 1e-6).all())
        # Independent float64 finite differences at start/middle/end views.
        raw_values = raw.detach().double().numpy()
        scales = np.array([10.]*6+[15.]*3)
        nominal64 = geometry_to_pmat(self.nominal, dtype=np.float64)
        def objective(values):
            matrix = compose_effective_parameters(nominal64, scales*np.tanh(values))
            q = matrix @ points.double().numpy().T
            return np.mean((q[:, :2]/q[:, 2:])**2)
        for view in (0, 15, 30):
            finite = []
            for j in range(9):
                plus, minus = raw_values.copy(), raw_values.copy()
                plus[view,j] += 1e-5; minus[view,j] -= 1e-5
                finite.append((objective(plus)-objective(minus))/2e-5)
            np.testing.assert_allclose(raw.grad[view].numpy(), finite, rtol=2e-4, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
