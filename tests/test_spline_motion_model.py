"""Independent spline values, coefficient gradients, and camera integration."""
import unittest
import numpy as np
import torch
from scipy.interpolate import BSpline

from spline_motion_model import BSplineMotion9, cubic_bspline_basis
from calibration_gauge import compose_effective_parameters
from calibration_geometry import geometry_to_pmat
from sinespin_geometry import build_icono_orbit
from run_sinespin_calibration import apply_motion


class SplineEstimatorTests(unittest.TestCase):
    def test_basis_endpoints_partition_and_local_support(self):
        for views, controls in ((31, 4), (546, 20), (61, 30)):
            b, knots = cubic_bspline_basis(views, controls)
            self.assertEqual(b.shape, (views, controls))
            np.testing.assert_allclose(b.sum(1), 1., atol=1e-14)
            self.assertTrue((b >= 0).all())
            self.assertTrue(((b > 0).sum(1) <= 4).all())
            np.testing.assert_array_equal(b[0], np.eye(controls)[0])
            np.testing.assert_array_equal(b[-1], np.eye(controls)[-1])
            # The last two views are independent, not an endpoint-copy shortcut.
            self.assertFalse(np.array_equal(b[-1], b[-2]))
            self.assertEqual(np.linalg.matrix_rank(b), controls)

    def test_physical_curve_matches_scipy_and_respects_bounds(self):
        model = BSplineMotion9(101, 20, dtype=torch.float64)
        rng = np.random.default_rng(9)
        with torch.no_grad():
            model.raw_coefficients.copy_(torch.tensor(rng.normal(0, 3, (20, 9))))
        coefficients = model.physical_coefficients().detach().numpy()
        expected = BSpline(model.knots.numpy(), coefficients, 3)(np.linspace(0, 1, 101))
        curve = model(torch.arange(101)).detach().numpy()
        np.testing.assert_allclose(curve, expected, atol=1e-14)
        self.assertTrue((abs(curve) <= model.scales.numpy() + 1e-13).all())
        # No forced zero mean or periodic endpoint; constant shifts are estimable.
        with torch.no_grad(): model.raw_coefficients.fill_(.2)
        np.testing.assert_allclose(model(torch.arange(101)).detach().numpy(),
                                   np.tile(np.tanh(.2)*model.scales.numpy(), (101, 1)), atol=1e-14)

    def test_shuffle_and_checkpoint_roundtrip(self):
        model = BSplineMotion9(31, 8)
        with torch.no_grad(): model.raw_coefficients.copy_(torch.arange(72).reshape(8, 9)/100.)
        ids = torch.tensor([30, 0, 15, 15, 2])
        torch.testing.assert_close(model(ids), model(torch.arange(31))[ids], rtol=0, atol=0)
        restored = BSplineMotion9(31, 8)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(ids), model(ids), rtol=0, atol=0)

    def test_all_coefficients_have_correct_gradient_at_zero(self):
        model = BSplineMotion9(61, 20, dtype=torch.float64)
        weights = torch.linspace(.1, 1., 61*9, dtype=torch.float64).reshape(61, 9)
        (model(torch.arange(61))*weights).sum().backward()
        expected = model.basis.T @ weights * model.scales
        torch.testing.assert_close(model.raw_coefficients.grad, expected, rtol=1e-13, atol=1e-13)
        self.assertTrue((model.raw_coefficients.grad.abs() > 0).all())

    def test_camera_path_gradient_against_independent_double_precision(self):
        views, controls = 31, 8
        model = BSplineMotion9(views, controls)
        rng = np.random.default_rng(44)
        values = rng.normal(0, .03, (controls, 9)).astype(np.float32)
        with torch.no_grad(): model.raw_coefficients.copy_(torch.from_numpy(values))
        nominal = geometry_to_pmat(build_icono_orbit('circular',n_views=views,scan_angle_deg=220,detector_bin=2), dtype=np.float64)
        xyz = np.c_[rng.uniform(-80, 80, (13, 3)), np.ones(13)]
        p, motion = apply_motion(torch.tensor(nominal), model(torch.arange(views)), {}, physical=True)
        q = p @ torch.tensor(xyz, dtype=torch.float32).T
        (q[:, :2]/q[:, 2:]).square().mean().backward()
        b = model.basis.double().numpy()
        def objective(raw):
            m = b @ (np.tanh(raw)*np.array([10.]*6+[15.]*3))
            q = compose_effective_parameters(nominal,m) @ xyz.T
            return np.mean((q[:, :2]/q[:, 2:])**2)
        for k in (0, 3, 7):
            for j in range(9):
                plus, minus = values.astype(float), values.astype(float)
                plus[k,j] += 1e-5; minus[k,j] -= 1e-5
                finite = (objective(plus)-objective(minus))/2e-5
                np.testing.assert_allclose(float(model.raw_coefficients.grad[k,j]),finite,rtol=1e-3,atol=.02)
        # Raw-MLP and direct-physical APIs produce the same camera for same motion.
        raw = torch.atanh(motion / model.scales)
        legacy, _ = apply_motion(torch.tensor(nominal),raw,dict(ts_max_mm=10.,tp_max_mm=10.,rot_max_deg=15.))
        torch.testing.assert_close(p,legacy,rtol=1e-6,atol=1e-3)

    def test_invalid_dimensions_and_bounds(self):
        for views, controls in ((10,3),(10,11),(1,4)):
            with self.assertRaises(ValueError): cubic_bspline_basis(views,controls)
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): BSplineMotion9(31,8,ts_max_mm=value)


if __name__ == '__main__': unittest.main()
