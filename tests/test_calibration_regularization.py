"""Independent value, derivative, and inverse-problem checks for geometry priors."""
import json
import unittest

import torch

from calibration_regularization import AppliedMotionRegularizer, RegularizationConfig


class AppliedMotionRegularizerTests(unittest.TestCase):
    def test_group_values_and_unequal_batch_aggregation(self):
        motion = torch.arange(-22, 23, dtype=torch.float64).reshape(5, 9)
        reg = AppliedMotionRegularizer(RegularizationConfig(
            intrinsic_weight=.3, translation_weight=.7, rotation_weight=1.1,
            intrinsic_scale_mm=2., translation_scale_mm=4., rotation_scale_deg=8.))
        parts = reg.components(motion)
        for group, columns, scale in (("intrinsic", range(3), 2.),
                                      ("translation", range(3, 6), 4.),
                                      ("rotation", range(6, 9), 8.)):
            expected = sum(float(motion[i, j]) ** 2 / scale ** 2
                           for i in range(5) for j in columns) / 15
            self.assertAlmostEqual(float(parts[group]), expected, places=12)
        self.assertAlmostEqual(float(parts["total"]),
                               .3 * parts["intrinsic"] + .7 * parts["translation"] + 1.1 * parts["rotation"],
                               places=12)
        torch.testing.assert_close(reg(motion), (2 * reg(motion[:2]) + 3 * reg(motion[2:])) / 5,
                                   rtol=1e-14, atol=1e-14)

    def test_analytic_gradient_and_disabled_group_independence(self):
        motion = torch.linspace(-6., 9., 36, dtype=torch.float64).reshape(4, 9).requires_grad_()
        reg = AppliedMotionRegularizer(RegularizationConfig(
            intrinsic_weight=.2, translation_weight=.5, rotation_weight=0.,
            intrinsic_scale_mm=10., translation_scale_mm=7., rotation_scale_deg=15.))
        gradient = torch.autograd.grad(reg(motion), motion)[0]
        expected = torch.zeros_like(motion)
        expected[:, :3] = 2 * .2 * motion[:, :3] / (12 * 10 ** 2)
        expected[:, 3:6] = 2 * .5 * motion[:, 3:6] / (12 * 7 ** 2)
        torch.testing.assert_close(gradient, expected, rtol=1e-14, atol=1e-14)
        self.assertTrue(torch.equal(gradient[:, 6:], torch.zeros_like(gradient[:, 6:])))
        changed = motion.detach().clone()
        changed[:, 6:] += 100.
        self.assertTrue(torch.equal(reg(changed), reg(motion)))

    def test_applied_bounds_tanh_chain_rule(self):
        raw = torch.linspace(-1.3, 1.1, 18, dtype=torch.float64).reshape(2, 9).requires_grad_()
        bounds = torch.tensor([10.] * 6 + [15.] * 3, dtype=torch.float64)
        reg = AppliedMotionRegularizer(RegularizationConfig(
            intrinsic_weight=.01, translation_weight=.03, rotation_weight=.02))
        gradient = torch.autograd.grad(reg(bounds * raw.tanh()), raw)[0]
        weights = torch.tensor([.01] * 3 + [.03] * 3 + [.02] * 3, dtype=torch.float64)
        # The group scales equal bounds, so the analytic prior is lambda*tanh(raw)^2 / 6.
        expected = 2 * weights * raw.tanh() * (1 - raw.tanh().square()) / 6
        torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-14)
        self.assertGreater(float(gradient.abs().min()), 0.)

    def test_zero_weights_preserve_image_object_and_gradients_exactly(self):
        reg = AppliedMotionRegularizer(RegularizationConfig())
        self.assertFalse(reg.active)
        for dtype in (torch.float32, torch.float64):
            image = torch.tensor([-.3, .6, 1.2], dtype=dtype, requires_grad=True)
            motion = torch.arange(18, dtype=dtype).reshape(2, 9).requires_grad_()
            original = image.square().mean()
            combined = reg.combine(original, motion)
            self.assertIs(combined, original)
            baseline_gradient = torch.autograd.grad(original, image, retain_graph=True)[0]
            image_gradient, motion_gradient = torch.autograd.grad(combined, (image, motion), allow_unused=True)
            self.assertTrue(torch.equal(image_gradient, baseline_gradient))
            self.assertIsNone(motion_gradient)
            self.assertEqual(float(reg(motion)), 0.)

    def test_active_combine_preserves_data_gradient_and_adds_only_selected_prior(self):
        image = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
        motion = torch.ones((1, 9), dtype=torch.float64, requires_grad=True)
        reg = AppliedMotionRegularizer(RegularizationConfig(intrinsic_weight=.01))
        image_gradient, motion_gradient = torch.autograd.grad(
            reg.combine(image.square().sum(), motion), (image, motion))
        torch.testing.assert_close(image_gradient, 2 * image, rtol=0, atol=0)
        self.assertTrue(torch.equal(motion_gradient[:, 3:], torch.zeros_like(motion_gradient[:, 3:])))
        torch.testing.assert_close(motion_gradient[:, :3], torch.full((1, 3), 2 * .01 / 300., dtype=torch.float64))

    def test_correlated_inverse_problem_selects_prior_without_changing_observable(self):
        # Only x+y is measured: intrinsic x and unpenalized translation y trade off.
        # A prior selects x=0, y=2; that is not evidence that the hidden GT has x=0.
        reg = AppliedMotionRegularizer(RegularizationConfig(intrinsic_weight=.2, intrinsic_scale_mm=1.))

        def objective(pair):
            zero = pair.new_zeros(())
            motion = torch.stack((pair[0], zero, zero, pair[1], zero, zero, zero, zero, zero))[None]
            return reg.combine((pair.sum() - 2).square(), motion)

        start = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        hessian = torch.autograd.functional.hessian(objective, start)
        gradient = torch.autograd.grad(objective(start), start)[0]
        optimum = torch.linalg.solve(hessian, -gradient)
        torch.testing.assert_close(optimum, torch.tensor([0., 2.], dtype=torch.float64), rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(optimum.sum()), 2., places=12)
        self.assertLess(float(objective(optimum)), float(objective(torch.ones(2, dtype=torch.float64))))

    def test_config_serialization_and_invalid_values(self):
        reg = AppliedMotionRegularizer(RegularizationConfig(intrinsic_weight=.01))
        config = json.loads(json.dumps(reg.get_config(), allow_nan=False))
        self.assertTrue(config["active"])
        self.assertEqual(config["intrinsic_scale_mm"], 10.)
        self.assertIn("not GT", config["target"])
        for name in ("intrinsic_weight", "translation_weight", "rotation_weight"):
            for value in (-.1, float("nan"), float("inf"), True, "0.1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RegularizationConfig(**{name: value})
        for name in ("intrinsic_scale_mm", "translation_scale_mm", "rotation_scale_deg"):
            for value in (0., -1., float("nan"), float("inf"), False, "10"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RegularizationConfig(**{name: value})

    def test_invalid_parameter_inputs_and_active_image_shape(self):
        reg = AppliedMotionRegularizer(RegularizationConfig(intrinsic_weight=.01))
        for value in (None, [[0.] * 9], torch.zeros(9), torch.zeros(0, 9),
                      torch.zeros(2, 8), torch.zeros(1, 2, 9), torch.zeros(2, 9, dtype=torch.int64),
                      torch.full((2, 9), float("nan")), torch.full((2, 9), float("inf"))):
            with self.subTest(shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                reg(value)
        with self.assertRaises(ValueError):
            reg.combine(torch.zeros(2), torch.zeros(2, 9))


if __name__ == "__main__":
    unittest.main()
