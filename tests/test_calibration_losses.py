"""CPU regression and independent mathematical checks for calibration losses."""
import unittest

import torch
from monai.losses import LocalNormalizedCrossCorrelationLoss

from calibration_losses import build_loss


class CalibrationLossTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(812)

    def samples(self, dtype=torch.float64):
        p = torch.rand((2, 37, 43), dtype=dtype, generator=self.rng)
        t = .8 * p + .1 * torch.rand(p.shape, dtype=dtype, generator=self.rng)
        return p, t

    def test_default_value_and_image_gradient_are_exactly_monai(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                pred, target = self.samples(dtype)
                left = pred.clone().requires_grad_(True)
                right = pred.clone().requires_grad_(True)
                actual = build_loss()(left, target)
                expected = 1.0 + LocalNormalizedCrossCorrelationLoss(
                    spatial_dims=2, kernel_size=31, kernel_type="rectangular", reduction="mean"
                )(right[:, None], target[:, None])
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.autograd.grad(actual, left)[0],
                                            torch.autograd.grad(expected, right)[0]))

    def test_per_view_and_batch_reduction_for_each_loss(self):
        pred, target = self.samples()
        counts = torch.round(44000 * torch.exp(-target)).long()
        for name in ("lncc", "signed_lncc", "global_ncc", "mse", "huber", "poisson"):
            with self.subTest(name=name):
                loss = build_loss(name)
                by_view = loss.per_view(pred, target, counts)
                individual = torch.stack([loss(pred[i:i+1], target[i:i+1], counts[i:i+1]) for i in range(2)])
                torch.testing.assert_close(by_view, individual, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(loss(pred, target, counts), by_view.mean(), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(build_loss(name, reduction="none")(pred, target, counts), by_view)

    def test_signed_correlation_distinguishes_contrast_inversion(self):
        # Zero-centered values and exact negation also invert padded local covariances.
        values = torch.randn((2, 37, 43), dtype=torch.float64, generator=self.rng)
        for name in ("signed_lncc", "global_ncc"):
            with self.subTest(name=name):
                loss = build_loss(name)
                self.assertAlmostEqual(float(loss(values, values)), 0., places=12)
                self.assertAlmostEqual(float(loss(-values, values)), 2., places=12)
        squared = build_loss("lncc")
        torch.testing.assert_close(squared(values, values), squared(-values, values), rtol=0, atol=1e-12)

    def test_signed_local_square_matches_monai_on_same_support(self):
        # Reconstructing squared correlation from signed local maps independently
        # checks that the local kernels, padding, sums and variance floor agree.
        pred, target = self.samples()
        for kernel_type in ("rectangular", "triangular"):
            with self.subTest(kernel_type=kernel_type):
                signed = build_loss("signed_lncc", kernel_size=7, kernel_type=kernel_type, smooth_dr=.002)
                correlation = 1.0 - signed._signed_local_map(pred, target)
                reference = LocalNormalizedCrossCorrelationLoss(
                    spatial_dims=2, kernel_size=7, kernel_type=kernel_type,
                    smooth_dr=.002, reduction="none")(pred[:, None], target[:, None])
                torch.testing.assert_close(correlation.square(), -reference, rtol=1e-12, atol=1e-12)

    def test_flat_images_have_finite_zero_gradient(self):
        for name in ("lncc", "signed_lncc", "global_ncc"):
            with self.subTest(name=name):
                pred = torch.zeros((2, 9, 11), dtype=torch.float64, requires_grad=True)
                value = build_loss(name)(pred, torch.zeros_like(pred))
                grad = torch.autograd.grad(value, pred)[0]
                self.assertEqual(float(value.detach()), 1.)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertEqual(float(grad.abs().max()), 0.)

    def test_signed_local_gradient_matches_finite_differences(self):
        target = torch.randn((1, 4, 5), dtype=torch.float64, generator=self.rng)
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                pred = (sign * target + .2 * torch.randn(
                    target.shape, dtype=target.dtype, generator=self.rng)).requires_grad_(True)
                loss = build_loss("signed_lncc", kernel_size=3, smooth_dr=.002)
                self.assertTrue(torch.autograd.gradcheck(
                    lambda x: loss(x, target), (pred,), eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_single_weight_local_kernels_are_rejected(self):
        for name in ("lncc", "signed_lncc"):
            for kind, size in (("rectangular", 1), ("triangular", 1), ("triangular", 3)):
                with self.subTest(name=name, kind=kind, size=size), self.assertRaises(ValueError):
                    build_loss(name, kernel_type=kind, kernel_size=size)
        pred, target = self.samples()
        for kind, size in (("rectangular", 3), ("triangular", 5)):
            x = pred.clone().requires_grad_(True)
            grad = torch.autograd.grad(build_loss("signed_lncc", kernel_type=kind, kernel_size=size)(x, target), x)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().max()), 0.)

    def test_poisson_zero_counts_finite_and_gradient_matches_likelihood(self):
        counts = torch.tensor([[[0, 1, 44000], [5, 60000, 200]]], dtype=torch.int64)
        pred = torch.tensor([[[2., 5., .1], [1., -.2, 4.]]], dtype=torch.float64, requires_grad=True)
        loss = build_loss("poisson", i0=44000.)
        value = loss(pred, counts=counts)
        actual_grad = torch.autograd.grad(value, pred)[0]
        expected_grad = 2 * (counts.double()/44000. - torch.exp(-pred.detach())) / pred.numel()
        self.assertTrue(torch.isfinite(value))
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
        zero_pred = torch.tensor([[[1000., 0., 3.]]], dtype=torch.float64, requires_grad=True)
        zero_loss = loss(zero_pred, counts=torch.zeros_like(zero_pred, dtype=torch.int64))
        self.assertTrue(torch.isfinite(zero_loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(zero_loss, zero_pred)[0]).all())
        torch.testing.assert_close(zero_loss, (2*torch.exp(-zero_pred)).mean())

    def test_poisson_deviance_matches_independent_float64_definition(self):
        counts = torch.tensor([[[0, 2, 70000, 300]]], dtype=torch.int64)
        pred = torch.tensor([[[.5, 4., -.1, 2.]]], dtype=torch.float64)
        rate = 44000. * torch.exp(-pred)
        n = counts.double()
        # xlogy(0,0)=0 gives the independent zero-count convention.
        expected = (2/44000. * (rate-n+torch.xlogy(n, n/rate))).mean()
        torch.testing.assert_close(build_loss("poisson")(pred, counts=counts), expected, rtol=1e-12, atol=1e-12)

    def test_poisson_minimum_at_matching_observed_count_rate(self):
        counts = torch.tensor([[[1, 100, 22000, 44000, 88000]]], dtype=torch.int64)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                pred = (-torch.log(counts.to(dtype)/44000.)).requires_grad_(True)
                loss = build_loss("poisson")(pred, counts=counts)
                grad = torch.autograd.grad(loss, pred)[0]
                self.assertEqual(float(loss.detach()), 0.)
                self.assertEqual(float(grad.abs().max()), 0.)

    def test_mse_and_huber_have_their_physical_residual_scale(self):
        pred = torch.tensor([[[0., .5, 2.]]], dtype=torch.float64)
        target = torch.zeros_like(pred)
        self.assertAlmostEqual(float(build_loss("mse")(pred, target)), (0+.25+4)/3)
        self.assertAlmostEqual(float(build_loss("huber", huber_delta=1.)(pred, target)), (0+.125+1.5)/3)

    def test_invalid_configuration_and_counts(self):
        invalid = [dict(name="unknown"), dict(kernel_size=0), dict(kernel_size=2),
                   dict(kernel_size=3.5), dict(kernel_size=True), dict(kernel_type="gaussian"),
                   dict(kernel_type="other"), dict(smooth_nr=-1), dict(smooth_dr=0),
                   dict(smooth_dr=float("nan")), dict(smooth_nr=float("inf")),
                   dict(i0=0), dict(i0=float("inf")), dict(huber_delta=-1),
                   dict(reduction="sum"), dict(name="signed_lncc", smooth_nr=.01)]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_loss(**config)
        loss = build_loss("poisson")
        pred = torch.zeros((1, 2, 3))
        for counts in (None, torch.zeros((2, 3), dtype=torch.int64),
                       -torch.ones_like(pred, dtype=torch.int64), torch.ones_like(pred),
                       torch.ones_like(pred, dtype=torch.bool)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                loss(pred, counts=counts)
        with self.assertRaises(ValueError):
            build_loss()(pred, torch.zeros((1, 2, 2)))

    def test_unsigned_prepared_counts_are_accepted(self):
        counts = torch.tensor([[[0, 44000]]], dtype=torch.uint32)
        pred = torch.zeros((1, 1, 2), dtype=torch.float64)
        self.assertEqual(float(build_loss("poisson")(pred, counts=counts)), 1.)


if __name__ == "__main__":
    unittest.main()
