"""Independent CPU checks of Grangeat units, signs, quadrature, and rebinning."""
from dataclasses import replace
import unittest

import numpy as np

trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

from grangeat_phantom import gaussian_cone_projection, gaussian_density, gaussian_radon
from grangeat_recon import (detector_derivatives, hemisphere_quadrature,
                           grangeat_view_numpy, rebin_radon_derivative)
from sinespin_geometry import build_icono_orbit


class GrangeatReconstructionTests(unittest.TestCase):
    def test_hemisphere_area_and_moments(self):
        n, w = hemisphere_quadrature(16, 64)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1., atol=2e-15)
        self.assertTrue(np.all(n[:, 2] > 0))
        self.assertAlmostEqual(w.sum(), 2*np.pi, places=13)
        self.assertAlmostEqual(w @ n[:, 2], np.pi, places=13)
        np.testing.assert_allclose(n.T @ (w[:, None]*n), np.eye(3)*(2*np.pi/3), atol=2e-14)

    def test_radon_inverse_quadrature_has_physical_amplitude(self):
        n, w = hemisphere_quadrature(32, 128)
        center = np.array([3., -5., 7.])
        covariance = np.array([[144., 30., 0.], [30., 225., 20.], [0., 20., 100.]])
        for point in (center, center+[5., -4., 8.], center+[-9., 7., 2.]):
            rpp = gaussian_radon(n, n @ point, center, covariance, .021, derivative=2)
            reconstructed = -(w @ rpp)/(4*np.pi**2)
            expected = gaussian_density(point, center, covariance, .021)
            self.assertAlmostEqual(reconstructed, float(expected), places=12)

    def test_flat_panel_gaussian_identity_with_anisotropic_binning(self):
        base = build_icono_orbit('sinespin', n_views=5)
        view, center, amplitude = 1, np.array([9., -7., 11.]), .023
        covariance = np.array([[400., 60., 0.], [60., 529., 25.], [0., 25., 324.]])
        alpha = np.array([.1, 1.2, 2.4, -.7, -2.8])
        c = np.array([-.02, .01, .025, -.015, .035])
        n = (np.sqrt(1-c*c)[:, None]
             * (np.cos(alpha)[:, None]*base.col_vectors[view]
                + np.sin(alpha)[:, None]*base.row_vectors[view])
             + c[:, None]*base.normals[view])
        # Opposite oriented planes must reverse the first Radon derivative.
        n = np.concatenate((n, -n))
        expected = gaussian_radon(n, n @ base.source_positions[view], center,
                                 covariance, amplitude, derivative=1)
        self.assertTrue(np.any(expected > 0) and np.any(expected < 0))
        errors = []
        for bin_factor in (2, 1):
            cols, rows = 720//bin_factor, 440//bin_factor
            g = replace(base, detector_cols=cols, detector_rows=rows,
                        pixel_width=base.detector_width_mm/cols,
                        pixel_height=base.detector_height_mm/rows)
            u = (np.arange(cols)-(cols-1)/2)*g.pixel_width
            v = (np.arange(rows)-(rows-1)/2)*g.pixel_height
            rays = (g.module_centers[view]-g.source_positions[view]
                    + u[None, :, None]*g.col_vectors[view]
                    + v[:, None, None]*g.row_vectors[view])
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            projection = gaussian_cone_projection(g.source_positions[view], rays,
                                                  center, covariance, amplitude)
            actual = grangeat_view_numpy(projection, g, view, n,
                                         line_step_mm=min(g.pixel_width, g.pixel_height)/2)
            errors.append(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            np.testing.assert_allclose(actual[:5], -actual[5:], atol=1e-12)
        self.assertLess(errors[0], .002)
        self.assertLess(errors[1], .0006)
        self.assertLess(errors[1], errors[0]*.4)

    def test_zero_continuation_preserves_detector_line_boundary_jump(self):
        # A uniform slab has g=A/cos(gamma), hence a constant cosine-weighted
        # image. Zero continuation makes Q(s) a rectangle of height A*panel_H.
        # Its positive and negative boundary derivatives must each integrate
        # to that height, independently of derivative padding or pixel pitch.
        g = replace(build_icono_orbit('circular', n_views=2),
                    detector_cols=64, detector_rows=40, pixel_width=1.1, pixel_height=1.9)
        u = (np.arange(g.detector_cols)-(g.detector_cols-1)/2)*g.pixel_width
        v = (np.arange(g.detector_rows)-(g.detector_rows-1)/2)*g.pixel_height
        amplitude = 2.5
        slab = amplitude*np.sqrt(g.sdd_mm**2+v[:, None]**2+u[None, :]**2)/g.sdd_mm
        projections = np.broadcast_to(slab, (g.n_views, *slab.shape))
        _, du, _ = detector_derivatives(projections, g)
        q_derivative = trapezoid(du[0], dx=g.pixel_height, axis=0)
        positive_jump = trapezoid(np.maximum(q_derivative, 0), dx=g.pixel_width)
        negative_jump = -trapezoid(np.minimum(q_derivative, 0), dx=g.pixel_width)
        expected = amplitude*g.detector_height_mm
        np.testing.assert_allclose([positive_jump, negative_jump], expected, rtol=2e-5)

    @staticmethod
    def rebin(p, values, good, offsets):
        sources = np.column_stack((p, np.zeros((len(p), 2))))
        return rebin_radon_derivative(sources, np.array([[1., 0., 0.]]),
                                     np.asarray([values]), np.asarray([good]), offsets)[0]

    def test_monotone_rebin_does_not_extrapolate(self):
        offsets = np.array([-3., -2., -.5, .5, 2., 3.])
        actual = self.rebin([-2., -1., 0., 1., 2.], [-5., -2., 1., 4., 7.],
                            [True]*5, offsets)
        np.testing.assert_allclose(actual[1:-1], 3*offsets[1:-1]+1)
        self.assertTrue(np.isnan(actual[[0, -1]]).all())

    def test_multiple_source_branches_are_averaged_not_summed(self):
        # Different branch perturbations cancel; the common turning sample is 4.
        actual = self.rebin([-2., 0., 2., 0., -2.], [-5., -1., 4., 1., -3.],
                            [True]*5, [-1., 0., 1.])
        np.testing.assert_allclose(actual, [-2., 0., 2.])

    def test_only_available_branches_receive_weight(self):
        actual = self.rebin([-2., 0., 2., 0., -2.], [-5., -1., 4., 1., -3.],
                            [False, False, True, True, True], [-1., 0., 1.])
        np.testing.assert_allclose(actual, [-1., 1., 2.5])

    def test_open_trajectory_cannot_bridge_its_endpoints(self):
        actual = self.rebin([-2., 0., 2.], [-2., 0., 2.],
                            [True, False, True], [-1., 0., 1.])
        self.assertTrue(np.isnan(actual).all())

    def test_missing_internal_sample_is_not_interpolated_across(self):
        actual = self.rebin([-2., -1., 0., 1., 2.], [-2., -1., 0., 1., 2.],
                            [True, True, False, True, True], [-1.5, 0., 1.5])
        np.testing.assert_allclose(actual[[0, 2]], [-1.5, 1.5])
        self.assertTrue(np.isnan(actual[1]))


if __name__ == '__main__':
    unittest.main()
