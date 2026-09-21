"""Independent CPU checks for analytic Grangeat validation phantoms."""
import unittest

import numpy as np

from grangeat_phantom import (
    ellipsoid_cone_projection, ellipsoid_density, ellipsoid_radon,
    gaussian_cone_projection, gaussian_density, gaussian_radon,
)


def rotation():
    a, b = .47, -.31
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rz @ ry


class GrangeatPhantomTests(unittest.TestCase):
    def setUp(self):
        self.center = np.array([1.7, -2.3, .9])
        self.rotation = rotation()
        self.axes = self.rotation @ np.diag([3., 2., 1.])
        self.covariance = self.rotation @ np.diag([1.5**2, 2.5**2, 4.**2]) @ self.rotation.T

    def test_rotated_displaced_ellipsoid_chords_and_half_rays(self):
        u, v = self.rotation[:, 0], self.rotation[:, 1]
        source = self.center - 10*u + v  # offset half the second semiaxis
        value = ellipsoid_cone_projection(source, u, self.center, self.axes, .02)
        self.assertAlmostEqual(float(value), .02 * 6 * np.sqrt(.75), places=13)
        self.assertEqual(float(ellipsoid_cone_projection(source, -u, self.center, self.axes)), 0.)
        self.assertAlmostEqual(float(ellipsoid_cone_projection(self.center, u, self.center, self.axes)), 3.)
        self.assertEqual(float(ellipsoid_cone_projection(source + 4*v, u, self.center, self.axes)), 0.)
        self.assertEqual(float(ellipsoid_density(self.center, self.center, self.axes, .02)), .02)
        self.assertEqual(float(ellipsoid_density(self.center + 4*u, self.center, self.axes)), 0.)

    def test_ellipsoid_radon_cross_section_mass_and_surface_derivatives(self):
        normal = self.rotation[:, 0]
        center_offset = normal @ self.center
        offsets = center_offset + np.array([0., 1.5, 3., 4.])
        actual = ellipsoid_radon(normal, offsets, self.center, self.axes)
        np.testing.assert_allclose(actual, [2*np.pi, 1.5*np.pi, 0., 0.], atol=1e-13)
        self.assertTrue(np.isnan(ellipsoid_radon(normal, center_offset + 3., self.center, self.axes, derivative=1)))
        self.assertTrue(np.isnan(ellipsoid_radon(normal, center_offset + 3., self.center, self.axes, derivative=2)))
        self.assertEqual(float(ellipsoid_radon(normal, center_offset + 4., self.center, self.axes, derivative=2)), 0.)
        nodes, weights = np.polynomial.legendre.leggauss(12)
        integrated = 3 * np.dot(weights, ellipsoid_radon(normal, center_offset + 3*nodes, self.center, self.axes))
        self.assertAlmostEqual(float(integrated), 4*np.pi*3*2*1/3, places=12)

    def test_signed_offset_derivatives_and_plane_orientation(self):
        normal = np.array([2., -1., 3.]) / np.sqrt(14.)
        offset = normal @ self.center + .4
        for function, matrix in ((ellipsoid_radon, self.axes), (gaussian_radon, self.covariance)):
            f = lambda p: function(normal, p, self.center, matrix)
            h = .0003
            first_fd = (f(offset+h) - f(offset-h)) / (2*h)
            second_fd = (f(offset+h) - 2*f(offset) + f(offset-h)) / h**2
            np.testing.assert_allclose(first_fd, function(normal, offset, self.center, matrix, derivative=1), rtol=2e-7)
            np.testing.assert_allclose(second_fd, function(normal, offset, self.center, matrix, derivative=2), rtol=3e-6)
            for derivative in (0, 1, 2):
                actual = function(-normal, -offset, self.center, matrix, derivative=derivative)
                expected = (-1)**derivative * function(normal, offset, self.center, matrix, derivative=derivative)
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_gaussian_ray_and_plane_integrals_against_density_quadrature(self):
        normal = np.array([.2, -.7, .4]); normal /= np.linalg.norm(normal)
        source = self.center + [7., -4., 2.]
        ray = self.center - source; ray /= np.linalg.norm(ray)
        nodes, weights = np.polynomial.legendre.leggauss(160)
        t = 40 * (nodes + 1)
        direct_ray = 40 * np.dot(weights, gaussian_density(source + t[:, None]*ray,
                                                         self.center, self.covariance, .02))
        np.testing.assert_allclose(gaussian_cone_projection(source, ray, self.center, self.covariance, .02),
                                   direct_ray, rtol=2e-13)
        u = np.cross(normal, [1., 0., 0.]); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        offset = normal @ self.center + .7
        plane_center = self.center + .7*normal
        grid = (plane_center + 36*nodes[:, None, None]*u + 36*nodes[None, :, None]*v)
        direct_plane = 36**2 * np.sum(weights[:, None]*weights[None, :]
                                     * gaussian_density(grid, self.center, self.covariance, .02))
        np.testing.assert_allclose(gaussian_radon(normal, offset, self.center, self.covariance, .02),
                                   direct_plane, rtol=2e-12)
        # A Gaussian source at its centre sees exactly half the infinite line.
        sigma = 2.
        actual = gaussian_cone_projection([0, 0, 0], [1, 0, 0], [0, 0, 0], sigma**2*np.eye(3))
        self.assertAlmostEqual(float(actual), sigma*np.sqrt(np.pi/2), places=13)

    def test_grangeat_first_derivative_identity_on_displaced_anisotropic_gaussian(self):
        # dOmega = dphi*dt for w(t,phi)=sqrt(1-t^2)*u(phi)+t*n.
        # Grangeat gives R'(n,n.S)=integral_0^2pi d/dt g(S,w)|t=0 dphi.
        normal = np.array([.2, .7, -.4]); normal /= np.linalg.norm(normal)
        source = np.array([5., -4., 2.])
        u = np.cross(normal, [1., 0., 0.]); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        phi = np.arange(2048) * (2*np.pi/2048)
        equator = np.cos(phi)[:, None]*u + np.sin(phi)[:, None]*v
        epsilon = 1e-5
        plus = np.sqrt(1-epsilon**2)*equator + epsilon*normal
        minus = np.sqrt(1-epsilon**2)*equator - epsilon*normal
        angular_derivative = (gaussian_cone_projection(source, plus, self.center, self.covariance)
                              - gaussian_cone_projection(source, minus, self.center, self.covariance)) / (2*epsilon)
        actual = 2*np.pi*np.mean(angular_derivative)
        expected = gaussian_radon(normal, normal @ source, self.center, self.covariance, derivative=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)

    def test_rejects_nonunit_directions_and_invalid_covariance(self):
        with self.assertRaises(ValueError):
            ellipsoid_cone_projection([0, 0, 0], [2, 0, 0], self.center, self.axes)
        with self.assertRaises(ValueError):
            gaussian_radon([1, 0, 0], 0, self.center, np.diag([1., -1., 2.]))
        with self.assertRaises(ValueError):
            ellipsoid_radon([1, 0, 0], 0, self.center, np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
