"""Analytic CPU oracles for testing Grangeat's cone-beam/Radon identity.

These functions verify geometry, signs and normalization independently of a
voxel projector. They are not a replacement for Joseph simulation in the head
experiments. All calculations use NumPy float64, xyz coordinates and physical
length units (normally mm); density amplitudes can be in 1/mm.

``cone_projection(S, w) = integral_0^infinity f(S + t*w) dt`` for unit ray w.
``radon(n, p) = integral f(x) delta(p - n.x) dx`` for unit plane normal n.
The derivative argument means differentiation with respect to signed p while
holding n fixed, not differentiation along the source trajectory.

For an ellipsoid x=c+A*u, |u|<=1, let h=|A.T*n| and q=p-n.c.
The Jacobian |det A| and transformed delta give
R=amplitude*pi*|det A|/h * (1-q^2/h^2), |q|<h.
R'=-2*amplitude*pi*|det A|*q/h^3 and
R''=-2*amplitude*pi*|det A|/h^3 in the interior. R' jumps at
|q|=h, so R'' additionally has surface delta terms as a distribution. These
are deliberately not represented by ordinary samples: derivative calls return
NaN on the support boundary. Use the smooth Gaussian oracle for quadrature or
full Radon inversion tests that must not omit those distributional terms.
"""
from __future__ import annotations

from math import erfc

import numpy as np


def _xyz(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and have final axis xyz (length 3)")
    return value


def _unit(value, name):
    value = _xyz(value, name)
    if not np.allclose(np.linalg.norm(value, axis=-1), 1., rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name} must contain unit vectors")
    return value


def _parameters(center_xyz, matrix_xyz, amplitude, *, covariance=False):
    center = _xyz(center_xyz, "center_xyz")
    matrix = np.asarray(matrix_xyz, dtype=np.float64)
    if center.shape != (3,) or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("One centre (3,) and one finite matrix (3,3) are required")
    if not np.isscalar(amplitude) or not np.isfinite(amplitude):
        raise ValueError("amplitude must be a finite scalar")
    if covariance:
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
            raise ValueError("covariance_xyz must be symmetric positive definite")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance_xyz must be symmetric positive definite") from error
    determinant = abs(float(np.linalg.det(matrix)))
    if not np.isfinite(determinant) or determinant == 0:
        raise ValueError("matrix_xyz must be nonsingular with finite determinant")
    return center, matrix, float(amplitude), determinant


def ellipsoid_density(points_xyz, center_xyz, axes_xyz, amplitude=1.):
    """Density of ``center + axes @ unit_ball``; axes may encode 3-D rotation."""
    center, axes, amplitude, _ = _parameters(center_xyz, axes_xyz, amplitude)
    local = (_xyz(points_xyz, "points_xyz") - center) @ np.linalg.inv(axes).T
    return amplitude * (np.sum(local * local, axis=-1) <= 1.)


def ellipsoid_cone_projection(source_xyz, unit_ray_xyz, center_xyz, axes_xyz, amplitude=1.):
    """Exact half-ray integral through an arbitrarily positioned ellipsoid.

    Source and ray arrays broadcast to (...,3); return shape is (...,).
    Sources inside the object integrate only the outgoing segment. Rays that
    point away from an object behind the source correctly return zero.
    """
    center, axes, amplitude, _ = _parameters(center_xyz, axes_xyz, amplitude)
    source, ray = np.broadcast_arrays(_xyz(source_xyz, "source_xyz"), _unit(unit_ray_xyz, "unit_ray_xyz"))
    inverse = np.linalg.inv(axes)
    origin = (source - center) @ inverse.T
    direction = ray @ inverse.T
    a = np.sum(direction * direction, axis=-1)
    b = np.sum(origin * direction, axis=-1)
    c = np.sum(origin * origin, axis=-1) - 1.
    discriminant = b * b - a * c
    root = np.sqrt(np.maximum(discriminant, 0.))
    enter = np.maximum((-b - root) / a, 0.)
    leave = (-b + root) / a
    return amplitude * np.where(discriminant > 0., np.maximum(leave - enter, 0.), 0.)


def ellipsoid_radon(unit_normal_xyz, offset, center_xyz, axes_xyz, amplitude=1., *, derivative=0):
    """Exact plane integral or first/second signed-offset derivative.

    Normals (...,3) and offsets (...) broadcast. Derivatives at the support
    surface return NaN; derivative=2 omits the distributional surface terms.
    Exterior ordinary derivatives are zero. No density grid is involved.
    """
    center, axes, amplitude, determinant = _parameters(center_xyz, axes_xyz, amplitude)
    normal = _unit(unit_normal_xyz, "unit_normal_xyz")
    offset = np.asarray(offset, dtype=np.float64)
    if not np.isfinite(offset).all() or derivative not in (0, 1, 2):
        raise ValueError("offset must be finite and derivative must be 0, 1, or 2")
    q = offset - normal @ center
    h = np.linalg.norm(normal @ axes, axis=-1)
    factor = np.pi * amplitude * determinant / h
    if derivative == 0:
        return factor * np.maximum(1. - (q / h) ** 2, 0.)
    interior = np.abs(q) < h
    ordinary = (-2. * factor / h**2) * (q if derivative == 1 else np.ones_like(q))
    result = np.where(interior, ordinary, 0.)
    surface = np.isclose(np.abs(q), h, rtol=2e-13, atol=1e-14)
    return np.where(surface, np.nan, result)


def gaussian_density(points_xyz, center_xyz, covariance_xyz, amplitude=1.):
    """Unnormalized anisotropic Gaussian ``amplitude*exp(-.5*q.T*invSigma*q)``."""
    center, covariance, amplitude, _ = _parameters(center_xyz, covariance_xyz, amplitude, covariance=True)
    q = _xyz(points_xyz, "points_xyz") - center
    exponent = np.sum((q @ np.linalg.inv(covariance)) * q, axis=-1)
    return amplitude * np.exp(-.5 * exponent)


def gaussian_cone_projection(source_xyz, unit_ray_xyz, center_xyz, covariance_xyz, amplitude=1.):
    """Exact anisotropic-Gaussian half-ray integral, including finite source distance.

    This keeps the erfc endpoint term: treating the ray as an infinite line
    would give the wrong Grangeat oracle when a source lies within the density.
    Only NumPy and Python's standard-library erfc are required.
    """
    center, covariance, amplitude, _ = _parameters(center_xyz, covariance_xyz, amplitude, covariance=True)
    source, ray = np.broadcast_arrays(_xyz(source_xyz, "source_xyz"), _unit(unit_ray_xyz, "unit_ray_xyz"))
    inverse = np.linalg.inv(covariance)
    q = source - center
    inverse_ray = ray @ inverse
    a = np.sum(inverse_ray * ray, axis=-1)
    b = np.sum(inverse_ray * q, axis=-1)
    c = np.sum((q @ inverse) * q, axis=-1)
    perpendicular = np.maximum(c - b * b / a, 0.)
    argument = np.asarray(b / np.sqrt(2. * a))
    endpoint = np.fromiter((erfc(float(t)) for t in argument.flat), dtype=np.float64,
                           count=argument.size).reshape(argument.shape)
    return amplitude * np.sqrt(np.pi / (2. * a)) * np.exp(-.5 * perpendicular) * endpoint


def gaussian_radon(unit_normal_xyz, offset, center_xyz, covariance_xyz, amplitude=1., *, derivative=0):
    """Smooth analytic plane Radon R, R', or R'' of an anisotropic Gaussian.

    With q=p-n.c and v=n.T*Sigma*n,
    R=2*pi*amplitude*sqrt(det(Sigma)/v)*exp(-q^2/(2*v)),
    R'=-(q/v)*R, and R''=(q^2/v^2-1/v)*R.
    """
    center, covariance, amplitude, determinant = _parameters(center_xyz, covariance_xyz, amplitude, covariance=True)
    normal = _unit(unit_normal_xyz, "unit_normal_xyz")
    offset = np.asarray(offset, dtype=np.float64)
    if not np.isfinite(offset).all() or derivative not in (0, 1, 2):
        raise ValueError("offset must be finite and derivative must be 0, 1, or 2")
    q = offset - normal @ center
    variance = np.sum((normal @ covariance) * normal, axis=-1)
    value = 2. * np.pi * amplitude * np.sqrt(determinant / variance) * np.exp(-q*q / (2. * variance))
    if derivative == 1:
        return -q / variance * value
    if derivative == 2:
        return (q*q / variance**2 - 1. / variance) * value
    return value
