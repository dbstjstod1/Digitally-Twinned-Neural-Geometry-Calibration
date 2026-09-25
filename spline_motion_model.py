"""Estimate nine physical motion curves from a fixed cubic B-spline basis.

Inspired by Flow_matching_motion_3D/fm3d/motion_estimation.py's
BasisMotionEstimator (theta = B @ c). The basis is constructed independently
with SciPy, for arbitrary view counts and correctly clamped endpoints.
No simulation knots, labels, mean-zero constraint, or GT amplitudes are used.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline
from torch import nn


def cubic_bspline_basis(n_views: int, n_control: int):
    """Return a float64 open-uniform, clamped cubic basis and its knot vector."""
    if not 4 <= n_control <= n_views:
        raise ValueError('Require 4 <= control points <= acquired views')
    interior = np.linspace(0., 1., n_control - 2)[1:-1]
    knots = np.r_[np.zeros(4), interior, np.ones(4)]
    basis = BSpline.design_matrix(np.linspace(0., 1., n_views), knots, 3).toarray()
    if not np.isfinite(basis).all() or np.any(basis < 0):
        raise ValueError('Invalid spline basis')
    np.testing.assert_allclose(basis.sum(1), 1., rtol=0, atol=1e-14)
    return basis, knots


class BSplineMotion9(nn.Module):
    """Physical INTERNAL parameters in mm/mm/degrees, not raw MLP outputs.

    Bound the coefficients before interpolation. Nonnegative partition-of-unity
    weights then preserve 10/10/15-style bounds without applying a nonlinear
    tanh to the final curve (which would cease to be a cubic spline).
    """
    def __init__(self, n_views, n_control=20, *, ts_max_mm=10., tp_max_mm=10.,
                 rot_max_deg=15., dtype=torch.float32):
        super().__init__()
        scales = np.array([ts_max_mm]*3 + [tp_max_mm]*3 + [rot_max_deg]*3)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Require finite positive physical bounds')
        basis, knots = cubic_bspline_basis(n_views, n_control)
        self.register_buffer('basis', torch.tensor(basis, dtype=dtype))
        self.register_buffer('knots', torch.tensor(knots, dtype=torch.float64))
        self.register_buffer('scales', torch.tensor(scales, dtype=dtype))
        self.raw_coefficients = nn.Parameter(torch.zeros(n_control, 9, dtype=dtype))

    def physical_coefficients(self):
        return self.raw_coefficients.tanh() * self.scales

    def forward(self, view_indices):
        return self.basis[view_indices] @ self.physical_coefficients()

    def get_config(self):
        return dict(kind='cubic_bspline', control_points=self.basis.shape[1],
                    parameter_count=self.raw_coefficients.numel(), degree=3,
                    knots_progress=self.knots.cpu().tolist(),
                    boundary='Open uniform, clamped; independent endpoint coefficients; not periodic',
                    parameterization='physical_motion9 = B @ (bounds9 * tanh(raw_coefficients))',
                    mean_centered=False, gt_knots_used=False,
                    smoothness='Finite spline space only; no additional smoothness penalty')


class SharedIntrinsicBSplineMotion9(BSplineMotion9):
    """Learn a single unknown K correction and six view-dependent pose curves."""
    def __init__(self, n_views, n_control=20, *, ts_max_mm=10., tp_max_mm=10.,
                 rot_max_deg=15., dtype=torch.float32):
        nn.Module.__init__(self)
        scales = np.array([ts_max_mm]*3 + [tp_max_mm]*3 + [rot_max_deg]*3)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Require finite positive physical bounds')
        basis, knots = cubic_bspline_basis(n_views, n_control)
        self.register_buffer('basis', torch.tensor(basis, dtype=dtype))
        self.register_buffer('knots', torch.tensor(knots, dtype=torch.float64))
        self.register_buffer('scales', torch.tensor(scales, dtype=dtype))
        self.raw_intrinsic = nn.Parameter(torch.zeros(3, dtype=dtype))
        self.raw_rigid = nn.Parameter(torch.zeros(n_control, 6, dtype=dtype))

    @property
    def raw_coefficients(self):
        # An expanded archive representation, not independent K coefficients.
        return torch.cat((self.raw_intrinsic[None].expand(self.basis.shape[1], -1),
                          self.raw_rigid), dim=1)

    def forward(self, view_indices):
        k = self.raw_intrinsic.tanh()*self.scales[:3]
        rigid = self.basis[view_indices] @ (self.raw_rigid.tanh()*self.scales[3:])
        return torch.cat((k[None].expand(len(view_indices), -1), rigid), dim=1)

    def get_config(self):
        config = super().get_config()
        config.update(kind='shared_intrinsic_cubic_bspline',
            parameter_count=sum(p.numel() for p in self.parameters()),
            intrinsic_model='Three unknown constants shared by all views',
            parameterization='K = boundsK*tanh(rawK); rigid = B @ (boundsRigid*tanh(rawRigid))',
            expanded_archive='raw_coefficients repeats rawK in each spline row; only 3 K variables are optimized')
        return config
