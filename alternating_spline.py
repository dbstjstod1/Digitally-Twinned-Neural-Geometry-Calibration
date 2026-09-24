"""Rigid-first block updates in the existing nine-curve B-spline space.

Disjoint Parameter objects and Adam states are essential: zeroing columns of a
shared gradient does not freeze Adam momentum in those columns.
"""
import torch
from torch import nn
from spline_motion_model import BSplineMotion9


class SplitBSplineMotion9(nn.Module):
    def __init__(self, n_views, n_control=20, **kwargs):
        super().__init__()
        template = BSplineMotion9(n_views, n_control, **kwargs)
        for name in ('basis', 'knots', 'scales'):
            self.register_buffer(name, getattr(template, name).clone())
        self.raw_intrinsic = nn.Parameter(template.raw_coefficients[:, :3].detach().clone())
        self.raw_rigid = nn.Parameter(template.raw_coefficients[:, 3:].detach().clone())

    @property
    def raw_coefficients(self):
        return torch.cat((self.raw_intrinsic, self.raw_rigid), dim=-1)

    physical_coefficients = BSplineMotion9.physical_coefficients
    get_config = BSplineMotion9.get_config

    def forward(self, view_indices, block=None):
        if block not in (None, 'rigid', 'intrinsic'):
            raise ValueError(f'Unknown motion block: {block}')
        k = self.raw_intrinsic.detach() if block == 'rigid' else self.raw_intrinsic
        r = self.raw_rigid.detach() if block == 'intrinsic' else self.raw_rigid
        coefficients = torch.cat((k, r), dim=-1).tanh() * self.scales
        return self.basis[view_indices] @ coefficients


class AlternatingSplineAdam:
    """Persistent per-block moments/clocks, with runtime inactive-block audit."""
    blocks = ('rigid', 'intrinsic')

    def __init__(self, model, lr):
        self.parameters = dict(rigid=model.raw_rigid, intrinsic=model.raw_intrinsic)
        self.optimizers = {name: torch.optim.Adam([parameter], lr=lr)
                           for name, parameter in self.parameters.items()}
        self.updates = {name: 0 for name in self.blocks}
        self.inactive_checks = 0

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, block):
        if block not in self.blocks:
            raise ValueError(f'Unknown optimizer block: {block}')
        inactive = 'intrinsic' if block == 'rigid' else 'rigid'
        other = self.parameters[inactive]
        if other.grad is not None:
            raise RuntimeError('Inactive block unexpectedly received a gradient')
        before = other.detach().clone()
        self.optimizers[block].step()
        if not torch.equal(before, other):
            raise RuntimeError('Inactive block changed during the other block update')
        self.updates[block] += 1
        self.inactive_checks += 1

    def state_dict(self):
        return dict(optimizers={name: opt.state_dict() for name,opt in self.optimizers.items()},
                    updates=self.updates.copy(), inactive_checks=self.inactive_checks)

    def load_state_dict(self, state):
        for name,opt in self.optimizers.items(): opt.load_state_dict(state['optimizers'][name])
        self.updates = state['updates'].copy()
        self.inactive_checks = state['inactive_checks']
