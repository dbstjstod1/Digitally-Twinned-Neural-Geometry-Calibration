"""Signed local normalized cross correlation used by the simulation recipe."""

import torch
from torch import nn
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.networks.layers.simplelayers import separable_filtering


class SignedLNCC(nn.Module):
    def __init__(self, kernel_size=31, smooth_dr=1e-5):
        super().__init__()
        self.smooth_dr = smooth_dr
        self.window = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=2,
            kernel_size=kernel_size,
            kernel_type="rectangular",
            smooth_nr=0.0,
            smooth_dr=smooth_dr,
            reduction="none",
        )

    def _signed_local_map(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # These sum and padding operations mirror the installed MONAI implementation.
        p, t = pred[:, None], target[:, None]
        kernel = self.window.kernel.to(p)
        mass = self.window.kernel_vol.to(p)
        kernels = [kernel, kernel]
        t_sum = separable_filtering(t, kernels=kernels)
        p_sum = separable_filtering(p, kernels=kernels)
        t2_sum = separable_filtering(t * t, kernels=kernels)
        p2_sum = separable_filtering(p * p, kernels=kernels)
        tp_sum = separable_filtering(t * p, kernels=kernels)
        cross = tp_sum - (p_sum / mass) * t_sum
        t_var = (t2_sum - (t_sum / mass) * t_sum).clamp_min(self.smooth_dr)
        p_var = (p2_sum - (p_sum / mass) * p_sum).clamp_min(self.smooth_dr)
        return 1.0 - cross / torch.sqrt(t_var * p_var)

    def forward(self, pred, target, counts=None):
        if pred.ndim != 3 or pred.shape != target.shape:
            raise ValueError("Expected equal [view, row, column] tensors")
        return self._signed_local_map(pred, target).flatten(1).mean(1).mean()
