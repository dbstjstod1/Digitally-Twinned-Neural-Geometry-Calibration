"""Single-resolution projection losses for geometry calibration.

Inputs are unmodified line integrals with shape ``[views, rows, columns]``.
No loss applies a mask, ROI, intensity normalization, or image pyramid.

``lncc`` preserves ``1 + MONAI LocalNormalizedCrossCorrelationLoss`` exactly
for its default mean reduction. MONAI clamps *each weighted variance sum* to
``smooth_dr``; this is not an epsilon added to their product. Rectangular
kernels have 2-D weight mass ``kernel_size**2``; MONAI triangular kernels have
different weight mass. Consequently an identical sum-domain ``smooth_dr``
does not represent an identical variance-mean floor across kernel types.

``signed_lncc`` uses the same local sums, kernels and zero padding, retaining
the sign of the cross covariance. ``global_ncc`` uses whole-image centered
sums. Both use the same per-variance-sum floor and require ``smooth_nr=0``.

``poisson`` requires the ORIGINAL integer photon counts, not exponentiated
post-log targets. It returns pixel-mean Poisson deviance divided by incident
fluence I0: ``2/I0 * (lambda - N + N*log(N/lambda))``, where
``lambda = I0*exp(-prediction)``. Its zero-count limit is evaluated explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.networks.layers.simplelayers import separable_filtering


LOSS_NAMES = ("lncc", "signed_lncc", "global_ncc", "mse", "huber", "poisson")


@dataclass(frozen=True)
class LossConfig:
    name: str = "lncc"
    kernel_size: int = 31
    kernel_type: str = "rectangular"
    smooth_nr: float = 0.0
    smooth_dr: float = 1e-5
    huber_delta: float = 1.0
    i0: float = 44000.0
    reduction: str = "mean"

    def __post_init__(self):
        if self.name not in LOSS_NAMES:
            raise ValueError(f"Unknown loss {self.name!r}; choose from {LOSS_NAMES}")
        if (isinstance(self.kernel_size, bool) or not isinstance(self.kernel_size, Integral)
                or self.kernel_size <= 0 or self.kernel_size % 2 != 1):
            raise ValueError("kernel_size must be a positive odd integer")
        if self.kernel_type == "gaussian":
            raise ValueError(
                "Gaussian LNCC kernels are disabled: MONAI 1.5.2 has a known "
                "off-center/truncated Gaussian implementation bug. Use rectangular or triangular."
            )
        if self.kernel_type not in ("rectangular", "triangular"):
            raise ValueError("kernel_type must be rectangular or triangular")
        if self.name in ("lncc", "signed_lncc"):
            minimum = 5 if self.kernel_type == "triangular" else 3
            if self.kernel_size < minimum:
                raise ValueError(
                    f"{self.kernel_type} local correlation requires kernel_size >= {minimum}; "
                    "a single nonzero kernel weight has no local variance or registration gradient"
                )
        for name in ("smooth_nr", "smooth_dr", "huber_delta", "i0"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            if value < 0 or (name != "smooth_nr" and value == 0):
                raise ValueError(f"{name} must be {'nonnegative' if name == 'smooth_nr' else 'positive'}")
        if self.name != "lncc" and self.smooth_nr != 0:
            raise ValueError("Nonzero smooth_nr is supported only by squared lncc")
        if self.reduction not in ("mean", "none"):
            raise ValueError("reduction must be mean or none (one loss per view)")


class CalibrationLoss(nn.Module):
    """Callable loss with an explicit ``per_view`` method for screening.

    ``forward(pred, target, counts=None)`` returns a scalar by default;
    ``reduction='none'`` and ``per_view(...)`` return ``[views]``.
    ``counts`` is used only for Poisson; image losses use ``target``.
    Poisson also accepts ``forward(pred, counts=counts)`` without a target.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        options = dict(spatial_dims=2, kernel_size=config.kernel_size,
                       kernel_type=config.kernel_type, smooth_nr=config.smooth_nr,
                       smooth_dr=config.smooth_dr)
        self._monai_mean = LocalNormalizedCrossCorrelationLoss(**options, reduction="mean")
        self._monai_none = LocalNormalizedCrossCorrelationLoss(**options, reduction="none")

    def get_config(self) -> dict:
        """Serializable configuration including the smoothing-unit convention."""
        result = asdict(self.config)
        if self.config.name in ("lncc", "signed_lncc"):
            result["kernel_weight_mass_2d"] = float(self._monai_none.kernel_vol)
            result["smooth_dr_units"] = "Floor for each weighted variance sum, not variance mean or variance product"
        elif self.config.name == "global_ncc":
            result["smooth_dr_units"] = "Floor for each whole-image centered sum of squares"
        if self.config.name == "poisson":
            result["normalization"] = "Poisson deviance divided by I0; mean over detector pixels"
        return result

    @staticmethod
    def _validate_prediction(pred: torch.Tensor):
        if not isinstance(pred, torch.Tensor) or pred.ndim != 3 or any(n == 0 for n in pred.shape):
            raise ValueError("pred must be a nonempty [views, rows, columns] tensor")
        if not pred.is_floating_point():
            raise ValueError("pred must contain floating-point line integrals")

    @staticmethod
    def _validate_target(pred: torch.Tensor, target: torch.Tensor | None):
        if not isinstance(target, torch.Tensor) or target.shape != pred.shape:
            raise ValueError("target must have the same [views, rows, columns] shape as pred")
        if target.device != pred.device or target.dtype != pred.dtype:
            raise ValueError("target and pred must have the same floating-point dtype and device")

    def _signed_local_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # These sum and padding operations mirror the installed MONAI implementation.
        p, t = pred[:, None], target[:, None]
        kernel = self._monai_none.kernel.to(p)
        mass = self._monai_none.kernel_vol.to(p)
        kernels = [kernel, kernel]
        t_sum = separable_filtering(t, kernels=kernels)
        p_sum = separable_filtering(p, kernels=kernels)
        t2_sum = separable_filtering(t * t, kernels=kernels)
        p2_sum = separable_filtering(p * p, kernels=kernels)
        tp_sum = separable_filtering(t * p, kernels=kernels)
        cross = tp_sum - (p_sum / mass) * t_sum
        t_var = (t2_sum - (t_sum / mass) * t_sum).clamp_min(self.config.smooth_dr)
        p_var = (p2_sum - (p_sum / mass) * p_sum).clamp_min(self.config.smooth_dr)
        return 1.0 - cross / torch.sqrt(t_var * p_var)

    def _poisson_map(self, pred: torch.Tensor, counts: torch.Tensor | None) -> torch.Tensor:
        if not isinstance(counts, torch.Tensor) or counts.shape != pred.shape:
            raise ValueError("Poisson loss requires original integer counts matching pred.shape")
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        integer_dtypes.update(getattr(torch, name) for name in ("uint16", "uint32", "uint64")
                              if hasattr(torch, name))
        if counts.dtype not in integer_dtypes or counts.device != pred.device:
            raise ValueError("counts must have integer dtype and be on the prediction device")
        observed = counts.to(dtype=pred.dtype)
        if bool((observed < 0).any()):
            raise ValueError("counts must be nonnegative")
        q = observed / self.config.i0
        # uint32 storage is supported by torch.from_numpy, but some unsigned
        # comparison kernels are unavailable on CUDA. Compare the converted data.
        positive = observed != 0
        log_q = torch.log(torch.where(positive, q, torch.ones_like(q)))
        a = pred + log_q
        # At q>0, q*(expm1(-a)+a) avoids subtracting nearly equal lambda and N.
        # For q=0, take the exact deviance limit instead of log(0) or 0*log(0).
        positive_term = 2.0 * q * (torch.expm1(-a) + a)
        return torch.where(positive, positive_term, 2.0 * torch.exp(-pred))

    def per_view(self, pred: torch.Tensor, target: torch.Tensor | None = None,
                 counts: torch.Tensor | None = None) -> torch.Tensor:
        """Return one spatially averaged loss per view, without view reduction."""
        self._validate_prediction(pred)
        name = self.config.name
        if name == "poisson":
            return self._poisson_map(pred, counts).mean(dim=(1, 2))
        self._validate_target(pred, target)
        if name == "lncc":
            return 1.0 + self._monai_none(pred[:, None], target[:, None]).mean(dim=(1, 2, 3))
        if name == "signed_lncc":
            return self._signed_local_map(pred, target).mean(dim=(1, 2, 3))
        if name == "global_ncc":
            p = pred - pred.mean(dim=(1, 2), keepdim=True)
            t = target - target.mean(dim=(1, 2), keepdim=True)
            cross = (p * t).sum(dim=(1, 2))
            vp = p.square().sum(dim=(1, 2)).clamp_min(self.config.smooth_dr)
            vt = t.square().sum(dim=(1, 2)).clamp_min(self.config.smooth_dr)
            return 1.0 - cross / torch.sqrt(vp * vt)
        if name == "mse":
            return (pred - target).square().mean(dim=(1, 2))
        return F.huber_loss(pred, target, reduction="none", delta=self.config.huber_delta).mean(dim=(1, 2))

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None = None,
                counts: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.name == "lncc" and self.config.reduction == "mean":
            # Preserve the original default reduction order and image gradients exactly.
            self._validate_prediction(pred)
            self._validate_target(pred, target)
            return 1.0 + self._monai_mean(pred[:, None], target[:, None])
        values = self.per_view(pred, target, counts)
        return values.mean() if self.config.reduction == "mean" else values


def build_loss(name: str = "lncc", *, kernel_size: int = 31,
               kernel_type: str = "rectangular", smooth_nr: float = 0.0,
               smooth_dr: float = 1e-5, huber_delta: float = 1.0,
               i0: float = 44000.0, reduction: str = "mean") -> CalibrationLoss:
    """Build one single-resolution loss. See ``LossConfig`` for recorded options."""
    return CalibrationLoss(LossConfig(name=name, kernel_size=kernel_size,
                           kernel_type=kernel_type, smooth_nr=smooth_nr,
                           smooth_dr=smooth_dr, huber_delta=huber_delta,
                           i0=i0, reduction=reduction))
