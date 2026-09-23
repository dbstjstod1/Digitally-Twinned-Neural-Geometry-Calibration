"""Optional minimum-norm priors on the APPLIED effective geometry parameters.

The input is bounds * tanh(network_output), not raw network outputs, network
weights, source positions, or GT-relative errors. Its groups are (du, df, dv),
effective object-space translation xyz, and Euler rotation xyz. Each group is
divided by its configured physical scale before averaging its squared entries.
Using the training bounds as scales makes these penalties dimensionless.

This is a nominal-geometry prior that changes the training objective. It can
prefer a smaller correction among weakly distinguished solutions; it does not
remove an exact gauge, guarantee recovery of GT, or preserve fitted P matrices.
No phantom pose, trajectory, smoothness term, or GT geometry is used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch
from torch import nn


@dataclass(frozen=True)
class RegularizationConfig:
    intrinsic_weight: float = 0.0
    translation_weight: float = 0.0
    rotation_weight: float = 0.0
    intrinsic_scale_mm: float = 10.0
    translation_scale_mm: float = 10.0
    rotation_scale_deg: float = 15.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            positive = "scale" in name
            if value < 0 or (positive and value == 0):
                raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
            # Keep provenance JSON-serializable even for numpy scalar inputs.
            object.__setattr__(self, name, float(value))

    @property
    def active(self) -> bool:
        return any(weight > 0 for weight in (
            self.intrinsic_weight, self.translation_weight, self.rotation_weight))

    def get_config(self) -> dict:
        return dict(
            **asdict(self), active=self.active,
            type="applied_motion_group_minimum_norm",
            input="Applied bounded nominal-relative motion9 = bounds * tanh(raw9); not raw9 or network weights",
            normalization="Mean squared (parameter / physical group scale) over views and three coordinates; scales may be set to training bounds",
            parameter_groups=dict(intrinsic=["delta_u_mm", "delta_f_mm", "delta_v_mm"],
                                  translation=["translation_x_mm", "translation_y_mm", "translation_z_mm"],
                                  rotation=["rotation_x_deg", "rotation_y_deg", "rotation_z_deg"]),
            target="Zero nominal correction, not GT geometry or an estimated source position",
            limitation="A training prior that can bias geometry; not exact gauge removal or a guarantee of accurate P",
        )


class AppliedMotionRegularizer(nn.Module):
    """Weighted sum of dimensionless group mean squares on [batch,9] motion.

    ``components`` returns unweighted ``intrinsic``, ``translation``, and
    ``rotation`` scalars plus their weighted ``total``. An unequal last batch
    must be weighted by its number of views when accumulating epoch averages.
    ``combine`` bypasses the regularizer entirely when every weight is zero,
    returning the original image-loss tensor object and unchanged graph.
    """

    def __init__(self, config: RegularizationConfig):
        super().__init__()
        if not isinstance(config, RegularizationConfig):
            raise ValueError("config must be a RegularizationConfig")
        self.config = config

    @property
    def active(self) -> bool:
        return self.config.active

    def get_config(self) -> dict:
        return self.config.get_config()

    @staticmethod
    def _validate(motion9: torch.Tensor):
        if (not isinstance(motion9, torch.Tensor) or motion9.ndim != 2
                or motion9.shape[0] == 0 or motion9.shape[1] != 9):
            raise ValueError("motion9 must be a nonempty [batch,9] applied-parameter tensor")
        if not motion9.is_floating_point():
            raise ValueError("motion9 must have a floating-point dtype")
        if not torch.isfinite(motion9).all():
            raise ValueError("motion9 must contain finite applied parameters")

    def components(self, motion9: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate(motion9)
        cfg = self.config
        result = {
            "intrinsic": (motion9[:, :3] / cfg.intrinsic_scale_mm).square().mean(),
            "translation": (motion9[:, 3:6] / cfg.translation_scale_mm).square().mean(),
            "rotation": (motion9[:, 6:9] / cfg.rotation_scale_deg).square().mean(),
        }
        total = motion9.new_zeros(())
        for name, weight in (("intrinsic", cfg.intrinsic_weight),
                             ("translation", cfg.translation_weight),
                             ("rotation", cfg.rotation_weight)):
            if weight > 0:
                total = total + weight * result[name]
        result["total"] = total
        return result

    def forward(self, motion9: torch.Tensor) -> torch.Tensor:
        return self.components(motion9)["total"]

    def combine(self, image_loss: torch.Tensor, motion9: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return image_loss
        if (not isinstance(image_loss, torch.Tensor) or image_loss.ndim != 0
                or not image_loss.is_floating_point()):
            raise ValueError("image_loss must be a scalar floating-point tensor")
        return image_loss + self(motion9)
