import torch
import torch.nn as nn
from models.hash_encoder import MLP_hash


class MotionNetHash_9DoF(nn.Module):
    def __init__(
        self,
        n_views: int,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
    ):
        super().__init__()
        self.n_views = n_views

        # IMPORTANT: HashGrid needs n_input_dims in {2,3,4}
        self.net = MLP_hash(
            n_inputs=2,
            output_dim=9,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )

    def forward(self, v_idx: torch.Tensor) -> torch.Tensor:
        if v_idx.dtype != torch.float32:
            v_idx = v_idx.float()

        denom = max(self.n_views - 1, 1)
        v_norm = v_idx / denom  # [0,1]

        # Make 2D input: (v_norm, 0)
        zeros = torch.zeros_like(v_norm)
        v_in = torch.stack([v_norm, zeros], dim=-1).contiguous()  # (B,2)

        p9_raw = self.net(v_in)
        return p9_raw
