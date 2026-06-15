import math
import torch
import torch.nn as nn
from models.hash_encoder import MLP_hash

class MotionNetHash_6DoF(nn.Module):
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
            n_inputs=2,             # <-- was 1
            output_dim=6,
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

        p6_raw = self.net(v_in)
        return p6_raw

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
            n_inputs=2,             # <-- was 1
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
    
class MotionNetHash_10DoF(nn.Module):
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

        self.net = MLP_hash(
            n_inputs=2,
            output_dim=10,
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
        v_norm = v_idx / denom

        zeros = torch.zeros_like(v_norm)
        v_in = torch.stack([v_norm, zeros], dim=-1).contiguous()

        p10_raw = self.net(v_in)
        return p10_raw
    
class MotionNetHash_10DoF_globalK(nn.Module):
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

        # View-dependent rigid motion only
        self.net = MLP_hash(
            n_inputs=2,
            output_dim=6,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )

        # Global K correction shared by all views
        # [alpha_u, alpha_v, delta_u, delta_v]
        self.kcorr_global = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def forward(self, v_idx: torch.Tensor) -> torch.Tensor:
        if v_idx.dtype != torch.float32:
            v_idx = v_idx.float()

        denom = max(self.n_views - 1, 1)
        v_norm = v_idx / denom

        zeros = torch.zeros_like(v_norm)
        v_in = torch.stack([v_norm, zeros], dim=-1).contiguous()

        p6_raw = self.net(v_in)  # (B,6)

        B = p6_raw.shape[0]
        kcorr = self.kcorr_global.unsqueeze(0).expand(B, 4)  # (B,4)

        p10_raw = torch.cat([kcorr, p6_raw], dim=-1)  # (B,10)
        return p10_raw

class FourierEncoding1D(nn.Module):
    """
    English comments only.

    1D Fourier feature encoding for view indices.

    Input:
        x_norm: (B, 1), normalized to [-1, 1]

    Output:
        feat: (B, 1 + 2 * num_frequencies)
              [x, sin(2^k * pi * x), cos(2^k * pi * x), ...]
    """
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)

        # Fixed frequency bands: [1, 2, 4, 8, ...]
        freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 2 or x_norm.shape[1] != 1:
            raise ValueError(f"x_norm must have shape (B,1), but got {tuple(x_norm.shape)}")

        # (B, F)
        angles = math.pi * x_norm * self.freq_bands.unsqueeze(0)

        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)

        out = []
        if self.include_input:
            out.append(x_norm)
        out.append(sin_feat)
        out.append(cos_feat)

        return torch.cat(out, dim=1)


class MotionFourierMLP_9DoF(nn.Module):
    """
    English comments only.

    Drop-in replacement for MotionNetHash_9DoF.

    Input:
        view_idx: (B,) or (B,1), integer or float view indices

    Output:
        p9_raw: (B, 9)
            Raw motion parameters.
            You can pass this output to your existing:
                motion9_to_ts_tp_rot(...)
    """
    def __init__(
        self,
        n_views: int,
        num_frequencies: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        include_input: bool = True,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        zero_init_last: bool = True,
    ):
        super().__init__()

        if n_views <= 1:
            raise ValueError("n_views must be > 1")

        self.n_views = int(n_views)

        self.encoder = FourierEncoding1D(
            num_frequencies=num_frequencies,
            include_input=include_input,
        )

        in_dim = (1 if include_input else 0) + 2 * num_frequencies

        layers = []
        prev_dim = in_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 9)

        if zero_init_last:
            # Start near zero motion for stability.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _normalize_view_idx(self, view_idx: torch.Tensor) -> torch.Tensor:
        """
        English comments only.

        Normalize view indices from [0, n_views-1] to [-1, 1].
        """
        if view_idx.ndim == 1:
            x = view_idx[:, None]
        elif view_idx.ndim == 2 and view_idx.shape[1] == 1:
            x = view_idx
        else:
            raise ValueError(f"view_idx must have shape (B,) or (B,1), but got {tuple(view_idx.shape)}")

        x = x.to(dtype=torch.float32)

        # Map [0, n_views-1] -> [-1, 1]
        denom = float(max(self.n_views - 1, 1))
        x_norm = 2.0 * (x / denom) - 1.0
        return x_norm

    def forward(self, view_idx: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalize_view_idx(view_idx)
        feat = self.encoder(x_norm)
        h = self.backbone(feat)
        p9_raw = self.head(h)
        return p9_raw