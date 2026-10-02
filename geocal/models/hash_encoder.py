"""CUDA hash-grid encoder and MLP used by the 9-DoF motion model."""

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
import tinycudann as tcnn


def _positive_integer(name, value, *, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        limit = (
            f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        )
        raise ValueError(f"{name} must be{limit}")
    return int(value)


class MLP_hash(nn.Module):
    def __init__(
        self,
        n_inputs,
        output_dim,
        n_levels,
        n_features_per_level,
        log2_hashmap_size,
        base_resolution,
        per_level_scale,
    ):
        super(MLP_hash, self).__init__()
        n_inputs = _positive_integer("n_inputs", n_inputs)
        if n_inputs not in (2, 3, 4):
            raise ValueError("This HashGrid supports n_inputs in {2, 3, 4}")
        output_dim = _positive_integer("output_dim", output_dim)
        n_levels = _positive_integer("n_levels", n_levels, maximum=128)
        n_features_per_level = _positive_integer(
            "n_features_per_level", n_features_per_level
        )
        if n_features_per_level not in (1, 2, 4, 8):
            raise ValueError("n_features_per_level must be 1, 2, 4, or 8")
        log2_hashmap_size = _positive_integer(
            "log2_hashmap_size", log2_hashmap_size, minimum=0, maximum=31
        )
        base_resolution = _positive_integer(
            "base_resolution", base_resolution, minimum=2
        )
        if (
            isinstance(per_level_scale, bool)
            or not isinstance(per_level_scale, Real)
            or not math.isfinite(per_level_scale)
            or per_level_scale < 1
        ):
            raise ValueError("per_level_scale must be finite and >= 1")
        per_level_scale = float(per_level_scale)
        # tiny-cuda-nn stores the lattice resolution in a uint32_t.
        if math.log(base_resolution) + (n_levels - 1) * math.log(
            per_level_scale
        ) >= math.log(2**32 - 1):
            raise ValueError("The finest HashGrid resolution must fit in uint32")

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        # The encoder uses its own native initialization seed, not torch's CPU
        # RNG. Construct locally to learn its actual (potentially padded) width;
        # register it after the MLP to retain legacy module/state-dict ordering.
        encoder = tcnn.Encoding(
            n_input_dims=n_inputs, encoding_config=encoding_config, dtype=torch.float32
        )
        hidden_dim = 64
        input_dim = int(encoder.n_output_dims)
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.hash_encoder = encoder
        self._config = dict(
            n_inputs=n_inputs,
            output_dim=output_dim,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
            encoding_output_dim=input_dim,
            mlp_hidden_dim=hidden_dim,
        )

    def get_config(self):
        """Return JSON-safe estimator metadata without adding checkpoint state."""
        config = dict(self._config)
        config["level_resolutions_nominal"] = [
            config["base_resolution"] * config["per_level_scale"] ** level
            for level in range(config["n_levels"])
        ]
        config["grid_resolution_convention"] = (
            "nominal r_l = base_resolution * per_level_scale**l; "
            "tiny-cuda-nn uses grid_scale = exp2f(l * log2f(per_level_scale)) "
            "* base_resolution - 1 and vertex_count = ceilf(grid_scale) + 1; "
            "nominal values omit native float32 rounding"
        )
        return config

    def forward(self, x):
        emb = self.hash_encoder(x)
        out = self.model(emb)
        return out
