"""CUDA hash-grid encoder and MLP used by the 9-DoF motion model."""

import torch
import torch.nn as nn
import tinycudann as tcnn


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

        hidden_dim = 64
        input_dim = 32
        output_dim = output_dim

        # Preserve layer order and names for existing motion checkpoints.
        layers = [
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ]

        self.model = nn.Sequential(*layers)

        self.hash_encoder = tcnn.Encoding(
            n_input_dims=n_inputs,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float32,
        )

    def forward(self, x):
        emb = self.hash_encoder(x)
        out = self.model(emb)
        return out
