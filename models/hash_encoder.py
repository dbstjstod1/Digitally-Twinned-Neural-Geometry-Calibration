from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn


#Original CUDA implemenation of Instant-NGP hash encoder
#https://github.com/NVlabs/tiny-cuda-nn
class MLP_hash(nn.Module):
    def __init__(self, n_inputs, output_dim, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, per_level_scale):
        super(MLP_hash, self).__init__()

        hidden_dim = 64
        input_dim = 32
        output_dim = output_dim       

        # layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        #           nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #           nn.Linear(hidden_dim, output_dim), nn.Sigmoid()] # tanh activation for output to be -1~1, sig for 0~1
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                   # nn.Linear(hidden_dim, output_dim), nn.Tanh()] # tanh activation for output to be -1~1
                  # nn.Linear(hidden_dim, output_dim),nn.ReLU()] # tanh activation for output to be -1~1
                  nn.Linear(hidden_dim, output_dim)] # no activation
        
        self.model = nn.Sequential(*layers)
        
        self.hash_encoder = tcnn.Encoding(
                                        n_input_dims=n_inputs,
                                        encoding_config={
                                            "otype": "HashGrid",
                                            "n_levels": n_levels,
                                            "n_features_per_level": n_features_per_level,
                                            "log2_hashmap_size": log2_hashmap_size,
                                            "base_resolution": base_resolution,
                                            "per_level_scale": per_level_scale
                                        },
                                        dtype = torch.float32
                                    )
    
    def forward(self, x):
        emb = self.hash_encoder(x)
        out = self.model(emb)
        return out


#pytorch implementation of Instant-NGP, however much slower than the CUDA based original
#adapted from https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
class HashEmbedder(nn.Module):
    def __init__(
        self,
        n_input_dims: int = 3,
        otype: str = "HashGrid",
        n_levels: int = 20,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.39,

        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(HashEmbedder, self).__init__()
        assert n_input_dims == 3 and otype == "HashGrid"

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.b = per_level_scale

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
                for _ in range(n_levels)
            ]
        )
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        self.register_buffer(
            "box_offsets",
            torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]),
        )

    def trilinear_interp(
        self,
        x: torch.Tensor,
        voxel_min_vertex: torch.Tensor,
        voxel_embedds: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = x - voxel_min_vertex

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 4] * weights[:, 0][:, None]
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = int(self.base_resolution * self.b**i)
            (
                voxel_min_vertex,
                hashed_voxel_indices,
                xi,
            ) = self.get_voxel_vertices(x, resolution)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(xi, voxel_min_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)

    def get_voxel_vertices(
        self, xyz: torch.Tensor, resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = xyz * resolution
        voxel_min_vertex = torch.floor(xyz).int()
        voxel_indices = voxel_min_vertex.unsqueeze(1) + self.box_offsets
        hashed_voxel_indices = _hash(voxel_indices, self.log2_hashmap_size)

        return voxel_min_vertex, hashed_voxel_indices, xyz


class HashEmbedder_2D(nn.Module):
    def __init__(
        self,
        n_input_dims: int = 2,
        otype: str = "HashGrid",
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 256,
        per_level_scale: float = 2,

        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(HashEmbedder_2D, self).__init__()
        assert n_input_dims == 2 and otype == "HashGrid"

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.b = per_level_scale

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
                for _ in range(n_levels)
            ]
        )
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        self.register_buffer(
            "box_offsets",
            torch.tensor([[[i, j] for i in [0, 1] for j in [0, 1]]]),
        )

    def trilinear_interp(
        self,
        x: torch.Tensor,
        voxel_min_vertex: torch.Tensor,
        voxel_embedds: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = x - voxel_min_vertex

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c0 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 2] * weights[:, 0][:, None]
        )
        c1 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 3] * weights[:, 0][:, None]
        )

        # step 3
        c = c0 * (1 - weights[:, 1][:, None]) + c1 * weights[:, 1][:, None]

        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = int(self.base_resolution * self.b**i)
            (
                voxel_min_vertex,
                hashed_voxel_indices,
                xi,
            ) = self.get_voxel_vertices(x, resolution)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(xi, voxel_min_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)

    def get_voxel_vertices(
        self, xyz: torch.Tensor, resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = xyz * resolution
        voxel_min_vertex = torch.floor(xyz).int() #(Bx1)
        voxel_indices = voxel_min_vertex.unsqueeze(1) + self.box_offsets #(Bx1)
        hashed_voxel_indices = _hash(voxel_indices, self.log2_hashmap_size)

        return voxel_min_vertex, hashed_voxel_indices, xyz


def _hash(coords: torch.Tensor, log2_hashmap_size: int) -> torch.Tensor:
    """
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return (
        torch.tensor((1 << log2_hashmap_size) - 1, device=xor_result.device)
        & xor_result
    )


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                """
                # 본 논문에서 첫 번째와 나머지 layer 의 weights 을 아래와 같이 설정하는 것이 가장 좋았다고 합니다.
                # 첫 번째 layer 를 제외하고는 +- sqrt(c/n) 사이로 초기화 하는 것이 sin을 적용했을 때 uniform 분포를 따르는 것을 보장할 수 있다고 합니다.
                # omega_0 로 sine 함수 정의에 의한 것인데, 자세한 내용은 논문을 참고하시길 바랍니다.
                """
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
    def forward_with_intermediate(self, x):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        net = []
        net += [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for i in range(hidden_layers):
            net += [SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)]
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            net += [final_linear]
        else:
            net += [SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)]
        
        self.net = nn.Sequential(*net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivatives
        output = self.net(coords)
        return output, coords
    
    def forward_with_activations(self, coords, retain_grad=False):
        """
        Returns not only model output, but also intermediate activations,
        Used for visualizing
        """
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join(str(layer.__class__), "%d" % activation_count)] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join(str(layer.__class__), "%d" % activation_count)]
            activation_count += 1
        return activations