"""Minimal PaiNN-style model implemented in PyTorch + PyG.

This module is intended as an **architecture-identical-ish** reference
implementation for benchmarking against Hadronis. It follows the core
PaiNN ideas (scalar and vector features, equivariant message passing)
without attempting to be a drop-in reproduction of any particular
codebase.

Dependencies (not installed by default):
- torch
- torch_geometric
- torch_scatter
- torch_cluster (for radius_graph)

The intended usage is to construct a `torch_geometric.data.Data` object
from atomic numbers and positions using `build_single_molecule_data`,
then run it through `PaiNNModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_cluster import radius_graph
from torch_scatter import scatter_add


@dataclass
class PaiNNConfig:
    hidden_dim: int = 128
    n_interactions: int = 3
    n_rbf: int = 32
    cutoff: float = 5.0
    max_z: int = 100


class RadialBasis(nn.Module):
    """Simple Gaussian radial basis as a function of pairwise distance.

    This is a lightweight implementation used only for the PyTorch
    baseline; Hadronis uses its own low-level implementation.
    """

    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        centers = torch.linspace(0.0, cutoff, num_rbf)
        # Use a fixed width based on spacing between centers
        delta = centers[1] - centers[0] if num_rbf > 1 else torch.tensor(cutoff)
        widths = torch.full_like(centers, delta)

        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distances: Tensor) -> Tensor:
        """Compute radial basis values for input distances.

        distances: shape [..., 1] or [...]
        returns: shape [..., num_rbf]
        """

        d = distances.unsqueeze(-1) if distances.dim() == 1 else distances
        # [..., 1] - [num_rbf] -> [..., num_rbf]
        diff = d - self.centers
        return torch.exp(-((diff / (self.widths + 1e-8)) ** 2))


class PaiNNInteraction(nn.Module):
    """Single PaiNN-style interaction block.

    Maintains scalar features s_i in R^{F} and vector features v_i in
    R^{3 x F}. Messages are computed along edges and aggregated with a
    scatter-add, preserving rotation equivariance via directional
    updates along the unit vectors r_ij / ||r_ij||.
    """

    def __init__(self, hidden_dim: int, n_rbf: int, radial: RadialBasis) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.radial = radial

        # MLP that turns concatenated [s_i, s_j, rbf(d_ij)] into scalar and
        # vector update magnitudes.
        in_dim = 2 * hidden_dim + n_rbf
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
        )

        # Separate linear projections for scalar and vector parts.
        self.to_scalar = nn.Linear(hidden_dim * 2, hidden_dim)
        self.to_vector = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, s: Tensor, v: Tensor, pos: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one interaction step.

        s: [N, F]
        v: [N, 3, F]
        pos: [N, 3]
        edge_index: [2, E]
        """

        src, dst = edge_index  # [E]

        # Pairwise relative vectors and distances
        rij = pos[dst] - pos[src]  # [E, 3]
        dist = torch.linalg.norm(rij, dim=-1, keepdim=True)  # [E, 1]

        rbf = self.radial(dist.squeeze(-1))  # [E, n_rbf]

        s_i = s[src]  # [E, F]
        s_j = s[dst]  # [E, F]

        h = torch.cat([s_i, s_j, rbf], dim=-1)  # [E, 2F + n_rbf]
        h = self.mlp(h)  # [E, 2F]

        ds_edge = self.to_scalar(h)  # [E, F]
        dv_mag = self.to_vector(h)  # [E, F]

        # Normalize rij to get directions; avoid division by zero.
        inv_dist = torch.where(dist > 0, 1.0 / dist, torch.zeros_like(dist))  # [E, 1]
        directions = rij * inv_dist  # [E, 3]

        # Turn per-edge vector magnitudes into full vectors aligned with rij.
        dv_edge = directions.unsqueeze(-1) * dv_mag.unsqueeze(-2)  # [E, 3, F]

        # Aggregate to destination nodes
        num_nodes = s.size(0)
        ds = scatter_add(ds_edge, dst, dim=0, dim_size=num_nodes)  # [N, F]
        dv = scatter_add(dv_edge, dst, dim=0, dim_size=num_nodes)  # [N, 3, F]

        s_out = s + ds
        v_out = v + dv
        return s_out, v_out


class PaiNNModel(nn.Module):
    """Minimal PaiNN-style model operating on PyG Data objects.

    Expected Data attributes:
    - data.z: atomic numbers, shape [N] (int64)
    - data.pos: positions, shape [N, 3]
    - data.batch: batch indices, shape [N] (optional, defaults to zeros)
    - data.edge_index: [2, E] (optional; built via radius_graph if missing)
    """

    def __init__(self, config: Optional[PaiNNConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = PaiNNConfig()
        self.config = config

        self.embedding = nn.Embedding(config.max_z, config.hidden_dim)
        self.radial = RadialBasis(config.n_rbf, config.cutoff)

        self.interactions = nn.ModuleList(
            [
                PaiNNInteraction(config.hidden_dim, config.n_rbf, self.radial)
                for _ in range(config.n_interactions)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    @property
    def cutoff(self) -> float:
        return self.config.cutoff

    def _build_edge_index(self, pos: Tensor, batch: Tensor) -> Tensor:
        # Use torch_cluster.radius_graph to build neighbor lists.
        return radius_graph(pos, r=self.cutoff, batch=batch, loop=False)

    def forward(self, data: Data) -> Tensor:
        if not hasattr(data, "z"):
            raise ValueError("Data object must have attribute 'z' with atomic numbers.")
        if not hasattr(data, "pos"):
            raise ValueError("Data object must have attribute 'pos' with positions.")

        z: Tensor = data.z
        pos: Tensor = data.pos
        batch: Tensor
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        if hasattr(data, "edge_index") and data.edge_index is not None:
            edge_index: Tensor = data.edge_index
        else:
            edge_index = self._build_edge_index(pos, batch)

        # Initial scalar features from atomic numbers; vector features start at zero.
        s = self.embedding(z.clamp(max=self.config.max_z - 1))  # [N, F]
        v = torch.zeros(z.size(0), 3, self.config.hidden_dim, device=z.device)  # [N, 3, F]

        for interaction in self.interactions:
            s, v = interaction(s, v, pos, edge_index)

        # Pool over atoms to get per-molecule representation.
        s_pool = global_add_pool(s, batch)  # [B, F]
        out = self.readout(s_pool).squeeze(-1)  # [B]
        return out


def build_single_molecule_data(
    atomic_numbers: Tensor,
    positions: Tensor,
    cutoff: float,
    device: Optional[torch.device] = None,
) -> Data:
    """Construct a PyG Data object for a single molecule.

    atomic_numbers: 1D tensor of shape [N] (int64 or int32)
    positions: tensor of shape [N, 3]
    cutoff: radius for neighbor construction (Angstrom)
    device: optional device; if provided, tensors are moved there.
    """

    if device is None:
        device = positions.device

    z = atomic_numbers.to(device=device, dtype=torch.long)
    pos = positions.to(device=device, dtype=torch.get_default_dtype())

    batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
    edge_index = radius_graph(pos, r=cutoff, batch=batch, loop=False)

    data = Data(z=z, pos=pos, batch=batch, edge_index=edge_index)
    return data
