"""
Energy Computation Layer — Module 4 of FAPT-GNN

Computes the System Energy Function E(t):

  E(t) = F_t^T (I + λ A_t) F_t
       = Σ_i F_i(t)^2  +  λ Σ_(i,j)∈E w_ij · F_i(t) · F_j(t)

Interpretation:
  - First term  → local instability accumulation
  - Second term → contagion coupling through network

This is the core physics-inspired novelty.

Also implements:
  - ΔE(t) = E(t) - E(t-1): energy change (crash precursor)
  - d²E/dt² > 0: accelerating instability (early warning signal)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EnergyLayer(nn.Module):
    """
    Computes system energy E(t) = F_t^T (I + λ A_t) F_t

    λ (lambda) is a learnable parameter initialized to 1.0.

    Also provides:
      - energy_gradient: ΔE sequence for temporal model
      - energy_acceleration: d²E/dt²
    """

    def __init__(self, lambda_init: float = 1.0):
        super().__init__()
        # λ is the contagion coupling strength — LEARNABLE
        self.log_lambda = nn.Parameter(torch.tensor(lambda_init).log())

    @property
    def lambda_(self) -> torch.Tensor:
        """Positive lambda via exp(log_lambda)."""
        return self.log_lambda.exp()

    def forward(
        self,
        fragility: torch.Tensor,     # (N,) or (B, N) — F_i(t)
        adj: torch.Tensor,           # (N, N) or (B, N, N) — adjacency matrix
    ) -> torch.Tensor:
        """
        Compute E(t) = F^T (I + λ A) F

        Args:
            fragility: scalar fragility per node (N,) or batch (B, N)
            adj      : weighted adjacency matrix (N, N) or (B, N, N)

        Returns:
            energy: scalar system energy E(t) — shape () or (B,)
        """
        lam = self.lambda_

        if fragility.dim() == 1:
            # Single graph: (N,)
            N = fragility.size(0)
            I = torch.eye(N, device=fragility.device)
            M = I + lam * adj              # (N, N)
            energy = fragility @ M @ fragility  # scalar
        else:
            # Batch: (B, N)
            B, N = fragility.shape
            I = torch.eye(N, device=fragility.device).unsqueeze(0).expand(B, -1, -1)
            M = I + lam * adj              # (B, N, N)
            # E = F^T M F for each batch item
            F_unsq = fragility.unsqueeze(-1)   # (B, N, 1)
            energy = torch.bmm(torch.bmm(F_unsq.transpose(-1, -2), M), F_unsq).squeeze(-1).squeeze(-1)  # (B,)

        return energy

    def energy_components(
        self,
        fragility: torch.Tensor,  # (N,)
        adj: torch.Tensor,        # (N, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose energy into:
          local_term = F^T F = Σ F_i^2
          coupling_term = λ * F^T A F = λ Σ w_ij F_i F_j

        Returns:
            (local_energy, coupling_energy)
        """
        local = (fragility ** 2).sum()
        coupling = self.lambda_ * (fragility @ adj @ fragility)
        return local, coupling


class EnergySequenceProcessor(nn.Module):
    """
    Processes a sequence of energy values E(t-k), ..., E(t)
    and computes derived signals:
      - ΔE(t) = E(t) - E(t-1)
      - ΔΔE(t) = ΔE(t) - ΔE(t-1)  [acceleration → early warning signal]
      - rolling statistics

    These are concatenated and fed to the Temporal Model.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Project energy scalar to richer representation
        self.energy_projector = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [E, ΔE, ΔΔE, E_norm]
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, energy_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energy_seq: (T,) or (B, T) — sequence of E(t) values

        Returns:
            energy_features: (T, hidden_dim) or (B, T, hidden_dim)
        """
        if energy_seq.dim() == 1:
            energy_seq = energy_seq.unsqueeze(0)  # (1, T)
            squeeze = True
        else:
            squeeze = False

        B, T = energy_seq.shape

        # ΔE(t) = E(t) - E(t-1)
        delta_e = torch.zeros_like(energy_seq)
        delta_e[:, 1:] = energy_seq[:, 1:] - energy_seq[:, :-1]

        # ΔΔE(t) = ΔE(t) - ΔE(t-1) — acceleration (paper Proposition 5)
        delta2_e = torch.zeros_like(energy_seq)
        delta2_e[:, 1:] = delta_e[:, 1:] - delta_e[:, :-1]

        # Normalized E within sequence
        e_min = energy_seq.min(dim=-1, keepdim=True).values
        e_max = energy_seq.max(dim=-1, keepdim=True).values
        e_norm = (energy_seq - e_min) / (e_max - e_min + 1e-8)

        # Stack: (B, T, 4)
        features = torch.stack([energy_seq, delta_e, delta2_e, e_norm], dim=-1)

        # Project: (B, T, hidden_dim)
        out = self.energy_projector(features)

        if squeeze:
            out = out.squeeze(0)  # (T, hidden_dim)

        return out


if __name__ == "__main__":
    N = 50

    energy_layer = EnergyLayer(lambda_init=1.0)
    fragility = torch.rand(N)          # F_i ∈ (0, 1)
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2            # symmetric
    adj.fill_diagonal_(0)

    E = energy_layer(fragility, adj)
    local, coupling = energy_layer.energy_components(fragility, adj)
    print(f"System Energy E(t) = {E.item():.4f}")
    print(f"  Local term:    {local.item():.4f}")
    print(f"  Coupling term: {coupling.item():.4f}")
    print(f"  λ (learned):   {energy_layer.lambda_.item():.4f}")

    # Batch test
    B, T = 4, 30
    energy_seq = torch.rand(B, T) * 10

    processor = EnergySequenceProcessor(hidden_dim=32)
    features = processor(energy_seq)
    print(f"\nEnergy sequence features: {features.shape}")  # (B, T, 32)
