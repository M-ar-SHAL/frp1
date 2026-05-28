import torch
import torch.nn as nn
from typing import Optional, Tuple

class EnergyLayer(nn.Module):

    def __init__(self, lambda_init: float = 1.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(lambda_init).log())

    @property
    def lambda_(self) -> torch.Tensor:
        
        return self.log_lambda.exp()

    def forward(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        
        lam = self.lambda_

        if fragility.dim() == 1:
            N = fragility.size(0)
            I = torch.eye(N, device=fragility.device)
            M = I + lam * adj
            energy = fragility @ M @ fragility
        else:
            B, N = fragility.shape
            I = torch.eye(N, device=fragility.device).unsqueeze(0).expand(B, -1, -1)
            M = I + lam * adj
            F_unsq = fragility.unsqueeze(-1)
            energy = torch.bmm(torch.bmm(F_unsq.transpose(-1, -2), M), F_unsq).squeeze(-1).squeeze(-1)

        energy = torch.nan_to_num(energy, nan=0.0, posinf=1.0, neginf=0.0)
        return energy

    def energy_components(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        local = (fragility ** 2).sum()
        coupling = self.lambda_ * (fragility @ adj @ fragility)
        return local, coupling

class EnergySequenceProcessor(nn.Module):

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.energy_projector = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, energy_seq: torch.Tensor) -> torch.Tensor:
        
        if energy_seq.dim() == 1:
            energy_seq = energy_seq.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T = energy_seq.shape

        delta_e = torch.zeros_like(energy_seq)
        delta_e[:, 1:] = energy_seq[:, 1:] - energy_seq[:, :-1]

        delta2_e = torch.zeros_like(energy_seq)
        delta2_e[:, 1:] = delta_e[:, 1:] - delta_e[:, :-1]

        e_min = energy_seq.min(dim=-1, keepdim=True).values
        e_max = energy_seq.max(dim=-1, keepdim=True).values
        e_norm = (energy_seq - e_min) / (e_max - e_min + 1e-8)

        features = torch.stack([energy_seq, delta_e, delta2_e, e_norm], dim=-1)

        out = self.energy_projector(features)

        if squeeze:
            out = out.squeeze(0)

        return out

if __name__ == "__main__":
    N = 50

    energy_layer = EnergyLayer(lambda_init=1.0)
    fragility = torch.rand(N)
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)

    E = energy_layer(fragility, adj)
    local, coupling = energy_layer.energy_components(fragility, adj)
    print(f"System Energy E(t) = {E.item():.4f}")
    print(f"  Local term:    {local.item():.4f}")
    print(f"  Coupling term: {coupling.item():.4f}")
    print(f"  λ (learned):   {energy_layer.lambda_.item():.4f}")

    B, T = 4, 30
    energy_seq = torch.rand(B, T) * 10

    processor = EnergySequenceProcessor(hidden_dim=32)
    features = processor(energy_seq)
    print(f"\nEnergy sequence features: {features.shape}")

