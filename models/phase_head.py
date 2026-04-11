import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PhaseTransitionHead(nn.Module):

    def __init__(
        self,
        temporal_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.15
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.crash_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        self.instability_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        z: torch.Tensor,
        energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        shared = self.shared(z)

        crash_prob = self.crash_head(shared).squeeze(-1)
        time_to_crash = self.tte_head(shared).squeeze(-1)

        inst_learned = self.instability_head(shared).squeeze(-1)
        
        energy_detached = energy.detach()
        energy_detached = torch.nan_to_num(energy_detached, nan=0.0, posinf=1.0, neginf=0.0)
        energy_min = torch.min(energy_detached)
        energy_max = torch.max(energy_detached)
        energy_range = torch.clamp(energy_max - energy_min, min=1e-8)
        energy_norm = torch.sigmoid((energy - energy_min) / energy_range)
        
        instability = 0.5 * inst_learned + 0.5 * energy_norm

        crash_prob = torch.nan_to_num(crash_prob, nan=0.5, posinf=1-1e-7, neginf=1e-7)
        crash_prob = torch.clamp(crash_prob, min=1e-7, max=1-1e-7)
        time_to_crash = torch.nan_to_num(time_to_crash, nan=0.0, posinf=100.0, neginf=0.0)
        time_to_crash = torch.clamp(time_to_crash, min=0.0)
        instability = torch.nan_to_num(instability, nan=0.5, posinf=1.0, neginf=0.0)
        instability = torch.clamp(instability, min=0.0, max=1.0)

        return crash_prob, time_to_crash, instability

class ShockSimulator(nn.Module):

    def __init__(self, energy_layer, shock_intensity: float = 0.5):
        super().__init__()
        self.energy_layer = energy_layer
        self.shock_intensity = nn.Parameter(torch.tensor(shock_intensity))

    def simulate_node_failure(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
        node_idx: int,
        failure_multiplier: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        E_baseline = self.energy_layer(fragility, adj)

        F_shocked = fragility.clone()
        F_shocked[node_idx] = 1.0

        neighbors = adj[node_idx]
        shock_propagation = failure_multiplier * neighbors * (1.0 - F_shocked)
        F_shocked = (F_shocked + shock_propagation).clamp(0, 1)

        E_shocked = self.energy_layer(F_shocked, adj)
        return E_baseline, E_shocked

    def simulate_liquidity_shock(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
        shock_fraction: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        E_baseline = self.energy_layer(fragility, adj)

        intensity = self.shock_intensity.abs().clamp(0.1, 1.0) * shock_fraction
        F_shocked = (fragility + intensity).clamp(0, 1)

        E_shocked = self.energy_layer(F_shocked, adj)
        return E_baseline, E_shocked

    def simulate_sentiment_crash(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
        panic_multiplier: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        E_baseline = self.energy_layer(fragility, adj)
        F_shocked = (fragility * panic_multiplier).clamp(0, 1)
        E_shocked = self.energy_layer(F_shocked, adj)
        return E_baseline, E_shocked

    def compute_resilience_score(
        self,
        E_baseline: torch.Tensor,
        E_shocked: torch.Tensor,
    ) -> torch.Tensor:
        
        delta_E = E_shocked - E_baseline
        norm_delta = delta_E / (E_baseline.abs() + 1e-8)
        return 1.0 - torch.sigmoid(norm_delta)

    def run_all_shocks(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
        tickers: Optional[list] = None
    ) -> dict:
        
        results = {}

        node_resiliences = []
        N = fragility.size(0)
        for idx in range(N):
            E_base, E_shock = self.simulate_node_failure(fragility, adj, idx)
            R = self.compute_resilience_score(E_base, E_shock)
            node_resiliences.append(R.item())
        results["node_resilience"] = torch.tensor(node_resiliences)

        E_base, E_shock = self.simulate_liquidity_shock(fragility, adj)
        results["liquidity_resilience"] = self.compute_resilience_score(E_base, E_shock)

        E_base, E_shock = self.simulate_sentiment_crash(fragility, adj)
        results["sentiment_resilience"] = self.compute_resilience_score(E_base, E_shock)

        sr = results["node_resilience"]
        sifi_indices = torch.argsort(sr)[:5]
        if tickers:
            results["SIFI_stocks"] = [tickers[i] for i in sifi_indices.tolist()]
        results["SIFI_indices"] = sifi_indices

        return results

if __name__ == "__main__":
    from models.energy_layer import EnergyLayer

    N = 50
    fragility = torch.rand(N) * 0.4
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    energy_layer = EnergyLayer()

    head = PhaseTransitionHead(temporal_dim=128)
    z = torch.randn(8, 128)
    E = torch.rand(8) * 5.0
    crash_prob, tte, instability = head(z, E)
    print(f"Crash prob:  {crash_prob.shape} → {crash_prob[:3].detach().numpy().round(3)}")
    print(f"Time to crash: {tte[:3].detach().numpy().round(2)} days")
    print(f"Instability: {instability[:3].detach().numpy().round(3)}")

    sim = ShockSimulator(energy_layer)
    results = sim.run_all_shocks(fragility, adj)
    print(f"\nShock Simulation:")
    print(f"  Node resilience range: [{results['node_resilience'].min():.3f}, {results['node_resilience'].max():.3f}]")
    print(f"  Liquidity resilience:  {results['liquidity_resilience'].item():.3f}")
    print(f"  Sentiment resilience:  {results['sentiment_resilience'].item():.3f}")

