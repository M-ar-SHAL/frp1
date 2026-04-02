"""
Phase Transition Head — Module 6 of FAPT-GNN

Outputs three quantities:
  ŷ_t = σ(g_θ(E_{t-k:t}))  → Crash Probability ∈ [0, 1]
  τ̂_t = h_θ(E_{t-k:t})      → Time-to-Collapse (days)
  Î_t = E(t)                 → Instability Index (energy-based)

Also: Shock Simulator — Module 8 (highly novel)
  Simulate node failure / liquidity shock → measure system resilience
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PhaseTransitionHead(nn.Module):
    """
    Three-output prediction head operating on the temporal representation.

    Inputs: last-timestep temporal representation z from Transformer/LSTM.
    
    Outputs:
      crash_prob:     ŷ_t ∈ [0, 1] — crash probability (binary classification)
      time_to_crash:  τ̂_t ≥ 0     — days until crash (regression)
      instability:    Î_t ∈ [0, 1] — system instability index
    """

    def __init__(
        self,
        temporal_dim: int = 128,   # Transformer/LSTM output dim
        hidden_dim: int = 64,
        dropout: float = 0.15
    ):
        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ŷ_t: Crash Probability Head
        self.crash_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # τ̂_t: Time-to-Crash Head (regression: positive output)
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()  # ensures positive output (days > 0)
        )

        # Î_t: Instability Index Head
        self.instability_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        z: torch.Tensor,          # (B, temporal_dim) — last-step temporal repr
        energy: torch.Tensor,     # (B,) — current E(t) for interpretable instability
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            crash_prob   : (B,) ∈ [0, 1]
            time_to_crash: (B,) ≥ 0 (in trading days)
            instability  : (B,) ∈ [0, 1] (energy-derived)
        """
        shared = self.shared(z)

        crash_prob = self.crash_head(shared).squeeze(-1)    # (B,)
        time_to_crash = self.tte_head(shared).squeeze(-1)  # (B,)

        # Î_t = normalized E(t) combined with learned instability
        inst_learned = self.instability_head(shared).squeeze(-1)
        
        # Energy-based instability: normalize energy to [0, 1]
        # Use safe normalization: (E - E_min) / (E_max - E_min + eps)
        energy_detached = energy.detach()
        energy_detached = torch.nan_to_num(energy_detached, nan=0.0, posinf=1.0, neginf=0.0)
        energy_min = torch.min(energy_detached)
        energy_max = torch.max(energy_detached)
        energy_range = torch.clamp(energy_max - energy_min, min=1e-8)
        energy_norm = torch.sigmoid((energy - energy_min) / energy_range)
        
        instability = 0.5 * inst_learned + 0.5 * energy_norm  # (B,)

        # Ensure outputs are in valid ranges (numerical stability)
        crash_prob = torch.nan_to_num(crash_prob, nan=0.5, posinf=1-1e-7, neginf=1e-7)
        crash_prob = torch.clamp(crash_prob, min=1e-7, max=1-1e-7)
        time_to_crash = torch.nan_to_num(time_to_crash, nan=0.0, posinf=100.0, neginf=0.0)
        time_to_crash = torch.clamp(time_to_crash, min=0.0)
        instability = torch.nan_to_num(instability, nan=0.5, posinf=1.0, neginf=0.0)
        instability = torch.clamp(instability, min=0.0, max=1.0)

        return crash_prob, time_to_crash, instability


class ShockSimulator(nn.Module):
    """
    Module 8: Shock Simulation Engine (HIGHLY NOVEL)

    Simulates three types of financial shocks and measures system resilience:
      1. Node failure (bank/firm bankruptcy)
      2. Liquidity shock (sudden market withdrawal)
      3. Sentiment crash (panic/sudden fear spike)

    For each shock, we:
      1. Apply the shock to fragility vector F
      2. Recompute E(t) via energy layer
      3. Measure energy spike = ΔE_shock = E_shocked - E_baseline

    Resilience Score R = 1 - sigmoid(ΔE_shock / E_baseline)
    High resilience → small ΔE_shock → system is robust.
    Low resilience  → large ΔE_shock → system is fragile.

    This is useful for:
      - Stress testing (RBI/SEBI applications)
      - Identifying systemically important stocks (for paper's interpretability section)
    """

    def __init__(self, energy_layer, shock_intensity: float = 0.5):
        super().__init__()
        self.energy_layer = energy_layer
        self.shock_intensity = nn.Parameter(torch.tensor(shock_intensity))

    def simulate_node_failure(
        self,
        fragility: torch.Tensor,  # (N,)
        adj: torch.Tensor,        # (N, N)
        node_idx: int,
        failure_multiplier: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate failure of node `node_idx` (e.g., a major bank collapses).
        Sets fragility of that node to max and propagates via adjacency.
        
        Returns:
            (E_baseline, E_shocked): energy before and after shock
        """
        E_baseline = self.energy_layer(fragility, adj)

        # Apply shock: set failed node to max fragility (= 1.0)
        F_shocked = fragility.clone()
        F_shocked[node_idx] = 1.0

        # Propagate shock: neighbors of failed node get increased fragility
        neighbors = adj[node_idx]  # (N,) — edge weights to neighbors
        shock_propagation = failure_multiplier * neighbors * (1.0 - F_shocked)
        F_shocked = (F_shocked + shock_propagation).clamp(0, 1)

        E_shocked = self.energy_layer(F_shocked, adj)
        return E_baseline, E_shocked

    def simulate_liquidity_shock(
        self,
        fragility: torch.Tensor,  # (N,)
        adj: torch.Tensor,        # (N, N)
        shock_fraction: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate simultaneous liquidity withdrawal from all nodes
        (e.g., FII sudden outflow like COVID March 2020).
        shock_fraction: fraction of market experiencing stress.
        """
        E_baseline = self.energy_layer(fragility, adj)

        # Apply uniform fragility spike to all nodes
        intensity = self.shock_intensity.abs().clamp(0.1, 1.0) * shock_fraction
        F_shocked = (fragility + intensity).clamp(0, 1)

        E_shocked = self.energy_layer(F_shocked, adj)
        return E_baseline, E_shocked

    def simulate_sentiment_crash(
        self,
        fragility: torch.Tensor,  # (N,)
        adj: torch.Tensor,        # (N, N)
        panic_multiplier: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate panic/sentiment crash:
        All fragility scores amplified by panic_multiplier.
        """
        E_baseline = self.energy_layer(fragility, adj)
        F_shocked = (fragility * panic_multiplier).clamp(0, 1)
        E_shocked = self.energy_layer(F_shocked, adj)
        return E_baseline, E_shocked

    def compute_resilience_score(
        self,
        E_baseline: torch.Tensor,
        E_shocked: torch.Tensor,
    ) -> torch.Tensor:
        """
        R = 1 - sigmoid(ΔE / E_baseline)
        High R → resilient; Low R → fragile.
        """
        delta_E = E_shocked - E_baseline
        norm_delta = delta_E / (E_baseline.abs() + 1e-8)
        return 1.0 - torch.sigmoid(norm_delta)

    def run_all_shocks(
        self,
        fragility: torch.Tensor,   # (N,)
        adj: torch.Tensor,         # (N, N)
        tickers: Optional[list] = None
    ) -> dict:
        """
        Run all three shock simulations and return resilience scores.

        Returns dict with:
          - node_resilience: per-node resilience (N,)
          - liquidity_resilience: scalar
          - sentiment_resilience: scalar
        """
        results = {}

        # Node failure simulation for each node
        node_resiliences = []
        N = fragility.size(0)
        for idx in range(N):
            E_base, E_shock = self.simulate_node_failure(fragility, adj, idx)
            R = self.compute_resilience_score(E_base, E_shock)
            node_resiliences.append(R.item())
        results["node_resilience"] = torch.tensor(node_resiliences)

        # Liquidity shock
        E_base, E_shock = self.simulate_liquidity_shock(fragility, adj)
        results["liquidity_resilience"] = self.compute_resilience_score(E_base, E_shock)

        # Sentiment crash
        E_base, E_shock = self.simulate_sentiment_crash(fragility, adj)
        results["sentiment_resilience"] = self.compute_resilience_score(E_base, E_shock)

        # Identify most systemically important nodes (lowest resilience = SIFI)
        sr = results["node_resilience"]
        sifi_indices = torch.argsort(sr)[:5]
        if tickers:
            results["SIFI_stocks"] = [tickers[i] for i in sifi_indices.tolist()]
        results["SIFI_indices"] = sifi_indices

        return results


if __name__ == "__main__":
    from models.energy_layer import EnergyLayer

    N = 50
    fragility = torch.rand(N) * 0.4  # mild fragility
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    energy_layer = EnergyLayer()

    # Test Phase Head
    head = PhaseTransitionHead(temporal_dim=128)
    z = torch.randn(8, 128)  # batch of 8
    E = torch.rand(8) * 5.0
    crash_prob, tte, instability = head(z, E)
    print(f"Crash prob:  {crash_prob.shape} → {crash_prob[:3].detach().numpy().round(3)}")
    print(f"Time to crash: {tte[:3].detach().numpy().round(2)} days")
    print(f"Instability: {instability[:3].detach().numpy().round(3)}")

    # Test Shock Simulator
    sim = ShockSimulator(energy_layer)
    results = sim.run_all_shocks(fragility, adj)
    print(f"\nShock Simulation:")
    print(f"  Node resilience range: [{results['node_resilience'].min():.3f}, {results['node_resilience'].max():.3f}]")
    print(f"  Liquidity resilience:  {results['liquidity_resilience'].item():.3f}")
    print(f"  Sentiment resilience:  {results['sentiment_resilience'].item():.3f}")
