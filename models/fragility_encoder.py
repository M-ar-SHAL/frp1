"""
Fragility Encoder — Module 2 of FAPT-GNN

Computes F_i(t): node-level latent fragility from raw features.

Formula from paper:
  h_i^frag = MLP([σ_i, C_i, S_i, L_i])
  F_i = FragilityHead(h_i^frag)

Extended form:
  F_i(t) = σ(W_f · [x_i(t) ‖ h_i(t)] + b_f)

Where h_i(t) is the GNN embedding (passed in from GNN core).
The Fragility Encoder runs BEFORE the attention-weighted GNN message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FragilityEncoder(nn.Module):
    """
    Encodes node features into a scalar fragility score F_i ∈ [0, 1].

    Architecture:
      Input → LayerNorm → MLP(d→hidden→hidden/2) → Linear(1) → Sigmoid

    Input features per node:
      [return, volatility, centrality, sentiment, liquidity, momentum, drawdown]
      → d = node_feature_dim

    After GNN pass:
      h_i^frag = MLP([x_i ‖ h_i^gnn])
      F_i = Sigmoid(FragilityHead(h_i^frag))
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        gnn_hidden_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        # Input: raw features only (used before GNN, for fragility-weighted attention)
        raw_input_dim = node_feature_dim
        self.raw_encoder = nn.Sequential(
            nn.LayerNorm(raw_input_dim),
            nn.Linear(raw_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Input: raw features + GNN embedding (post-GNN fragility refinement)
        combined_dim = node_feature_dim + gnn_hidden_dim
        self.combined_encoder = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Fragility head: outputs scalar fragility per node
        self.fragility_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()   # F_i ∈ [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fragility from RAW features only (before GNN pass).
        Used to inject fragility into GNN attention as initial prior.

        Args:
            x: (N, node_feature_dim) node features

        Returns:
            h_frag: (N, hidden//2) fragility embedding
            F_raw:  (N,) scalar fragility scores (initial estimate)
        """
        h_frag = self.raw_encoder(x)
        F_raw = self.fragility_head(h_frag).squeeze(-1)
        return h_frag, F_raw

    def forward(
        self,
        x: torch.Tensor,
        h_gnn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refined fragility computation using GNN embedding.

        Args:
            x    : (N, node_feature_dim) raw node features
            h_gnn: (N, gnn_hidden_dim) GNN node embeddings

        Returns:
            h_frag: (N, hidden//2) fragility embedding
            F:      (N,) refined scalar fragility scores F_i(t)
        """
        combined = torch.cat([x, h_gnn], dim=-1)  # [x_i ‖ h_i^gnn]
        h_frag = self.combined_encoder(combined)
        F = self.fragility_head(h_frag).squeeze(-1)
        return h_frag, F


class FragilityInterpretability(nn.Module):
    """
    Decomposes fragility F_i into its 4 components:
      σ_i (volatility), C_i (centrality), S_i (sentiment), L_i (liquidity)

    Useful for ablation studies and paper interpretability analysis.
    Each component gets a learned weight: F_i = α·σ_i + β·C_i + γ·S_i + δ·L_i
    """

    def __init__(self):
        super().__init__()
        # Learnable weights for each fragility component
        self.weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def forward(
        self,
        volatility: torch.Tensor,   # (N,) σ_i
        centrality: torch.Tensor,   # (N,) C_i
        sentiment: torch.Tensor,    # (N,) S_i
        liquidity: torch.Tensor,    # (N,) L_i
    ) -> torch.Tensor:
        """
        Returns interpretable fragility: F_i = softmax_weights · [σ,C,S,L]
        """
        w = F.softmax(self.weights, dim=0)  # ensure sum=1
        components = torch.stack([volatility, centrality, sentiment, liquidity], dim=1)  # (N, 4)
        F = (components * w).sum(dim=-1)  # (N,)
        return torch.sigmoid(F)


if __name__ == "__main__":
    # Quick test
    N = 50  # 50 NIFTY stocks
    d = 7   # 7 features per stock

    encoder = FragilityEncoder(node_feature_dim=d, gnn_hidden_dim=64)
    x = torch.randn(N, d)
    h_gnn = torch.randn(N, 64)

    h_frag, F_raw = encoder.forward_raw(x)
    h_frag2, F = encoder(x, h_gnn)

    print(f"Raw Fragility F_raw: shape={F_raw.shape}, range=[{F_raw.min():.3f}, {F_raw.max():.3f}]")
    print(f"Refined Fragility F: shape={F.shape},   range=[{F.min():.3f}, {F.max():.3f}]")

    # Interpretable version
    interp = FragilityInterpretability()
    F_interp = interp(x[:, 1], x[:, 2], x[:, 3], x[:, 4])
    print(f"Interpretable F: {F_interp.shape}, weights={F.softmax(-1)}")
