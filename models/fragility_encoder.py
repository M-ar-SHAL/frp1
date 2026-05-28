import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FragilityEncoder(nn.Module):

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

        raw_input_dim = node_feature_dim
        self.raw_encoder = nn.Sequential(
            nn.LayerNorm(raw_input_dim),
            nn.Linear(raw_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        combined_dim = node_feature_dim + gnn_hidden_dim
        self.combined_encoder = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        self.fragility_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h_frag = self.raw_encoder(x)
        F_raw = self.fragility_head(h_frag).squeeze(-1)
        return h_frag, F_raw

    def forward(
        self,
        x: torch.Tensor,
        h_gnn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        combined = torch.cat([x, h_gnn], dim=-1)
        h_frag = self.combined_encoder(combined)
        F = self.fragility_head(h_frag).squeeze(-1)
        return h_frag, F

class FragilityInterpretability(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def forward(
        self,
        volatility: torch.Tensor,
        centrality: torch.Tensor,
        sentiment: torch.Tensor,
        liquidity: torch.Tensor,
    ) -> torch.Tensor:
        
        w = F.softmax(self.weights, dim=0)
        components = torch.stack([volatility, centrality, sentiment, liquidity], dim=1)
        F = (components * w).sum(dim=-1)
        return torch.sigmoid(F)

if __name__ == "__main__":
    N = 50
    d = 7

    encoder = FragilityEncoder(node_feature_dim=d, gnn_hidden_dim=64)
    x = torch.randn(N, d)
    h_gnn = torch.randn(N, 64)

    h_frag, F_raw = encoder.forward_raw(x)
    h_frag2, F = encoder(x, h_gnn)

    print(f"Raw Fragility F_raw: shape={F_raw.shape}, range=[{F_raw.min():.3f}, {F_raw.max():.3f}]")
    print(f"Refined Fragility F: shape={F.shape},   range=[{F.min():.3f}, {F.max():.3f}]")

    interp = FragilityInterpretability()
    F_interp = interp(x[:, 1], x[:, 2], x[:, 3], x[:, 4])
    print(f"Interpretable F: {F_interp.shape}, weights={F.softmax(-1)}")

