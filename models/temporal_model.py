"""
Temporal Transition Detector — Module 5 of FAPT-GNN

Processes the sequence of energy values E(t-k:t) and graph embeddings
over time to detect phase transition precursors.

Input:  sequence of [E(t), graph_embedding(t)] for t = t-k...t
Output: instability trajectory → fed to Phase Transition Head

Architecture:
  Option A (default): Transformer Encoder (best for long sequences)
  Option B:           Bidirectional LSTM (simpler, faster)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTemporalModel(nn.Module):
    """
    Transformer-based temporal model for energy sequence modeling.

    Input per timestep: [E_features(t) ‖ graph_embedding(t)]
    where graph_embedding(t) is the mean-pooled GNN output.

    Architecture:
      Input → Linear projection → Positional Encoding
      → Transformer Encoder (N layers, causal mask)
      → CLS token output → rich temporal representation

    The causal mask ensures no future information leakage (important for financial prediction).
    """

    def __init__(
        self,
        energy_feature_dim: int = 32,   # from EnergySequenceProcessor
        graph_embed_dim: int = 64,      # mean-pooled GNN hidden dim
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 100,
    ):
        super().__init__()

        input_dim = energy_feature_dim + graph_embed_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        energy_features: torch.Tensor,   # (B, T, energy_feature_dim)
        graph_embeddings: torch.Tensor,  # (B, T, graph_embed_dim)
    ) -> torch.Tensor:
        """
        Args:
            energy_features : energy sequence features (B, T, E_dim)
            graph_embeddings: time-varying graph embeddings (B, T, G_dim)

        Returns:
            temporal_repr: (B, T, d_model) — temporal representations
                           Use temporal_repr[:, -1, :] for prediction (last timestep)
        """
        B, T, _ = energy_features.shape

        # Concatenate energy + graph features
        x = torch.cat([energy_features, graph_embeddings], dim=-1)  # (B, T, E+G)
        x = self.input_proj(x)      # (B, T, d_model)
        x = self.pos_encoding(x)    # (B, T, d_model)

        # Causal mask: upper triangular masked to prevent future leakage
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.output_norm(x)

        return x  # (B, T, d_model)


class LSTMTemporalModel(nn.Module):
    """
    Simpler LSTM-based temporal model.
    Use as ablation baseline vs Transformer.
    """

    def __init__(
        self,
        energy_feature_dim: int = 32,
        graph_embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_dim = energy_feature_dim + graph_embed_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # causal (no future leakage)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        energy_features: torch.Tensor,    # (B, T, E_dim)
        graph_embeddings: torch.Tensor,   # (B, T, G_dim)
    ) -> torch.Tensor:
        """Returns: (B, T, hidden_dim)"""
        x = torch.cat([energy_features, graph_embeddings], dim=-1)
        out, _ = self.lstm(x)
        return self.output_norm(out)


if __name__ == "__main__":
    B, T = 8, 30
    E_dim, G_dim = 32, 64

    energy_feats = torch.randn(B, T, E_dim)
    graph_embs = torch.randn(B, T, G_dim)

    # Transformer
    transformer_model = TransformerTemporalModel(
        energy_feature_dim=E_dim,
        graph_embed_dim=G_dim,
        d_model=128,
        nhead=4,
        num_layers=3
    )
    out_t = transformer_model(energy_feats, graph_embs)
    print(f"Transformer output: {out_t.shape}")  # (B, T, 128)

    # LSTM
    lstm_model = LSTMTemporalModel(energy_feature_dim=E_dim, graph_embed_dim=G_dim)
    out_l = lstm_model(energy_feats, graph_embs)
    print(f"LSTM output: {out_l.shape}")  # (B, T, 128)
