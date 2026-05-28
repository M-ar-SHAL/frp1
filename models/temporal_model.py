import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

class PositionalEncoding(nn.Module):

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerTemporalModel(nn.Module):

    def __init__(
        self,
        energy_feature_dim: int = 32,
        graph_embed_dim: int = 64,
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
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        energy_features: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        
        B, T, _ = energy_features.shape

        x = torch.cat([energy_features, graph_embeddings], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.output_norm(x)

        return x

class LSTMTemporalModel(nn.Module):

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
            bidirectional=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        energy_features: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        
        x = torch.cat([energy_features, graph_embeddings], dim=-1)
        out, _ = self.lstm(x)
        return self.output_norm(out)

if __name__ == "__main__":
    B, T = 8, 30
    E_dim, G_dim = 32, 64

    energy_feats = torch.randn(B, T, E_dim)
    graph_embs = torch.randn(B, T, G_dim)

    transformer_model = TransformerTemporalModel(
        energy_feature_dim=E_dim,
        graph_embed_dim=G_dim,
        d_model=128,
        nhead=4,
        num_layers=3
    )
    out_t = transformer_model(energy_feats, graph_embs)
    print(f"Transformer output: {out_t.shape}")

    lstm_model = LSTMTemporalModel(energy_feature_dim=E_dim, graph_embed_dim=G_dim)
    out_l = lstm_model(energy_feats, graph_embs)
    print(f"LSTM output: {out_l.shape}")

