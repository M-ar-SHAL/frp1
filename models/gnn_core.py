"""
GNN Core — Module 3 of FAPT-GNN

Implements the Fragility-Aware Graph Attention Network.

KEY NOVELTY: Standard attention is modified to inject fragility scores:
  α_ij = softmax(f(h_i, h_j) + η·F_j)

Nodes with higher fragility influence their neighbors more —
physically grounded (fragile nodes are more systemically important).

Architecture:
  Multi-head Graph Attention (GAT-based)
  + Fragility-weighted attention modification
  + Residual connections
  + Layer normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, add_self_loops, degree
from typing import Optional, Tuple


class FragilityAwareGATLayer(nn.Module):
    """
    Single Graph Attention layer with fragility injection.

    Novel attention mechanism:
      e_ij = LeakyReLU(a^T [W*h_i ‖ W*h_j]) + η * F_j
      α_ij = softmax_j(e_ij)

    This is Equation (Module 3) from the paper.
    
    NOTE: Implemented WITHOUT MessagePassing to avoid PyG code generation issues.
    Uses manual edge-wise attention computation instead.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        fragility_weight: float = 1.0,  # η in the paper
        edge_dim: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout_rate = dropout
        self.fragility_weight = nn.Parameter(torch.tensor(fragility_weight))
        self.negative_slope = negative_slope

        # Linear transformation W
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention vectors (one per head)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_dim))

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)

        # Bias
        self.bias = nn.Parameter(torch.Tensor(heads * out_dim if concat else out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,           # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
        fragility: torch.Tensor,   # (N,) — F_i(t) scores
        edge_attr: Optional[torch.Tensor] = None,  # (E, edge_dim)
    ) -> torch.Tensor:
        """
        Args:
            x         : node features (N, in_dim)
            edge_index: edge indices (2, E)
            fragility : scalar fragility per node (N,)
            edge_attr : edge weights from multi-layer graph (E, 1)

        Returns:
            h_out: updated node embeddings (N, heads*out_dim) if concat else (N, out_dim)
        """
        N = x.size(0)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        if edge_attr is not None:
            # Pad self-loop edges with weight 1.0
            self_loop_attr = torch.ones(N, edge_attr.size(-1), device=x.device)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Linear transform: (N, heads, out_dim)
        h = self.W(x).view(N, self.heads, self.out_dim)

        # Manual edge-wise attention (avoids PyG MessagePassing issues)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        E = src_idx.size(0)

        # Get node embeddings for each edge
        h_i = h[dst_idx]  # (E, heads, out_dim)
        h_j = h[src_idx]  # (E, heads, out_dim)

        # Compute attention scores per edge
        alpha = (h_i * self.att_src).sum(dim=-1) + (h_j * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # ★ KEY NOVELTY: inject fragility of source node j
        fragility_j = fragility[src_idx]  # (E,)
        fragility_bias = self.fragility_weight * fragility_j.unsqueeze(-1)
        alpha = alpha + fragility_bias

        # Add edge weight contribution
        if edge_attr is not None:
            edge_weight = self.edge_proj(edge_attr)  # (E, heads)
            alpha = alpha + edge_weight

        # Softmax attention per destination node
        alpha = softmax(alpha, dst_idx, num_nodes=N)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)

        # Weight messages: (E, heads, out_dim)
        weighted_msg = h_j * alpha.unsqueeze(-1)

        # Aggregate by destination node
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst_idx, weighted_msg)

        # Output reshape & bias
        if self.concat:
            out = out.view(N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out


class FragilityAwareGNN(nn.Module):
    """
    Full GNN core: stacked FragilityAwareGATLayers with residuals.

    Architecture:
      Input (N, in_dim)
        ↓ [Linear projection]
      (N, hidden_dim)
        ↓ [L layers of GAT + residual + norm]
      (N, hidden_dim)
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Stacked GAT layers with layer norm
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim // heads if heads > 1 else hidden_dim

            gat = FragilityAwareGATLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True,  # Always concatenate heads
            )
            self.gat_layers.append(gat)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (N, d)
        edge_index: torch.Tensor,  # (2, E)
        fragility: torch.Tensor,   # (N,)
        edge_attr: Optional[torch.Tensor] = None,  # (E, 1)
    ) -> torch.Tensor:
        """
        Args:
            x         : node features
            edge_index: edge connectivity
            fragility : per-node fragility scores F_i(t)
            edge_attr : multi-layer edge weights

        Returns:
            h: (N, hidden_dim) — updated node embeddings
        """
        h = self.input_proj(x)  # (N, hidden_dim)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h, edge_index, fragility, edge_attr)  # (N, hidden_dim)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)  # Residual connection

        return h
