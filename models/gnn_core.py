import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, add_self_loops, degree
from typing import Optional, Tuple

class FragilityAwareGATLayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        fragility_weight: float = 1.0,
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

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_dim))

        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)

        self.bias = nn.Parameter(torch.Tensor(heads * out_dim if concat else out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        fragility: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        N = x.size(0)

        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        if edge_attr is not None:
            self_loop_attr = torch.ones(N, edge_attr.size(-1), device=x.device)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        h = self.W(x).view(N, self.heads, self.out_dim)

        src_idx, dst_idx = edge_index[0], edge_index[1]
        E = src_idx.size(0)

        h_i = h[dst_idx]
        h_j = h[src_idx]

        alpha = (h_i * self.att_src).sum(dim=-1) + (h_j * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        fragility_j = fragility[src_idx]
        fragility_bias = self.fragility_weight * fragility_j.unsqueeze(-1)
        alpha = alpha + fragility_bias

        if edge_attr is not None:
            edge_weight = self.edge_proj(edge_attr)
            alpha = alpha + edge_weight

        alpha = softmax(alpha, dst_idx, num_nodes=N)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)

        weighted_msg = h_j * alpha.unsqueeze(-1)

        out = torch.zeros(N, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst_idx, weighted_msg)

        if self.concat:
            out = out.view(N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out

class FragilityAwareGNN(nn.Module):

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

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

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
                concat=True,
            )
            self.gat_layers.append(gat)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        fragility: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        h = self.input_proj(x)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h, edge_index, fragility, edge_attr)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)

        return h

