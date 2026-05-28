import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    
    def __init__(self, node_feature_dim: int = 7, seq_len: int = 30, hidden: int = 128):
        super().__init__()
        self.seq_len = seq_len
        input_dim = node_feature_dim * seq_len
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
        self.tte_head = nn.Sequential(
            nn.Linear(hidden // 2, 16), nn.GELU(), nn.Linear(16, 1), nn.Softplus()
        )

    def forward(self, graph_sequence: List[Data]):
        T = len(graph_sequence)
        seq = torch.stack([g.x.mean(dim=0) for g in graph_sequence], dim=0)
        seq_flat = seq.flatten().unsqueeze(0)
        out = self.net(seq_flat)
        crash_prob = out.squeeze(-1)
        tte = torch.tensor([30.0], device=seq.device)
        instability = torch.tensor([0.0], device=seq.device)
        energy_seq = torch.zeros(T, device=seq.device)
        return crash_prob, tte, instability, energy_seq, []

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    
    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=node_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.crash_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus()
        )

    def forward(self, graph_sequence: List[Data]):
        seq = torch.stack([g.x.mean(dim=0) for g in graph_sequence], dim=0).unsqueeze(0)
        out, _ = self.lstm(seq)
        z = out[:, -1, :]
        crash_prob = self.crash_head(z).squeeze(-1)
        tte = self.tte_head(z).squeeze(-1)
        T = len(graph_sequence)
        device = seq.device
        instability = torch.tensor([0.0], device=device)
        energy_seq = torch.zeros(T, device=device)
        return crash_prob, tte, instability, energy_seq, []

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

class GNNOnlyBaseline(nn.Module):
    
    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        in_dim = node_feature_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // heads
            self.convs.append(
                GATConv(in_dim, out_dim, heads=heads,
                        dropout=dropout, concat=(i < num_layers - 1))
            )
            in_dim = out_dim * heads if i < num_layers - 1 else hidden_dim

        self.crash_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus()
        )
        self.dropout = dropout

    def forward(self, graph_sequence: List[Data]):
        g = graph_sequence[-1]
        x = g.x
        ei = g.edge_index
        T = len(graph_sequence)

        for conv in self.convs:
            x = conv(x, ei)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = x.mean(dim=0, keepdim=True)
        crash_prob = self.crash_head(h).squeeze(-1)
        tte = self.tte_head(h).squeeze(-1)
        device = x.device
        instability = torch.tensor([0.0], device=device)
        energy_seq = torch.zeros(T, device=device)
        return crash_prob, tte, instability, energy_seq, []

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

class GNNLSTMBaseline(nn.Module):
    
    def __init__(
        self,
        node_feature_dim: int = 7,
        gnn_hidden: int = 64,
        lstm_hidden: int = 128,
        num_gnn_layers: int = 2,
        num_lstm_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn_layers = nn.ModuleList()
        in_dim = node_feature_dim
        for i in range(num_gnn_layers):
            out_dim = gnn_hidden if i < num_gnn_layers - 1 else gnn_hidden // heads
            self.gnn_layers.append(
                GATConv(in_dim, out_dim, heads=heads,
                        dropout=dropout, concat=(i < num_gnn_layers - 1))
            )
            in_dim = out_dim * heads if i < num_gnn_layers - 1 else gnn_hidden

        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )
        self.crash_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.tte_head = nn.Sequential(
            nn.Linear(lstm_hidden, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus()
        )
        self.dropout = dropout

    def _gnn_forward(self, graph: Data) -> torch.Tensor:
        x = graph.x
        ei = graph.edge_index
        for conv in self.gnn_layers:
            x = conv(x, ei)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x.mean(dim=0)

    def forward(self, graph_sequence: List[Data]):
        T = len(graph_sequence)
        embeds = torch.stack([self._gnn_forward(g) for g in graph_sequence], dim=0)
        embeds = embeds.unsqueeze(0)
        out, _ = self.lstm(embeds)
        z = out[:, -1, :]
        crash_prob = self.crash_head(z).squeeze(-1)
        tte = self.tte_head(z).squeeze(-1)
        device = embeds.device
        instability = torch.tensor([0.0], device=device)
        energy_seq = torch.zeros(T, device=device)
        return crash_prob, tte, instability, energy_seq, []

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "MLP": MLPBaseline,
    "LSTM": LSTMBaseline,
    "GNN-Only": GNNOnlyBaseline,
    "GNN-LSTM": GNNLSTMBaseline,
}

def build_baseline(name: str, config: dict):
    
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINE_REGISTRY.keys())}")
    cls = BASELINE_REGISTRY[name]
    return cls(
        node_feature_dim=config.get("node_feature_dim", 7),
        hidden_dim=config.get("gnn_hidden_dim", 64),
    ) if name == "MLP" else cls(
        node_feature_dim=config.get("node_feature_dim", 7),
    )

if __name__ == "__main__":
    from torch_geometric.data import Data

    N, d, T = 50, 7, 30
    graphs = []
    for _ in range(T):
        ei = torch.randint(0, N, (2, 100))
        g = Data(x=torch.randn(N, d), edge_index=ei,
                 edge_attr=torch.rand(100, 1), num_nodes=N)
        g.adj = torch.eye(N)
        graphs.append(g)

    for name, cls in BASELINE_REGISTRY.items():
        model = cls(node_feature_dim=d)
        cp, tte, inst, E, _ = model(graphs)
        print(f"{name:12s} | crash_prob={cp.item():.4f} | tte={tte.item():.2f}")

