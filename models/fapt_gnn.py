import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from torch_geometric.data import Data, Batch

from models.fragility_encoder import FragilityEncoder
from models.gnn_core import FragilityAwareGNN
from models.energy_layer import EnergyLayer, EnergySequenceProcessor
from models.temporal_model import TransformerTemporalModel, LSTMTemporalModel
from models.phase_head import PhaseTransitionHead, ShockSimulator

class FAPT_GNN(nn.Module):

    def __init__(
        self,
        node_feature_dim: int = 7,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 2,
        gnn_heads: int = 4,
        energy_hidden_dim: int = 32,
        temporal_d_model: int = 128,
        temporal_num_layers: int = 3,
        temporal_nhead: int = 4,
        seq_len: int = 30,
        dropout: float = 0.2,
        use_transformer: bool = True,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.seq_len = seq_len

        # ── Module 2: Fragility Encoder ──────────────────────────────
        self.fragility_encoder = FragilityEncoder(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            hidden_dim=128,
            dropout=dropout,
        )

        # ── Module 3: Fragility-Aware GNN ────────────────────────────
        self.gnn = FragilityAwareGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            heads=gnn_heads,
            dropout=dropout,
            edge_dim=1,
        )

        # ── Module 4: Energy Computation Layer ───────────────────────
        self.energy_layer = EnergyLayer(lambda_init=1.0)
        self.energy_processor = EnergySequenceProcessor(hidden_dim=energy_hidden_dim)

        # ── Module 5: Temporal Transition Detector ───────────────────
        if use_transformer:
            self.temporal_model = TransformerTemporalModel(
                energy_feature_dim=energy_hidden_dim,
                graph_embed_dim=gnn_hidden_dim,
                d_model=temporal_d_model,
                nhead=temporal_nhead,
                num_layers=temporal_num_layers,
                dim_feedforward=temporal_d_model * 2,
                dropout=dropout,
                max_seq_len=seq_len + 10,
            )
            temporal_out_dim = temporal_d_model
        else:
            self.temporal_model = LSTMTemporalModel(
                energy_feature_dim=energy_hidden_dim,
                graph_embed_dim=gnn_hidden_dim,
                hidden_dim=temporal_d_model,
                num_layers=temporal_num_layers,
                dropout=dropout,
            )
            temporal_out_dim = temporal_d_model

        # ── Module 6: Phase Transition Head ──────────────────────────
        self.phase_head = PhaseTransitionHead(
            temporal_dim=temporal_out_dim,
            hidden_dim=64,
            dropout=dropout,
        )

        # ── Module 8: Shock Simulator ─────────────────────────────────
        self.shock_simulator = ShockSimulator(self.energy_layer)

        # ── Graph mean-pooling ────────────────────────────────────────
        self.graph_pool_proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.GELU()
        )

        print(f"[FAPT-GNN] Initialized with {self._count_params():,} parameters")

    def _count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def process_single_graph(
        self,
        graph: Data,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        adj = graph.adj if hasattr(graph, 'adj') else None

        _, F_raw = self.fragility_encoder.forward_raw(x)

        h_gnn = self.gnn(x, edge_index, F_raw, edge_attr)

        _, F = self.fragility_encoder(x, h_gnn)

        h_graph = self.graph_pool_proj(h_gnn.mean(dim=0))

        if adj is not None:
            E = self.energy_layer(F, adj)
        else:
            N = x.size(0)
            adj_approx = torch.zeros(N, N, device=x.device)
            if edge_attr is not None:
                adj_approx[edge_index[0], edge_index[1]] = edge_attr.squeeze(-1)
            else:
                adj_approx[edge_index[0], edge_index[1]] = 1.0
            E = self.energy_layer(F, adj_approx)

        return F, h_graph, E

    def forward(
        self,
        graph_sequence: List[Data],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        
        T = len(graph_sequence)

        energy_values = []
        graph_embeddings = []
        fragility_seq = []

        for t, graph in enumerate(graph_sequence):
            F, h_graph, E = self.process_single_graph(graph)
            energy_values.append(E)
            graph_embeddings.append(h_graph)
            fragility_seq.append(F)

        energy_seq_tensor = torch.stack(energy_values, dim=0)
        graph_embed_seq = torch.stack(graph_embeddings, dim=0)

        energy_seq_batch = energy_seq_tensor.unsqueeze(0)
        graph_embed_batch = graph_embed_seq.unsqueeze(0)

        energy_features = self.energy_processor(energy_seq_batch)

        temporal_out = self.temporal_model(energy_features, graph_embed_batch)

        z = temporal_out[:, -1, :]
        E_current = energy_seq_tensor[-1].unsqueeze(0)

        crash_prob, time_to_crash, instability = self.phase_head(z, E_current)

        return crash_prob, time_to_crash, instability, energy_seq_tensor, fragility_seq

    def predict(
        self,
        graph_sequence: List[Data],
        return_shock_analysis: bool = False,
        tickers: Optional[list] = None,
    ) -> Dict:
        
        self.eval()
        with torch.no_grad():
            crash_prob, tte, instability, energy_seq, fragility_seq = self(graph_sequence)

            result = {
                "crash_probability": crash_prob.item(),
                "time_to_crash_days": tte.item(),
                "instability_index": instability.item(),
                "system_energy": energy_seq.tolist(),
                "fragility_per_stock": fragility_seq[-1].tolist(),
                "energy_acceleration": (energy_seq[-1] - energy_seq[-2]).item() if len(energy_seq) > 1 else 0.0,
            }

            if return_shock_analysis:
                last_graph = graph_sequence[-1]
                F_last = fragility_seq[-1]
                adj_last = last_graph.adj if hasattr(last_graph, 'adj') else None

                if adj_last is not None:
                    shock_results = self.shock_simulator.run_all_shocks(F_last, adj_last, tickers)
                    result["shock_analysis"] = {
                        "node_resilience": shock_results["node_resilience"].tolist(),
                        "liquidity_resilience": shock_results["liquidity_resilience"].item(),
                        "sentiment_resilience": shock_results["sentiment_resilience"].item(),
                        "SIFI_stocks": shock_results.get("SIFI_stocks", []),
                    }

        return result

def build_model(config: dict) -> FAPT_GNN:
    
    return FAPT_GNN(
        node_feature_dim=config.get("node_feature_dim", 7),
        gnn_hidden_dim=config.get("gnn_hidden_dim", 64),
        gnn_num_layers=config.get("gnn_num_layers", 2),
        gnn_heads=config.get("gnn_heads", 4),
        energy_hidden_dim=config.get("energy_hidden_dim", 32),
        temporal_d_model=config.get("temporal_d_model", 128),
        temporal_num_layers=config.get("temporal_num_layers", 3),
        temporal_nhead=config.get("temporal_nhead", 4),
        seq_len=config.get("seq_len", 30),
        dropout=config.get("dropout", 0.2),
        use_transformer=config.get("use_transformer", True),
    )

if __name__ == "__main__":
    from torch_geometric.data import Data

    N, d, T = 50, 7, 30
    graph_seq = []
    for t in range(T):
        E_fake = 100
        edge_index = torch.randint(0, N, (2, E_fake))
        edge_attr = torch.rand(E_fake, 1)
        x = torch.randn(N, d)
        adj = torch.rand(N, N)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)
        adj = adj / (adj.sum(-1, keepdim=True) + 1e-8)

        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g.adj = adj
        graph_seq.append(g)

    model = FAPT_GNN(node_feature_dim=d, seq_len=T)

    crash_prob, tte, instability, energy_seq, fragility_seq = model(graph_seq)
    print(f"\n{'='*50}")
    print(f"FAPT-GNN Output:")
    print(f"  Crash Probability : {crash_prob.item():.4f}")
    print(f"  Time to Crash     : {tte.item():.2f} days")
    print(f"  Instability Index : {instability.item():.4f}")
    print(f"  Energy sequence   : len={len(energy_seq)}, last={energy_seq[-1].item():.4f}")
    print(f"  Fragility (last t): min={fragility_seq[-1].min():.3f}, max={fragility_seq[-1].max():.3f}")
    print(f"{'='*50}")

