import os
import sys
import json
import argparse
import copy
import time
import warnings

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from data.data_pipeline import load_all_data
from data.gdelt_sentiment import load_or_build_sentiment
from data.feature_engineering import build_all_features, build_node_feature_matrix, compute_returns
from data.graph_builder import build_graph_sequence, build_correlation_graph, build_pyg_graph
from data.crash_labeler import create_labels
from models.fapt_gnn import FAPT_GNN, build_model
from models.baselines import MLPBaseline, LSTMBaseline, GNNOnlyBaseline, GNNLSTMBaseline
from training.losses import FAPTGNNLoss
from training.trainer import (
    build_sliding_window_dataset, walk_forward_split,
    compute_pos_weight, train, eval_epoch
)
from training.evaluate import Evaluator, print_evaluation_report
import yaml

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

class FAPT_GNN_NoFragility(FAPT_GNN):
    
    def process_single_graph(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        N = x.size(0)
        F_raw = torch.full((N,), 0.5, device=x.device)

        h_gnn = self.gnn(x, edge_index, F_raw, edge_attr)

        _, F = self.fragility_encoder(x, h_gnn)
        h_graph = self.graph_pool_proj(h_gnn.mean(dim=0))

        adj = graph.adj if hasattr(graph, 'adj') else None
        if adj is not None:
            E = self.energy_layer(F, adj)
        else:
            adjm = torch.zeros(N, N, device=x.device)
            if edge_attr is not None:
                adjm[edge_index[0], edge_index[1]] = edge_attr.squeeze(-1)
            E = self.energy_layer(F, adjm)

        return F, h_graph, E

class FAPT_GNN_NoEnergy(FAPT_GNN):
    
    def process_single_graph(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        _, F_raw = self.fragility_encoder.forward_raw(x)
        h_gnn = self.gnn(x, edge_index, F_raw, edge_attr)
        _, F = self.fragility_encoder(x, h_gnn)
        h_graph = self.graph_pool_proj(h_gnn.mean(dim=0))

        E = torch.tensor(0.0, device=x.device)
        return F, h_graph, E

class FAPT_GNN_CorrOnlyGraph(FAPT_GNN):
    
    pass

def build_correlation_only_graphs(graphs):
    
    from torch_geometric.data import Data
    new_graphs = []
    for g in graphs:
        edge_attr = g.edge_attr
        if edge_attr is not None:
            weights = edge_attr.squeeze(-1)
            threshold = weights.median()
            mask = weights >= threshold
            new_edge_index = g.edge_index[:, mask]
            new_edge_attr = edge_attr[mask]
        else:
            new_edge_index = g.edge_index
            new_edge_attr = g.edge_attr

        new_g = Data(
            x=g.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            num_nodes=g.num_nodes
        )
        if hasattr(g, 'adj'):
            adj = g.adj.clone()
            adj_mean = adj.mean()
            adj = torch.where(adj >= adj_mean, adj, torch.zeros_like(adj))
            row_sums = adj.sum(1, keepdim=True).clamp(min=1e-8)
            new_g.adj = adj / row_sums
        if hasattr(g, 'date'):
            new_g.date = g.date
        new_graphs.append(new_g)
    return new_graphs

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

class SimpleCrashLoss(torch.nn.Module):
    
    def __init__(self, pos_weight: float = 9.0, use_focal: bool = True):
        super().__init__()
        from training.losses import CrashClassificationLoss
        self.cls_loss = CrashClassificationLoss(pos_weight=pos_weight, use_focal=use_focal)

    def forward(self, crash_prob, time_to_crash_pred, energy_seq,
                fragility_seq, adj, crash_label, time_to_crash_true, energy_proxy):
        L_cls = self.cls_loss(crash_prob, crash_label)
        loss_dict = {
            "total": L_cls.item(),
            "cls": L_cls.item(),
            "time": 0.0, "energy": 0.0, "smooth": 0.0, "temporal": 0.0
        }
        return L_cls, loss_dict

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def get_ablation_variants(config: dict, pos_weight: float, graphs_full, graphs_corr_only):
    
    model_cfg = config["model"]
    loss_cfg = config["loss"]

    full_criterion = FAPTGNNLoss(
        alpha=loss_cfg["alpha"], beta=loss_cfg["beta"],
        gamma=loss_cfg["gamma"], delta=loss_cfg["delta"],
        eta=loss_cfg["eta"], pos_weight=pos_weight,
        use_focal=loss_cfg["use_focal"]
    )
    simple_criterion = SimpleCrashLoss(pos_weight=pos_weight, use_focal=loss_cfg["use_focal"])

    variants = [
        {
            "name": "B1: MLP",
            "model": MLPBaseline(
                node_feature_dim=model_cfg["node_feature_dim"],
                seq_len=model_cfg["seq_len"]
            ),
            "criterion": simple_criterion,
            "graphs": graphs_full,
            "description": "No graph, no temporal",
        },
        {
            "name": "B2: LSTM",
            "model": LSTMBaseline(
                node_feature_dim=model_cfg["node_feature_dim"]
            ),
            "criterion": simple_criterion,
            "graphs": graphs_full,
            "description": "Temporal only (no graph)",
        },
        {
            "name": "B3: GNN-Only",
            "model": GNNOnlyBaseline(
                node_feature_dim=model_cfg["node_feature_dim"],
                hidden_dim=model_cfg["gnn_hidden_dim"],
                num_layers=model_cfg["gnn_num_layers"],
                heads=model_cfg["gnn_heads"]
            ),
            "criterion": simple_criterion,
            "graphs": graphs_full,
            "description": "Graph only (no temporal, no energy)",
        },
        {
            "name": "B4: GNN-LSTM",
            "model": GNNLSTMBaseline(
                node_feature_dim=model_cfg["node_feature_dim"],
                gnn_hidden=model_cfg["gnn_hidden_dim"],
                lstm_hidden=model_cfg["temporal_d_model"],
                num_gnn_layers=model_cfg["gnn_num_layers"],
                heads=model_cfg["gnn_heads"]
            ),
            "criterion": simple_criterion,
            "graphs": graphs_full,
            "description": "GNN + LSTM (no fragility, no energy)",
        },
        {
            "name": "B5: FAPT-GNN w/o Fragility",
            "model": FAPT_GNN_NoFragility(
                node_feature_dim=model_cfg["node_feature_dim"],
                gnn_hidden_dim=model_cfg["gnn_hidden_dim"],
                gnn_num_layers=model_cfg["gnn_num_layers"],
                gnn_heads=model_cfg["gnn_heads"],
                energy_hidden_dim=model_cfg["energy_hidden_dim"],
                temporal_d_model=model_cfg["temporal_d_model"],
                temporal_num_layers=model_cfg["temporal_num_layers"],
                temporal_nhead=model_cfg["temporal_nhead"],
                seq_len=model_cfg["seq_len"],
                dropout=model_cfg["dropout"],
                use_transformer=model_cfg["use_transformer"],
            ),
            "criterion": full_criterion,
            "graphs": graphs_full,
            "description": "Full model minus fragility encoder (uniform F_i)",
        },
        {
            "name": "B6: FAPT-GNN w/o Energy",
            "model": FAPT_GNN_NoEnergy(
                node_feature_dim=model_cfg["node_feature_dim"],
                gnn_hidden_dim=model_cfg["gnn_hidden_dim"],
                gnn_num_layers=model_cfg["gnn_num_layers"],
                gnn_heads=model_cfg["gnn_heads"],
                energy_hidden_dim=model_cfg["energy_hidden_dim"],
                temporal_d_model=model_cfg["temporal_d_model"],
                temporal_num_layers=model_cfg["temporal_num_layers"],
                temporal_nhead=model_cfg["temporal_nhead"],
                seq_len=model_cfg["seq_len"],
                dropout=model_cfg["dropout"],
                use_transformer=model_cfg["use_transformer"],
            ),
            "criterion": full_criterion,
            "graphs": graphs_full,
            "description": "Full model minus energy layer (E(t)=0)",
        },
        {
            "name": "B7: FAPT-GNN Corr-Only Graph",
            "model": build_model(model_cfg),
            "criterion": full_criterion,
            "graphs": graphs_corr_only,
            "description": "Full model on correlation-only (single-layer) graph",
        },
        {
            "name": "B8: FAPT-GNN FULL (Proposed)",
            "model": build_model(model_cfg),
            "criterion": full_criterion,
            "graphs": graphs_full,
            "description": "Complete proposed model",
        },
    ]
    return variants

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def run_variant(variant: dict, labels, energy_proxy, config: dict, device: str) -> dict:
    
    name = variant["name"]
    model = variant["model"]
    criterion = variant["criterion"]
    graphs = variant["graphs"]
    train_cfg = config["training"]

    print(f"\n{'─'*60}")
    print(f"  Running: {name}")
    print(f"  ({variant['description']})")
    print(f"{'─'*60}")

    dataset = build_sliding_window_dataset(
        graph_sequence=graphs,
        labels=labels,
        energy_proxy=energy_proxy,
        seq_len=config["model"]["seq_len"],
        stride=train_cfg.get("stride", 1),
    )
    train_ds, val_ds, test_ds = walk_forward_split(
        dataset,
        train_ratio=train_cfg["train_ratio"],
        val_ratio=train_cfg["val_ratio"],
    )

    if not train_ds or not val_ds or not test_ds:
        print(f"  [Skip] Insufficient data for {name}")
        return {}

    ablation_config = copy.deepcopy(train_cfg)
    ablation_config["epochs"] = train_cfg.get("ablation_epochs",
                                               min(train_cfg.get("epochs", 50), 30))
    ablation_config["patience"] = min(train_cfg.get("patience", 10), 7)

    t0 = time.time()
    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        criterion=criterion,
        config=ablation_config,
        device=device,
        checkpoint_dir=os.path.join(
            config["output"]["checkpoint_dir"],
            name.replace(" ", "_").replace(":", "").replace("/", "_")
        ),
    )

    test_evaluator = Evaluator()
    test_losses, test_results = eval_epoch(model, test_ds, criterion, device, test_evaluator)
    elapsed = time.time() - t0

    metrics = test_results["metrics"]
    ews = test_results["ews"]
    corr = test_results.get("energy_corr", {})

    result = {
        "name": name,
        "description": variant["description"],
        "auc_roc": metrics["auc_roc"],
        "pr_auc": metrics["pr_auc"],
        "f1_score": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "EWS@5": ews.get("EWS@5", 0.0),
        "EWS@10": ews.get("EWS@10", 0.0),
        "EWS@15": ews.get("EWS@15", 0.0),
        "energy_crash_corr": corr.get("energy_crash_corr_contemporaneous", 0.0),
        "best_val_auc": history["best_val_auc"],
        "train_time_sec": round(elapsed, 1),
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    print(f"\n  ✅ {name} → AUC={metrics['auc_roc']:.4f} | "
          f"F1={metrics['f1_score']:.4f} | EWS@10={ews.get('EWS@10', 0):.4f}")
    return result

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def print_ablation_table(results: list):
    
    print(f"\n\n{'='*90}")
    print("  ABLATION STUDY RESULTS — Table 2 (FAPT-GNN vs Baselines, NIFTY 50)")
    print(f"{'='*90}")
    print(f"{'Model':<35} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1':>7} "
          f"{'EWS@5':>7} {'EWS@10':>8} {'EWS@15':>8} {'Params':>10}")
    print("─" * 90)

    for r in results:
        marker = " ← PROPOSED" if "FULL" in r["name"] else ""
        print(f"{r['name']:<35} "
              f"{r.get('auc_roc', 0):>8.4f} "
              f"{r.get('pr_auc', 0):>8.4f} "
              f"{r.get('f1_score', 0):>7.4f} "
              f"{r.get('EWS@5', 0):>7.4f} "
              f"{r.get('EWS@10', 0):>8.4f} "
              f"{r.get('EWS@15', 0):>8.4f} "
              f"{r.get('num_params', 0):>10,}{marker}")
    print(f"{'='*90}\n")

    print("📄 LaTeX Table Snippet:")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Model & AUC-ROC & PR-AUC & F1 & EWS@5 & EWS@10 & EWS@15 \\\\")
    print("\\hline")
    for r in results:
        bold = r.get("name", "").endswith("(Proposed)")
        vals = [
            r.get('auc_roc', 0), r.get('pr_auc', 0), r.get('f1_score', 0),
            r.get('EWS@5', 0), r.get('EWS@10', 0), r.get('EWS@15', 0)
        ]
        name_clean = r['name'].replace("B1: ", "").replace("B2: ", "").replace(
            "B3: ", "").replace("B4: ", "").replace("B5: ", "").replace(
            "B6: ", "").replace("B7: ", "").replace("B8: ", "")
        row = " & ".join([f"{v:.4f}" for v in vals])
        if bold:
            print(f"\\textbf{{{name_clean}}} & {row} \\\\")
        else:
            print(f"{name_clean} & {row} \\\\")
    print("\\hline")
    print("\\end{tabular}")

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def print_increment_analysis(results: list):
    
    lookup = {r["name"]: r for r in results if r}

    full_auc = lookup.get("B8: FAPT-GNN FULL (Proposed)", {}).get("auc_roc", 0)
    comparisons = [
        ("B4: GNN-LSTM",                   "Graph + Temporal only"),
        ("B5: FAPT-GNN w/o Fragility",     "Add: Fragility Encoder"),
        ("B6: FAPT-GNN w/o Energy",        "Add: Energy Layer"),
        ("B7: FAPT-GNN Corr-Only Graph",   "Add: Multi-layer Graph"),
        ("B8: FAPT-GNN FULL (Proposed)",   "Add: All components"),
    ]

    print("\n📊 Component Contribution Analysis (AUC-ROC improvement):")
    print("─" * 60)
    prev_auc = lookup.get("B4: GNN-LSTM", {}).get("auc_roc", 0)
    for model_key, label in comparisons:
        auc = lookup.get(model_key, {}).get("auc_roc", 0)
        delta = auc - prev_auc if model_key != "B4: GNN-LSTM" else 0
        sign = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {label:<35} AUC={auc:.4f}  {sign} {abs(delta):.4f}")
        prev_auc = auc
    print()

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def run_ablation(config: dict, fast: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  FAPT-GNN ABLATION STUDY — NIFTY 50 Crash Prediction")
    print(f"  Device: {device.upper()} | Fast mode: {fast}")
    print(f"{'='*70}\n")

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    print("📊 Loading data (shared across all ablations)...")
    data = load_all_data(
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        cache_dir=config["data"]["cache_dir"]
    )
    sentiment_features = load_or_build_sentiment(
        price_index=data["prices"].index,
        vix_series=data["vix"],
        use_gdelt=config["data"]["use_gdelt"],
        cache_path=os.path.join(config["data"]["cache_dir"], "gdelt_sentiment.parquet")
    )
    features = build_all_features(
        prices=data["prices"], vix=data["vix"],
        macro=data["macro"], sentiment_features=sentiment_features,
        vol_window=config["features"]["vol_window"],
        centrality_window=config["features"]["centrality_window"],
        liquidity_window=config["features"]["liquidity_window"],
        centrality_threshold=config["features"]["centrality_threshold"],
    )
    node_feature_dict = build_node_feature_matrix(features)

    print("\n🕸️  Building multi-layer graphs...")
    graphs_full, graph_dates = build_graph_sequence(
        node_features_dict=node_feature_dict,
        features_raw=features,
        sentiment_features=sentiment_features,
        window=config["graph"]["graph_window"],
    )

    print("🕸️  Building correlation-only graphs (for B7 ablation)...")
    graphs_corr_only = build_correlation_only_graphs(graphs_full)

    print("\n🏷️  Creating crash labels...")
    returns = features["returns"]
    nifty_aligned = data["nifty"].reindex(returns.index).ffill().bfill()
    labels = create_labels(
        nifty_series=nifty_aligned,
        returns=returns,
        percentile=config["labels"]["percentile"],
        drawdown_threshold=config["labels"]["drawdown_threshold"],
        forward_days=config["labels"]["forward_days"],
        dd_window=config["labels"]["dd_window"],
        max_tte_horizon=config["labels"]["max_tte_horizon"],
    )
    vix_proxy = data["vix"].reindex(returns.index).ffill().bfill()

    crashes = labels["crash_label"].sum()
    normals = len(labels) - crashes
    pos_weight = float(normals / crashes) if crashes > 0 else 9.0
    print(f"\n[Ablation] Class imbalance weight: {pos_weight:.2f}x")

    if fast:
        config["training"]["ablation_epochs"] = 5
        config["training"]["patience"] = 3
        print("[Ablation] FAST MODE: 5 epochs per variant")

    variants = get_ablation_variants(config, pos_weight, graphs_full, graphs_corr_only)

    all_results = []
    for variant in variants:
        try:
            result = run_variant(variant, labels, vix_proxy, config, device)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ❌ Error in {variant['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": variant["name"],
                "description": variant["description"],
                "auc_roc": 0.0, "f1_score": 0.0,
                "error": str(e),
            })

    print_ablation_table(all_results)
    print_increment_analysis(all_results)

    results_path = os.path.join(config["output"]["results_dir"], "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"💾 Ablation results saved to {results_path}")

    csv_path = os.path.join(config["output"]["results_dir"], "ablation_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"📊 CSV saved to {csv_path}")

    return all_results

def main():
    parser = argparse.ArgumentParser(description="FAPT-GNN Ablation Study")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 5 epochs per variant (for testing)")
    parser.add_argument("--epochs", type=int, help="Override ablation epochs")
    parser.add_argument("--variants", nargs="+",
                        help="Only run specific variant indices (e.g. --variants 7 8)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["ablation_epochs"] = args.epochs

    run_ablation(config, fast=args.fast)

if __name__ == "__main__":
    main()

