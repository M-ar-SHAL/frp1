"""
Minimal step-by-step debug runner — no emoji, captures full traceback.
Writes results to debug_result.txt
"""
import sys, os, traceback, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG = open("debug_result.txt", "w", encoding="ascii", errors="replace")

def log(msg):
    print(msg)
    LOG.write(str(msg) + "\n")
    LOG.flush()

try:
    # ── Config ────────────────────────────────────────────────────────────────
    with open("experiments/config.yaml") as f:
        config = yaml.safe_load(f)

    config["data"]["start_date"] = "2019-01-01"
    config["training"]["epochs"] = 2
    config["model"]["seq_len"] = 10
    config["output"]["results_dir"] = "experiments/results"
    config["output"]["checkpoint_dir"] = "experiments/checkpoints"
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    device = "cpu"
    log(f"[DEBUG] Device: {device}")

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    log("[DEBUG] Step 1: Loading data...")
    from data.data_pipeline import load_all_data
    data = load_all_data(start=config["data"]["start_date"])
    log(f"[DEBUG] prices shape: {data['prices'].shape}")

    # ── Step 2: Sentiment ─────────────────────────────────────────────────────
    log("[DEBUG] Step 2: Sentiment...")
    from data.gdelt_sentiment import load_or_build_sentiment
    sentiment = load_or_build_sentiment(data["prices"].index, data["vix"], use_gdelt=False)
    log(f"[DEBUG] sentiment shape: {sentiment.shape}")

    # ── Step 3: Features ──────────────────────────────────────────────────────
    log("[DEBUG] Step 3: Feature engineering...")
    from data.feature_engineering import build_all_features, build_node_feature_matrix
    features = build_all_features(
        data["prices"], data["vix"], data["macro"], sentiment,
        vol_window=20, centrality_window=60, liquidity_window=20, centrality_threshold=0.3
    )
    node_feat = build_node_feature_matrix(features)
    log(f"[DEBUG] Tickers: {len(node_feat['tickers'])}")

    # ── Step 4: Graphs ────────────────────────────────────────────────────────
    log("[DEBUG] Step 4: Building graphs...")
    from data.graph_builder import build_graph_sequence
    graphs, graph_dates = build_graph_sequence(node_feat, features, sentiment, window=60)
    log(f"[DEBUG] Graphs built: {len(graphs)}")

    # ── Step 5: Labels ────────────────────────────────────────────────────────
    log("[DEBUG] Step 5: Labels...")
    from data.crash_labeler import create_labels
    import pandas as pd
    returns = features["returns"]
    nifty_aligned = data["nifty"].reindex(returns.index).ffill().bfill()
    labels = create_labels(
        nifty_aligned, returns,
        percentile=5.0, drawdown_threshold=-0.07,
        forward_days=5, dd_window=10, max_tte_horizon=60
    )
    log(f"[DEBUG] Labels shape: {labels.shape}, crashes: {labels['crash_label'].sum()}")

    # ── Step 6: Dataset ───────────────────────────────────────────────────────
    log("[DEBUG] Step 6: Dataset...")
    from training.trainer import build_sliding_window_dataset, walk_forward_split, compute_pos_weight
    vix_proxy = data["vix"].reindex(returns.index).ffill().bfill()
    dataset = build_sliding_window_dataset(graphs, labels, vix_proxy, seq_len=10, stride=1)
    train_ds, val_ds, test_ds = walk_forward_split(dataset, 0.7, 0.15)
    log(f"[DEBUG] Dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ── Step 7: Model ─────────────────────────────────────────────────────────
    log("[DEBUG] Step 7: Model...")
    from models.fapt_gnn import FAPT_GNN
    model = FAPT_GNN(**config["model"])
    log(f"[DEBUG] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Step 8: Loss ──────────────────────────────────────────────────────────
    log("[DEBUG] Step 8: Loss...")
    from training.losses import FAPTGNNLoss
    pos_weight = compute_pos_weight(train_ds)
    criterion = FAPTGNNLoss(
        alpha=1.0, beta=0.3, gamma=0.2, delta=0.1, eta=0.1,
        pos_weight=pos_weight, use_focal=True
    )
    log(f"[DEBUG] pos_weight={pos_weight:.2f}")

    # ── Step 9: Train (1 sample forward pass test) ────────────────────────────
    log("[DEBUG] Step 9: Single forward pass test...")
    import torch
    model.train()
    sample = train_ds[0]
    graphs_s = [g for g in sample["graphs"]]
    crash_label = sample["crash_label"]
    tte_true = sample["time_to_crash"]
    ep = sample["energy_proxy"]
    adj = sample["adj"] if sample["adj"] is not None else torch.eye(graphs_s[-1].num_nodes)
    crash_prob, tte_pred, instability, energy_seq, fragility_seq = model(graphs_s)
    loss, ld = criterion(crash_prob, tte_pred, energy_seq, fragility_seq, adj, crash_label, tte_true, ep)
    log(f"[DEBUG] Forward pass OK! loss={loss.item():.4f}")
    log(f"[DEBUG] crash_prob={crash_prob.item():.4f}")

    # ── Step 10: Full train ────────────────────────────────────────────────────
    log("[DEBUG] Step 10: Full training (2 epochs)...")
    from training.trainer import train
    history = train(model, train_ds, val_ds, criterion, config["training"],
                    device=device, checkpoint_dir=config["output"]["checkpoint_dir"])
    log(f"[DEBUG] Training complete. Best val AUC: {history['best_val_auc']:.4f}")
    log("[DEBUG] === ALL STEPS PASSED ===")

except Exception:
    log("\n[ERROR] TRACEBACK:")
    tb = traceback.format_exc()
    log(tb)

finally:
    LOG.close()
print("[DEBUG] Results written to debug_result.txt")
