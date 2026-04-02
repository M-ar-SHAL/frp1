"""
Step-by-step debug — redirects stdout entirely to file, no emoji.
Run: python debug_step.py
"""
import sys, os, traceback

# Redirect ALL output to file from the start  
sys.stdout = open("debug_step_out.txt", "w", encoding="ascii", errors="replace")
sys.stderr = sys.stdout

import yaml

sys.path.insert(0, ".")

def step(n, msg):
    print(f"[STEP {n}] {msg}", flush=True)


try:
    # Config
    with open("experiments/config.yaml") as f:
        config = yaml.safe_load(f)
    config["data"]["start_date"] = "2019-01-01"
    config["training"]["epochs"] = 2
    config["model"]["seq_len"] = 10
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)

    step(1, "Loading data...")
    from data.data_pipeline import load_all_data
    data = load_all_data(start=config["data"]["start_date"])
    step(1, f"OK - prices {data['prices'].shape}, vix {data['vix'].shape}")

    step(2, "Sentiment...")
    from data.gdelt_sentiment import load_or_build_sentiment
    sentiment = load_or_build_sentiment(data["prices"].index, data["vix"], use_gdelt=False)
    step(2, f"OK - sentiment {sentiment.shape}")

    step(3, "Feature engineering...")
    from data.feature_engineering import build_all_features, build_node_feature_matrix
    features = build_all_features(
        data["prices"], data["vix"], data["macro"], sentiment,
        vol_window=20, centrality_window=60, liquidity_window=20, centrality_threshold=0.3
    )
    node_feat = build_node_feature_matrix(features)
    step(3, f"OK - tickers: {len(node_feat['tickers'])}")

    step(4, "Building graphs...")
    from data.graph_builder import build_graph_sequence
    graphs, graph_dates = build_graph_sequence(node_feat, features, sentiment, window=60)
    step(4, f"OK - {len(graphs)} graphs")

    step(5, "Creating labels...")
    from data.crash_labeler import create_labels
    returns = features["returns"]
    nifty_aligned = data["nifty"].reindex(returns.index).ffill().bfill()
    labels = create_labels(
        nifty_aligned, returns,
        percentile=5.0, drawdown_threshold=-0.07,
        forward_days=5, dd_window=10, max_tte_horizon=60
    )
    step(5, f"OK - crashes: {int(labels['crash_label'].sum())}")

    step(6, "Building dataset...")
    from training.trainer import build_sliding_window_dataset, walk_forward_split, compute_pos_weight
    vix_proxy = data["vix"].reindex(returns.index).ffill().bfill()
    dataset = build_sliding_window_dataset(graphs, labels, vix_proxy, seq_len=10, stride=1)
    train_ds, val_ds, test_ds = walk_forward_split(dataset, 0.7, 0.15)
    step(6, f"OK - train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    step(7, "Building model...")
    from models.fapt_gnn import FAPT_GNN
    model = FAPT_GNN(**config["model"])
    step(7, f"OK - params: {sum(p.numel() for p in model.parameters()):,}")

    step(8, "Building loss...")
    from training.losses import FAPTGNNLoss
    pos_weight = compute_pos_weight(train_ds)
    criterion = FAPTGNNLoss(
        alpha=1.0, beta=0.3, gamma=0.2, delta=0.1, eta=0.1,
        pos_weight=pos_weight, use_focal=True
    )
    step(8, f"OK - pos_weight={pos_weight:.2f}")

    step(9, "Single-sample forward pass...")
    import torch
    model.train()
    sample = train_ds[0]
    gs = sample["graphs"]
    adj = sample["adj"] if sample["adj"] is not None else torch.eye(gs[-1].num_nodes)
    cp, tte_p, inst, eseq, fseq = model(gs)
    loss, ld = criterion(cp, tte_p, eseq, fseq, adj,
                         sample["crash_label"], sample["time_to_crash"], sample["energy_proxy"])
    step(9, f"OK - loss={loss.item():.4f}, crash_prob={cp.item():.4f}")

    step(10, "2-epoch training...")
    from training.trainer import train
    history = train(model, train_ds, val_ds, criterion, config["training"],
                    device="cpu", checkpoint_dir="experiments/checkpoints")
    step(10, f"OK - best val AUC: {history['best_val_auc']:.4f}")

    print("\n=== ALL STEPS PASSED ===")

except Exception:
    print("\n=== ERROR ===")
    traceback.print_exc()

finally:
    sys.stdout.close()
