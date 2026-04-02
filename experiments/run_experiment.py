"""
FAPT-GNN End-to-End Experiment Runner

Usage:
  python experiments/run_experiment.py
  python experiments/run_experiment.py --config experiments/config.yaml
  python experiments/run_experiment.py --start 2018-01-01 --end 2023-12-31

This script runs the full pipeline:
  1. Download & cache NIFTY 50 data
  2. Feature engineering
  3. Build dynamic multi-layer graphs
  4. Create crash labels
  5. Build sliding window dataset
  6. Train FAPT-GNN
  7. Evaluate on test set
  8. Run shock simulation analysis
  9. Save results
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_pipeline import load_all_data
from data.gdelt_sentiment import load_or_build_sentiment
from data.feature_engineering import build_all_features, build_node_feature_matrix, compute_returns
from data.graph_builder import build_graph_sequence
from data.crash_labeler import create_labels
from models.fapt_gnn import build_model
from training.losses import FAPTGNNLoss
from training.trainer import (
    build_sliding_window_dataset, walk_forward_split,
    compute_pos_weight, train
)
from training.evaluate import (
    Evaluator, print_evaluation_report,
    compute_early_warning_score, compute_energy_crash_correlation
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  FAPT-GNN Experiment — NIFTY 50 Crash Prediction")
    print(f"  Device: {device.upper()}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    # ─────────────────────────────────────────
    # STEP 1: Load Data
    # ─────────────────────────────────────────
    print("📊 STEP 1: Loading market data...")
    data = load_all_data(
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        cache_dir=config["data"]["cache_dir"]
    )

    # ─────────────────────────────────────────
    # STEP 2: Sentiment
    # ─────────────────────────────────────────
    print("\n📰 STEP 2: Loading sentiment features...")
    sentiment_features = load_or_build_sentiment(
        price_index=data["prices"].index,
        vix_series=data["vix"],
        use_gdelt=config["data"]["use_gdelt"],
        cache_path=os.path.join(config["data"]["cache_dir"], "gdelt_sentiment.parquet")
    )

    # ─────────────────────────────────────────
    # STEP 3: Feature Engineering
    # ─────────────────────────────────────────
    print("\n⚙️  STEP 3: Computing fragility features...")
    features = build_all_features(
        prices=data["prices"],
        vix=data["vix"],
        macro=data["macro"],
        sentiment_features=sentiment_features,
        vol_window=config["features"]["vol_window"],
        centrality_window=config["features"]["centrality_window"],
        liquidity_window=config["features"]["liquidity_window"],
        centrality_threshold=config["features"]["centrality_threshold"],
    )
    node_feature_dict = build_node_feature_matrix(features)

    # ─────────────────────────────────────────
    # STEP 4: Build Graphs
    # ─────────────────────────────────────────
    print("\n🕸️  STEP 4: Building dynamic multi-layer graphs...")
    graphs, graph_dates = build_graph_sequence(
        node_features_dict=node_feature_dict,
        features_raw=features,
        sentiment_features=sentiment_features,
        window=config["graph"]["graph_window"],
    )

    # ─────────────────────────────────────────
    # STEP 5: Create Labels
    # ─────────────────────────────────────────
    print("\n🏷️  STEP 5: Creating crash labels...")
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

    # ─────────────────────────────────────────
    # STEP 6: Build Dataset
    # ─────────────────────────────────────────
    print("\n📦 STEP 6: Building sliding window dataset...")
    # VIX as energy proxy (aligned to graph dates)
    vix_proxy = data["vix"].reindex(returns.index).ffill().bfill()

    dataset = build_sliding_window_dataset(
        graph_sequence=graphs,
        labels=labels,
        energy_proxy=vix_proxy,
        seq_len=config["model"]["seq_len"],
        stride=config["training"].get("stride", 1),
    )
    train_ds, val_ds, test_ds = walk_forward_split(
        dataset,
        train_ratio=config["training"]["train_ratio"],
        val_ratio=config["training"]["val_ratio"],
    )

    # ─────────────────────────────────────────
    # STEP 7: Build Model
    # ─────────────────────────────────────────
    print("\n🧠 STEP 7: Building FAPT-GNN model...")
    model = build_model(config["model"])

    # Compute class weight for imbalanced data
    pos_weight = compute_pos_weight(train_ds)
    print(f"[Loss] Class weight (pos/neg): {pos_weight:.2f}x")

    criterion = FAPTGNNLoss(
        alpha=config["loss"]["alpha"],
        beta=config["loss"]["beta"],
        gamma=config["loss"]["gamma"],
        delta=config["loss"]["delta"],
        eta=config["loss"]["eta"],
        pos_weight=pos_weight,
        use_focal=config["loss"]["use_focal"],
    )

    # ─────────────────────────────────────────
    # STEP 8: Train
    # ─────────────────────────────────────────
    print("\n🚀 STEP 8: Training FAPT-GNN...")
    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        criterion=criterion,
        config=config["training"],
        device=device,
        checkpoint_dir=config["output"]["checkpoint_dir"],
    )

    # ─────────────────────────────────────────
    # STEP 9: Test Evaluation
    # ─────────────────────────────────────────
    print("\n📈 STEP 9: Evaluating on test set...")

    # Load best checkpoint
    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded best model from epoch {ckpt['epoch']} (val AUC={ckpt['val_auc']:.4f})")

    from training.trainer import eval_epoch
    test_evaluator = Evaluator()
    test_losses, test_results = eval_epoch(model, test_ds, criterion, device, test_evaluator)

    ews = test_results["ews"]
    corr = test_results.get("energy_corr", {})
    metrics = test_results["metrics"]
    print_evaluation_report(metrics, ews, corr, model_name="FAPT-GNN (NIFTY 50)")

    # ─────────────────────────────────────────
    # STEP 10: Shock Simulation
    # ─────────────────────────────────────────
    print("\n⚠️  STEP 10: Running shock simulations on last test sample...")
    if test_ds:
        last_sample = test_ds[-1]
        last_graphs = [g.to(device) for g in last_sample["graphs"]]
        tickers = node_feature_dict["tickers"]

        shock_result = model.predict(last_graphs, return_shock_analysis=True, tickers=tickers)
        print(f"\n📉 Crash Probability (latest): {shock_result['crash_probability']*100:.1f}%")
        print(f"⚡ Instability Index:          {shock_result['instability_index']:.4f}")
        print(f"🔋 System Energy:              {shock_result['system_energy'][-1]:.4f}")
        print(f"⏱  Time-to-Crash Estimate:     {shock_result['time_to_crash_days']:.1f} days")

        if "shock_analysis" in shock_result:
            sa = shock_result["shock_analysis"]
            print(f"\n🔥 Shock Analysis:")
            print(f"  Liquidity Shock Resilience:  {sa['liquidity_resilience']:.4f}")
            print(f"  Sentiment Crash Resilience:  {sa['sentiment_resilience']:.4f}")
            if sa["SIFI_stocks"]:
                print(f"  Top Systemically Important Stocks (SIFI):")
                for s in sa["SIFI_stocks"]:
                    print(f"    → {s}")

    # ─────────────────────────────────────────
    # STEP 11: Save Results
    # ─────────────────────────────────────────
    results_path = os.path.join(config["output"]["results_dir"], "test_results.json")
    save_results = {
        "metrics": metrics,
        "ews": ews,
        "energy_corr": corr,
        "test_losses": test_losses,
        "best_val_auc": history["best_val_auc"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")
    print(f"\n✅ Experiment complete!\n")
    return save_results


def main():
    parser = argparse.ArgumentParser(description="FAPT-GNN Experiment Runner")
    parser.add_argument("--config", default="experiments/config.yaml", help="Path to config file")
    parser.add_argument("--start", type=str, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--use-gdelt", action="store_true", help="Use GDELT sentiment (slower)")
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.start:
        config["data"]["start_date"] = args.start
    if args.end:
        config["data"]["end_date"] = args.end
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.use_gdelt:
        config["data"]["use_gdelt"] = True

    run_experiment(config)


if __name__ == "__main__":
    main()
