"""
Fast Training Script for FAPT-GNN
Uses optimized configuration for quick experimentation.

Usage:
    python train_fast.py

Expected time: ~2-3 minutes for 10 epochs (vs 30+ minutes with default config)
"""

import os
import yaml
import torch
import time
from pathlib import Path

# Load FAST config instead of default
CONFIG_PATH = "experiments/config_fast.yaml"

print("\n" + "="*80)
print("FAPT-GNN FAST TRAINING")
print("="*80)
print(f"Config: {CONFIG_PATH}")
print("Expected: 10 epochs in ~2-3 minutes")
print("="*80 + "\n")

start_time = time.time()

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Data loading
print("[1] Loading data (2020-2026)...")
from data.data_pipeline import load_all_data
from data.feature_engineering import build_all_features, build_node_feature_matrix
from data.crash_labeler import create_labels
from data.graph_builder import build_graph_sequence

data = load_all_data(start=config['data']['start_date'])

# Feature engineering
print("[2] Building features (optimized windows)...")
sent = __import__('pandas').Series(
    0.0, 
    index=data['vix'].index
).to_frame('sentiment_diverge').assign(vix_norm=0.0)

feats = build_all_features(
    data["prices"], data["vix"], data["macro"], sent, 
    **config['features']
)

# Labels
print("[3] Creating crash labels...")
from data.feature_engineering import compute_returns
returns = compute_returns(data["prices"])
labels = create_labels(
    data["nifty"], returns,
    percentile=config['labels']['percentile'],
    drawdown_threshold=config['labels']['drawdown_threshold'],
    forward_days=config['labels']['forward_days'],
    dd_window=config['labels']['dd_window']
)

# Graph sequence
print("[4] Building graph sequence (fast)...")
node_feats = build_node_feature_matrix(feats)
graph_sequence, _ = build_graph_sequence(
    node_feats, feats, sent, 
    window=config['graph']['graph_window']
)

# Dataset
print("[5] Building dataset with stride=5...")
from training.trainer import build_sliding_window_dataset, walk_forward_split
dataset = build_sliding_window_dataset(
    graph_sequence, labels, data['vix'],
    seq_len=config['model']['seq_len'],
    stride=config['training']['stride']  # stride=5 means 5x fewer samples
)
train_ds, val_ds, test_ds = walk_forward_split(dataset)

# Model setup
print("[6] Initializing fast model...")
from models.fapt_gnn import FAPT_GNN
from training.losses import FAPTGNNLoss
from training.trainer import (
    build_sliding_window_dataset, walk_forward_split, train_epoch, eval_epoch,
    compute_pos_weight
)
from training.evaluate import Evaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    Device: {device}")

model = FAPT_GNN(**config['model'])
print(f"    Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

pos_weight = compute_pos_weight(train_ds)
criterion = FAPTGNNLoss(pos_weight=pos_weight, **config['loss'])

# Training
print("\n[7] TRAINING...\n")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['training']['lr'],
    weight_decay=config['training']['weight_decay']
)
scaler = torch.cuda.amp.GradScaler()

model = model.to(device)

for epoch in range(1, config['training']['epochs'] + 1):
    epoch_start = time.time()
    evaluator = Evaluator()
    
    train_losses = train_epoch(
        model, train_ds, criterion, optimizer, scaler, device, evaluator,
        max_grad_norm=config['training']['max_grad_norm']
    )
    
    val_losses, val_metrics = eval_epoch(model, val_ds, criterion, device, Evaluator())
    
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch:2d} | Time: {epoch_time:5.1f}s | "
          f"Train Loss: {train_losses.get('total', 0):.4f} | "
          f"Val AUC: {val_metrics.get('auc', 0):.4f}")

elapsed = time.time() - start_time
print(f"\n{'='*80}")
print(f"Total training time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
print(f"Average per epoch: {elapsed/config['training']['epochs']:.1f}s")
print(f"{'='*80}\n")

# Save checkpoint
os.makedirs("experiments/checkpoints", exist_ok=True)
torch.save({
    'epoch': config['training']['epochs'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_auc': val_metrics.get('auc', 0),
    'config': config,
}, "experiments/checkpoints/best_model.pt")

print("[8] Model saved to experiments/checkpoints/best_model.pt")
print("\nFast training complete! Use dashboard or test_gnn_fix.py to test predictions.\n")
