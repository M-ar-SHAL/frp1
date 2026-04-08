import torch
import os

CHECKPOINT_PATH = "experiments/checkpoints/best_model.pt"
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Best Val AUC: {ckpt.get('val_auc', 0.0):.4f}")
    if 'val_metrics' in ckpt:
        print(f"Full Metrics: {ckpt['val_metrics'].get('metrics', {})}")
else:
    print(f"Checkpoint not found at {CHECKPOINT_PATH}")
