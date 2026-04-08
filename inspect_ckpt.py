import torch
import os

CHECKPOINT_PATH = "experiments/checkpoints/best_model.pt"
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    print("Keys:", ckpt.keys())
    if "val_metrics" in ckpt:
        print("Val Metrics keys:", ckpt["val_metrics"].keys())
        if "metrics" in ckpt["val_metrics"]:
            print("Metric values:", ckpt["val_metrics"]["metrics"])
else:
    print("Checkpoint not found")
