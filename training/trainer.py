import os
import time
import torch
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
import warnings

warnings.filterwarnings("ignore")

def build_sliding_window_dataset(
    graph_sequence: List[Data],
    labels: "pd.DataFrame",
    energy_proxy: "pd.Series",
    seq_len: int = 30,
    stride: int = 1,
) -> List[Dict]:
    
    dataset = []
    T = len(graph_sequence)

    graph_dates = [g.date for g in graph_sequence]
    labels_aligned = labels.reindex(graph_dates).ffill().bfill().fillna(0)
    proxy_aligned = energy_proxy.reindex(graph_dates).ffill().bfill().fillna(0)

    for i in range(0, T - seq_len, stride):
        window_graphs = graph_sequence[i: i + seq_len]
        label_date_idx = i + seq_len - 1

        crash_label = labels_aligned["crash_label"].iloc[label_date_idx]
        tte = labels_aligned["time_to_crash"].iloc[label_date_idx]
        energy_proxy_window = proxy_aligned.iloc[i: i + seq_len].values

        adj = window_graphs[-1].adj if hasattr(window_graphs[-1], 'adj') else None

        dataset.append({
            "graphs": window_graphs,
            "crash_label": torch.tensor([crash_label], dtype=torch.float32),
            "time_to_crash": torch.tensor([tte], dtype=torch.float32),
            "energy_proxy": torch.tensor(energy_proxy_window, dtype=torch.float32),
            "adj": adj,
        })

    crash_count = sum(int(d["crash_label"].item()) for d in dataset)
    if len(dataset) > 0:
        crash_pct = 100 * crash_count / len(dataset)
    else:
        crash_pct = 0.0
    print(f"[Dataset] {len(dataset)} samples | Crashes: {crash_count} ({crash_pct:.1f}%)")
    return dataset

def walk_forward_split(dataset: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15):
    
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]

    print(f"[Split] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def compute_pos_weight(dataset: List[Dict]) -> float:
    
    crashes = sum(int(d["crash_label"].item()) for d in dataset)
    normals = len(dataset) - crashes
    if crashes == 0:
        return 1.0
    ratio = normals / crashes
    return np.sqrt(ratio)

def train_epoch(
    model,
    dataset: List[Dict],
    criterion,
    optimizer,
    scaler,
    device: str,
    evaluator,
    max_grad_norm: float = 1.0,
) -> Dict:
    
    model.train()
    total_losses = {}

    for sample in dataset:
        graphs = [g.to(device) for g in sample["graphs"]]
        crash_label = sample["crash_label"].to(device)
        tte_true = sample["time_to_crash"].to(device)
        energy_proxy = sample["energy_proxy"].to(device)
        adj = sample["adj"].to(device) if sample["adj"] is not None else None

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            crash_prob, tte_pred, instability, energy_seq, fragility_seq = model(graphs)

            if adj is None:
                adj = graphs[-1].adj if hasattr(graphs[-1], "adj") else torch.eye(graphs[-1].num_nodes, device=device)

            loss, loss_dict = criterion(
                crash_prob=crash_prob,
                time_to_crash_pred=tte_pred,
                energy_seq=energy_seq,
                fragility_seq=fragility_seq,
                adj=adj,
                crash_label=crash_label,
                time_to_crash_true=tte_true,
                energy_proxy=energy_proxy,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v

        evaluator.update(crash_prob.detach(), crash_label.detach(),
                          tte_pred.detach(), tte_true.detach(),
                          energy_seq[-1].item())

    n = len(dataset)
    return {k: round(v / n, 5) for k, v in total_losses.items()}

@torch.no_grad()
def eval_epoch(model, dataset: List[Dict], criterion, device: str, evaluator) -> Tuple[Dict, Dict]:
    
    model.eval()
    total_losses = {}

    for sample in dataset:
        graphs = [g.to(device) for g in sample["graphs"]]
        crash_label = sample["crash_label"].to(device)
        tte_true = sample["time_to_crash"].to(device)
        energy_proxy = sample["energy_proxy"].to(device)
        adj = sample["adj"].to(device) if sample["adj"] is not None else None

        crash_prob, tte_pred, instability, energy_seq, fragility_seq = model(graphs)

        if adj is None:
            adj = torch.eye(graphs[-1].num_nodes, device=device)

        _, loss_dict = criterion(
            crash_prob=crash_prob,
            time_to_crash_pred=tte_pred,
            energy_seq=energy_seq,
            fragility_seq=fragility_seq,
            adj=adj,
            crash_label=crash_label,
            time_to_crash_true=tte_true,
            energy_proxy=energy_proxy,
        )

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v

        evaluator.update(crash_prob.detach(), crash_label.detach(),
                          tte_pred.detach(), tte_true.detach(),
                          energy_seq[-1].item())

    n = len(dataset)
    avg_losses = {k: round(v / n, 5) for k, v in total_losses.items()}
    eval_results = evaluator.compute()
    return avg_losses, eval_results

def train(
    model,
    train_dataset: List[Dict],
    val_dataset: List[Dict],
    criterion,
    config: dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "experiments/checkpoints",
    epoch_callback = None,
) -> Dict:
    
    from training.evaluate import Evaluator

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("epochs", 50),
        eta_min=config.get("lr_min", 1e-5),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    best_val_auc = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    print(f"\n[Trainer] Starting training on {device} for {epochs} epochs")
    print(f"[Trainer] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_eval = Evaluator()
        val_eval = Evaluator()

        train_losses = train_epoch(model, train_dataset, criterion, optimizer, scaler, device, train_eval)
        val_losses, val_results = eval_epoch(model, val_dataset, criterion, device, val_eval)

        scheduler.step()

        val_auc = val_results["metrics"]["auc_roc"]
        val_f1 = val_results["metrics"]["f1_score"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train L={train_losses['total']:.4f} [cls={train_losses.get('cls', 0):.4f}] | "
              f"Val L={val_losses['total']:.4f} | "
              f"Val AUC={val_auc:.4f} F1={val_f1:.4f} | "
              f"EWS@5={val_results['ews'].get('EWS@5', 0):.3f} | "
              f"{elapsed:.1f}s")

        history["train"].append(train_losses)
        history["val"].append({**val_losses, **val_results["metrics"]})

        if epoch_callback:
            epoch_callback(epoch, epochs, train_losses, val_losses, val_results)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_metrics": val_results,
                "config": config,
            }, ckpt_path)
            print(f"  ✅ New best! AUC={val_auc:.4f} → saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\n[Trainer] Training complete. Best Val AUC: {best_val_auc:.4f}")
    return {"history": history, "best_val_auc": best_val_auc}

