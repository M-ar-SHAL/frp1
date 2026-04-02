"""
Evaluation Metrics for FAPT-GNN

Metrics for research paper:
  1. AUC-ROC
  2. F1-score (critical due to class imbalance)
  3. Early Warning Score (EWS) at t+5, t+10, t+15 days
  4. Precision-Recall AUC
  5. Energy-Crash correlation (interpretability metric)
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve,
    auc, confusion_matrix, classification_report
)
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def compute_all_metrics(
    crash_probs: np.ndarray,     # (N,) predicted crash probabilities
    crash_labels: np.ndarray,    # (N,) true binary labels
    tte_preds: Optional[np.ndarray] = None,   # time-to-crash predictions
    tte_true: Optional[np.ndarray] = None,    # true time-to-crash
    threshold: float = 0.5,
) -> Dict:
    """
    Compute full evaluation metrics suite.
    """
    # Optimal threshold via F1 maximization
    best_f1, best_thresh = 0, threshold
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (crash_probs >= t).astype(int)
        f1 = f1_score(crash_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (crash_probs >= best_thresh).astype(int)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(crash_labels, crash_probs)
    except ValueError:
        auc_roc = 0.0

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(crash_labels, crash_probs)
    pr_auc = auc(recall, precision)

    # F1 at best threshold
    f1 = f1_score(crash_labels, preds_binary, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(crash_labels, preds_binary, labels=[0, 1]).ravel() if crash_labels.sum() > 0 else (0, 0, 0, 0)

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "best_threshold": round(best_thresh, 3),
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

    # Time-to-crash metrics (if provided)
    if tte_preds is not None and tte_true is not None:
        mask = crash_labels == 1
        if mask.sum() > 0:
            mae_tte = np.abs(tte_preds[mask] - tte_true[mask]).mean()
            metrics["tte_mae_on_crashes"] = round(float(mae_tte), 2)

    return metrics


def compute_early_warning_score(
    crash_probs: np.ndarray,   # (T,) full time series of predictions
    crash_labels: np.ndarray,  # (T,) binary crash labels
    horizons: List[int] = [5, 10, 15],
) -> Dict:
    """
    Early Warning Score (EWS): measures how well the model warns
    BEFORE a crash occurs.

    For each horizon h, check if the model's crash probability was high
    in the h-day window BEFORE each crash event.

    EWS@h = fraction of crash events preceded by high alarm (>0.5) within h days.
    """
    T = len(crash_labels)
    crash_dates = np.where(crash_labels == 1)[0]  # indices of crash days

    ews = {}
    for h in horizons:
        caught = 0
        for crash_idx in crash_dates:
            # Check window [crash_idx - h, crash_idx)
            start = max(0, crash_idx - h)
            window = crash_probs[start:crash_idx]
            if len(window) > 0 and window.max() > 0.5:
                caught += 1
        recall_at_h = caught / len(crash_dates) if len(crash_dates) > 0 else 0
        ews[f"EWS@{h}"] = round(recall_at_h, 4)

    return ews


def compute_energy_crash_correlation(
    energy_seq: np.ndarray,     # (T,) E(t) values
    crash_labels: np.ndarray,   # (T,) binary crash labels
    lead_days: int = 10,
) -> Dict:
    """
    Compute correlation between energy E(t) and crash labels.
    Tests the paper's core hypothesis: energy spikes BEFORE crashes.

    Returns:
      - contemporaneous correlation (at same day)
      - lead correlation (E at t-lead predicts crash at t)
    """
    T = min(len(energy_seq), len(crash_labels))
    E = energy_seq[:T]
    Y = crash_labels[:T]

    # Contemporaneous
    corr_now = np.corrcoef(E, Y)[0, 1]

    # Lead correlation
    if lead_days < T:
        E_lead = E[:-lead_days]
        Y_future = Y[lead_days:]
        corr_lead = np.corrcoef(E_lead, Y_future)[0, 1]
    else:
        corr_lead = np.nan

    return {
        "energy_crash_corr_contemporaneous": round(float(corr_now), 4),
        f"energy_crash_corr_lead{lead_days}d": round(float(corr_lead), 4),
    }


def print_evaluation_report(metrics: Dict, ews: Dict, corr: Dict, model_name: str = "FAPT-GNN"):
    """Print a formatted evaluation report (suitable for paper Table)."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")
    print(f"\n📊 Classification Metrics:")
    print(f"  AUC-ROC Score    : {metrics['auc_roc']:.4f}")
    print(f"  PR-AUC Score     : {metrics['pr_auc']:.4f}")
    print(f"  F1 Score         : {metrics['f1_score']:.4f}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")
    print(f"  Best Threshold   : {metrics['best_threshold']:.3f}")
    print(f"\n⚡ Early Warning Scores:")
    for k, v in ews.items():
        print(f"  {k}             : {v:.4f}")
    print(f"\n🔋 Energy-Crash Correlation (Paper Hypothesis):")
    for k, v in corr.items():
        print(f"  {k}: {v:.4f}")
    if "tte_mae_on_crashes" in metrics:
        print(f"\n⏱️  Time-to-Crash MAE (crash events only): {metrics['tte_mae_on_crashes']:.2f} days")
    print(f"{'='*60}\n")


class Evaluator:
    """Stateful evaluator to collect predictions over full epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.crash_probs = []
        self.crash_labels = []
        self.tte_preds = []
        self.tte_trues = []
        self.energy_values = []

    def update(
        self,
        crash_prob: torch.Tensor,
        crash_label: torch.Tensor,
        tte_pred: Optional[torch.Tensor] = None,
        tte_true: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ):
        self.crash_probs.extend(crash_prob.detach().cpu().numpy().tolist())
        self.crash_labels.extend(crash_label.detach().cpu().numpy().tolist())
        if tte_pred is not None:
            self.tte_preds.extend(tte_pred.detach().cpu().numpy().tolist())
        if tte_true is not None:
            self.tte_trues.extend(tte_true.detach().cpu().numpy().tolist())
        if energy is not None:
            self.energy_values.append(energy)

    def compute(self) -> Dict:
        probs = np.array(self.crash_probs)
        labels = np.array(self.crash_labels)

        metrics = compute_all_metrics(probs, labels,
                                       np.array(self.tte_preds) if self.tte_preds else None,
                                       np.array(self.tte_trues) if self.tte_trues else None)
        ews = compute_early_warning_score(probs, labels)

        corr = {}
        if self.energy_values:
            E = np.array(self.energy_values[:len(labels)])
            corr = compute_energy_crash_correlation(E, labels)

        return {"metrics": metrics, "ews": ews, "energy_corr": corr}


if __name__ == "__main__":
    # Simulate outputs
    np.random.seed(42)
    T = 500
    labels = np.zeros(T)
    labels[np.random.choice(T, 25, replace=False)] = 1  # 5% crash rate

    probs = labels * 0.7 + np.random.rand(T) * 0.3  # noisy predictions

    metrics = compute_all_metrics(probs, labels)
    ews = compute_early_warning_score(probs, labels)
    energy = np.random.rand(T)
    corr = compute_energy_crash_correlation(energy, labels)

    print_evaluation_report(metrics, ews, corr)
