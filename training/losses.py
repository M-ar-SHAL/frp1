"""
Multi-Objective Loss Functions for FAPT-GNN

Five loss components as derived in the paper:
  L = α*L_cls + β*L_time + γ*L_energy + δ*L_smooth + η*L_temp

  L_cls    : Binary cross-entropy for crash classification
  L_time   : MSE for time-to-crash regression
  L_energy : Energy regularization (align E(t) with volatility proxy)
  L_smooth : Graph Laplacian smoothness for fragility
  L_temp   : Temporal consistency of fragility over time

Also implements class imbalance handling (crash days ≈ 5-10% of data).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrashClassificationLoss(nn.Module):
    """
    L_cls: Binary cross-entropy with class balancing.
    
    Crash events are rare (≈5-10% of days) → use:
      1. Focal Loss (better than weighted BCE for imbalanced data)
      2. Positive class weight = N_neg / N_pos
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,  # set based on dataset imbalance
        focal_gamma: float = 2.0,            # focal loss focusing parameter
        use_focal: bool = True,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

    def forward(
        self,
        pred: torch.Tensor,   # (B,) predicted crash probability ∈ [0, 1]
        target: torch.Tensor, # (B,) binary crash label {0, 1}
    ) -> torch.Tensor:
        # Standard BCE
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=pred.device)
            bce = F.binary_cross_entropy(pred, target.float(),
                                          weight=target * pw + (1 - target))
        else:
            bce = F.binary_cross_entropy(pred, target.float())

        if not self.use_focal:
            return bce

        # Focal loss modification: downweight easy negatives
        pt = torch.where(target == 1, pred, 1 - pred)  # p_t
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_bce = focal_weight * F.binary_cross_entropy(pred, target.float(), reduction="none")
        return focal_bce.mean()


class TimeToEventLoss(nn.Module):
    """
    L_time: Regression loss for time-to-crash τ_t.
    
    Options:
      - MSE (simple)
      - Huber (robust to outliers — recommended)
    """

    def __init__(self, delta: float = 5.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(
        self,
        pred: torch.Tensor,   # (B,) predicted τ̂_t (days)
        target: torch.Tensor, # (B,) true τ_t (days)
    ) -> torch.Tensor:
        return self.huber(pred, target.float())


class EnergyRegularizationLoss(nn.Module):
    """
    L_energy: Align learned E(t) with external volatility proxy.

    E_proxy(t) = India VIX (or rolling variance of NIFTY 50)
    We want E(t) to be monotonically related to E_proxy(t).
    
    Loss = |E(t) - α * E_proxy(t)| (MAE with learned scale)
    
    This grounds the abstract energy function to observable signals
    (important for paper's empirical validation section).
    """

    def __init__(self):
        super().__init__()
        # Learnable scale to align energy magnitude with proxy
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        energy: torch.Tensor,       # (B,) or (T,) — model E(t)
        energy_proxy: torch.Tensor, # (B,) or (T,) — VIX or variance proxy
    ) -> torch.Tensor:
        # Normalize proxy to positive range
        proxy_norm = F.relu(energy_proxy) / (energy_proxy.abs().max() + 1e-8)
        scale = F.softplus(self.scale)  # ensure positive
        loss = (energy - scale * proxy_norm).abs().mean()
        return loss


class SmoothnessLoss(nn.Module):
    """
    L_smooth: Graph Laplacian smoothness for fragility.

    L_smooth = Σ w_ij (F_i - F_j)^2 = F^T L F

    Where L = D - A (graph Laplacian).
    Prevents noisy, discontinuous fragility assignments.
    Encourages connected stocks to have similar fragility levels.
    """

    def forward(
        self,
        fragility: torch.Tensor,  # (N,) fragility scores
        adj: torch.Tensor,        # (N, N) adjacency matrix
    ) -> torch.Tensor:
        """F^T L F = F^T (D - A) F"""
        # Degree matrix D
        degree = adj.sum(dim=-1)  # (N,)
        D = torch.diag(degree)

        # Laplacian L = D - A
        L = D - adj

        # Smoothness: F^T L F
        loss = fragility @ L @ fragility
        # Normalize by number of edges
        num_edges = (adj > 0).sum().float().clamp(min=1)
        return loss / num_edges


class TemporalConsistencyLoss(nn.Module):
    """
    L_temp: Temporal smoothness of fragility over time.

    L_temp = Σ_t ||F_t - F_{t-1}||^2

    Fragility should not jump discontinuously between timesteps
    (markets don't become fragile instantaneously).
    """

    def forward(
        self,
        fragility_seq: list,  # list of (N,) tensors over time
    ) -> torch.Tensor:
        if len(fragility_seq) < 2:
            return torch.tensor(0.0)

        total = torch.tensor(0.0, device=fragility_seq[0].device)
        for t in range(1, len(fragility_seq)):
            diff = fragility_seq[t] - fragility_seq[t-1]
            total = total + (diff ** 2).mean()

        return total / (len(fragility_seq) - 1)


class FAPTGNNLoss(nn.Module):
    """
    Full multi-objective loss for FAPT-GNN:
    
    L = α*L_cls + β*L_time + γ*L_energy + δ*L_smooth + η*L_temp
    
    Default weights from paper recommendations:
      α = 1.0 (primary task — crash classification)
      β = 0.3 (secondary — time-to-crash)
      γ = 0.2 (energy alignment)
      δ = 0.1 (smoothness)
      η = 0.1 (temporal consistency)

    All weights can be tuned as hyperparameters.
    """

    def __init__(
        self,
        alpha: float = 1.0,    # crash classification weight
        beta: float = 0.3,     # time-to-crash weight
        gamma: float = 0.2,    # energy regularization weight
        delta: float = 0.1,    # smoothness weight
        eta: float = 0.1,      # temporal consistency weight
        pos_weight: float = 9.0,  # crash:normal ratio ≈ 1:9
        use_focal: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta

        self.loss_cls = CrashClassificationLoss(pos_weight=pos_weight, use_focal=use_focal)
        self.loss_time = TimeToEventLoss(delta=5.0)
        self.loss_energy = EnergyRegularizationLoss()
        self.loss_smooth = SmoothnessLoss()
        self.loss_temp = TemporalConsistencyLoss()

    def forward(
        self,
        # Model outputs
        crash_prob: torch.Tensor,        # (B,)
        time_to_crash_pred: torch.Tensor,# (B,)
        energy_seq: torch.Tensor,        # (T,)
        fragility_seq: list,             # list of (N,) tensors
        adj: torch.Tensor,               # (N, N)
        # Targets
        crash_label: torch.Tensor,       # (B,)
        time_to_crash_true: torch.Tensor,# (B,)
        energy_proxy: torch.Tensor,      # (T,) — VIX or variance proxy
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss : scalar loss for backward()
            loss_dict  : dict of individual loss values (for logging)
        """
        L_cls = self.loss_cls(crash_prob, crash_label)
        L_time = self.loss_time(time_to_crash_pred, time_to_crash_true)
        L_energy = self.loss_energy(energy_seq, energy_proxy)
        L_smooth = self.loss_smooth(fragility_seq[-1], adj)
        L_temp = self.loss_temp(fragility_seq)

        total = (self.alpha * L_cls +
                 self.beta * L_time +
                 self.gamma * L_energy +
                 self.delta * L_smooth +
                 self.eta * L_temp)

        loss_dict = {
            "total": total.item(),
            "cls": L_cls.item(),
            "time": L_time.item(),
            "energy": L_energy.item(),
            "smooth": L_smooth.item(),
            "temporal": L_temp.item(),
        }

        return total, loss_dict


if __name__ == "__main__":
    B, N, T = 4, 50, 30

    # Simulate outputs
    crash_prob = torch.rand(B)
    tte_pred = torch.rand(B) * 30
    energy_seq = torch.rand(T)
    fragility_seq = [torch.rand(N) for _ in range(T)]
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)

    # Simulate targets
    crash_label = torch.zeros(B)
    crash_label[0] = 1  # 1 crash out of 4 samples
    tte_true = torch.randint(0, 60, (B,)).float()
    energy_proxy = torch.rand(T)

    criterion = FAPTGNNLoss(pos_weight=9.0)
    total_loss, loss_dict = criterion(
        crash_prob, tte_pred, energy_seq, fragility_seq, adj,
        crash_label, tte_true, energy_proxy
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  L_{k}: {v:.4f}")

class MultiObjectiveFAPTLoss(FAPTGNNLoss):
    """Compatibility alias for the original loss class name used in the dashboard.
    Inherits all behavior from FAPTGNNLoss without modification.
    """
    pass
