import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CrashClassificationLoss(nn.Module):

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=pred.device)
            bce = F.binary_cross_entropy(pred, target.float(),
                                          weight=target * pw + (1 - target))
        else:
            bce = F.binary_cross_entropy(pred, target.float())

        if not self.use_focal:
            return bce

        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_bce = focal_weight * F.binary_cross_entropy(pred, target.float(), reduction="none")
        return focal_bce.mean()

class TimeToEventLoss(nn.Module):

    def __init__(self, delta: float = 5.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.huber(pred, target.float())

class EnergyRegularizationLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        energy: torch.Tensor,
        energy_proxy: torch.Tensor,
    ) -> torch.Tensor:
        proxy_norm = F.relu(energy_proxy) / (energy_proxy.abs().max() + 1e-8)
        scale = F.softplus(self.scale)
        loss = (energy - scale * proxy_norm).abs().mean()
        return loss

class SmoothnessLoss(nn.Module):

    def forward(
        self,
        fragility: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        
        degree = adj.sum(dim=-1)
        D = torch.diag(degree)

        L = D - adj

        loss = fragility @ L @ fragility
        num_edges = (adj > 0).sum().float().clamp(min=1)
        return loss / num_edges

class TemporalConsistencyLoss(nn.Module):

    def forward(
        self,
        fragility_seq: list,
    ) -> torch.Tensor:
        if len(fragility_seq) < 2:
            return torch.tensor(0.0)

        total = torch.tensor(0.0, device=fragility_seq[0].device)
        for t in range(1, len(fragility_seq)):
            diff = fragility_seq[t] - fragility_seq[t-1]
            total = total + (diff ** 2).mean()

        return total / (len(fragility_seq) - 1)

class FAPTGNNLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.1,
        eta: float = 0.1,
        pos_weight: float = 9.0,
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
        crash_prob: torch.Tensor,
        time_to_crash_pred: torch.Tensor,
        energy_seq: torch.Tensor,
        fragility_seq: list,
        adj: torch.Tensor,
        crash_label: torch.Tensor,
        time_to_crash_true: torch.Tensor,
        energy_proxy: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        
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

    crash_prob = torch.rand(B)
    tte_pred = torch.rand(B) * 30
    energy_seq = torch.rand(T)
    fragility_seq = [torch.rand(N) for _ in range(T)]
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)

    crash_label = torch.zeros(B)
    crash_label[0] = 1
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
    
    pass

