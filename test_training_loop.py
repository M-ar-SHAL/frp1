import sys
sys.path.insert(0, '.')

import torch
from torch_geometric.data import Data
from models.fapt_gnn import FAPT_GNN
from training.losses import FAPTGNNLoss

print("[Test] Creating mock training batch...")
N, d, T = 50, 7, 30

graph_seq = []
for t in range(T):
    edge_index = torch.randint(0, N, (2, 100))
    edge_attr = torch.ones(100, 1)
    x = torch.randn(N, d)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.adj = torch.eye(N)
    graph_seq.append(g)

print("[Test] Running model forward pass...")
model = FAPT_GNN(node_feature_dim=d, seq_len=T)
crash_prob, tte, instability, energy_seq, fragility_seq = model(graph_seq)

print(f"  Crash Prob range: [{crash_prob.min().item():.6f}, {crash_prob.max().item():.6f}]")
print(f"  TTE range: [{tte.min().item():.6f}, {tte.max().item():.6f}]")
print(f"  Instability range: [{instability.min().item():.6f}, {instability.max().item():.6f}]")

assert 0 <= crash_prob.min().item() <= 1, f"crash_prob min out of range: {crash_prob.min().item()}"
assert 0 <= crash_prob.max().item() <= 1, f"crash_prob max out of range: {crash_prob.max().item()}"
assert tte.min().item() >= 0, f"tte has negative values: {tte.min().item()}"
assert 0 <= instability.min().item() <= 1, f"instability min out of range: {instability.min().item()}"
assert 0 <= instability.max().item() <= 1, f"instability max out of range: {instability.max().item()}"
print("[OK] All outputs in valid ranges")

print("\n[Test] Computing loss...")
criterion = FAPTGNNLoss(alpha=1.0, beta=0.3, gamma=0.2, delta=0.1, eta=0.1, pos_weight=10.0)

batch = {
    'crash_label': torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32),
    'time_to_crash': torch.tensor([5.0, 10.0, 3.0, 15.0, 7.0], dtype=torch.float32),
    'energy_proxy': torch.randn(5),
    'adj': torch.eye(N)
}

crash_prob_batch = crash_prob.repeat(5)[:5]
tte_batch = tte.repeat(5)[:5]
instability_batch = instability.repeat(5)[:5]
energy_batch = energy_seq.repeat(5)[:5]
fragility_batch = [fragility_seq[-1] for _ in range(5)]

try:
    loss, loss_dict = criterion(
        crash_prob=crash_prob_batch,
        crash_label=batch['crash_label'],
        time_to_crash_pred=tte_batch,
        time_to_crash_true=batch['time_to_crash'],
        energy_seq=energy_batch,
        energy_proxy=batch['energy_proxy'],
        fragility_seq=fragility_batch,
        adj=batch['adj']
    )
    print(f"[OK] Loss computed: {loss.item():.4f}")
    print(f"  Classification loss: {loss_dict.get('L_cls', 'N/A')}")
    print(f"  Time-to-crash loss: {loss_dict.get('L_time', 'N/A')}")
    print(f"  Energy loss: {loss_dict.get('L_energy', 'N/A')}")
except Exception as e:
    print(f"[FAIL] Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] Training loop test passed - no BCE errors")

