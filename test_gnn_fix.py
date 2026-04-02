"""
Quick test to verify the GNN fix works without PyG code generation errors.
"""
import sys
import torch
from torch_geometric.data import Data

# Add project to path
sys.path.insert(0, '.')

print("[Test] Importing GNN modules...")
try:
    from models.gnn_core import FragilityAwareGATLayer, FragilityAwareGNN
    print("[OK] Imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n[Test] Creating test data...")
N, d, E = 50, 7, 200
x = torch.randn(N, d)
edge_index = torch.randint(0, N, (2, E))
fragility = torch.rand(N)
edge_attr = torch.rand(E, 1)

print(f"  Nodes: {N}, Features: {d}, Edges: {E}")

print("\n[Test] Testing FragilityAwareGATLayer...")
try:
    gat_layer = FragilityAwareGATLayer(
        in_dim=d, 
        out_dim=32, 
        heads=4,
        edge_dim=1
    )
    h_out = gat_layer(x, edge_index, fragility, edge_attr)
    print(f"[OK] GAT Layer forward pass successful")
    print(f"  Output shape: {h_out.shape} (expected: {N}, {4*32})")
except Exception as e:
    print(f"[FAIL] GAT Layer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Test] Testing FragilityAwareGNN...")
try:
    gnn = FragilityAwareGNN(
        node_feature_dim=d,
        hidden_dim=32,
        num_layers=2,
        heads=4,
        edge_dim=1
    )
    h_gnn = gnn(x, edge_index, fragility, edge_attr)
    print(f"[OK] GNN forward pass successful")
    print(f"  Output shape: {h_gnn.shape} (expected: {N}, 32)")
except Exception as e:
    print(f"[FAIL] GNN failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Test] Testing with graph sequence (FAPT integration)...")
try:
    from models.fapt_gnn import FAPT_GNN
    
    # Create dummy graph sequence
    T = 30  # Time steps
    graph_seq = []
    for t in range(T):
        g = Data(x=torch.randn(N, d), edge_index=edge_index, edge_attr=edge_attr)
        g.adj = torch.rand(N, N)
        g.adj = (g.adj + g.adj.T) / 2
        g.adj.fill_diagonal_(0)
        graph_seq.append(g)
    
    model = FAPT_GNN(node_feature_dim=d, seq_len=T)
    crash_prob, tte, instability, energy_seq, fragility_seq = model(graph_seq)
    
    print(f"[OK] FAPT-GNN forward pass successful")
    print(f"  Crash Prob: {crash_prob.item():.4f}")
    print(f"  Time-to-Crash: {tte.item():.2f} days")
    print(f"  Instability: {instability.item():.4f}")
    
except Exception as e:
    print(f"[FAIL] FAPT-GNN failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("[PASS] All tests passed! GNN fix is working correctly.")
print("="*60)
