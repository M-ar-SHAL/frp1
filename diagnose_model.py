"""
Diagnostic script to identify issues with model inference vs market context.
"""
import torch
import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

CONFIG_PATH = "experiments/config.yaml"
CHECKPOINT_DIR = "experiments/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

print("=" * 80)
print("FAPT-GNN DIAGNOSTIC REPORT")
print("=" * 80)

# 1. Check if checkpoint exists
print("\n[1] CHECKPOINT STATUS")
print(f"    Path: {CHECKPOINT_PATH}")
print(f"    Exists: {os.path.exists(CHECKPOINT_PATH)}")

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    print(f"    Keys: {ckpt.keys()}")
    print(f"    Val AUC: {ckpt.get('val_auc', 'N/A')}")
    print(f"    Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"    Model State Keys: {list(ckpt['model_state_dict'].keys())[:5]}...")

# 2. Load config
print("\n[2] MODEL CONFIGURATION")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

print(f"    Node features: {config['model']['node_feature_dim']}")
print(f"    Seq length: {config['model']['seq_len']}")
print(f"    GNN hidden: {config['model']['gnn_hidden_dim']}")

# 3. Load data and check feature shapes
print("\n[3] LIVE DATA SHAPE CHECK")
try:
    from data.data_pipeline import load_all_data
    from data.feature_engineering import build_all_features
    from data.crash_labeler import create_labels
    from data.graph_builder import build_graph_sequence
    
    data = load_all_data(start='2020-01-01')
    print(f"    NIFTY shape: {data['nifty'].shape}")
    print(f"    VIX shape: {data['vix'].shape}")
    
    # Create proper sentiment series with required columns
    sent_index = data['vix'].index
    sent = pd.DataFrame({
        'sentiment_diverge': np.zeros(len(sent_index)),
        'vix_norm': (data['vix'].values - data['vix'].mean()) / (data['vix'].std() + 1e-8)
    }, index=sent_index)
    feats_dict = build_all_features(data["prices"], data["vix"], data["macro"], sent, **config['features'])
    print(f"    Features keys: {feats_dict.keys()}")
    
    from data.feature_engineering import build_node_feature_matrix
    node_feats_dict = build_node_feature_matrix(feats_dict)
    print(f"    Node features keys: {node_feats_dict.keys()}")
    per_stock = node_feats_dict['per_stock']
    print(f"    Per-stock features shape: {per_stock['return'].shape}")
    
    graph_sequence, _ = build_graph_sequence(node_feats_dict, feats_dict, sent, window=config['graph']['graph_window'])
    print(f"    Graph sequence length: {len(graph_sequence)}")
    
    if len(graph_sequence) > 0:
        latest_seq = graph_sequence[-config['model']['seq_len']:]
        print(f"    Latest seq length: {len(latest_seq)}")
        print(f"    First graph nodes: {latest_seq[0].x.shape if hasattr(latest_seq[0], 'x') else 'N/A'}")
        print(f"    First graph edges: {latest_seq[0].edge_index.shape if hasattr(latest_seq[0], 'edge_index') else 'N/A'}")
        
        # 4. Diagnose graph connectivity
        print("\n[4] GRAPH CONNECTIVITY ANALYSIS")
        from data.graph_builder import build_multilayer_adjacency, build_sector_graph
        
        # Check a single graph adjacency
        test_graph = graph_sequence[-1]
        sector_adj = build_sector_graph(node_feats_dict["tickers"])
        
        # Rebuild adjacency for diagnosis
        ret_window = feats_dict['returns'].iloc[-61:-1]  # Last 60 days
        vol_window = feats_dict['volatility'].iloc[-61:-1]
        sent_series = pd.Series(np.zeros(60), index=ret_window.index)
        
        adj_multilayer = build_multilayer_adjacency(
            ret_window, vol_window, sent_series, node_feats_dict["tickers"], sector_adj
        )
        
        print(f"    Adjacency matrix shape: {adj_multilayer.shape}")
        print(f"    Adjacency range: [{adj_multilayer.min():.8f}, {adj_multilayer.max():.8f}]")
        print(f"    Adjacency mean: {adj_multilayer.mean():.8f}")
        print(f"    Non-zero entries: {np.count_nonzero(adj_multilayer)}")
        print(f"    Non-zero ratio: {np.count_nonzero(adj_multilayer) / adj_multilayer.size:.6f}")
        print(f"    Values > 0.001: {np.sum(adj_multilayer > 0.001)}")
        print(f"    Values > 0.0001: {np.sum(adj_multilayer > 0.0001)}")
        print("\n[4] MODEL INFERENCE TEST")
        from models.fapt_gnn import FAPT_GNN
        
        model = FAPT_GNN(**config['model'])
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print("    [OK] Checkpoint loaded")
        else:
            print("    ✗ No checkpoint - using random initialization")
        
        model.eval()
        with torch.no_grad():
            crash_prob, tte_pred, instability, energy_seq, fragility_seq = model(latest_seq)
        
        print(f"    Crash Probability: {crash_prob.item():.4f}")
        print(f"    Time-to-Crash: {tte_pred.item():.4f} days")
        print(f"    Instability: {instability.item():.4f}")
        print(f"    Energy sequence length: {len(energy_seq)}")
        print(f"    Energy range: [{energy_seq.min().item():.6f}, {energy_seq.max().item():.6f}]")
        print(f"    Energy mean: {energy_seq.mean().item():.6f}")
        print(f"    Last energy: {energy_seq[-1].item():.6f}")
        
        # 5. Diagnose energy computation
        print("\n[5] ENERGY COMPUTATION ANALYSIS")
        print(f"    Energy values (last 5): {energy_seq[-5:].numpy().tolist()}")
        
        # Check if energy is frozen at 0
        if abs(energy_seq[-1].item()) < 1e-6:
            print("    [WARN] WARNING: Energy is essentially ZERO")
            print("       This suggests:")
            print("       - Node features may be zero/constant")
            print("       - Fragility computation is broken")
            print("       - Energy layer has numerical issues")
        
        # 6. Check feature statistics
        print("\n[6] NODE FEATURE STATISTICS (latest snapshot)")
        latest_node_feats = latest_seq[-1].x
        print(f"    Shape: {latest_node_feats.shape}")
        print(f"    Mean: {latest_node_feats.mean(dim=0).numpy().tolist()}")
        print(f"    Std:  {latest_node_feats.std(dim=0).numpy().tolist()}")
        print(f"    Min:  {latest_node_feats.min(dim=0).values.numpy().tolist()}")
        print(f"    Max:  {latest_node_feats.max(dim=0).values.numpy().tolist()}")
        
        # Check for NaN
        nan_count = torch.isnan(latest_node_feats).sum().item()
        print(f"    NaN count: {nan_count}")
        
except Exception as e:
    import traceback
    print(f"    ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("END DIAGNOSTIC REPORT")
print("=" * 80)
