import sys
import os
sys.path.insert(0, '.')

print("[Test] Starting FAPT-GNN integration test...")

print("\n[Test 1] Loading data...")
try:
    from data.data_pipeline import load_all_data
    from data.crash_labeler import create_labels
    from data.feature_engineering import build_all_features, build_node_feature_matrix
    from data.graph_builder import build_graph_sequence
    from data.gdelt_sentiment import load_or_build_sentiment
    
    data = load_all_data(start='2023-01-01', end='2024-01-01')
    print(f"✓ Data loaded: {len(data['prices'])} days")
    
    labels = create_labels(
        nifty_series=data['nifty'],
        returns=data['prices'].pct_change(),
        percentile=10, 
        drawdown_threshold=-5,
        forward_days=5
    )
    print(f"✓ Labels created: {labels['crash_label'].sum()} crash days")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Test 2] Feature engineering...")
try:
    sentiment = load_or_build_sentiment(
        price_index=data['prices'].index,
        vix_series=data['vix'],
        use_gdelt=False
    )
    
    features = build_all_features(
        prices=data['prices'],
        vix=data['vix'],
        macro=data['macro'],
        sentiment_features=sentiment,
        vol_window=20,
        centrality_window=30,
        liquidity_window=20,
        centrality_threshold=0.3
    )
    print(f"✓ Features computed")
    
    node_features = build_node_feature_matrix(features)
    print(f"✓ Node features: {len(node_features['tickers'])} tickers")
    
except Exception as e:
    print(f"✗ Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Test 3] Building graphs...")
try:
    graphs, dates = build_graph_sequence(
        node_features_dict=node_features,
        features_raw=features,
        sentiment_features=sentiment,
        window=20
    )
    print(f"✓ Graphs built: {len(graphs)} snapshots")
    
except Exception as e:
    print(f"✗ Graph building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Test 4] Training FAPT-GNN...")
try:
    from training.trainer import build_sliding_window_dataset, walk_forward_split, train
    from models.fapt_gnn import build_model
    from training.losses import FAPTGNNLoss
    import torch
    
    dataset = build_sliding_window_dataset(
        graph_sequence=graphs,
        labels=labels,
        energy_proxy=data['vix'],
        seq_len=20,
        stride=1
    )
    train_ds, val_ds, test_ds = walk_forward_split(dataset)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    model = build_model({
        'node_feature_dim': 7,
        'gnn_hidden_dim': 32,
        'gnn_num_layers': 2,
        'temporal_d_model': 64,
        'seq_len': 20
    })
    print(f"✓ Model initialized: {model._count_params():,} parameters")
    
    criterion = FAPTGNNLoss(alpha=1.0, beta=0.3, gamma=0.2, delta=0.1, eta=0.1, pos_weight=10.0)
    
    history = train(
        model=model,
        train_dataset=train_ds[:10],
        val_dataset=val_ds[:5],
        criterion=criterion,
        config={'epochs': 1, 'lr': 0.001, 'batch_size': 1},
        device='cpu',
        checkpoint_dir='./tmp_checkpoint'
    )
    print(f"✓ Training completed: {history['train_losses']}")
    
except Exception as e:
    print(f"✗ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("[OK] All integration tests passed!")
print("="*60)

