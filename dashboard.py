import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
import os
import torch
import warnings

warnings.filterwarnings('ignore')

# Provide absolute paths if necessary, assuming script runs from root
CONFIG_PATH = "experiments/config_fast.yaml"  # Use fast config - matches Colab training
CHECKPOINT_DIR = "experiments/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model1.pt")

st.set_page_config(page_title="FAPT-GNN Dashboard", layout="wide", page_icon=":chart_with_upwards_trend:")

# ---------------------------------------------------------
# IMPORT FAPT-GNN LOGIC
# ---------------------------------------------------------
from data.data_pipeline import load_all_data
from data.feature_engineering import build_all_features
from data.crash_labeler import create_labels
from data.graph_builder import build_graph_sequence
from models.fapt_gnn import FAPT_GNN
from training.trainer import build_sliding_window_dataset, walk_forward_split, train

@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Setup Sidebar
st.sidebar.title("System Configuration")
st.sidebar.markdown("### Data Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(config['data']['start_date']))
use_gdelt = st.sidebar.checkbox("Use GDELT Sentiment", value=config['data']['use_gdelt'])

st.sidebar.markdown("### Crash Definition")
crash_percentile = st.sidebar.slider("Crash Percentile Threshold", min_value=1.0, max_value=15.0, value=config['labels']['percentile'], step=0.5)
drawdown_threshold = st.sidebar.slider("Forward Drawdown Threshold (%)", min_value=-15.0, max_value=-2.0, value=config['labels']['drawdown_threshold']*100, step=1.0) / 100.0

st.title("FAPT-GNN: Fragility-Aware Phase Transition GNN")
st.markdown("""
This dashboard answers the FAPT-GNN review:
- **Extended Constraints:** Evaluates models on 10+ years of robust market data.
- **Dynamic Definition:** Crash percentiles and drawdowns are adjustable to not capture "normal volatility".
- **Real-Time Prediction:** Loads live current-day market data, merging historical context, executing genuine walk-forward dynamics.
""")

# ---------------------------------------------------------
# 1. DATA PREPARATION (CACHED)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data(start_d, gdelt, cp, dt):
    d = load_all_data(start=str(start_d))
    nifty = d["nifty"]
    from data.feature_engineering import compute_returns
    returns = compute_returns(d["prices"])
    labels = create_labels(
        nifty,
        returns,
        percentile=cp, 
        drawdown_threshold=dt, 
        forward_days=config['labels']['forward_days'], 
        dd_window=config['labels']['dd_window']
    )
    return d, labels

with st.spinner("Fetching Real-Time Market Data & Building Features (10+ Yr Horizon)..."):
    try:
        data, labels = get_data(start_date, use_gdelt, crash_percentile, drawdown_threshold)
        st.success(f"Data Loaded: {len(data['nifty'])} Trading Days | {start_date} to {data['nifty'].index[-1].date()}")
    except Exception as e:
        import traceback
        st.error(f"Error loading data: {e}")
        st.code(traceback.format_exc())
        st.stop()

# ---------------------------------------------------------
# UI TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Market Context & Data", "Model Training", "Prediction & Phase Transition", "Validation Matrix"])

with tab1:
    st.header("Historical Market Data & Labels")
    st.markdown("We address **Data Scarcity** by leveraging over a decade of market indices, ensuring diverse crisis exposure.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['nifty'].index, y=data['nifty'].values, mode='lines', name='NIFTY 50', line=dict(color='blue')))
    
    # Highlight Crashes
    crash_indices = labels[labels['crash_label'] == 1].index
    if not crash_indices.empty:
        crash_prices = data['nifty'].loc[crash_indices]
        fig.add_trace(go.Scatter(x=crash_indices, y=crash_prices, mode='markers', name='Target Crashes', marker=dict(color='red', size=8, symbol='x')))

    fig.update_layout(title="NIFTY 50 Index + Dynamic Crash Highlights", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Total Trading Days", len(data['nifty']))
    metrics_cols[1].metric("Registered Systemic Crashes", len(crash_indices))
    metrics_cols[2].metric("Crash Ratio", f"{len(crash_indices)/len(data['nifty'])*100:.2f}%")

with tab2:
    st.header("Model Pipeline & State")
    st.write(f"Looking for pre-trained model checkpoint in `{CHECKPOINT_PATH}` ...")
    
    model_exists = os.path.exists(CHECKPOINT_PATH)
    if model_exists:
        st.success("Trained Checkpoint Located!")
        checkpoint_data = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        st.metric("Best Cross-Valid AUC", f"{checkpoint_data.get('val_auc', 0.0):.4f}")
    else:
        st.warning("No pretrained checkpoint found. Model requires initial background training.")

    if st.button("Train Real-Time FAPT-GNN (Resource Intensive)"):
        with st.spinner("Building Graph Sequences and Triggering Walk-Forward Pipeline..."):
            from data.gdelt_sentiment import load_or_build_sentiment
            from data.feature_engineering import build_node_feature_matrix
            sent = load_or_build_sentiment(data["prices"].index, data["vix"], use_gdelt=use_gdelt)
            feats = build_all_features(data["prices"], data["vix"], data["macro"], sent, **config['features'])
            node_feats = build_node_feature_matrix(feats)
            graph_sequence, _ = build_graph_sequence(node_feats, feats, sent, window=config['graph']['graph_window'])
            
            dataset = build_sliding_window_dataset(
                graph_sequence, labels, data['vix'], 
                seq_len=config['model']['seq_len'], stride=config['training']['stride']
            )
            train_ds, val_ds, test_ds = walk_forward_split(dataset, train_ratio=0.7, val_ratio=0.15)
            
            from training.losses import FAPTGNNLoss
            from training.trainer import compute_pos_weight
            from models.fapt_gnn import FAPT_GNN
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = FAPT_GNN(**config['model'])
            pos_weight = compute_pos_weight(train_ds)
            loss_cfg = {k: v for k, v in config['loss'].items()}
            criterion = FAPTGNNLoss(pos_weight=pos_weight, **loss_cfg)
            
            # Short training for UI interaction speed (but more thorough than before)
            config['training']['epochs'] = 20 
            
            history = train(model, train_ds, val_ds, criterion, config['training'], device=device, checkpoint_dir=CHECKPOINT_DIR)
            st.success("Training sequence complete! Refreshing state...")
            st.rerun()

with tab3:
    st.header("Systemic Fragility & Real-Time Alert Engine")
    
    if model_exists:
        st.markdown("**Executing Forward Pass over today's rolling window sequence.**")
        
        # We simulate the extraction of today's inference block 
        with st.spinner("Extracting recent dynamics and executing graph traversal..."):
            try:
                from data.gdelt_sentiment import load_or_build_sentiment
                from data.feature_engineering import build_node_feature_matrix
                sent = load_or_build_sentiment(data["prices"].index, data["vix"], use_gdelt=use_gdelt)
                feats = build_all_features(data["prices"], data["vix"], data["macro"], sent, **config['features'])
                node_feats = build_node_feature_matrix(feats)
                graph_sequence, _ = build_graph_sequence(node_feats, feats, sent, window=config['graph']['graph_window'])
                
                # slice the input logic ... given this dashboard must be dynamic:
                latest_seq = graph_sequence[-config['model']['seq_len']:]
                
                model = FAPT_GNN(**config['model'])
                ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                
                with torch.no_grad():
                    crash_prob, tte_pred, instability, energy_seq, fragility_seq = model(latest_seq)
                
                prob_val = crash_prob.item()
                energy_val = energy_seq[-1].item()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Present Day Crash Probability", f"{prob_val*100:.2f}%", "--")
                c2.metric("Systemic Energy Level E(t)", f"{energy_val:.4f}", "--")
                ds_horizon = int(tte_pred.item())
                c3.metric("Expected Time Horizon (Days)", f"~ {ds_horizon}", "--")

                if prob_val > 0.65:
                   st.error("🚨 CRITICAL ALERT: FAPT-GNN Detects Heightened Fragility and Imminent Phase Transition.")
                elif prob_val > 0.4:
                   st.warning(">> WARNING: Elevated systemic energy detected in the NIFTY correlations structure.")
                else:
                   st.success("✓ Normal Market Phase. Component fragility is within stable manifold boundaries.")

                st.subheader("Energy Landscape Transition")
                energy_tensor = torch.cat([e.unsqueeze(0) for e in energy_seq]).squeeze().numpy()
                dates_seq = [g.date.date() for g in latest_seq]
                
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(x=dates_seq, y=energy_tensor, mode='lines+markers', name='Accumulating Real-Time Energy', line=dict(color='orange')))
                fig_energy.update_layout(title="Phase Transition Indicator (T-30 Days Trajectory)", yaxis_title="Systemic GNN Energy Formulation")
                st.plotly_chart(fig_energy, use_container_width=True)

            except Exception as e:
                st.error(f"Inference failed: {e}")
    else:
        st.info("Train the model in the 'Model Training' tab first to unlock Live Predictions.")

with tab4:
    st.header("Methodology Validation Overview")
    st.markdown("""
    To counter overfitting and small-sample critiques, the pipeline uses a strict **Walk-Forward Validation** (Temporal Splitting).
    - **No Look-Ahead Bias**: The training split never looks into the future graph linkages.
    - **Varying Crisis Exposures**: Spanning back to 2010 covers the 2015 Chinese stock market crash, the 2020 COVID flash crash, and the 2022 rate transitions.
    """)
    if model_exists:
        try:
           ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
           val_metrics = ckpt.get("val_metrics", {}).get("metrics", {})
           st.json(val_metrics)
        except Exception:
           pass
