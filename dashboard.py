import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yaml
import os
import torch
import warnings

warnings.filterwarnings('ignore')

# Provide absolute paths if necessary, assuming script runs from root
CONFIG_PATH = "experiments/config_fast.yaml"  # Use fast config - matches Colab training
CHECKPOINT_DIR = "experiments/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

st.set_page_config(page_title="FAPT-GNN Dashboard", layout="wide", page_icon=":chart_with_upwards_trend:")

st.markdown("""
<style>
/* Glassmorphism for metrics */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00e5ff;
}
div[data-testid="metric-container"] {
    background: rgba(21, 26, 35, 0.7);
    border: 1px solid rgba(0, 229, 255, 0.3);
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

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

with st.sidebar.expander("Model Hyperparameters", expanded=False):
    config['model']['gnn_hidden_dim'] = st.number_input("GNN Hidden Dim", value=config['model']['gnn_hidden_dim'])
    config['training']['epochs'] = st.number_input("Epochs", value=config['training']['epochs'])
    config['training']['lr'] = float(st.number_input("Learning Rate", value=config['training']['lr'], format="%.4f"))

st.markdown("""
    <h1 style='text-align: center; background: -webkit-linear-gradient(#00e5ff, #0077ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    FAPT-GNN: Fragility-Aware Phase Transition Graph Neural Network
    </h1>
    <h4 style='text-align: center; color: #888;'>AI-Powered Systemic Risk & Market Phase Indicator</h4>
    <hr>
""", unsafe_allow_html=True)
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
tab1, tab2, tab3 = st.tabs(["Market Context & Data", "Model Training", "Prediction & Phase Transition"])

with tab1:
    st.header("Historical Market Data & Labels")
    st.markdown("We address **Data Scarcity** by leveraging over a decade of market indices, ensuring diverse crisis exposure.")

    from plotly.subplots import make_subplots
    
    crash_indices = labels[labels['crash_label'] == 1].index
    
    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Total Trading Days", len(data['nifty']))
    metrics_cols[1].metric("Registered Systemic Crashes", len(crash_indices))
    metrics_cols[2].metric("Crash Ratio", f"{len(crash_indices)/len(data['nifty'])*100:.2f}%")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data['nifty'].index, y=data['nifty'].values, mode='lines', name='NIFTY 50', line=dict(color='#00e5ff', width=2), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.05)'), secondary_y=False)
    
    fig.add_trace(go.Scatter(x=data['vix'].index, y=data['vix'].values, mode='lines', name='India VIX', line=dict(color='rgba(255, 255, 255, 0.2)', width=1), fill='tozeroy', fillcolor='rgba(255, 255, 255, 0.02)'), secondary_y=True)

    if not crash_indices.empty:
        crash_prices = data['nifty'].loc[crash_indices]
        fig.add_trace(go.Scatter(x=crash_indices, y=crash_prices, mode='markers', name='Target Crashes', marker=dict(color='red', size=8, symbol='x')), secondary_y=False)

    fig.update_layout(
        title="NIFTY 50 Index & Volatility + Dynamic Crash Highlights", 
        xaxis_title="Date",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    fig.update_yaxes(title_text="Price (NIFTY 50)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Volatility (VIX)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

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
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            chart_data = {"Train Loss": [], "Val Loss": []}
            
            def update_ui(epoch, epochs, train_losses, val_losses, val_results):
                progress = epoch / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch}/{epochs} | Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f} | Val AUC: {val_results['metrics']['auc_roc']:.4f}")
                
                chart_data["Train Loss"].append(train_losses["total"])
                chart_data["Val Loss"].append(val_losses["total"])
                
                df = pd.DataFrame(chart_data, index=range(1, epoch + 1))
                chart_placeholder.line_chart(df)
            
            history = train(
                model, train_ds, val_ds, criterion, config['training'], 
                device=device, checkpoint_dir=CHECKPOINT_DIR,
                epoch_callback=update_ui
            )
            progress_bar.empty()
            status_text.empty()
            st.success("Training sequence complete! Refreshing state...")
            st.rerun()

with tab3:
    st.header("Systemic Fragility & Real-Time Alert Engine")
    
    if model_exists:
        st.markdown("**Executing Forward Pass over rolling window sequence.**")
        
        with st.spinner("Extracting recent dynamics and executing graph traversal..."):
            try:
                from data.gdelt_sentiment import load_or_build_sentiment
                from data.feature_engineering import build_node_feature_matrix
                import networkx as nx
                
                sent = load_or_build_sentiment(data["prices"].index, data["vix"], use_gdelt=use_gdelt)
                feats = build_all_features(data["prices"], data["vix"], data["macro"], sent, **config['features'])
                node_feats = build_node_feature_matrix(feats)
                graph_sequence, _ = build_graph_sequence(node_feats, feats, sent, window=config['graph']['graph_window'])
                
                # TIME TRAVEL SLIDER
                min_idx = config['model']['seq_len']
                dates_list = [pd.to_datetime(g.date).strftime("%Y-%m-%d") for g in graph_sequence]
                if len(dates_list) > min_idx:
                    selected_date = st.select_slider("Select Historical Inference Date", options=dates_list[min_idx:], value=dates_list[-1])
                    target_idx = dates_list.index(selected_date)
                    latest_seq = graph_sequence[target_idx + 1 - config['model']['seq_len']: target_idx + 1]
                else:
                    st.warning("Not enough data to run sequence length model")
                    st.stop()
                
                model = FAPT_GNN(**config['model'])
                ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                
                with torch.no_grad():
                    crash_prob, tte_pred, instability, energy_seq, fragility_seq = model(latest_seq)
                
                prob_val = crash_prob.item()
                energy_val = energy_seq[-1].item()
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Crash Probability ({selected_date})", f"{prob_val*100:.2f}%", "--")
                c2.metric("Systemic Energy Level E(t)", f"{energy_val:.4f}", "--")
                ds_horizon = int(tte_pred.item())
                c3.metric("Expected Time Horizon (Days)", f"~ {ds_horizon}", "--")

                if prob_val > 0.65:
                   st.error("🚨 CRITICAL ALERT: FAPT-GNN Detects Heightened Fragility and Imminent Phase Transition.")
                elif prob_val > 0.4:
                   st.warning(">> WARNING: Elevated systemic energy detected in the NIFTY correlations structure.")
                else:
                   st.success("✓ Normal Market Phase. Component fragility is within stable manifold boundaries.")

                # NETWORK VISUALIZATION
                target_graph = latest_seq[-1]
                adj = target_graph.adj.numpy() if hasattr(target_graph, 'adj') else None
                if adj is not None:
                    st.subheader("Network Fragility Graph")
                    G = nx.from_numpy_array(np.abs(adj))
                    pos = nx.spring_layout(G, dim=3, seed=42)
                    
                    edge_x, edge_y, edge_z = [], [], []
                    for edge in G.edges():
                        x0, y0, z0 = pos[edge[0]]
                        x1, y1, z1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_z.extend([z0, z1, None])
                    
                    edge_trace = go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        line=dict(width=2, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    node_x, node_y, node_z = [], [], []
                    node_text = []
                    fragilities = fragility_seq[-1].squeeze().detach().numpy()
                    tickers = node_feats['tickers']
                    
                    for node in G.nodes():
                        x, y, z = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_z.append(z)
                        
                        tck = tickers[node] if node < len(tickers) else str(node)
                        
                        # Find Most/Least Impacted neighbors (outgoing edges from this node)
                        row = np.abs(adj[node])
                        # Mask self-loop
                        mask = np.ones(len(row), dtype=bool)
                        mask[node] = False
                        
                        # neighbors indices (excluding self)
                        neighbor_indices = np.where(mask)[0]
                        neighbor_weights = row[neighbor_indices]
                        
                        if len(neighbor_weights) > 0:
                            # 1. Most Affects
                            max_idx_local = np.argmax(neighbor_weights)
                            max_idx = neighbor_indices[max_idx_local]
                            max_val = neighbor_weights[max_idx_local]
                            max_tck = tickers[max_idx] if max_idx < len(tickers) else str(max_idx)
                            
                            # 2. Least Affects (minimum of the rest after excluding most affected)
                            mask_least = np.ones(len(neighbor_weights), dtype=bool)
                            mask_least[max_idx_local] = False
                            
                            rem_indices = neighbor_indices[mask_least]
                            rem_weights = neighbor_weights[mask_least]
                            
                            if len(rem_weights) > 0:
                                # Prioritize lowest non-zero to provide informative labels
                                nz_mask = rem_weights > 1e-7
                                if np.any(nz_mask):
                                    nz_indices = rem_indices[nz_mask]
                                    nz_weights = rem_weights[nz_mask]
                                    min_idx_rem = np.argmin(nz_weights)
                                    min_idx = nz_indices[min_idx_rem]
                                    min_val = nz_weights[min_idx_rem]
                                else:
                                    # Fallback to absolute minimum if all are zero
                                    min_idx_rem = np.argmin(rem_weights)
                                    min_idx = rem_indices[min_idx_rem]
                                    min_val = rem_weights[min_idx_rem]
                                    
                                min_tck = tickers[min_idx] if min_idx < len(tickers) else str(min_idx)
                                
                                hover_info = (
                                    f"<b>{tck}</b><br>"
                                    f"Fragility Score: {fragilities[node]:.4f}<br>"
                                    f"───────────────────<br>"
                                    f"Most Affects: {max_tck} ({max_val:.4f})<br>"
                                    f"Least Affects: {min_tck} ({min_val:.4f})"
                                )
                            else:
                                hover_info = (
                                    f"<b>{tck}</b><br>"
                                    f"Fragility Score: {fragilities[node]:.4f}<br>"
                                    f"Most Affects: {max_tck} ({max_val:.4f})"
                                )
                        else:
                            hover_info = f"<b>{tck}</b><br>Fragility: {fragilities[node]:.4f}"
                            
                        node_text.append(hover_info)
                        
                    node_trace = go.Scatter3d(
                        x=node_x, y=node_y, z=node_z,
                        mode='markers',
                        hoverinfo='text',
                        text=node_text,
                        marker=dict(
                            showscale=True,
                            colorscale='YlOrRd',
                            color=fragilities,
                            size=8,
                            line=dict(width=1, color='#222'),
                            colorbar=dict(thickness=15, title='Fragility', xanchor='left')
                        )
                    )
                    
                    fig_net = go.Figure(data=[edge_trace, node_trace])
                    fig_net.update_layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='')
                        )
                    )
                    st.plotly_chart(fig_net, use_container_width=True)

                st.subheader("Energy Landscape Transition")
                energy_tensor = torch.cat([e.unsqueeze(0) for e in energy_seq]).squeeze().numpy()
                dates_seq = [g.date.date() for g in latest_seq]
                
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(x=dates_seq, y=energy_tensor, mode='lines+markers', name='Accumulating Energy', line=dict(color='#ff9100', width=3), fill='tozeroy', fillcolor='rgba(255, 145, 0, 0.1)'))
                fig_energy.update_layout(
                    title="Phase Transition Indicator (Trajectory)", 
                    yaxis_title="Systemic GNN Energy Formulation",
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_energy, use_container_width=True)

            except Exception as e:
                import traceback
                st.error(f"Inference failed: {e}\n{traceback.format_exc()}")
    else:
        st.info("Train the model in the 'Model Training' tab first to unlock Live Predictions.")


