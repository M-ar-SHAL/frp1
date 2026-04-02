"""
Multi-Layer Dynamic Graph Builder for FAPT-GNN

Constructs 4 graph layers as described in the paper:

  Layer 1: Correlation Graph     — w_ij = corr(r_i, r_j)
  Layer 2: Sector Graph          — w_ij = 1 if same NIFTY sector
  Layer 3: Sentiment Graph       — w_ij = similarity of sentiment trends
  Layer 4: Volatility Spillover  — w_ij = corr(σ_i, σ_j)

Final adjacency:
  A_t = α*A_corr + β*A_sector + γ*A_sent + δ*A_vol

Also builds PyG (PyTorch Geometric) format graph objects
for use in the GNN model.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────
# NIFTY 50 SECTOR MAPPING
# ────────────────────────────────────────────────
SECTOR_MAP = {
    # Banking & Finance
    "HDFCBANK.NS": 0, "ICICIBANK.NS": 0, "SBIN.NS": 0, "KOTAKBANK.NS": 0,
    "AXISBANK.NS": 0, "BAJFINANCE.NS": 0, "BAJAJFINSV.NS": 0,
    "INDUSINDBK.NS": 0, "SBILIFE.NS": 0, "HDFCLIFE.NS": 0,

    # IT & Technology
    "TCS.NS": 1, "INFY.NS": 1, "HCLTECH.NS": 1, "WIPRO.NS": 1,
    "TECHM.NS": 1, "LTIM.NS": 1,

    # Energy & Oil
    "RELIANCE.NS": 2, "ONGC.NS": 2, "BPCL.NS": 2, "COALINDIA.NS": 2,
    "NTPC.NS": 2, "POWERGRID.NS": 2, "ADANIENT.NS": 2,

    # Consumer & FMCG
    "HINDUNILVR.NS": 3, "ITC.NS": 3, "NESTLEIND.NS": 3, "BRITANNIA.NS": 3,
    "TATACONSUM.NS": 3, "TITAN.NS": 3,

    # Auto & Manufacturing
    "MARUTI.NS": 4, "TATAMOTORS.NS": 4, "EICHERMOT.NS": 4,
    "HEROMOTOCO.NS": 4, "M&M.NS": 4, "BAJAJ-AUTO.NS": 4,

    # Metals & Materials
    "JSWSTEEL.NS": 5, "TATASTEEL.NS": 5, "HINDALCO.NS": 5,
    "ADANIPORTS.NS": 5, "GRASIM.NS": 5, "ULTRACEMCO.NS": 5,

    # Pharma & Healthcare
    "SUNPHARMA.NS": 6, "CIPLA.NS": 6, "DRREDDY.NS": 6,
    "DIVISLAB.NS": 6, "APOLLOHOSP.NS": 6,

    # Others
    "ASIANPAINT.NS": 7, "LT.NS": 7, "UPL.NS": 7,
}

# Layer weights (can be tuned as hyperparameters)
DEFAULT_LAYER_WEIGHTS = {
    "alpha": 0.4,   # correlation graph
    "beta": 0.25,   # sector graph
    "gamma": 0.2,   # sentiment graph
    "delta": 0.15,  # volatility spillover
}


# ────────────────────────────────────────────────
# LAYER 1: CORRELATION GRAPH
# ────────────────────────────────────────────────

def build_correlation_graph(
    returns_window: pd.DataFrame,
    threshold: float = 0.2,
    method: str = "spearman"
) -> np.ndarray:
    """
    Layer 1: Price correlation graph.
    w_ij = max(0, corr(r_i, r_j)) if corr > threshold else 0
    Only keep positive correlations (co-movement risk).
    """
    n = returns_window.shape[1]
    if method == "spearman":
        corr = returns_window.corr(method="spearman").values
    else:
        corr = returns_window.corr(method="pearson").values

    np.fill_diagonal(corr, 0)
    adj = np.where(corr > threshold, corr, 0)
    return adj.astype(np.float32)


# ────────────────────────────────────────────────
# LAYER 2: SECTOR GRAPH
# ────────────────────────────────────────────────

def build_sector_graph(tickers: list) -> np.ndarray:
    """
    Layer 2: Sector exposure graph (static).
    w_ij = 1.0 if same NIFTY sector, 0.3 if adjacent sector, else 0.
    """
    n = len(tickers)
    adj = np.zeros((n, n), dtype=np.float32)

    for i, t_i in enumerate(tickers):
        for j, t_j in enumerate(tickers):
            if i == j:
                continue
            s_i = SECTOR_MAP.get(t_i, -1)
            s_j = SECTOR_MAP.get(t_j, -2)
            if s_i == s_j and s_i != -1:
                adj[i, j] = 1.0

    return adj


# ────────────────────────────────────────────────
# LAYER 3: SENTIMENT GRAPH
# ────────────────────────────────────────────────

def build_sentiment_graph(
    sentiment_window: pd.Series,
    tickers: list,
    volatility_window: pd.DataFrame
) -> np.ndarray:
    """
    Layer 3: Sentiment similarity graph.
    w_ij = similarity of sentiment × volatility co-movement.
    (Since sentiment is market-wide, we modulate by vol similarity.)
    """
    n = len(tickers)

    # Sentiment is scalar (market-wide); differentiate by volatility correlation
    vol_corr = volatility_window.corr(method="spearman").values
    np.fill_diagonal(vol_corr, 0)

    # Sentiment magnitude (absolute fear level)
    sent_magnitude = np.abs(sentiment_window.mean()) if len(sentiment_window) > 0 else 0.5
    sent_magnitude = min(sent_magnitude / 10.0, 1.0)  # normalize to [0, 1]

    # Sentiment graph: vol corr × sentiment magnitude
    adj = np.maximum(vol_corr, 0) * sent_magnitude
    return adj.astype(np.float32)


# ────────────────────────────────────────────────
# LAYER 4: VOLATILITY SPILLOVER
# ────────────────────────────────────────────────

def build_volatility_spillover_graph(
    volatility_window: pd.DataFrame,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Layer 4: Volatility spillover graph.
    w_ij = corr(σ_i, σ_j) — captures cross-stock volatility clustering.
    """
    corr = volatility_window.corr(method="spearman").values
    np.fill_diagonal(corr, 0)
    adj = np.where(corr > threshold, corr, 0)
    return adj.astype(np.float32)


# ────────────────────────────────────────────────
# COMBINED MULTI-LAYER ADJACENCY
# ────────────────────────────────────────────────

def build_multilayer_adjacency(
    returns_window: pd.DataFrame,
    volatility_window: pd.DataFrame,
    sentiment_window: pd.Series,
    tickers: list,
    sector_adj: Optional[np.ndarray] = None,
    weights: Dict[str, float] = None,
    corr_threshold: float = 0.2,
) -> np.ndarray:
    """
    Build combined multi-layer adjacency matrix:
    A_t = α*A_corr + β*A_sector + γ*A_sent + δ*A_vol

    Normalized to [0, 1] range.
    """
    if weights is None:
        weights = DEFAULT_LAYER_WEIGHTS

    A_corr = build_correlation_graph(returns_window, corr_threshold)
    A_sector = sector_adj if sector_adj is not None else build_sector_graph(tickers)
    A_sent = build_sentiment_graph(sentiment_window, tickers, volatility_window)
    A_vol = build_volatility_spillover_graph(volatility_window, corr_threshold)

    A = (weights["alpha"] * A_corr +
         weights["beta"] * A_sector +
         weights["gamma"] * A_sent +
         weights["delta"] * A_vol)

    # Handle NaN values before normalization
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize rows to sum to 1 (stochastic adjacency)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    A_norm = A / row_sums
    
    # Final NaN cleanup
    A_norm = np.nan_to_num(A_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return A_norm.astype(np.float32)


# ────────────────────────────────────────────────
# CONVERT TO PyTorch Geometric FORMAT
# ────────────────────────────────────────────────

def adj_to_edge_index(adj: np.ndarray, threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert adjacency matrix to edge_index and edge_weight for PyG.
    Only include edges above threshold for sparsity.
    """
    rows, cols = np.where(adj > threshold)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float32)
    return edge_index, edge_weight


def build_pyg_graph(
    node_features: np.ndarray,  # (N, d) — node feature matrix at time t
    adj: np.ndarray,            # (N, N) — adjacency matrix at time t
    fragility: Optional[np.ndarray] = None,  # (N,) — ground-truth fragility if known
    threshold: float = 0.001  # Reduced from 0.01 to capture more edges
) -> Data:
    """
    Build a PyTorch Geometric Data object for a single timestep.
    """
    edge_index, edge_weight = adj_to_edge_index(adj, threshold)
    x = torch.tensor(node_features, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(-1),
        num_nodes=x.shape[0],
    )
    if fragility is not None:
        data.fragility = torch.tensor(fragility, dtype=torch.float32)

    return data


# ────────────────────────────────────────────────
# ROLLING GRAPH SEQUENCE BUILDER
# ────────────────────────────────────────────────

def build_graph_sequence(
    node_features_dict: Dict,   # from feature_engineering.build_node_feature_matrix()
    features_raw: Dict,          # full features dict
    sentiment_features: pd.DataFrame,
    window: int = 60,            # rolling window for graph construction
    node_feature_keys: List[str] = None,
) -> List[Data]:
    """
    Build a time-ordered list of PyG graph objects.
    One graph per trading day (after warm-up window).

    Each graph represents the state of the financial network at time t.
    """
    if node_feature_keys is None:
        node_feature_keys = ["return", "volatility", "centrality", "liquidity",
                              "momentum", "drawdown", "sentiment"]

    tickers = node_features_dict["tickers"]
    per_stock = node_features_dict["per_stock"]
    n_stocks = len(tickers)

    # Static sector graph (computed once)
    sector_adj = build_sector_graph(tickers)

    # Common index
    returns = features_raw["returns"]
    volatility = features_raw["volatility"]
    T = len(returns)

    graphs = []
    print(f"[Graph] Building {T - window} graph snapshots...")

    for i in range(window, T):
        date = returns.index[i]

        # Rolling windows for graph construction
        ret_window = returns.iloc[i - window:i]
        vol_window = volatility.iloc[i - window:i]

        # Sentiment window (scalar series)
        sent_window = sentiment_features["sentiment_raw"].iloc[i - window:i] \
            if "sentiment_raw" in sentiment_features.columns \
            else pd.Series([0.0])

        # Build adjacency matrix for this timestep
        adj = build_multilayer_adjacency(
            returns_window=ret_window,
            volatility_window=vol_window,
            sentiment_window=sent_window,
            tickers=tickers,
            sector_adj=sector_adj,
        )

        # Build node feature matrix: (N, d)
        feat_cols = []
        for key in node_feature_keys:
            if key in per_stock:
                col = per_stock[key].loc[date].values
                feat_cols.append(col)

        node_feat_matrix = np.stack(feat_cols, axis=1)  # (N, d)
        # Clip & normalize
        node_feat_matrix = np.clip(node_feat_matrix, -10, 10)
        node_feat_matrix = np.nan_to_num(node_feat_matrix, nan=0.0)

        pyg_data = build_pyg_graph(node_feat_matrix, adj)
        pyg_data.date = date
        pyg_data.adj = torch.tensor(adj, dtype=torch.float32)

        graphs.append(pyg_data)

    print(f"[Graph] Built {len(graphs)} graph snapshots.")
    return graphs, returns.index[window:]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.data_pipeline import load_all_data
    from data.gdelt_sentiment import load_or_build_sentiment
    from data.feature_engineering import build_all_features, build_node_feature_matrix

    data = load_all_data(start="2020-01-01", end="2022-12-31")
    sent = load_or_build_sentiment(data["prices"].index, data["vix"], use_gdelt=False)
    feats = build_all_features(data["prices"], data["vix"], data["macro"], sent)
    node_feats = build_node_feature_matrix(feats)

    graphs, dates = build_graph_sequence(node_feats, feats, sent, window=30)
    g = graphs[0]
    print(f"\nFirst graph:")
    print(f"  Nodes: {g.num_nodes}, Edges: {g.edge_index.shape[1]}")
    print(f"  Node features shape: {g.x.shape}")
    print(f"  Edge weights shape: {g.edge_attr.shape}")
