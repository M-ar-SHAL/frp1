"""
Feature Engineering for FAPT-GNN
Computes the four FRAGILITY DRIVERS per node (stock) per timestep:

  σ_i(t) — Volatility Instability
  C_i(t) — Network Centrality  
  S_i(t) — Sentiment Divergence
  L_i(t) — Liquidity Stress

Plus additional features: returns, momentum, drawdown, correlation breakdown.

Final node feature vector:
  x_i(t) = [return, σ_i, C_i, S_i, L_i, volume, macro...]
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional, Tuple
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────
# 1. RETURNS & VOLATILITY (σ_i)
# ────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns for each stock. Fill isolated NaN (holiday gaps) with 0."""
    ret = np.log(prices / prices.shift(1))
    # dropna(how='all') only removes rows where EVERY stock is NaN.
    # fillna(0) replaces individual missing entries with 0 (no-change assumption).
    return ret.dropna(how='all').fillna(0)


def compute_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling std of log returns = σ_i(t).
    Window=20 ≈ 1 trading month.
    """
    return returns.rolling(window=window, min_periods=5).std().fillna(0)


def compute_garch_proxy(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    GARCH-like volatility clustering proxy:
    Exponentially weighted variance (fast, no GARCH fitting needed).
    """
    return returns.ewm(span=window, min_periods=5).std().fillna(0)


# ────────────────────────────────────────────────
# 2. LIQUIDITY STRESS (L_i)
# ────────────────────────────────────────────────

def compute_volume_spike_ratio(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume spike ratio: V_t / rolling_mean(V, window).
    Proxy for liquidity disruption (high spike = stressed liquidity).
    Note: uses price-based Amihud proxy when volume unavailable.
    """
    returns = compute_returns(prices)
    # Amihud illiquidity proxy: |return| / |price change| (no volume needed)
    illiquidity = returns.abs()
    liquidity_stress = illiquidity.rolling(window=window, min_periods=5).mean()
    # Normalize: spike above rolling baseline
    rolling_mean = liquidity_stress.rolling(window=window, min_periods=5).mean()
    rolling_std = liquidity_stress.rolling(window=window, min_periods=5).std().replace(0, 1e-8)
    spike = (liquidity_stress - rolling_mean) / rolling_std
    return spike.fillna(0)


def compute_turnover_proxy(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Intraday range as turnover proxy: (High-Low)/Close."""
    # Using close price changes as proxy (if only Close data available)
    returns = compute_returns(prices)
    return returns.rolling(window=window, min_periods=3).apply(
        lambda x: np.percentile(np.abs(x), 90), raw=True
    ).fillna(0)


# ────────────────────────────────────────────────
# 3. NETWORK CENTRALITY (C_i) — Per timestep
# ────────────────────────────────────────────────

def compute_correlation_matrix(returns_window: pd.DataFrame, threshold: float = 0.3) -> np.ndarray:
    """
    Compute Spearman correlation matrix for a returns window.
    Threshold: only keep edges with |corr| > threshold (sparse graph).
    """
    corr = returns_window.corr(method="spearman").values.copy()
    np.fill_diagonal(corr, 0)
    adj = np.where(np.abs(corr) > threshold, corr, 0)
    return adj


def compute_network_centrality(adj_matrix: np.ndarray, tickers: list) -> Dict[str, float]:
    """
    Compute degree centrality for each node from the adjacency matrix.
    Returns dict: {ticker: centrality_score}
    """
    G = nx.from_numpy_array(np.abs(adj_matrix))
    # Weighted degree centrality: sum of edge weights
    centrality = {}
    for i, ticker in enumerate(tickers):
        centrality[ticker] = np.sum(np.abs(adj_matrix[i]))  # weighted degree
    # Normalize
    max_c = max(centrality.values()) if max(centrality.values()) > 0 else 1
    return {k: v / max_c for k, v in centrality.items()}


def compute_rolling_centrality(
    returns: pd.DataFrame,
    window: int = 60,
    step: int = 1,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Compute C_i(t) for every timestep using rolling window correlation graph.
    Builds a list of {ticker: centrality} rows indexed by date to avoid
    column-mismatch from pd.DataFrame(dict).T when the dict is empty or
    the tickers differ from what pandas infers as column names.
    """
    tickers = returns.columns.tolist()
    rows = []
    dates = []

    for i in range(window, len(returns)):
        date = returns.index[i]
        window_data = returns.iloc[i - window:i]
        adj = compute_correlation_matrix(window_data, threshold)
        cent = compute_network_centrality(adj, tickers)  # {ticker: score}
        rows.append(cent)
        dates.append(date)

    if rows:
        centrality_df = pd.DataFrame(rows, index=dates, columns=tickers)
    else:
        # Edge case: not enough data for even one window
        centrality_df = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)

    # Backfill the first `window` rows that had no data yet
    centrality_df = centrality_df.reindex(returns.index).bfill().ffill()
    return centrality_df.fillna(0)


# ────────────────────────────────────────────────
# 4. MOMENTUM & DRAWDOWN
# ────────────────────────────────────────────────

def compute_momentum(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling cumulative return (momentum)."""
    return returns.rolling(window=window, min_periods=5).sum().fillna(0)


def compute_drawdown(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling maximum drawdown per stock."""
    rolling_max = prices.rolling(window=window, min_periods=10).max()
    drawdown = (prices - rolling_max) / rolling_max.replace(0, 1e-8)
    return drawdown.fillna(0)


def compute_correlation_breakdown(returns: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Correlation breakdown: mean pairwise correlation increasing = contagion rising.
    System-level signal (same value for all nodes at time t).
    During crashes, correlations spike toward 1 (everything falls together).
    """
    breakdown = []
    for i in range(window, len(returns)):
        window_data = returns.iloc[i - window:i]
        corr_matrix = window_data.corr().values
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = np.nanmean(corr_matrix[mask])
        breakdown.append((returns.index[i], mean_corr))

    if breakdown:
        idx, vals = zip(*breakdown)
        series = pd.Series(vals, index=idx, name="corr_breakdown")
        # Align to full returns index, forward/back fill missing early values
        series = series.reindex(returns.index).bfill().ffill().fillna(0)
    else:
        # Not enough data for a single window – return zeros for every timestep
        series = pd.Series(0.0, index=returns.index, name="corr_breakdown")
    return series


# ────────────────────────────────────────────────
# 5. MACRO FEATURES
# ────────────────────────────────────────────────

def compute_macro_features(macro_df: pd.DataFrame, returns_index: pd.Index) -> pd.DataFrame:
    """
    Normalize macro indicators and align to returns index.
    Returns a DataFrame with same index as returns.
    """
    macro = macro_df.reindex(returns_index).ffill().bfill()
    # Compute macro returns (changes)
    macro_ret = macro.pct_change().fillna(0)
    macro_ret.columns = [f"macro_{c}_ret" for c in macro_ret.columns]
    # Rolling z-score normalization
    rolling_mean = macro_ret.rolling(60, min_periods=10).mean()
    rolling_std = macro_ret.rolling(60, min_periods=10).std().replace(0, 1e-8)
    macro_z = ((macro_ret - rolling_mean) / rolling_std).fillna(0).clip(-5, 5)
    return macro_z


# ────────────────────────────────────────────────
# 6. MASTER FEATURE BUILDER
# ────────────────────────────────────────────────

def build_all_features(
    prices: pd.DataFrame,
    vix: pd.Series,
    macro: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    vol_window: int = 20,
    centrality_window: int = 60,
    liquidity_window: int = 20,
    centrality_threshold: float = 0.3
) -> Dict[str, pd.DataFrame]:
    """
    Build all node features and system-level features.

    Returns dict with:
      - 'returns'     : log returns (N stocks)
      - 'volatility'  : σ_i(t) — volatility instability
      - 'centrality'  : C_i(t) — network centrality
      - 'liquidity'   : L_i(t) — liquidity stress
      - 'momentum'    : momentum signals
      - 'drawdown'    : drawdown signals
      - 'macro'       : macro feature matrix
      - 'corr_breakdown': system correlation breakdown scalar
      - 'sentiment'   : S_i(t) — sentiment divergence (from sentiment_features)
    """
    print("[Features] Computing log returns...")
    returns = compute_returns(prices)

    # Align all to returns index
    common_idx = returns.index

    print("[Features] Computing volatility sigma_i(t)...")
    volatility = compute_volatility(returns, vol_window)
    garch_vol = compute_garch_proxy(returns, vol_window)

    print("[Features] Computing liquidity stress L_i(t)...")
    liquidity = compute_volume_spike_ratio(prices.reindex(common_idx), liquidity_window)

    print("[Features] Computing network centrality C_i(t) (this takes a few minutes)...")
    centrality = compute_rolling_centrality(returns, centrality_window, threshold=centrality_threshold)

    print("[Features] Computing momentum & drawdown...")
    momentum = compute_momentum(returns, 20)
    drawdown = compute_drawdown(prices.reindex(common_idx), 60)

    print("[Features] Computing correlation breakdown signal...")
    corr_breakdown = compute_correlation_breakdown(returns, 30)

    print("[Features] Processing macro features...")
    macro_feats = compute_macro_features(macro, common_idx)

    # Sentiment: align to common index
    sentiment = sentiment_features.reindex(common_idx).ffill().bfill()

    print(f"\n[Features] [OK] Feature engineering complete.")
    print(f"  Shape: {returns.shape[0]} timesteps × {returns.shape[1]} stocks")

    return {
        "returns": returns,
        "volatility": volatility,
        "garch_vol": garch_vol,
        "centrality": centrality,
        "liquidity": liquidity,
        "momentum": momentum,
        "drawdown": drawdown,
        "corr_breakdown": corr_breakdown,
        "macro": macro_feats,
        "sentiment": sentiment,
    }


def build_node_feature_matrix(features: Dict) -> Dict[str, pd.DataFrame]:
    """
    Build the final node feature matrix x_i(t) for each timestep.

    x_i(t) = [return, sigma_i, centrality_i, sentiment_diverge, liquidity_i,
               momentum_i, drawdown_i]  (7 features per node)

    Also returns system-level features (same for all nodes):
    z(t) = [E_VIX, corr_breakdown, macro...]
    """
    returns = features["returns"]
    tickers = returns.columns.tolist()
    common_idx = returns.index

    # Per-stock features: each is (T x N)
    per_stock = {
        "return": returns,
        "volatility": features["volatility"],
        "centrality": features["centrality"],
        "liquidity": features["liquidity"],
        "momentum": features["momentum"],
        "drawdown": features["drawdown"],
    }

    # Add sentiment divergence per stock (broadcast scalar to all stocks)
    sent_div = features["sentiment"]["sentiment_diverge"]
    per_stock["sentiment"] = pd.DataFrame(
        np.outer(sent_div.reindex(common_idx).values, np.ones(len(tickers))),
        index=common_idx,
        columns=tickers
    )

    # System-level series (T,) — will be broadcast to all nodes in model
    system_features = pd.DataFrame(index=common_idx)
    system_features["corr_breakdown"] = features["corr_breakdown"]
    system_features["vix_sentiment"] = features["sentiment"]["vix_norm"]
    system_features = system_features.join(features["macro"])
    system_features = system_features.fillna(0)

    return {
        "per_stock": per_stock,    # dict of (T x N) DataFrames
        "system": system_features, # (T x macro_dim) DataFrame
        "tickers": tickers,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.data_pipeline import load_all_data
    from data.gdelt_sentiment import load_or_build_sentiment

    data = load_all_data(start="2018-01-01", end="2021-12-31")
    sentiment_features = load_or_build_sentiment(
        price_index=data["prices"].index,
        vix_series=data["vix"],
        use_gdelt=False
    )
    features = build_all_features(
        prices=data["prices"],
        vix=data["vix"],
        macro=data["macro"],
        sentiment_features=sentiment_features
    )
    node_feat = build_node_feature_matrix(features)
    print("\nNode feature keys:", list(node_feat["per_stock"].keys()))
    print("System features:", node_feat["system"].shape)
    print("Sample centrality:\n", node_feat["per_stock"]["centrality"].tail(3))
