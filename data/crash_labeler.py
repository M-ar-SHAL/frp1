"""
Crash Labeler for FAPT-GNN
Creates binary crash labels for NIFTY 50 based on:

  Method 1 (Primary) : Top 5% worst returns (threshold-based)
  Method 2 (Secondary): Drawdown > 10% over 10 days
  Method 3 (Combined) : Union of both (for robust labeling)

Also creates:
  - τ_t: time-to-crash (days until next crash event)
  - I_t: instability index (rolling worst-return percentile)

These match the paper's output head requirements.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def label_crashes_percentile(
    nifty_series: pd.Series,
    percentile: float = 5.0,
    window: int = 1,
    forward_days: int = 5
) -> pd.Series:
    """
    Method 1: Label a day as crash if the FORWARD return over `forward_days`
    falls in the bottom `percentile`% of all returns.

    Args:
        nifty_series : NIFTY 50 index close prices
        percentile   : bottom X% = crash (default 5%)
        forward_days : forward-looking window for return
        
    Returns:
        crash_label: binary Series (1=crash, 0=normal)
    """
    # Compute forward returns
    fwd_returns = nifty_series.pct_change(forward_days).shift(-forward_days)

    threshold = np.percentile(fwd_returns.dropna(), percentile)
    crash_label = (fwd_returns < threshold).astype(int)
    crash_label.name = "crash_label_pct"
    return crash_label


def label_crashes_drawdown(
    nifty_series: pd.Series,
    drawdown_threshold: float = -0.07,
    window: int = 10
) -> pd.Series:
    """
    Method 2: Label a day as crash if max drawdown over next `window` days
    exceeds `drawdown_threshold` (e.g., -7% = Indian market correction threshold).
    """
    rolling_max = nifty_series.rolling(window=window).max()

    def fwd_drawdown(series, i, window):
        end = min(i + window, len(series))
        future_slice = series.iloc[i:end]
        peak = series.iloc[i]
        if peak == 0:
            return 0
        dd = (future_slice.min() - peak) / peak
        return dd

    dds = []
    for i in range(len(nifty_series)):
        dd = fwd_drawdown(nifty_series, i, window)
        dds.append(dd)

    dd_series = pd.Series(dds, index=nifty_series.index, name="fwd_drawdown")
    crash_label = (dd_series < drawdown_threshold).astype(int)
    crash_label.name = "crash_label_dd"
    return crash_label


def label_crashes_combined(
    nifty_series: pd.Series,
    percentile: float = 5.0,
    drawdown_threshold: float = -0.07,
    forward_days: int = 5,
    dd_window: int = 10
) -> pd.Series:
    """
    Method 3 (Paper's recommended approach):
    Union of percentile method + drawdown method.
    More robust for capturing all market stress events.
    """
    label_pct = label_crashes_percentile(nifty_series, percentile, forward_days=forward_days)
    label_dd = label_crashes_drawdown(nifty_series, drawdown_threshold, dd_window)

    combined = ((label_pct == 1) | (label_dd == 1)).astype(int)
    combined.name = "crash_label"
    return combined


def compute_time_to_crash(crash_labels: pd.Series, max_horizon: int = 60) -> pd.Series:
    """
    Compute τ_t: number of days until next crash event.
    τ = 0 if today is a crash day.
    τ = max_horizon if no crash within horizon.

    This is the time-to-event regression target (τ_t in the paper).
    """
    labels = crash_labels.values
    n = len(labels)
    tte = np.full(n, max_horizon, dtype=np.float32)

    for i in range(n):
        if labels[i] == 1:
            tte[i] = 0
        else:
            found = False
            for j in range(i + 1, min(i + max_horizon + 1, n)):
                if labels[j] == 1:
                    tte[i] = j - i
                    found = True
                    break
            if not found:
                tte[i] = max_horizon

    return pd.Series(tte, index=crash_labels.index, name="time_to_crash")


def compute_instability_index(
    returns: pd.DataFrame,
    window: int = 20
) -> pd.Series:
    """
    Compute I_t: rolling system-level instability index.
    Defined as mean of |portfolio loss percentile rank| across stocks.
    Corresponds to the Instability Index output head in the paper.
    """
    # Rolling worst-return rank (0=calm, 1=extreme stress)
    mean_market_return = returns.mean(axis=1)  # equal-weight portfolio return
    rolling_min = mean_market_return.rolling(window=window, min_periods=5).min()
    rolling_max = mean_market_return.rolling(window=window, min_periods=5).max()
    denom = (rolling_max - rolling_min).replace(0, 1e-8)
    instability = 1 - (mean_market_return - rolling_min) / denom  # 1 = very bad
    instability = instability.clip(0, 1)
    instability.name = "instability_index"
    return instability.fillna(0.5)


def create_labels(
    nifty_series: pd.Series,
    returns: pd.DataFrame,
    percentile: float = 5.0,
    drawdown_threshold: float = -0.07,
    forward_days: int = 5,
    dd_window: int = 10,
    max_tte_horizon: int = 60
) -> pd.DataFrame:
    """
    Master label creator. Returns DataFrame with:
      - crash_label      : binary (0/1)
      - time_to_crash    : τ_t (days to next crash)
      - instability_index: I_t (0-1 score)
    """
    print("[Labels] Computing crash labels...")
    crash_label = label_crashes_combined(
        nifty_series,
        percentile=percentile,
        drawdown_threshold=drawdown_threshold,
        forward_days=forward_days,
        dd_window=dd_window
    )

    print("[Labels] Computing time-to-crash (τ_t)...")
    tte = compute_time_to_crash(crash_label, max_horizon=max_tte_horizon)

    print("[Labels] Computing instability index (I_t)...")
    instability = compute_instability_index(returns)

    labels = pd.DataFrame({
        "crash_label": crash_label,
        "time_to_crash": tte,
        "instability_index": instability
    }, index=crash_label.index)

    crash_rate = crash_label.mean() * 100
    print(f"\n[Labels] ✅ Label stats:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Crash days: {crash_label.sum()} ({crash_rate:.1f}%)")
    print(f"  Mean time-to-crash: {tte.mean():.1f} days")
    print(f"  Mean instability: {instability.mean():.3f}")

    return labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.data_pipeline import load_all_data
    from data.feature_engineering import compute_returns

    data = load_all_data(start="2015-01-01", end="2023-12-31")
    returns = compute_returns(data["prices"])

    labels = create_labels(
        nifty_series=data["nifty"],
        returns=returns
    )
    print("\nLabel sample:\n", labels.tail(10))

    # Show crash events
    crashes = labels[labels["crash_label"] == 1]
    print(f"\nCrash dates ({len(crashes)} events):")
    print(crashes.index[:20].tolist())
