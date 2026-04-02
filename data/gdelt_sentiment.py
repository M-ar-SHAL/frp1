"""
GDELT Sentiment Pipeline for FAPT-GNN
Fetches news sentiment signals for Indian financial markets
using the GDELT 2.0 GKG (Global Knowledge Graph) API.

GDELT is 100% FREE — no registration, no API key required.
It is a legitimate source for research papers.

Reference: "The GDELT Project" — https://www.gdeltproject.org
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time
import warnings

warnings.filterwarnings("ignore")

# GDELT GKG API endpoint (free)
GDELT_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Keywords targeting Indian financial market news
INDIA_FINANCE_KEYWORDS = [
    "India stock market",
    "NIFTY 50",
    "BSE Sensex",
    "RBI monetary policy",
    "Indian economy crash",
    "India market crash",
    "Dalal Street",
    "NSE India",
    "SEBI",
    "India financial crisis"
]


def fetch_gdelt_sentiment_day(date: str, keywords: list = None) -> float:
    """
    Fetch average sentiment tone for Indian market news on a given date.
    GDELT tone: negative = negative sentiment, positive = positive.

    Args:
        date: "YYYY-MM-DD" format
        keywords: list of search keywords

    Returns:
        avg_tone: float (negative = bearish, positive = bullish)
    """
    if keywords is None:
        keywords = ["India stock market crash", "NIFTY", "Indian economy"]

    # Format date for GDELT API (YYYYMMDD)
    date_fmt = date.replace("-", "")
    query = " OR ".join([f'"{kw}"' for kw in keywords[:3]])  # Keep query short

    params = {
        "query": f"{query} sourcelang:english",
        "mode": "ArtList",
        "maxrecords": "50",
        "startdatetime": f"{date_fmt}000000",
        "enddatetime": f"{date_fmt}235959",
        "format": "json",
    }

    try:
        resp = requests.get(GDELT_API_BASE, params=params, timeout=15)
        if resp.status_code != 200:
            return np.nan

        data = resp.json()
        articles = data.get("articles", [])

        if not articles:
            return np.nan

        # Extract tone scores from articles
        tones = []
        for art in articles:
            # GDELT tone: comma-separated "tone,pos,neg,polarity,..."
            tone_str = art.get("tone", "")
            if tone_str:
                try:
                    tone_val = float(tone_str.split(",")[0])
                    tones.append(tone_val)
                except ValueError:
                    continue

        return float(np.mean(tones)) if tones else np.nan

    except Exception as e:
        return np.nan


def build_sentiment_series(
    start: str = "2015-01-01",
    end: str = None,
    cache_path: str = "data/cache/gdelt_sentiment.parquet",
    sleep_sec: float = 0.5
) -> pd.Series:
    """
    Build a daily sentiment time series using GDELT.
    Caches results to avoid re-fetching.

    NOTE: For a large date range, this can take a while due to rate limiting.
    We use India VIX as a fallback/supplement (see below).
    """
    import os
    os.makedirs("data/cache", exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[Sentiment] Loading cached GDELT data from {cache_path}")
        return pd.read_parquet(cache_path)["sentiment"]

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start, end=end, freq="B")  # Business days
    sentiment_dict = {}

    print(f"[Sentiment] Fetching GDELT sentiment for {len(dates)} trading days...")
    print("[Sentiment] This may take a while due to rate limits. Results will be cached.")

    for i, dt in enumerate(dates):
        date_str = dt.strftime("%Y-%m-%d")
        tone = fetch_gdelt_sentiment_day(date_str)
        sentiment_dict[dt] = tone

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dates)}] Date: {date_str}, Tone: {tone:.3f}" if not np.isnan(tone) else f"  [{i+1}/{len(dates)}] Date: {date_str}, Tone: N/A")

        time.sleep(sleep_sec)  # Be polite to the free API

    sentiment = pd.Series(sentiment_dict, name="sentiment")
    sentiment = sentiment.interpolate(method="linear").ffill().bfill()

    # Save cache
    pd.DataFrame({"sentiment": sentiment}).to_parquet(cache_path)
    print(f"[Sentiment] Saved GDELT sentiment to {cache_path}")

    return sentiment


def derive_sentiment_features(
    sentiment_series: pd.Series,
    vix_series: pd.Series,
    window: int = 5
) -> pd.DataFrame:
    """
    Derive sentiment features used as node feature S_i(t).

    Features:
      - sentiment_raw     : raw GDELT tone (or VIX-derived)
      - sentiment_ma      : rolling mean (market-wide sentiment)
      - sentiment_diverge : deviation from rolling mean
      - sentiment_vol     : rolling std (disagreement/uncertainty)
      - vix_norm          : normalized India VIX

    These map to S_i in our fragility field formula.
    """
    df = pd.DataFrame(index=sentiment_series.index)

    # Fill missing with VIX-derived proxy (negative VIX → bearish sentiment)
    # Normalize VIX to (-1, 0) range so it acts as negative sentiment proxy
    vix_aligned = vix_series.reindex(sentiment_series.index).ffill().bfill()
    vix_norm = -((vix_aligned - vix_aligned.min()) / (vix_aligned.max() - vix_aligned.min()))  # 0=calm, -1=extreme fear

    # Fill NaN sentiment with VIX proxy
    sent = sentiment_series.copy()
    sent = sent.fillna(vix_norm * 10)  # scale to GDELT tone range

    df["sentiment_raw"] = sent
    df["sentiment_ma"] = sent.rolling(window=window, min_periods=1).mean()
    df["sentiment_diverge"] = sent - df["sentiment_ma"]  # S_i in paper
    df["sentiment_vol"] = sent.rolling(window=window, min_periods=1).std().fillna(0)
    df["vix_norm"] = vix_norm

    return df


def load_or_build_sentiment(
    price_index: pd.Index,
    vix_series: pd.Series,
    use_gdelt: bool = False,  # Set True to fetch GDELT (slow for long ranges)
    cache_path: str = "data/cache/gdelt_sentiment.parquet"
) -> pd.DataFrame:
    """
    Main entry: load or build sentiment features.

    If use_gdelt=False (default), uses India VIX as sentiment proxy.
    This is scientifically valid for research papers:
      - India VIX reflects implied volatility = market fear = negative sentiment
      - Cited in finance literature as behavioral/sentiment indicator

    Set use_gdelt=True to fetch actual GDELT news sentiment (recommended for paper).
    """
    if use_gdelt:
        start = price_index[0].strftime("%Y-%m-%d")
        end = price_index[-1].strftime("%Y-%m-%d")
        sentiment_raw = build_sentiment_series(start, end, cache_path)
    else:
        print("[Sentiment] Using India VIX as sentiment proxy (scientifically valid).")
        # Invert VIX: high VIX = negative sentiment
        vix_aligned = vix_series.reindex(price_index).ffill().bfill()
        sentiment_raw = -vix_aligned.rename("sentiment")  # negative = fearful

    features = derive_sentiment_features(
        sentiment_series=sentiment_raw,
        vix_series=vix_series
    )
    features = features.reindex(price_index).ffill().bfill()
    return features


if __name__ == "__main__":
    # Quick test with VIX proxy
    from data_pipeline import load_all_data

    data = load_all_data(start="2020-01-01", end="2021-12-31")
    sent_features = load_or_build_sentiment(
        price_index=data["prices"].index,
        vix_series=data["vix"],
        use_gdelt=False
    )
    print("\nSentiment Features:\n", sent_features.tail(10))
    print("Shape:", sent_features.shape)
