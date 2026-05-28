import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time
import warnings
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

warnings.filterwarnings("ignore")

GDELT_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

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
    
    if keywords is None:
        keywords = ["India stock market crash", "NIFTY", "Indian economy"]

    date_fmt = date.replace("-", "")
    query = " OR ".join([f'"{kw}"' for kw in keywords[:3]])

    params = {
        "query": query,
        "mode": "TimelineTone",
        "startdatetime": s,
        "enddatetime": e,
        "format": "json",
    }

    try:
        # Increased timeout to 45s for heavy timeline responses
        resp = session.get(GDELT_API_BASE, params=params, timeout=45)
        
        if resp.status_code != 200:
            return np.nan

        data = resp.json()
        articles = data.get("articles", [])

        if not articles:
            return np.nan

        tones = []
        for art in articles:
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
    sleep_sec: float = 1.0,
) -> pd.Series:
    
    import os
    os.makedirs("data/cache", exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[Sentiment] Loading cached GDELT data from {cache_path}")
        return pd.read_parquet(cache_path)["sentiment"]

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start, end=end, freq="B")
    sentiment_dict = {}

    print(f"[Sentiment] Fetching GDELT sentiment for {len(dates)} trading days...")
    print("[Sentiment] This may take a while due to rate limits. Results will be cached.")

    for i, dt in enumerate(dates):
        date_str = dt.strftime("%Y-%m-%d")
        tone = fetch_gdelt_sentiment_day(date_str)
        sentiment_dict[dt] = tone

    # Persist to parquet
    pd.DataFrame({"sentiment": result}).to_parquet(cache_path)
    print(f"[Sentiment] Saved to {cache_path}")
    return result

        time.sleep(sleep_sec)

def build_sentiment_series(
    start: str = "2015-01-01",
    end: str = None,
    cache_path: str = "data/cache/gdelt_sentiment.parquet",
    sleep_sec: float = 0.5
) -> pd.Series:
    """Legacy wrapper — delegates to build_recent_gdelt_sentiment."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    return build_recent_gdelt_sentiment(
        full_start=start, full_end=end,
        cache_path=cache_path, sleep_sec=sleep_sec
    )

    pd.DataFrame({"sentiment": sentiment}).to_parquet(cache_path)
    print(f"[Sentiment] Saved GDELT sentiment to {cache_path}")


def derive_sentiment_features(
    sentiment_series: pd.Series,
    vix_series: pd.Series,
    window: int = 5
) -> pd.DataFrame:
    
    df = pd.DataFrame(index=sentiment_series.index)

    vix_aligned = vix_series.reindex(sentiment_series.index).ffill().bfill()
    vix_norm = -((vix_aligned - vix_aligned.min()) / (vix_aligned.max() - vix_aligned.min()))

    sent = sentiment_series.copy()
    sent = sent.fillna(vix_norm * 10)

    df["sentiment_raw"] = sent
    df["sentiment_ma"] = sent.rolling(window=window, min_periods=1).mean()
    df["sentiment_diverge"] = sent - df["sentiment_ma"]
    df["sentiment_vol"] = sent.rolling(window=window, min_periods=1).std().fillna(0)
    df["vix_norm"] = vix_norm

    return df

def load_or_build_sentiment(
    price_index: pd.Index,
    vix_series: pd.Series,
    use_gdelt: bool = False,
    cache_path: str = "data/cache/gdelt_sentiment.parquet"
) -> pd.DataFrame:
    
    if use_gdelt:
        start = price_index[0].strftime("%Y-%m-%d")
        end   = price_index[-1].strftime("%Y-%m-%d")
        # Returns NaN for historical dates, GDELT tone for recent `gdelt_years`
        sentiment_raw = build_recent_gdelt_sentiment(
            full_start=start, full_end=end,
            gdelt_years=gdelt_years,
            cache_path=cache_path,
        )
    else:
        print("[Sentiment] Using India VIX as sentiment proxy (scientifically valid).")
        vix_aligned = vix_series.reindex(price_index).ffill().bfill()
        sentiment_raw = -vix_aligned.rename("sentiment")

    features = derive_sentiment_features(
        sentiment_series=sentiment_raw,
        vix_series=vix_series,
    )
    features = features.reindex(price_index).ffill().bfill()
    return features

if __name__ == "__main__":
    from data_pipeline import load_all_data

    data = load_all_data(start="2020-01-01", end="2021-12-31")
    sent_features = load_or_build_sentiment(
        price_index=data["prices"].index,
        vix_series=data["vix"],
        use_gdelt=False
    )
    print("\nSentiment Features:\n", sent_features.tail(10))
    print("Shape:", sent_features.shape)

