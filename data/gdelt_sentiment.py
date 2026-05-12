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
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

warnings.filterwarnings("ignore")

# GDELT DOC 2.0 API endpoint (free, no key needed)
GDELT_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT 2.0 data begins on this date — querying before it returns "Invalid query start date"
GDELT_MIN_DATE = "20150218"

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
    Uses TimelineTone for robustness.
    """
    series = fetch_gdelt_sentiment_range(date, date, keywords)
    if series.empty:
        return np.nan
    return float(series.iloc[0])


def fetch_gdelt_sentiment_range(start_date: str, end_date: str, keywords: list = None) -> pd.Series:
    """
    Fetch average sentiment tone for a range using TimelineTone.
    Includes robust retries for network stability.
    """
    if keywords is None:
        keywords = ["India stock market", "NIFTY 50", "Indian economy"]

    # Setup session with retries
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,  # Exponential backoff: 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Format dates for GDELT API (YYYYMMDDHHMMSS)
    # Clamp start date: GDELT 2.0 has no data before 2015-02-18
    raw_s = start_date.replace("-", "")
    if raw_s < GDELT_MIN_DATE:
        raw_s = GDELT_MIN_DATE
    s = raw_s + "000000"
    e = end_date.replace("-", "") + "235959"

    # GDELT requires OR'd terms to be wrapped in parentheses: ("A" OR "B" OR "C")
    inner = " OR ".join([f'"{kw}"' for kw in keywords[:3]])
    query = f"({inner}) sourcelang:english"

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
            print(f"      [!] API Error {resp.status_code}")
            return pd.Series()

        if not resp.text.strip():
            print(f"      [!] Empty response from API")
            return pd.Series()

        try:
            data = resp.json()
        except Exception:
            print(f"      [!] Response is not valid JSON. First 50 chars: {resp.text[:50]}")
            return pd.Series()

        timeline = data.get("timeline", [])
        
        if not timeline:
            # Fallback: single-keyword query (no OR needed, no parentheses required)
            print(f"      [!] No results for main query, trying 'NIFTY 50' fallback...")
            params["query"] = '"NIFTY 50" sourcelang:english'
            resp = session.get(GDELT_API_BASE, params=params, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                try:
                    data = resp.json()
                    timeline = data.get("timeline", [])
                except:
                    timeline = []
            else:
                print(f"      [!] Fallback also failed (status={resp.status_code})")

        if not timeline:
            return pd.Series()
        
        points = timeline[0].get("data", [])
        if not points:
            return pd.Series()
            
        results = {}
        for pt in points:
            dt_str = pt["date"][:8]
            dt = pd.to_datetime(dt_str, format="%Y%m%d")
            results[dt] = pt["value"]
            
        return pd.Series(results).sort_index()

    except Exception as ex:
        print(f"      [!] API Exception: {str(ex)[:100]}")
        return pd.Series()


def build_recent_gdelt_sentiment(
    full_start: str,
    full_end: str,
    gdelt_years: int = 2,
    cache_path: str = "data/cache/gdelt_sentiment.parquet",
    sleep_sec: float = 1.0,
) -> pd.Series:
    """
    Fetch GDELT sentiment for the most recent `gdelt_years` years only.

    Rationale: fetching 10+ years times out on the GDELT API; fetching
    2 years completes in ~30 seconds and is reliably available.
    Returns a Series indexed over ALL trading days in [full_start, full_end]:
      - recent `gdelt_years`: real GDELT tone values
      - historical: NaN  (caller fills with VIX proxy in derive_sentiment_features)
    Result is cached to parquet so re-runs are instant.
    """
    import os
    os.makedirs("data/cache", exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[Sentiment] Loading cached GDELT data from {cache_path}")
        cached = pd.read_parquet(cache_path)["sentiment"]
        # Re-index to current price index range (handles date range changes)
        all_days = pd.date_range(start=full_start, end=full_end, freq="B")
        return cached.reindex(all_days)

    if full_end is None:
        full_end = datetime.today().strftime("%Y-%m-%d")

    all_days = pd.date_range(start=full_start, end=full_end, freq="B")

    # Determine the GDELT window (last N years only)
    end_dt = pd.to_datetime(full_end)
    gdelt_start_dt = end_dt - pd.DateOffset(years=gdelt_years)
    # Clamp to GDELT 2.0 launch date
    gdelt_start_dt = max(gdelt_start_dt, pd.to_datetime("2015-02-18"))
    gdelt_start = gdelt_start_dt.strftime("%Y-%m-%d")

    print(f"[Sentiment] Fetching GDELT for last {gdelt_years} yr window: {gdelt_start} -> {full_end}")
    print("[Sentiment] (Historical data before this window uses VIX proxy)")

    # Yearly chunks within the GDELT window
    chunk_starts = pd.date_range(start=gdelt_start, end=full_end, freq="YS")
    if chunk_starts.empty or chunk_starts[0] > gdelt_start_dt:
        chunk_starts = chunk_starts.insert(0, gdelt_start_dt)

    all_sentiment = []
    for i, s_dt in enumerate(chunk_starts):
        e_dt = s_dt + pd.offsets.YearEnd(0)
        if e_dt > pd.to_datetime(full_end):
            e_dt = pd.to_datetime(full_end)
        s_str = s_dt.strftime("%Y-%m-%d")
        e_str = e_dt.strftime("%Y-%m-%d")
        print(f"  Chunk {i+1}: {s_str} to {e_str}")
        chunk = fetch_gdelt_sentiment_range(s_str, e_str)
        if not chunk.empty:
            all_sentiment.append(chunk)
            print(f"    Loaded {len(chunk)} days. Avg tone: {chunk.mean():.3f}")
        else:
            print(f"    Warning: no data for this chunk.")
        time.sleep(sleep_sec)

    # Start with all-NaN (historical portion stays NaN → VIX filled later)
    result = pd.Series(np.nan, index=all_days, name="sentiment")

    if all_sentiment:
        gdelt_data = pd.concat(all_sentiment).sort_index()
        gdelt_data = gdelt_data[~gdelt_data.index.duplicated(keep="first")]
        # Reindex to trading days in the GDELT window and interpolate gaps
        gdelt_days = all_days[all_days >= gdelt_start_dt]
        gdelt_data = gdelt_data.reindex(gdelt_days).interpolate(method="linear").ffill().bfill()
        result.loc[gdelt_data.index] = gdelt_data.values
        gdelt_coverage = gdelt_data.notna().sum()
        print(f"[Sentiment] GDELT covers {gdelt_coverage} of {len(all_days)} trading days.")
    else:
        print("[Sentiment] [ERROR] No GDELT data fetched — chart will fall back to VIX proxy.")

    # Persist to parquet
    pd.DataFrame({"sentiment": result}).to_parquet(cache_path)
    print(f"[Sentiment] Saved to {cache_path}")
    return result


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
    use_gdelt: bool = False,
    cache_path: str = "data/cache/gdelt_sentiment.parquet",
    gdelt_years: int = 2,
) -> pd.DataFrame:
    """
    Main entry: load or build sentiment features.

    If use_gdelt=False: uses India VIX (inverted) as the sentiment proxy.
    If use_gdelt=True : fetches real GDELT news tone for the last `gdelt_years`
                        years (default 2) and blends with VIX proxy for earlier
                        dates.  Only fetching the recent window keeps API calls
                        fast (30 s) instead of timing out over a 10-year range.

    The two modes produce genuinely different sentiment_raw signals:
      - VIX mode   : monotone negative series tracking market fear
      - GDELT mode : oscillates positive/negative driven by news tone
    """
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
        print("[Sentiment] Using India VIX as sentiment proxy.")
        vix_aligned = vix_series.reindex(price_index).ffill().bfill()
        sentiment_raw = -vix_aligned.rename("sentiment")

    features = derive_sentiment_features(
        sentiment_series=sentiment_raw,
        vix_series=vix_series,
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
