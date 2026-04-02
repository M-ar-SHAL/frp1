"""
Data Pipeline for FAPT-GNN
Fetches NIFTY 50 stock prices, India VIX, and macro indicators.
All data sources are FREE — no API keys required.
  - yfinance: NIFTY 50 OHLCV + India VIX
  - pandas_datareader / FRED: Macro (interest rate, USD/INR, oil)
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Dict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# NIFTY 50 Constituent Tickers (Yahoo Finance)
# ─────────────────────────────────────────────
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
    "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "WIPRO.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "TECHM.NS", "HINDALCO.NS", "ADANIENT.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "M&M.NS", "SBILIFE.NS", "TATACONSUM.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "ADANIPORTS.NS", "UPL.NS"
]


def fetch_price_data(
    tickers: list = NIFTY50_TICKERS,
    start: str = "2015-01-01",
    end: str = None,
    cache_dir: str = "data/cache"
) -> pd.DataFrame:
    """
    Fetch OHLCV data for all NIFTY 50 stocks.
    Returns MultiIndex DataFrame: (Date) x (Ticker, OHLCV).
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"prices_{start}_{end}.parquet")

    if os.path.exists(cache_file):
        print(f"[Data] Loading cached prices from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"[Data] Downloading NIFTY 50 prices: {start} -> {end}")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=True
    )
    raw.to_parquet(cache_file)
    print(f"[Data] Saved to cache: {cache_file}")
    return raw


def extract_close_prices(raw: pd.DataFrame, tickers: list = NIFTY50_TICKERS) -> pd.DataFrame:
    """Extract Close prices into a clean (Date x Ticker) DataFrame."""
    frames = {}
    for ticker in tickers:
        try:
            frames[ticker] = raw[ticker]["Close"]
        except KeyError:
            print(f"[Warning] Missing data for {ticker}")
    df = pd.DataFrame(frames).dropna(how="all")
    # Forward fill missing days (holidays)
    df = df.ffill().bfill()
    return df


def _get_close(df):
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        return close.iloc[:, 0]
    return close

def fetch_india_vix(start: str = "2015-01-01", end: str = None) -> pd.Series:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    print("[Data] Downloading India VIX...")
    vix = yf.download("^INDIAVIX", start=start, end=end, auto_adjust=True, progress=False)
    s = _get_close(vix)
    s.name = "INDIA_VIX"
    return s

def fetch_nifty_index(start: str = "2015-01-01", end: str = None) -> pd.Series:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    print("[Data] Downloading NIFTY 50 Index...")
    nifty = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
    s = _get_close(nifty)
    s.name = "NIFTY50"
    return s

def fetch_macro_data(start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print("[Data] Downloading macro indicators...")
    macro_symbols = {
        "USDINR": "INR=X",        
        "OIL_BRENT": "BZ=F",      
        "US10Y_YIELD": "^TNX",    
        "GOLD": "GC=F",           
    }

    frames = {}
    for name, sym in macro_symbols.items():
        try:
            data = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            frames[name] = _get_close(data)
        except Exception as e:
            print(f"[Warning] Could not fetch {name}: {e}")

    macro_df = pd.DataFrame(frames).ffill().bfill()
    return macro_df


def load_all_data(
    start: str = "2015-01-01",
    end: str = None,
    cache_dir: str = "data/cache"
) -> Dict[str, pd.DataFrame]:
    """
    Master function: loads ALL data needed for the pipeline.

    Returns dict with keys:
      - 'prices'   : (Date x Ticker) close prices
      - 'vix'      : India VIX Series
      - 'nifty'    : NIFTY 50 index Series
      - 'macro'    : (Date x macro_vars) DataFrame
    """
    # yfinance's `end` argument is *exclusive*, so to include today we request tomorrow.
    if end is None:
        # add one day to today
        end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = end
    raw = fetch_price_data(NIFTY50_TICKERS, start, end_date, cache_dir)
    prices = extract_close_prices(raw, NIFTY50_TICKERS)
    vix = fetch_india_vix(start, end_date)
    nifty = fetch_nifty_index(start, end_date)
    macro = fetch_macro_data(start, end_date)

    # Align all on common dates
    common_idx = prices.index
    vix = vix.reindex(common_idx).ffill().bfill()
    nifty = nifty.reindex(common_idx).ffill().bfill()
    macro = macro.reindex(common_idx).ffill().bfill()

    print(f"\n[Data] Loaded {len(prices)} trading days × {len(prices.columns)} stocks")
    print(f"[Data] Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")

    return {
        "prices": prices,
        "vix": vix,
        "nifty": nifty,
        "macro": macro
    }


if __name__ == "__main__":
    data = load_all_data(start="2015-01-01")
    print("\nPrices shape:", data["prices"].shape)
    print("VIX sample:\n", data["vix"].tail(5))
    print("Macro sample:\n", data["macro"].tail(5))
