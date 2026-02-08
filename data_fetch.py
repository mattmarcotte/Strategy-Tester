from __future__ import annotations
from typing import Any, Dict

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import yaml

import indicator_registry

def fetch_ohlcv_from_yfinance(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker using yfinance.

    yfinance returns a DataFrame with columns like: Open, High, Low, Close, Adj Close, Volume.
    We'll standardize column names to lowercase for consistency.
    """
    df = yf.download(
        tickers=symbol,
        interval=timeframe,
        start=start,
        end=end,
        auto_adjust=False,  # keep raw OHLCV; you can change later
        progress=False,     # cleaner output
        threads=False, 
        rounding=True# simpler, fewer surprises while learning
    )

    if df.empty:
        raise ValueError(
            f"No data returned from yfinance for {symbol=} {timeframe=} {start=} {end=}. "
            "Check the ticker, interval, and date range."
        )
    
    if isinstance(df.columns, pd.MultiIndex):
        # Keep only the first level: Open, High, Low, Close, Volume
        df.columns = df.columns.get_level_values(0)
    
    # Standardize columns to lowercase, replace spaces with underscores
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]


    # Ensure the core OHLCV columns exist (Adj Close may not always be present)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing}. Got: {list(df.columns)}")

    return df

def apply_indicators(df: pd.DataFrame, indicators_cfg: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Loop through the 'indicators' section of the YAML and append computed columns.

    indicators_cfg looks like:
      {
        "rsi14": {"kind": "rsi", "length": 14, "source": "close"},
        "ema21": {"kind": "ema", "length": 21, "source": "close"},
      }

    For each indicator alias:
      - call indicator_registry.compute_indicator(df, kind, spec)
      - if result is a Series -> store in df[alias]
      - if result is a DataFrame -> store in df[alias_<colname>] for each output column
    """
    out = df.copy()

    for alias, spec in indicators_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Indicator '{alias}' must have a dict spec (got {type(spec)}).")

        kind = spec.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"Indicator '{alias}' missing/invalid 'kind' (expected string).")

        # Compute indicator using your registry module.
        # The registry defines what indicator kinds are supported and how to compute them.
        result = indicator_registry.compute_indicator(out, kind, spec)

        # Case A: indicator returns a single Series (most do)
        if isinstance(result, pd.Series):
            out[alias] = result

        # Case B: indicator returns multiple columns (DataFrame)
        # Example later: MACD returns macd, signal, hist series.
        elif isinstance(result, pd.DataFrame):
            kind_lc = kind.strip().lower()
            alias_main_col = None
            if kind_lc == "adx":
                alias_main_col = next((c for c in result.columns if "ADX_" in str(c)), None)
            elif kind_lc == "macd":
                alias_main_col = next((c for c in result.columns if "MACDh_" in str(c)), None)

            if alias_main_col is not None:
                out[alias] = result[alias_main_col]

            for col in result.columns:
                safe_col = str(col).replace(" ", "_").replace(".", "_")
                out[f"{alias}_{safe_col}"] = result[col]

        else:
            raise TypeError(
                f"Indicator '{alias}' returned unexpected type {type(result)}. "
                "Expected pandas Series or DataFrame."
            )

    return out
