# indicator_registry.py

from typing import Dict, Any
import pandas as pd

def compute_indicator(
    df: pd.DataFrame,
    kind: str,
    spec: Dict[str, Any],
) -> pd.Series | pd.DataFrame:
    """
    Compute a supported indicator using pandas_ta.

    Parameters
    ----------
    df : DataFrame
        Must already contain standardized columns (close, high, low, volume, etc.)
    kind : str
        Indicator name (e.g., 'rsi', 'ema')
    spec : dict
        Indicator parameters from YAML

    Returns
    -------
    Series or DataFrame
        Indicator output
    """
    kind = kind.lower().strip()
    source = spec.get("source", "close")

    if source not in df.columns:
        raise ValueError(f"Source '{source}' not found in DataFrame columns")

    if kind == "rsi":
        return df.ta.rsi(
            length=int(spec["length"]),
            close=source,
        )

    if kind == "ema":
        return df.ta.ema(
            length=int(spec["length"]),
            close=source,
        )

    if kind == "supertrend":
        length = int(spec.get("length", 7))
        multiplier = float(spec.get("multiplier", 3.0))

        if length <= 0:
            raise ValueError("'length' must be a positive integer for supertrend.")
        if multiplier <= 0:
            raise ValueError("'multiplier' must be a positive number for supertrend.")

        required_cols = {"high", "low", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Supertrend requires columns {sorted(required_cols)}. Missing: {sorted(missing)}"
            )

        return df.ta.supertrend(
            length=length,
            multiplier=multiplier,
        )

    if kind == "adx":
        length = int(spec.get("length", 14))

        if length <= 0:
            raise ValueError("'length' must be a positive integer for adx.")

        required_cols = {"high", "low", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"ADX requires columns {sorted(required_cols)}. Missing: {sorted(missing)}"
            )

        return df.ta.adx(
            length=length,
        )

    if kind == "macd":
        fast = int(spec.get("fast", 12))
        slow = int(spec.get("slow", 26))
        signal = int(spec.get("signal", 9))

        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("'fast', 'slow', and 'signal' must be positive integers for macd.")
        if fast >= slow:
            raise ValueError("For macd, 'fast' must be less than 'slow'.")

        return df.ta.macd(
            close=source,
            fast=fast,
            slow=slow,
            signal=signal,
        )


    # Add more indicators here as needed:
    # if kind == "macd": ...
    # if kind == "bbands": ...

    raise ValueError(f"Unsupported indicator kind: '{kind}'")
