# indicator_registry.py

from typing import Dict, Any
import pandas as pd
import pandas_ta as ta  # registers df.ta


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


    # Add more indicators here as needed:
    # if kind == "macd": ...
    # if kind == "bbands": ...

    raise ValueError(f"Unsupported indicator kind: '{kind}'")
