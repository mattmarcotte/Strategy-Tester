from __future__ import annotations
from typing import Any, Dict

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import yaml

import indicator_registry

def load_strategy_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary.

    YAML is just a structured text format. After loading, you get nested
    dictionaries/lists you can read from like normal Python objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Basic sanity checks: we expect the YAML root to be a dict (mapping).
    if not isinstance(cfg, dict):
        raise ValueError("Strategy YAML root must be a mapping/dict.")

    return cfg

def validate_minimum_config(cfg: Dict[str, Any]) -> None:
   
    required_keys = ["symbol", "timeframe", "start", "end", "indicators"]

    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required config key: '{k}'")

    if not isinstance(cfg["symbol"], str) or not cfg["symbol"].strip():
        raise ValueError("'symbol' must be a non-empty string (e.g., 'SPY').")

    if not isinstance(cfg["timeframe"], str) or not cfg["timeframe"].strip():
        raise ValueError("'timeframe' must be a non-empty string (e.g., '1d').")

    if not isinstance(cfg["start"], str) or not cfg["start"].strip():
        raise ValueError("'start' must be a non-empty string like 'YYYY-MM-DD'.")

    if not isinstance(cfg["end"], str) or not cfg["end"].strip():
        raise ValueError("'end' must be a non-empty string like 'YYYY-MM-DD'.")

    if not isinstance(cfg["indicators"], dict):
        raise ValueError("'indicators' must be a mapping (dict) of alias -> indicator spec.")

def validate_strategy_config(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Validate optional strategy settings for backtesting.

    Expected structure:
      strategy:
        starting_cash: 10000
        position_size_pct: 0.25
        price_column: "close"        # optional (default "close")
        entry_rule: "ema21 > ema50"
        exit_rule: "ema21 < ema50"
        force_close_end: true        # optional (default true)
    """
    if "strategy" not in cfg:
        return None

    strat = cfg["strategy"]
    if not isinstance(strat, dict):
        raise ValueError("'strategy' must be a mapping (dict).")

    required = ["starting_cash", "position_size_pct", "entry_rule", "exit_rule"]
    for key in required:
        if key not in strat:
            raise ValueError(f"Missing required strategy key: '{key}'")

    starting_cash = strat["starting_cash"]
    if not isinstance(starting_cash, (int, float)) or starting_cash <= 0:
        raise ValueError("'starting_cash' must be a positive number.")

    position_size_pct = strat["position_size_pct"]
    if not isinstance(position_size_pct, (int, float)) or not (0 < position_size_pct <= 1):
        raise ValueError("'position_size_pct' must be a number between 0 and 1 (e.g., 0.25).")

    entry_rule = strat["entry_rule"]
    if not isinstance(entry_rule, str) or not entry_rule.strip():
        raise ValueError("'entry_rule' must be a non-empty string.")

    exit_rule = strat["exit_rule"]
    if not isinstance(exit_rule, str) or not exit_rule.strip():
        raise ValueError("'exit_rule' must be a non-empty string.")

    price_column = strat.get("price_column", "close")
    if not isinstance(price_column, str) or not price_column.strip():
        raise ValueError("'price_column' must be a non-empty string if provided.")

    force_close_end = strat.get("force_close_end", True)
    if not isinstance(force_close_end, bool):
        raise ValueError("'force_close_end' must be a boolean if provided.")

    return strat

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
            for col in result.columns:
                out[f"{alias}_{col}"] = result[col]

        else:
            raise TypeError(
                f"Indicator '{alias}' returned unexpected type {type(result)}. "
                "Expected pandas Series or DataFrame."
            )

    return out

def evaluate_rule(df: pd.DataFrame, rule: str, label: str) -> pd.Series:
    """
    Evaluate a boolean rule against the DataFrame columns.

    Example rule: "(ema21 > ema50) & (rsi14 < 70)"
    Note: Use & and | for logical operations, and ~ for NOT.
    """
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(f"{label} must be a non-empty string.")

    try:
        result = df.eval(rule, engine="python")
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate {label}='{rule}'. "
            "Use DataFrame column names and operators like &, |, ~. "
            f"Original error: {exc}"
        ) from exc

    if not isinstance(result, pd.Series):
        raise ValueError(f"{label} must evaluate to a pandas Series of booleans.")

    return result.fillna(False).astype(bool)

def backtest_strategy(df: pd.DataFrame, strategy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple long-only backtest using entry/exit rules.

    Buys with a fixed % of available cash at the bar's price_column
    and sells the full position on exit.
    """
    starting_cash = float(strategy_cfg["starting_cash"])
    position_size_pct = float(strategy_cfg["position_size_pct"])
    price_column = strategy_cfg.get("price_column", "close")
    force_close_end = bool(strategy_cfg.get("force_close_end", True))

    if price_column not in df.columns:
        raise ValueError(
            f"price_column '{price_column}' not found in DataFrame columns: {list(df.columns)}"
        )

    entry_signal = evaluate_rule(df, strategy_cfg["entry_rule"], "entry_rule")
    exit_signal = evaluate_rule(df, strategy_cfg["exit_rule"], "exit_rule")

    price_series = df[price_column].ffill()
    clean_prices = price_series.dropna()

    cash = starting_cash
    position = 0.0
    entry_price = None
    entry_date = None

    trades: list[Dict[str, Any]] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []

    for idx, price in price_series.items():
        if pd.isna(price):
            continue

        if position == 0.0:
            if entry_signal.loc[idx]:
                allocation = cash * position_size_pct
                if allocation > 0:
                    shares = allocation / price
                    cash -= shares * price
                    position = shares
                    entry_price = float(price)
                    entry_date = idx
        else:
            if exit_signal.loc[idx]:
                cash += position * price
                pnl = (float(price) - float(entry_price)) * position
                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": float(entry_price),
                        "exit_date": idx,
                        "exit_price": float(price),
                        "shares": float(position),
                        "pnl": float(pnl),
                        "return_pct": float(pnl / (entry_price * position)),
                        "forced_exit": False,
                    }
                )
                position = 0.0
                entry_price = None
                entry_date = None

        equity_points.append((idx, cash + position * price))

    if position > 0.0 and force_close_end:
        if clean_prices.empty:
            raise ValueError(f"No valid prices found in '{price_column}' to close the position.")
        last_price = float(clean_prices.iloc[-1])
        last_idx = clean_prices.index[-1]
        cash += position * last_price
        pnl = (last_price - float(entry_price)) * position
        trades.append(
            {
                "entry_date": entry_date,
                "entry_price": float(entry_price),
                "exit_date": last_idx,
                "exit_price": float(last_price),
                "shares": float(position),
                "pnl": float(pnl),
                "return_pct": float(pnl / (entry_price * position)),
                "forced_exit": True,
            }
        )
        position = 0.0

    equity_curve = pd.Series(
        {idx: equity for idx, equity in equity_points}, name="equity"
    )
    final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else starting_cash
    total_return_pct = (final_equity - starting_cash) / starting_cash

    return {
        "starting_cash": starting_cash,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": len(trades),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def main() -> None:
    """
    Orchestrates the full flow:
      YAML -> validate -> fetch OHLCV -> compute indicators -> preview output
    """
    # Step 1: Load and validate the YAML configuration
    cfg = load_strategy_yaml("strategy.yaml")
    validate_minimum_config(cfg)

    symbol = cfg["symbol"].strip()
    timeframe = cfg["timeframe"].strip()
    start = cfg["start"].strip()
    end = cfg["end"].strip()
    indicators_cfg = cfg["indicators"]
    strategy_cfg = validate_strategy_config(cfg)

    # Step 2: Fetch price data
    df = fetch_ohlcv_from_yfinance(symbol, timeframe, start, end)

    # Step 3: Compute indicators and append to DataFrame
    df = apply_indicators(df, indicators_cfg)

    # Step 4: Print quick sanity checks
    # (When debugging, these are your first checks: do we have data? do the new columns exist?)
    print("\n=== DataFrame columns ===")
    print(list(df.columns))

    print("\n=== First 5 rows ===")
    print(df.head(5))

    print("\n=== Last 5 rows ===")
    print(df.tail(5))
    
    percentage_change = round(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100, 2)
    print("\nThis ticker rose by", percentage_change, "% over this period.\n")

    # Optional: show how many NaNs each new indicator has (warm-up periods)
    print("\n=== NaN counts (useful for indicator warm-up) ===")
    print(df.isna().sum())

    if strategy_cfg:
        print("\n=== Backtest Summary ===")
        results = backtest_strategy(df, strategy_cfg)
        print(f"Starting cash: {results['starting_cash']:.2f}")
        print(f"Final equity:  {results['final_equity']:.2f}")
        print(f"Buy and hold: {percentage_change:.2f}%, {results['starting_cash']*(1 + percentage_change/100):.2f}$")
        print(f"Strategy return: {results['total_return_pct']*100:.2f}%, {results['final_equity']:.2f}$")
        print(f"This strategy beat buy-and-hold by {results['total_return_pct']*100 - percentage_change:.2f}% ({results['final_equity'] - results['starting_cash']*(1 + percentage_change/100):.2f}$) over this period.")
        print(f"Trades:       {results['num_trades']}")

        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"])
            print("\n=== Trades (first 5) ===")
            print(trades_df.head(5))


if __name__ == "__main__":
    main()







