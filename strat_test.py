import pandas as pd
from typing import Any, Dict, Optional, Union

def _resolve_operand(df: pd.DataFrame, operand: Any, label: str) -> Union[pd.Series, float, bool]:
    if isinstance(operand, (int, float, bool)):
        return operand

    if isinstance(operand, str):
        token = operand.strip()
        if not token:
            raise ValueError(f"{label}: operand cannot be empty.")

        if token in df.columns:
            return df[token]

        lower_token = token.lower()
        if lower_token == "true":
            return True
        if lower_token == "false":
            return False

        try:
            return float(token)
        except ValueError:
            pass

        raise ValueError(f"{label}: operand '{token}' is neither a DataFrame column nor a numeric literal.")

    raise ValueError(f"{label}: unsupported operand type {type(operand)}")


def _as_signal_series(df: pd.DataFrame, value: Any, label: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(df.index).fillna(False).astype(bool)
    if isinstance(value, bool):
        return pd.Series([value] * len(df), index=df.index, dtype=bool)
    raise ValueError(f"{label} did not resolve to a boolean series.")


def _evaluate_condition(df: pd.DataFrame, condition: Dict[str, Any], label: str) -> pd.Series:
    for key in ("left", "op", "right"):
        if key not in condition:
            raise ValueError(f"{label}: condition missing key '{key}'.")

    left = _resolve_operand(df, condition["left"], label)
    right = _resolve_operand(df, condition["right"], label)
    op = str(condition["op"]).strip()

    if op in ("=", "=="):
        out = left == right
    elif op == "!=":
        out = left != right
    elif op == ">":
        out = left > right
    elif op == "<":
        out = left < right
    elif op == ">=":
        out = left >= right
    elif op == "<=":
        out = left <= right
    else:
        raise ValueError(f"{label}: unsupported operator '{op}'.")

    return _as_signal_series(df, out, label)


def _evaluate_structured_rule(df: pd.DataFrame, rule_obj: Dict[str, Any], label: str) -> pd.Series:
    if {"left", "op", "right"}.issubset(rule_obj.keys()):
        return _evaluate_condition(df, rule_obj, label)

    has_all = "all" in rule_obj
    has_any = "any" in rule_obj
    if has_all == has_any:
        raise ValueError(f"{label} dict must contain exactly one of 'all' or 'any'.")

    mode = "all" if has_all else "any"
    children = rule_obj[mode]
    if not isinstance(children, list) or not children:
        raise ValueError(f"{label}.{mode} must be a non-empty list.")

    if mode == "all":
        combined = pd.Series([True] * len(df), index=df.index, dtype=bool)
        for child in children:
            if not isinstance(child, dict):
                raise ValueError(f"{label}.{mode} entries must be mappings.")
            combined &= _evaluate_structured_rule(df, child, label)
        return combined

    combined = pd.Series([False] * len(df), index=df.index, dtype=bool)
    for child in children:
        if not isinstance(child, dict):
            raise ValueError(f"{label}.{mode} entries must be mappings.")
        combined |= _evaluate_structured_rule(df, child, label)
    return combined


def evaluate_rule(df: pd.DataFrame, rule: Any, label: str) -> pd.Series:
    """
    Evaluate a boolean rule against the DataFrame columns.

    Supported forms:
      1) String expression, e.g. "(ema21 > ema50) & (rsi14 < 70)"
      2) Structured mapping with all/any and condition blocks:
         {"all": [{"left": "rsi14", "op": "<", "right": 40}, ...]}
    """
    if isinstance(rule, str):
        if not rule.strip():
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

    if isinstance(rule, dict):
        return _evaluate_structured_rule(df, rule, label)

    raise ValueError(f"{label} must be a string expression or a structured rule mapping.")

def backtest_strategy(df: pd.DataFrame, strategy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    
    starting_cash = float(strategy_cfg["starting_cash"])
    position_size_pct = float(strategy_cfg["position_size_pct"])
    price_column = strategy_cfg.get("price_column", "close")
    force_close_end = bool(strategy_cfg.get("force_close_end", True))
    allow_short = bool(strategy_cfg.get("allow_short", False))

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
    entry_side: Optional[str] = None

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
                    entry_side = "long"
            elif allow_short and exit_signal.loc[idx]:
                allocation = cash * position_size_pct
                if allocation > 0:
                    shares = allocation / price
                    # Opening a short sells borrowed shares and credits cash.
                    cash += shares * price
                    position = -shares
                    entry_price = float(price)
                    entry_date = idx
                    entry_side = "short"
        else:
            if position > 0.0 and exit_signal.loc[idx]:
                shares = position
                cash += shares * price
                pnl = (float(price) - float(entry_price)) * shares
                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": float(entry_price),
                        "exit_date": idx,
                        "exit_price": float(price),
                        "shares": float(shares),
                        "side": "long",
                        "pnl": float(pnl),
                        "return_pct": float(pnl / (entry_price * shares)),
                        "forced_exit": False,
                    }
                )
                position = 0.0
                entry_price = None
                entry_date = None
                entry_side = None
            elif position < 0.0 and entry_signal.loc[idx]:
                shares = abs(position)
                # Closing a short buys shares back and reduces cash.
                cash -= shares * price
                pnl = (float(entry_price) - float(price)) * shares
                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": float(entry_price),
                        "exit_date": idx,
                        "exit_price": float(price),
                        "shares": float(shares),
                        "side": "short",
                        "pnl": float(pnl),
                        "return_pct": float(pnl / (entry_price * shares)),
                        "forced_exit": False,
                    }
                )
                position = 0.0
                entry_price = None
                entry_date = None
                entry_side = None

        equity_points.append((idx, cash + position * price))

    if position != 0.0 and force_close_end:
        if clean_prices.empty:
            raise ValueError(f"No valid prices found in '{price_column}' to close the position.")
        last_price = float(clean_prices.iloc[-1])
        last_idx = clean_prices.index[-1]

        shares = abs(position)
        side = entry_side or ("long" if position > 0 else "short")

        if side == "long":
            cash += shares * last_price
            pnl = (last_price - float(entry_price)) * shares
        else:
            cash -= shares * last_price
            pnl = (float(entry_price) - last_price) * shares

        trades.append(
            {
                "entry_date": entry_date,
                "entry_price": float(entry_price),
                "exit_date": last_idx,
                "exit_price": float(last_price),
                "shares": float(shares),
                "side": side,
                "pnl": float(pnl),
                "return_pct": float(pnl / (entry_price * shares)),
                "forced_exit": True,
            }
        )
        position = 0.0
        entry_side = None

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
