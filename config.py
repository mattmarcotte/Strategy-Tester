from __future__ import annotations
from typing import Any, Dict

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import yaml

def _validate_structured_rule(rule: Dict[str, Any], label: str) -> None:
    if {"left", "op", "right"}.issubset(rule.keys()):
        op = rule.get("op")
        if not isinstance(op, str) or not op.strip():
            raise ValueError(f"{label}: 'op' must be a non-empty string.")
        return

    has_all = "all" in rule
    has_any = "any" in rule
    if has_all == has_any:
        raise ValueError(f"{label} mapping must contain exactly one of 'all' or 'any'.")

    mode = "all" if has_all else "any"
    items = rule.get(mode)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{label}.{mode} must be a non-empty list of conditions.")

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{label}.{mode}[{idx}] must be a mapping.")
        _validate_structured_rule(item, f"{label}.{mode}[{idx}]")


def _validate_rule_value(rule: Any, label: str) -> None:
    if isinstance(rule, str):
        if not rule.strip():
            raise ValueError(f"'{label}' must be a non-empty string.")
        return

    if isinstance(rule, dict):
        _validate_structured_rule(rule, label)
        return

    raise ValueError(f"'{label}' must be either a string expression or a structured mapping.")

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
        entry_rule: "ema21 > ema50"  # or structured mapping with all/any conditions
        exit_rule: "ema21 < ema50"   # or structured mapping with all/any conditions
        force_close_end: true        # optional (default true)
        allow_short: false           # optional (default false)
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

    _validate_rule_value(strat["entry_rule"], "entry_rule")
    _validate_rule_value(strat["exit_rule"], "exit_rule")

    price_column = strat.get("price_column", "close")
    if not isinstance(price_column, str) or not price_column.strip():
        raise ValueError("'price_column' must be a non-empty string if provided.")

    force_close_end = strat.get("force_close_end", True)
    if not isinstance(force_close_end, bool):
        raise ValueError("'force_close_end' must be a boolean if provided.")

    allow_short = strat.get("allow_short", False)
    if not isinstance(allow_short, bool):
        raise ValueError("'allow_short' must be a boolean if provided.")

    return strat
