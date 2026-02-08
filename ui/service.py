from __future__ import annotations

import math
from typing import Any

import pandas as pd

import config
import data_fetch
import strat_test


def run_strategy_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the existing strategy pipeline and return UI/API-friendly payloads."""
    config.validate_minimum_config(cfg)
    strategy_cfg = config.validate_strategy_config(cfg) if "strategy" in cfg else None

    symbol = str(cfg["symbol"]).strip().upper()
    timeframe = str(cfg["timeframe"]).strip()
    start = str(cfg["start"]).strip()
    end = str(cfg["end"]).strip()
    indicators_cfg = cfg["indicators"]

    df = data_fetch.fetch_ohlcv_from_yfinance(symbol, timeframe, start, end)
    df = data_fetch.apply_indicators(df, indicators_cfg)

    percentage_change = _percentage_change(df)

    results = None
    if strategy_cfg:
        results = strat_test.backtest_strategy(df, strategy_cfg)
    metrics = _build_strategy_metrics(results, timeframe) if results else None

    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "rows": int(len(df)),
        "buy_and_hold_pct": round(percentage_change, 2),
        "starting_cash": _safe_round(results.get("starting_cash"), 2) if results else None,
        "final_equity": _safe_round(results.get("final_equity"), 2) if results else None,
        "strategy_return_pct": _safe_round(float(results.get("total_return_pct", 0.0)) * 100, 2) if results else None,
        "num_trades": int(results.get("num_trades", 0)) if results else 0,
    }

    return {
        "summary": summary,
        "metrics": metrics,
        "metric_rows": _metric_rows(metrics),
        "trades": _serialize_trades(results.get("trades", []) if results else []),
        "price_chart_payload": _build_price_payload(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            indicators_cfg=indicators_cfg,
            results=results,
        ),
        "equity_chart_payload": _build_equity_payload(
            results=results,
            df=df,
            symbol=symbol,
            benchmark_source=(strategy_cfg.get("price_column", "close") if strategy_cfg else "close"),
        )
        if results
        else None,
        "drawdown_chart_payload": _build_drawdown_payload(results) if results else None,
    }


def _percentage_change(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    first = float(df["close"].iloc[0])
    last = float(df["close"].iloc[-1])
    if first == 0:
        return 0.0
    return ((last - first) / first) * 100


def _safe_round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _serialize_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in trades:
        out.append(
            {
                "side": str(trade.get("side", "long")),
                "entry_date": _to_date_str(trade.get("entry_date")),
                "entry_price": _safe_round(trade.get("entry_price"), 4),
                "exit_date": _to_date_str(trade.get("exit_date")),
                "exit_price": _safe_round(trade.get("exit_price"), 4),
                "shares": _safe_round(trade.get("shares"), 4),
                "pnl": _safe_round(trade.get("pnl"), 2),
                "return_pct": _safe_round(float(trade.get("return_pct", 0.0)) * 100, 2),
                "forced_exit": bool(trade.get("forced_exit", False)),
            }
        )
    return out


def _to_date_str(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is None:
        return ""
    return str(value)


def _indicator_columns(df: pd.DataFrame, alias: str) -> list[str]:
    if alias in df.columns:
        return [alias]
    prefix = f"{alias}_"
    return [col for col in df.columns if str(col).startswith(prefix)]


def _to_float_series(values: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for value in values.tolist():
        if pd.isna(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _to_date_list(index: Any) -> list[str]:
    out: list[str] = []
    for value in index:
        if isinstance(value, pd.Timestamp):
            out.append(value.strftime("%Y-%m-%d"))
        else:
            out.append(str(value))
    return out


def _periods_per_year_from_timeframe(timeframe: str) -> float:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": 252 * 390,
        "2m": 252 * 195,
        "5m": 252 * 78,
        "15m": 252 * 26,
        "30m": 252 * 13,
        "60m": 252 * 6.5,
        "90m": 252 * (390 / 90),
        "1h": 252 * 6.5,
        "1d": 252,
        "5d": 52,
        "1wk": 52,
        "1mo": 12,
        "3mo": 4,
    }
    return float(mapping.get(tf, 252))


def _infer_periods_per_year(index: Any, timeframe: str) -> float:
    idx = pd.to_datetime(pd.Index(index), errors="coerce")
    idx = idx[~idx.isna()]
    if len(idx) >= 3:
        deltas = pd.Series(idx).diff().dropna()
        if not deltas.empty:
            median_seconds = deltas.dt.total_seconds().median()
            if median_seconds and median_seconds > 0:
                return (365.25 * 24 * 60 * 60) / float(median_seconds)
    return _periods_per_year_from_timeframe(timeframe)


def _compute_cagr(start_value: float, end_value: float, years: float) -> float | None:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def _max_streak(values: list[float], positive: bool) -> int:
    best = 0
    streak = 0
    for value in values:
        matched = value > 0 if positive else value < 0
        if matched:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _build_strategy_metrics(results: dict[str, Any], timeframe: str) -> dict[str, Any]:
    equity_curve = results.get("equity_curve")
    if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
        return {}

    equity_curve = equity_curve.astype(float)
    returns = equity_curve.pct_change().dropna()
    periods_per_year = _infer_periods_per_year(equity_curve.index, timeframe)

    start_equity = float(results.get("starting_cash", equity_curve.iloc[0]))
    final_equity = float(results.get("final_equity", equity_curve.iloc[-1]))

    first_ts = pd.to_datetime(equity_curve.index[0], errors="coerce")
    last_ts = pd.to_datetime(equity_curve.index[-1], errors="coerce")
    years = None
    if not pd.isna(first_ts) and not pd.isna(last_ts):
        years = max((last_ts - first_ts).total_seconds() / (365.25 * 24 * 60 * 60), 0.0)

    cagr = _compute_cagr(start_equity, final_equity, years or 0.0)

    volatility = None
    sharpe = None
    sortino = None
    if not returns.empty:
        mean_return = float(returns.mean())
        std_return = float(returns.std(ddof=0))
        if std_return > 0:
            sharpe = (mean_return / std_return) * math.sqrt(periods_per_year)
            volatility = std_return * math.sqrt(periods_per_year)

        downside = returns[returns < 0]
        if not downside.empty:
            downside_std = float(downside.std(ddof=0))
            if downside_std > 0:
                sortino = (mean_return / downside_std) * math.sqrt(periods_per_year)

    drawdown = (equity_curve / equity_curve.cummax()) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    calmar = None
    if cagr is not None and max_drawdown is not None and max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)

    trades = results.get("trades", [])
    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    returns_by_trade = [float(t.get("return_pct", 0.0)) for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    total_trades = len(pnls)

    win_rate = (len(winners) / total_trades) if total_trades > 0 else None
    avg_win = (sum(winners) / len(winners)) if winners else None
    avg_loss = (sum(losers) / len(losers)) if losers else None

    payoff_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff_ratio = avg_win / abs(avg_loss)

    gross_profit = sum(winners) if winners else 0.0
    gross_loss = sum(losers) if losers else 0.0
    profit_factor = None
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0 and total_trades > 0:
        profit_factor = float("inf")

    expectancy_pnl = (sum(pnls) / total_trades) if total_trades > 0 else None
    avg_trade_return = (sum(returns_by_trade) / total_trades) if total_trades > 0 else None

    best_trade = max(returns_by_trade) if returns_by_trade else None
    worst_trade = min(returns_by_trade) if returns_by_trade else None

    durations_days: list[float] = []
    for trade in trades:
        entry_date = pd.to_datetime(trade.get("entry_date"), errors="coerce")
        exit_date = pd.to_datetime(trade.get("exit_date"), errors="coerce")
        if pd.isna(entry_date) or pd.isna(exit_date):
            continue
        duration = (exit_date - entry_date).total_seconds() / (24 * 60 * 60)
        if duration >= 0:
            durations_days.append(duration)

    avg_trade_duration_days = (
        sum(durations_days) / len(durations_days) if durations_days else None
    )

    return {
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "annualized_volatility": volatility,
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "expectancy_pnl": expectancy_pnl,
        "avg_trade_return": avg_trade_return,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_trade_duration_days": avg_trade_duration_days,
        "max_win_streak": _max_streak(pnls, positive=True),
        "max_loss_streak": _max_streak(pnls, positive=False),
        "total_trades": total_trades,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _fmt_cash(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _fmt_days(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}d"


def _metric_rows(metrics: dict[str, Any] | None) -> list[dict[str, str]]:
    if not metrics:
        return []

    return [
        {"label": "CAGR", "value": _fmt_pct(metrics.get("cagr"))},
        {"label": "Sharpe Ratio", "value": _fmt_ratio(metrics.get("sharpe_ratio"))},
        {"label": "Sortino Ratio", "value": _fmt_ratio(metrics.get("sortino_ratio"))},
        {"label": "Max Drawdown", "value": _fmt_pct(metrics.get("max_drawdown"))},
        {"label": "Calmar Ratio", "value": _fmt_ratio(metrics.get("calmar_ratio"))},
        {"label": "Annualized Volatility", "value": _fmt_pct(metrics.get("annualized_volatility"))},
        {"label": "Win Rate", "value": _fmt_pct(metrics.get("win_rate"))},
        {"label": "Payoff Ratio", "value": _fmt_ratio(metrics.get("payoff_ratio"))},
        {"label": "Profit Factor", "value": _fmt_ratio(metrics.get("profit_factor"))},
        {"label": "Expectancy / Trade", "value": _fmt_cash(metrics.get("expectancy_pnl"))},
        {"label": "Avg Trade Return", "value": _fmt_pct(metrics.get("avg_trade_return"))},
        {"label": "Best Trade", "value": _fmt_pct(metrics.get("best_trade"))},
        {"label": "Worst Trade", "value": _fmt_pct(metrics.get("worst_trade"))},
        {"label": "Avg Trade Duration", "value": _fmt_days(metrics.get("avg_trade_duration_days"))},
        {"label": "Max Win Streak", "value": str(metrics.get("max_win_streak", 0))},
        {"label": "Max Loss Streak", "value": str(metrics.get("max_loss_streak", 0))},
        {"label": "Gross Profit", "value": _fmt_cash(metrics.get("gross_profit"))},
        {"label": "Gross Loss", "value": _fmt_cash(metrics.get("gross_loss"))},
        {"label": "Trade Count", "value": str(metrics.get("total_trades", 0))},
    ]


def _build_price_payload(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    indicators_cfg: dict[str, dict[str, Any]],
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overlay_kinds = {
        "ema",
        "sma",
        "wma",
        "vwma",
        "dema",
        "tema",
        "hma",
        "rma",
        "supertrend",
        "bbands",
        "kc",
        "donchian",
    }

    overlay_cols: list[str] = []
    osc_cols: list[str] = []

    for alias, spec in indicators_cfg.items():
        kind = str(spec.get("kind", "")).lower()
        cols = _indicator_columns(df, alias)
        if not cols:
            continue

        if kind in overlay_kinds:
            if kind == "supertrend":
                st_cols = [
                    c for c in cols if "SUPERTd" not in c and "SUPERTl" not in c and "SUPERTs" not in c
                ]
                overlay_cols.extend(st_cols if st_cols else cols)
            else:
                overlay_cols.extend(cols)
        else:
            osc_cols.extend(cols)

    dates = _to_date_list(df.index)
    candles = [
        [float(row.open), float(row.close), float(row.low), float(row.high)]
        for row in df[["open", "close", "low", "high"]].itertuples(index=False)
    ]

    overlay_series = [{"name": col, "data": _to_float_series(df[col])} for col in overlay_cols]
    osc_series = [{"name": col, "data": _to_float_series(df[col])} for col in osc_cols]

    entries: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    if results and isinstance(results.get("trades"), list):
        for trade in results["trades"]:
            entry_date = trade.get("entry_date")
            exit_date = trade.get("exit_date")
            entry_price = trade.get("entry_price")
            exit_price = trade.get("exit_price")

            if entry_date is not None and entry_price is not None:
                entries.append({"date": _to_date_str(entry_date), "price": float(entry_price)})
            if exit_date is not None and exit_price is not None:
                exits.append({"date": _to_date_str(exit_date), "price": float(exit_price)})

    return {
        "title": f"{symbol} {timeframe} | {start} to {end}",
        "dates": dates,
        "candles": candles,
        "overlays": overlay_series,
        "oscillators": osc_series,
        "entries": entries,
        "exits": exits,
    }


def _build_equity_payload(
    results: dict[str, Any] | None,
    df: pd.DataFrame,
    symbol: str,
    benchmark_source: str = "close",
) -> dict[str, Any] | None:
    if not results:
        return None

    equity_curve = results.get("equity_curve")
    if equity_curve is None or not isinstance(equity_curve, pd.Series) or equity_curve.empty:
        return None

    payload: dict[str, Any] = {
        "title": "Portfolio Equity Curve",
        "dates": _to_date_list(equity_curve.index),
        "values": _to_float_series(equity_curve),
    }

    source_col = benchmark_source if benchmark_source in df.columns else "close"
    if source_col in df.columns:
        benchmark_prices = df[source_col].astype(float).reindex(equity_curve.index).ffill().bfill()
        if not benchmark_prices.empty and not pd.isna(benchmark_prices.iloc[0]):
            first_price = float(benchmark_prices.iloc[0])
            starting_cash = float(results.get("starting_cash", equity_curve.iloc[0]))
            if first_price != 0 and starting_cash > 0:
                benchmark_values = starting_cash * (benchmark_prices / first_price)
                payload["benchmark_values"] = _to_float_series(benchmark_values)
                payload["benchmark_label"] = f"Buy & Hold ({symbol})"

    return payload


def _build_drawdown_payload(results: dict[str, Any] | None) -> dict[str, Any] | None:
    if not results:
        return None

    equity_curve = results.get("equity_curve")
    if equity_curve is None or not isinstance(equity_curve, pd.Series) or equity_curve.empty:
        return None

    equity_curve = equity_curve.astype(float)
    drawdown = round((equity_curve / equity_curve.cummax()) - 1.0, 2)

    return {
        "title": "Drawdown",
        "dates": _to_date_list(drawdown.index),
        "values": _to_float_series(drawdown),
    }
