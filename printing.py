import pandas as pd
from typing import Any, Dict, Iterable

def print_rows(df: pd.DataFrame) -> float:
    
    print("\n=== DataFrame columns ===")
    print(list(df.columns))

    print("\n=== First 5 rows ===")
    print(df.head(5))

    print("\n=== Last 5 rows ===")
    print(df.tail(5))
    
    percentage_change = round(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100, 2)
    print("\nThis ticker rose by", percentage_change, "% over this period.\n")

    # Show how many NaNs each new indicator has (warm-up periods)
    print("\n=== NaN counts (useful for indicator warm-up) ===")
    print(df.isna().sum())
    
    return percentage_change

def print_strat_results(results: dict, symbol: str, timeframe: str, start: str, end: str, percentage_change: float) -> None:
    print("\n=== Backtest Summary ===")
    print(f"Tcker: {symbol} {timeframe} {start} to {end}")
    print(f"Starting cash: {results['starting_cash']:.2f}")
    print(f"Final equity:  {results['final_equity']:.2f}$")
    print(f"Buy and hold: {percentage_change:.2f}%, {results['starting_cash']*(1 + percentage_change/100):.2f}$")
    print(f"Strategy return: {results['total_return_pct']*100:.2f}%, {results['final_equity']:.2f}$")
    print(f"This strategy beat buy-and-hold by {results['total_return_pct']*100 - percentage_change:.2f}% ({results['final_equity'] - results['starting_cash']*(1 + percentage_change/100):.2f}$) over this period.")
    print(f"Trades:       {results['num_trades']}")

    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        print("\n=== Trades (first 5) ===")
        print(trades_df.head(5))

def _indicator_columns(df: pd.DataFrame, alias: str) -> list[str]:
    if alias in df.columns:
        return [alias]
    prefix = f"{alias}_"
    return [col for col in df.columns if str(col).startswith(prefix)]

def plot_strategy_charts(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    indicators_cfg: Dict[str, Dict[str, Any]] | None = None,
    results: Dict[str, Any] | None = None,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    if indicators_cfg:
        for alias, spec in indicators_cfg.items():
            kind = str(spec.get("kind", "")).lower()
            cols = _indicator_columns(df, alias)
            if not cols:
                continue

            if kind in overlay_kinds:
                if kind == "supertrend":
                    st_cols = [
                        c for c in cols
                        if "SUPERTd" not in c and "SUPERTl" not in c and "SUPERTs" not in c
                    ]
                    overlay_cols.extend(st_cols if st_cols else cols)
                else:
                    overlay_cols.extend(cols)
            else:
                osc_cols.extend(cols)

    has_osc = bool(osc_cols)
    rows = 2 if has_osc else 1
    row_heights = [0.7, 0.3] if has_osc else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"{symbol} OHLC",
        ),
        row=1,
        col=1,
    )

    for col in overlay_cols:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=1.2),
            ),
            row=1,
            col=1,
        )

    if has_osc:
        for col in osc_cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.2),
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        title=f"{symbol} {timeframe} {start} to {end}",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=800 if has_osc else 600,
    )

    fig.show()

    if results and isinstance(results.get("equity_curve"), pd.Series):
        nav_fig = go.Figure()
        nav_fig.add_trace(
            go.Scatter(
                x=results["equity_curve"].index,
                y=results["equity_curve"].values,
                mode="lines",
                name="NAV",
                line=dict(width=2),
            )
        )
        nav_fig.update_layout(
            title=f"Portfolio NAV ({symbol})",
            xaxis_title="Date",
            yaxis_title="Equity",
            height=450,
        )
        nav_fig.show()
