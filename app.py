from __future__ import annotations
from typing import Any, Dict

import config
import data_fetch
import printing
import strat_test


def main() -> None:

    # Load and validate the YAML configuration
    cfg = config.load_strategy_yaml("strategy.yaml")
    config.validate_minimum_config(cfg)

    symbol = cfg["symbol"].strip()
    timeframe = cfg["timeframe"].strip()
    start = cfg["start"].strip()
    end = cfg["end"].strip()
    indicators_cfg = cfg["indicators"]
    strategy_cfg = config.validate_strategy_config(cfg)

    # Fetch price data
    df = data_fetch.fetch_ohlcv_from_yfinance(symbol, timeframe, start, end)

    # Compute indicators and append to DataFrame
    df = data_fetch.apply_indicators(df, indicators_cfg)
    
    percentage_change = printing.print_rows(df)

    if strategy_cfg:
        results = strat_test.backtest_strategy(df, strategy_cfg)
        printing.print_strat_results(
            results=results,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            percentage_change=percentage_change,
        )
        printing.plot_strategy_charts(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            indicators_cfg=indicators_cfg,
            results=results,
        )


if __name__ == "__main__":
    main()






