"""
Main loader entrypoint (multi-source dispatcher).

Loads prices, returns, membership, and risk-free data from:
    - "data": local parquet files in data/
    - "bloomberg": Bloomberg API

All outputs are returned as Polars frames.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

import config
from loaders.base import (
    DATE_COL,
    normalize_wide,
    filter_date_range,
    restrict_universe,
)


def _as_list(index_id):
    if index_id is None:
        return None
    if isinstance(index_id, str):
        return [index_id]
    return list(index_id)


def load_membership(source="data", index_id=None) -> pl.DataFrame:
    """Index membership in long format [date, index_id, ticker]."""
    if source == "bloomberg":
        from loaders.bloomberg_loader import BloombergLoader
        mem = BloombergLoader().load_membership(index_id=index_id)
    else:
        mem = pl.read_parquet(config.MEMBERSHIP_PATH)
        mem = mem.with_columns(pl.col("date").cast(pl.Datetime))

    ids = _as_list(index_id)
    if ids is not None:
        mem = mem.filter(pl.col("index_id").is_in(ids))
    return mem.sort(["index_id", "date", "ticker"])


def _universe_tickers(source, index_id):
    """Universe tickers across all dates, used to restrict price data."""
    if index_id is None:
        return None
    mem = load_membership(source="data", index_id=index_id)
    return mem.get_column("ticker").unique().to_list()


def load_prices(source="data", index_id=None, start=None, end=None,
                tickers=None, prices_path=None) -> pl.DataFrame:
    """Prix (PX_LAST) au format wide Polars [date, <tickers...>].

    Parameters
    ----------
    source : "data" | "bloomberg"
    index_id : str | list[str] | None
        Restreint l'univers aux membres de ce(s) indice(s).
    start, end : str | datetime | None
        Plage de dates inclusive.
    tickers : list[str] | None
        Sous-ensemble explicite de tickers (prioritaire sur index_id).
    prices_path : str | Path | None
        Override the local parquet file (e.g. config.RAW_PRICES_PATH for raw
        unadjusted closes). Only used when source == "data".
    """
    if source == "bloomberg":
        from loaders.bloomberg_loader import BloombergLoader
        if tickers is None:
            tickers = _universe_tickers("data", index_id)
        prices = BloombergLoader().load_prices(tickers, start=start, end=end)
    else:
        prices = pl.read_parquet(prices_path or config.PRICES_PATH)
        prices = normalize_wide(prices)

    prices = normalize_wide(prices)

    # Universe restriction.
    if tickers is not None:
        prices = restrict_universe(prices, tickers)
    elif index_id is not None:
        univ = _universe_tickers(source, index_id)
        if univ:
            prices = restrict_universe(prices, univ)

    prices = filter_date_range(prices, start, end)
    return prices


def load_returns(source="data", index_id=None, start=None, end=None,
                 tickers=None, from_prices=False) -> pl.DataFrame:
    """Simple daily returns in wide Polars format.

    If `from_prices=True` (or source != "data"), recompute returns from prices
    via pct_change. Otherwise read the local returns.parquet file.
    """
    if source == "data" and not from_prices:
        rets = pl.read_parquet(config.RETURNS_PATH)
        rets = normalize_wide(rets)
        if tickers is not None:
            rets = restrict_universe(rets, tickers)
        elif index_id is not None:
            univ = _universe_tickers("data", index_id)
            if univ:
                rets = restrict_universe(rets, univ)
        return filter_date_range(rets, start, end)

    prices = load_prices(source=source, index_id=index_id, start=start,
                         end=end, tickers=tickers)
    return prices_to_returns(prices)


def prices_to_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Simple daily returns computed from a wide price frame."""
    value_cols = [c for c in prices.columns if c != DATE_COL]
    rets = prices.with_columns([
        (pl.col(c) / pl.col(c).shift(1) - 1.0).alias(c) for c in value_cols
    ])
    # The first line has no prior observation, so fill with 0.
    rets = rets.with_columns([pl.col(c).fill_null(0.0) for c in value_cols])
    return rets


def load_riskfree(source="data", currency=None) -> pl.DataFrame:
    """Daily risk-free rates in wide Polars format [date, <currencies...>]."""
    if source == "bloomberg":
        from loaders.bloomberg_loader import BloombergLoader
        rf = BloombergLoader().load_riskfree()
    else:
        rf = pl.read_parquet(config.RISKFREE_PATH)
    rf = normalize_wide(rf)
    if currency is not None:
        cur = currency if isinstance(currency, list) else [currency]
        keep = [DATE_COL] + [c for c in cur if c in rf.columns]
        rf = rf.select(keep)
    return rf
