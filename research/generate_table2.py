"""
Generate Table 2 — Basic Statistics for the Return Series.

Reproduces Table 2 from Eroglu, Yener & Yigit (2023) using local Bloomberg data.

For each GICS sector the script builds an equal-weight portfolio of sector stocks
and computes time-series statistics on that portfolio's daily return series:
    - Annualised mean return  (portfolio daily mean × 252, expressed as %)
    - Annualised std           (portfolio daily std  × √252, expressed as %)
    - Skewness of portfolio daily returns
    - Excess kurtosis of portfolio daily returns  (Fisher, normal = 0)

The total row uses the full equal-weight portfolio of all 415 tickers.

NOTE: the paper computes sector-portfolio statistics (equal-weight per sector),
NOT per-ticker statistics averaged within sector. Using per-ticker averages gives
std ≈ 25% vs paper's 16% because diversification within a sector portfolio lowers
the std substantially. The equal-weight portfolio approach matches the paper's
mean/std/skew values closely.

Output: research/outputs/tables/table2_basic_statistics_for_the_return_series.csv

Run:
    py research/generate_table2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from config import config_paper as paper_cfg

TRADING_DAYS = paper_cfg.TRADING_DAYS_PER_YEAR   # 252


def _daily_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute daily simple returns from a wide price DataFrame."""
    date_col = "date"
    cols = [c for c in prices.columns if c != date_col]
    rets = prices.with_columns([
        ((pl.col(c) / pl.col(c).shift(1)) - 1.0).alias(c) for c in cols
    ])
    # Drop the first row (NaN for all tickers).
    return rets.slice(1)


def _portfolio_stats(returns: pl.DataFrame, ticker_list: list) -> dict:
    """Equal-weight portfolio return stats for a list of tickers."""
    avail = [t for t in ticker_list if t in returns.columns]
    if not avail:
        return None
    # Equal-weight portfolio: mean across tickers at each day, drop nulls.
    port_ret = returns.select(avail).mean_horizontal().drop_nulls()
    if port_ret.len() < 10:
        return None
    return {
        "n_firms":  len(avail),
        "mean_pct": float(port_ret.mean()) * TRADING_DAYS * 100,
        "std_pct":  float(port_ret.std())  * (TRADING_DAYS ** 0.5) * 100,
        "skewness": float(port_ret.skew()),
        "kurtosis": float(port_ret.kurtosis()),   # excess kurtosis (normal = 0)
    }


def main() -> None:
    prices_path   = paper_cfg.PAPER_PRICES_ADJUSTED_415_PATH
    universe_path = paper_cfg.PAPER_UNIVERSE_415_TICKERS_PATH

    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} not found — run  py research/build_paper_dataset.py  first."
        )
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found — run  py research/build_paper_dataset.py  first."
        )

    prices   = pl.read_parquet(prices_path)
    universe = pl.read_csv(universe_path)   # columns: ticker, gics_sector, ...

    returns = _daily_returns(prices)

    # Build sector → ticker mapping.
    sector_map: dict[str, list[str]] = {}
    for row in universe.to_dicts():
        sec = row["gics_sector"]
        sector_map.setdefault(sec, []).append(row["ticker"])

    # Per-sector equal-weight portfolio stats.
    sector_rows = []
    for sec in sorted(sector_map):
        stats = _portfolio_stats(returns, sector_map[sec])
        if stats is None:
            continue
        sector_rows.append({"gics_sector": sec, **stats})

    sector_table = (
        pl.DataFrame(sector_rows)
        .with_columns(pl.col("n_firms").cast(pl.UInt32))
        .with_columns([
            pl.col("mean_pct").round(2),
            pl.col("std_pct").round(2),
            pl.col("skewness").round(4),
            pl.col("kurtosis").round(4),
        ])
    )

    # Total row — full equal-weight portfolio of all 415 tickers.
    all_tickers = universe.get_column("ticker").to_list()
    total_stats = _portfolio_stats(returns, all_tickers)
    total = pl.DataFrame([{
        "gics_sector": "TOTAL",
        "n_firms":     len([t for t in all_tickers if t in returns.columns]),
        "mean_pct":    round(total_stats["mean_pct"], 2),
        "std_pct":     round(total_stats["std_pct"],  2),
        "skewness":    round(total_stats["skewness"], 4),
        "kurtosis":    round(total_stats["kurtosis"], 4),
    }]).with_columns(pl.col("n_firms").cast(pl.UInt32))

    table2 = pl.concat([sector_table, total])

    # Write to outputs.
    out_path = paper_cfg.TABLES_DIR / "table2_basic_statistics_for_the_return_series.csv"
    paper_cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table2.write_csv(out_path)

    print(f"Table 2 written to {out_path.relative_to(ROOT)}")
    print()

    # Pretty-print comparison against paper values.
    PAPER = {
        "Communication Services":  (15, 19.25, 16.90, -0.4116, 5.9928),
        "Consumer Discretionary":  (51, 17.71, 17.97, -0.3172, 6.2322),
        "Consumer Staples":        (26, 13.16, 12.24, -0.3128, 5.3612),
        "Energy":                  (24,  8.61, 25.87, -0.1321, 5.5094),
        "Financials":              (60, 14.91, 20.59, -0.3279, 7.9465),
        "Health Care":             (55, 20.01, 16.23, -0.4824, 6.0463),
        "Industrials":             (57, 16.97, 18.05, -0.3298, 7.1194),
        "Information Technology":  (48, 19.70, 17.87, -0.2552, 6.1649),
        "Materials":               (23, 14.24, 19.42, -0.3281, 5.8789),
        "Real Estate":             (32, 12.59, 18.15,  0.0857, 8.8627),
        "Utilities":               (24,  9.37, 14.60, -0.3464, 5.4850),
        "TOTAL":                  (415, 15.91, 16.00, -0.3924, 8.0039),
    }
    HDR = f"{'Sector':<30} {'N':>4}  {'Mean%':>7}  {'Std%':>7}  {'Skew':>8}  {'Kurt':>8}"
    SEP = "-" * len(HDR)
    print(f"{'':30}  {'--- LOCAL ---':^30}  {'--- PAPER ---':^30}")
    print(HDR)
    print(SEP)

    rows_map = {
        r["gics_sector"]: r
        for r in table2.to_dicts()
    }
    for sector, (pn, pm, ps, psk, pku) in PAPER.items():
        r = rows_map.get(sector)
        if r is None:
            ln, lm, ls, lsk, lku = "—", "—", "—", "—", "—"
        else:
            ln, lm, ls = r["n_firms"], r["mean_pct"], r["std_pct"]
            lsk, lku = r["skewness"], r["kurtosis"]
        print(f"{sector:<30}  local: {ln:>4} {lm:>7.2f} {ls:>7.2f} {lsk:>8.4f} {lku:>8.4f}"
              f"  paper: {pn:>4} {pm:>7.2f} {ps:>7.2f} {psk:>8.4f} {pku:>8.4f}")
    print(SEP)


if __name__ == "__main__":
    main()
