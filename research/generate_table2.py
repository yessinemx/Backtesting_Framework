"""
Generate Table 2 — Basic Statistics for the Return Series.

Reproduces Table 2 from Eroglu, Yener & Yigit (2023) using local Bloomberg data.

For each stock in the 415-ticker universe the script computes:
    - Annualised mean return  (daily mean × 252, expressed as %)
    - Annualised std           (daily std  × √252, expressed as %)
    - Skewness of daily returns
    - Excess kurtosis of daily returns  (kurtosis − 3, so 0 for a normal distribution)

These per-ticker statistics are then averaged within each GICS sector.
The total row uses the cross-sectional average across all 415 tickers.

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


def _ticker_stats(returns: pl.DataFrame) -> pl.DataFrame:
    """Per-ticker annualised mean/std and daily skewness/excess-kurtosis."""
    date_col = "date"
    tickers = [c for c in returns.columns if c != date_col]

    rows = []
    for t in tickers:
        s = returns.get_column(t).drop_nulls()
        if s.len() < 10:
            continue
        mean_ann = float(s.mean()) * TRADING_DAYS * 100        # %
        std_ann  = float(s.std())  * (TRADING_DAYS ** 0.5) * 100  # %
        # Polars skewness / kurtosis (Fisher, excess kurtosis)
        sk   = float(s.skew())
        kurt = float(s.kurtosis())   # excess kurtosis (normal = 0)
        rows.append({"ticker": t, "mean_pct": mean_ann, "std_pct": std_ann,
                     "skewness": sk, "kurtosis": kurt})
    return pl.DataFrame(rows)


def main() -> None:
    prices_path  = paper_cfg.PAPER_PRICES_ADJUSTED_415_PATH
    universe_path = paper_cfg.PAPER_UNIVERSE_415_TICKERS_PATH

    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} not found — run  py research/build_paper_dataset.py  first."
        )
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found — run  py research/build_paper_dataset.py  first."
        )

    prices  = pl.read_parquet(prices_path)
    universe = pl.read_csv(universe_path)   # columns: ticker, gics_sector, ...

    returns = _daily_returns(prices)
    stats   = _ticker_stats(returns)

    # Join with sector labels.
    stats = stats.join(
        universe.select(["ticker", "gics_sector"]),
        on="ticker",
        how="left",
    )

    # Per-sector averages.
    sector_table = (
        stats.group_by("gics_sector")
        .agg([
            pl.len().alias("n_firms"),
            pl.col("mean_pct").mean().round(2),
            pl.col("std_pct").mean().round(2),
            pl.col("skewness").mean().round(4),
            pl.col("kurtosis").mean().round(4),
        ])
        .sort("gics_sector")
    )

    # Total row — cross-sectional average over all 415 tickers.
    total = pl.DataFrame([{
        "gics_sector": "TOTAL",
        "n_firms":     stats.height,
        "mean_pct":    round(float(stats["mean_pct"].mean()), 2),
        "std_pct":     round(float(stats["std_pct"].mean()),  2),
        "skewness":    round(float(stats["skewness"].mean()), 4),
        "kurtosis":    round(float(stats["kurtosis"].mean()), 4),
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
