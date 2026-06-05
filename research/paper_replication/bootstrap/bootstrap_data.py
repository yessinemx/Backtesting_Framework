"""
Full paper dataset bootstrap.

Steps
-----
1. Re-extract SPX membership (Bloomberg INDX_MWEIGHT) for the paper window.
2. Re-extract raw close prices for every SPX member that traded in the window.
3. Extract GICS sector metadata for all members.
4. Derive the 415-ticker universe by matching the Table 2 sector distribution.
5. Write everything into  research/data/  (overwrites previous slice).

Run:
    py research/paper_replication/bootstrap/bootstrap_data.py

The resulting files become the single source of truth for the paper replication.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from config import config_backtester as backtester_cfg
from config import config_paper as paper_cfg
from extraction.bloomberg_api import BloombergConnector
from extraction.bbg_members import extract_membership
from extraction.bbg_sectors import extract_sectors, SECTORS_PATH

# --------------------------------------------------------------------------- #
# Paper Table 2 reference values (transcribed from the paper)                 #
# --------------------------------------------------------------------------- #
# Canonical GICS names as returned by Bloomberg GICS_SECTOR_NAME.
# We compare each candidate universe to this distribution.
TABLE2_TARGET = {
    "Communication Services":  15,
    "Consumer Discretionary":  51,
    "Consumer Staples":        26,
    "Energy":                  24,
    "Financials":              60,
    "Health Care":             55,
    "Industrials":             57,
    "Information Technology":  48,
    "Materials":               23,
    "Real Estate":             32,
    "Utilities":               24,
}
TABLE2_TOTAL = 415


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _price_window_filter(prices: pl.DataFrame) -> pl.DataFrame:
    start = datetime.fromisoformat(paper_cfg.REPORT_START_DATE)
    end = datetime.fromisoformat(paper_cfg.REPORT_END_DATE)
    return prices.filter((pl.col("date") >= start) & (pl.col("date") <= end)).sort("date")


def _membership_window(membership: pl.DataFrame) -> pl.DataFrame:
    start = datetime.fromisoformat(paper_cfg.REPORT_START_DATE)
    end = datetime.fromisoformat(paper_cfg.REPORT_END_DATE)
    return (
        membership
        .filter(pl.col("index_id") == paper_cfg.PAIRS_CONFIG["index_id"])
        .filter((pl.col("date") >= start) & (pl.col("date") <= end))
    )


def _score_universe(tickers: list[str], sectors: pl.DataFrame) -> dict:
    """Score a candidate set of tickers against Table 2."""
    sec = sectors.filter(pl.col("ticker").is_in(tickers))
    counts = (
        sec.group_by("gics_sector")
        .agg(pl.len().alias("n"))
        .sort("gics_sector")
    )
    observed = dict(zip(
        counts.get_column("gics_sector").to_list(),
        counts.get_column("n").to_list(),
    ))
    total_delta = abs(len(tickers) - TABLE2_TOTAL)
    sector_delta = sum(
        abs(observed.get(sec, 0) - target)
        for sec, target in TABLE2_TARGET.items()
    )
    return {
        "n_tickers": len(tickers),
        "total_delta": total_delta,
        "sector_delta": sector_delta,
        "score": total_delta + sector_delta,
        "observed": observed,
    }


def _derive_415(
    membership_window: pl.DataFrame,
    raw_prices: pl.DataFrame,
    sectors: pl.DataFrame,
) -> tuple[list[str], dict]:
    """
    Find the best-matching 415-ticker candidate universe.

    Strategy
    --------
    1. From the full SPX membership window, keep only tickers that have:
       - a recognised GICS sector (filters Bloomberg stub identifiers)
       - at least ``min_periods`` paper-period windows with zero null prices
    2. Among those, rank by total membership-snapshot count (persistence).
    3. Try several top-N cuts around 415 and return the one whose sector
       distribution is closest to Table 2.
    """
    spx_tickers = set(membership_window.get_column("ticker").unique().to_list())
    price_cols = [c for c in raw_prices.columns if c != "date" and c in spx_tickers]

    # Tickers with a known GICS sector.
    with_sector = set(
        sectors
        .filter(pl.col("gics_sector").is_not_null())
        .get_column("ticker")
        .to_list()
    )
    price_cols = [c for c in price_cols if c in with_sector]

    # Count how many paper-period 2-year windows each ticker has complete prices.
    period_ok: dict[str, int] = {c: 0 for c in price_cols}
    for tr_s, _tr_e, _td_s, td_e in paper_cfg.PAIRS_CONFIG["paper_periods"]:
        window = raw_prices.filter(
            (pl.col("date") >= datetime.fromisoformat(tr_s))
            & (pl.col("date") <= datetime.fromisoformat(td_e))
        )
        if window.height == 0:
            continue
        nulls = window.select([pl.col(c).null_count().alias(c) for c in price_cols])
        for c in price_cols:
            if nulls[c][0] == 0:
                period_ok[c] += 1

    # Membership persistence.
    persistence = (
        membership_window.group_by("ticker")
        .agg(pl.col("date").n_unique().alias("n_snapshots"))
        .filter(pl.col("ticker").is_in(price_cols))
        .sort(["n_snapshots", "ticker"], descending=[True, False])
    )
    persistence_map = dict(zip(
        persistence.get_column("ticker").to_list(),
        persistence.get_column("n_snapshots").to_list(),
    ))

    # Score different (min_periods, top_N) combinations.
    best_score = 10**9
    best_tickers: list[str] = []
    best_info: dict = {}
    for min_periods in [1, 2, 3, 4, 5, 6, 7]:
        eligible = sorted(
            [c for c in price_cols if period_ok[c] >= min_periods],
            key=lambda c: (-persistence_map.get(c, 0), c),
        )
        for top_n in [380, 390, 400, 405, 410, 415, 420, 425, 430, 435, 440, 450]:
            candidate = eligible[:top_n]
            if len(candidate) < top_n:
                break
            info = _score_universe(candidate, sectors)
            if info["score"] < best_score:
                best_score = info["score"]
                best_tickers = candidate
                best_info = {"min_periods": min_periods, "top_n": top_n, **info}

    return best_tickers, best_info


def _period_counts(
    membership_window: pl.DataFrame,
    raw_prices: pl.DataFrame,
    tickers: list[str],
) -> pl.DataFrame:
    ticker_set = set(tickers)
    rows = []
    for period, (tr_s, _tr_e, _td_s, td_e) in enumerate(
        paper_cfg.PAIRS_CONFIG["paper_periods"], start=1
    ):
        start_dt = datetime.fromisoformat(tr_s)
        end_dt = datetime.fromisoformat(td_e)
        window_mem = membership_window.filter(
            (pl.col("date") >= start_dt)
            & (pl.col("date") <= end_dt)
            & pl.col("ticker").is_in(tickers)
        )
        n_snapshots = window_mem.get_column("date").n_unique()
        counts = window_mem.group_by("ticker").agg(pl.col("date").n_unique().alias("n"))
        continuous = counts.filter(pl.col("n") == n_snapshots).height if n_snapshots else 0
        window_prices = raw_prices.filter(
            (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
        )
        cols = [c for c in tickers if c in raw_prices.columns]
        if cols and window_prices.height:
            nulls = window_prices.select([pl.col(c).null_count().alias(c) for c in cols])
            full_prices = sum(1 for c in cols if nulls[c][0] == 0)
        else:
            full_prices = 0
        rows.append({
            "period": period,
            "paper_target": 415,
            "continuous_members_in_415": continuous,
            "full_prices_in_415": full_prices,
        })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    paper_cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    bbg = BloombergConnector()
    bbg.connect()
    if not bbg.connected:
        print("Bloomberg not available. Aborting.")
        return

    # ------------------------------------------------------------------ #
    # 1. Refresh SPX membership over the paper window.                    #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Refreshing SPX membership …")
    membership = extract_membership(bbg, index_ids=["SPX"])
    membership_window = _membership_window(membership)
    spx_window_tickers = sorted(membership_window.get_column("ticker").unique().to_list())
    print(f"      {membership_window.height} rows | {len(spx_window_tickers)} unique tickers in window")

    # ------------------------------------------------------------------ #
    # 2. Extract GICS sectors.                                            #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Extracting GICS sectors …")
    existing_tickers: set[str] = set()
    if SECTORS_PATH.exists():
        existing_tickers = set(pl.read_parquet(SECTORS_PATH).get_column("ticker").to_list())
    need_sectors = [t for t in spx_window_tickers if t not in existing_tickers]
    if need_sectors:
        new_sectors = extract_sectors(bbg, need_sectors)
        if SECTORS_PATH.exists():
            old_sec = pl.read_parquet(SECTORS_PATH).filter(~pl.col("ticker").is_in(need_sectors))
            sectors_full = pl.concat([old_sec, new_sectors], how="diagonal").sort("ticker")
        else:
            sectors_full = new_sectors.sort("ticker")
        sectors_full.write_parquet(SECTORS_PATH)
    else:
        sectors_full = pl.read_parquet(SECTORS_PATH)
    sectors = sectors_full.filter(pl.col("ticker").is_in(spx_window_tickers))
    n_with = sectors.filter(pl.col("gics_sector").is_not_null()).height
    print(f"      {sectors.height} rows | {n_with} with GICS sector")

    # ------------------------------------------------------------------ #
    # 3. Re-extract raw prices for the paper window.                      #
    # ------------------------------------------------------------------ #
    print("\n[3/5] Re-extracting raw prices for paper window …")
    from extraction.bbg_returns import extract_prices
    extract_prices(
        bbg,
        tickers=spx_window_tickers,
        adjust=False,
        prices_path=backtester_cfg.RAW_PRICES_PATH,
        returns_path=backtester_cfg.RAW_RETURNS_PATH,
    )
    raw_all = pl.read_parquet(backtester_cfg.RAW_PRICES_PATH)
    raw_window = _price_window_filter(raw_all)
    print(f"      raw_prices shape: {raw_window.shape}")

    # ------------------------------------------------------------------ #
    # 4. Derive the 415-ticker universe matching Table 2.                 #
    # ------------------------------------------------------------------ #
    print("\n[4/5] Deriving best-matching 415-ticker universe …")
    best_tickers, best_info = _derive_415(membership_window, raw_window, sectors)
    print(f"      Rule: min_periods={best_info['min_periods']}  top_n={best_info['top_n']}")
    print(f"      n_tickers={best_info['n_tickers']}  total_delta={best_info['total_delta']}  sector_delta={best_info['sector_delta']}")
    print("      Observed vs Table 2:")
    for sector, target in sorted(TABLE2_TARGET.items()):
        obs = best_info["observed"].get(sector, 0)
        print(f"        {sector:<35} obs={obs:3d}  paper={target:3d}  delta={obs-target:+d}")

    # ------------------------------------------------------------------ #
    # 5. Write research/data/ artefacts.                                  #
    # ------------------------------------------------------------------ #
    print("\n[5/5] Writing research/data/ artefacts …")

    # Membership window.
    membership_window.write_parquet(paper_cfg.PAPER_MEMBERSHIP_PATH)
    print(f"      {paper_cfg.PAPER_MEMBERSHIP_PATH.name}: {membership_window.shape}")

    # Adjusted prices (full window, all SPX).
    adj_all = pl.read_parquet(backtester_cfg.PRICES_PATH)
    adj_window = _price_window_filter(adj_all)
    adj_window_spx = adj_window.select(
        ["date"] + [c for c in adj_window.columns if c != "date" and c in set(spx_window_tickers)]
    )
    adj_window_spx.write_parquet(paper_cfg.PAPER_PRICES_ADJUSTED_PATH)
    print(f"      {paper_cfg.PAPER_PRICES_ADJUSTED_PATH.name}: {adj_window_spx.shape}")

    raw_window_spx = raw_window.select(
        ["date"] + [c for c in raw_window.columns if c != "date" and c in set(spx_window_tickers)]
    )
    raw_window_spx.write_parquet(paper_cfg.PAPER_PRICES_RAW_PATH)
    print(f"      {paper_cfg.PAPER_PRICES_RAW_PATH.name}: {raw_window_spx.shape}")

    # 415-ticker universe.
    ticker_df = (
        sectors.filter(pl.col("ticker").is_in(best_tickers))
        .select(["ticker", "gics_sector", "gics_industry_group", "gics_industry", "gics_sub_industry"])
        .sort("ticker")
        .with_row_index("selection_rank", offset=1)
    )
    ticker_df.write_csv(paper_cfg.PAPER_UNIVERSE_415_TICKERS_PATH)
    print(f"      {paper_cfg.PAPER_UNIVERSE_415_TICKERS_PATH.name}: {ticker_df.shape}")

    adj_415 = adj_window_spx.select(["date"] + [c for c in best_tickers if c in adj_window_spx.columns])
    raw_415 = raw_window_spx.select(["date"] + [c for c in best_tickers if c in raw_window_spx.columns])
    adj_415.write_parquet(paper_cfg.PAPER_PRICES_ADJUSTED_415_PATH)
    raw_415.write_parquet(paper_cfg.PAPER_PRICES_RAW_415_PATH)
    print(f"      {paper_cfg.PAPER_PRICES_ADJUSTED_415_PATH.name}: {adj_415.shape}")
    print(f"      {paper_cfg.PAPER_PRICES_RAW_415_PATH.name}: {raw_415.shape}")

    bbg.disconnect()
    print("\nDone. Paper dataset written to", paper_cfg.DATA_DIR)
    print("Run  py research/main.py --full  to regenerate every paper table and figure.")


if __name__ == "__main__":
    main()
