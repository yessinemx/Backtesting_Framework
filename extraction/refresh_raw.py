"""
Download RAW (unadjusted) Bloomberg closing prices for membership tickers
into a separate file (data/prices_raw.parquet), leaving the adjusted
prices.parquet untouched.

This exists for the wavelet replication study: total-return-adjusted series
are too smooth for level-1 MODWT filtering to correct the regression
coefficients, so the paper's effect cannot appear. Raw close prices carry
the microstructure noise the wavelet feeds on.

Usage:
    py extraction/refresh_raw.py                # all indices
    py extraction/refresh_raw.py SPX            # one or more indices
    py extraction/refresh_raw.py SPX NDX
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from config import MEMBERSHIP_PATH, RAW_PRICES_PATH, RAW_RETURNS_PATH
from extraction.bloomberg_api import BloombergConnector
from extraction.bbg_returns import extract_prices


def missing_tickers(index_ids=None):
    mem = pl.read_parquet(MEMBERSHIP_PATH)
    if index_ids:
        mem = mem.filter(pl.col("index_id").is_in(list(index_ids)))
    want = set(mem["ticker"].unique().to_list())

    have = set()
    if RAW_PRICES_PATH.exists():
        have = set(pl.read_parquet(RAW_PRICES_PATH).columns) - {"date"}
    return sorted(want - have), len(want), len(want & have)


def main(argv):
    index_ids = argv[1:] or None
    missing, total, present = missing_tickers(index_ids)
    label = ",".join(index_ids) if index_ids else "ALL"
    print(f"[{label}] members={total}  in_raw={present}  missing={len(missing)}")
    if not missing:
        print("Nothing to download.")
        return

    bbg = BloombergConnector()
    bbg.connect()
    extract_prices(
        bbg,
        tickers=missing,
        adjust=False,
        prices_path=RAW_PRICES_PATH,
        returns_path=RAW_RETURNS_PATH,
    )


if __name__ == "__main__":
    main(sys.argv)
