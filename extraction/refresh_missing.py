"""
Download Bloomberg price history for membership tickers that are missing
from the local prices.parquet file.

Usage:
    py extraction/refresh_missing.py                # all indices
    py extraction/refresh_missing.py SPX            # one or more indices
    py extraction/refresh_missing.py SPX NDX
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from config import MEMBERSHIP_PATH, PRICES_PATH
from extraction.bloomberg_api import BloombergConnector
from extraction.bbg_returns import extract_prices


def missing_tickers(index_ids=None):
    mem = pl.read_parquet(MEMBERSHIP_PATH)
    if index_ids:
        mem = mem.filter(pl.col("index_id").is_in(list(index_ids)))
    want = set(mem["ticker"].unique().to_list())

    have = set()
    if PRICES_PATH.exists():
        have = set(pl.read_parquet(PRICES_PATH).columns) - {"date"}
    return sorted(want - have), len(want), len(want & have)


def main(argv):
    index_ids = argv[1:] or None
    missing, total, present = missing_tickers(index_ids)
    label = ",".join(index_ids) if index_ids else "ALL"
    print(f"[{label}] members={total}  in_prices={present}  missing={len(missing)}")
    if not missing:
        print("Nothing to download.")
        return

    bbg = BloombergConnector()
    bbg.connect()
    extract_prices(bbg, tickers=missing)


if __name__ == "__main__":
    main(sys.argv)
