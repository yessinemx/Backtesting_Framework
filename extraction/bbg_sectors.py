"""
Extract GICS sector / sub-industry metadata from Bloomberg for a list of tickers.

Saves to  data/sectors.parquet  with columns:
    ticker, gics_sector, gics_industry_group, gics_industry, gics_sub_industry

Usage:
    py extraction/bbg_sectors.py          # all SPX tickers in membership.parquet
    py extraction/bbg_sectors.py SPX      # only one index
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from pathlib import Path

from config import MEMBERSHIP_PATH, DATA_DIR
from extraction.bloomberg_api import BloombergConnector

SECTORS_PATH = DATA_DIR / "sectors.parquet"

BBG_FIELDS = [
    "GICS_SECTOR_NAME",
    "GICS_INDUSTRY_GROUP_NAME",
    "GICS_INDUSTRY_NAME",
    "GICS_SUB_INDUSTRY_NAME",
]
COL_RENAME = {
    "GICS_SECTOR_NAME": "gics_sector",
    "GICS_INDUSTRY_GROUP_NAME": "gics_industry_group",
    "GICS_INDUSTRY_NAME": "gics_industry",
    "GICS_SUB_INDUSTRY_NAME": "gics_sub_industry",
}


def extract_sectors(bbg, tickers: list[str], batch_size: int = 100) -> pl.DataFrame:
    rows = []
    batches = [tickers[i: i + batch_size] for i in range(0, len(tickers), batch_size)]
    for b_idx, batch in enumerate(batches):
        try:
            data = bbg.bdp(batch, BBG_FIELDS)
            for ticker in batch:
                row = {"ticker": ticker}
                for field, col in COL_RENAME.items():
                    row[col] = data.get(field, {}).get(ticker)
                rows.append(row)
        except Exception as exc:
            print(f"  Batch {b_idx}: {exc}")
            for ticker in batch:
                rows.append({"ticker": ticker, **{v: None for v in COL_RENAME.values()}})
        print(f"  sectors batch {b_idx + 1}/{len(batches)}")
    return pl.DataFrame(rows, schema={
        "ticker": pl.Utf8,
        "gics_sector": pl.Utf8,
        "gics_industry_group": pl.Utf8,
        "gics_industry": pl.Utf8,
        "gics_sub_industry": pl.Utf8,
    })


def main(argv):
    index_ids = argv[1:] or None
    membership = pl.read_parquet(MEMBERSHIP_PATH)
    if index_ids:
        membership = membership.filter(pl.col("index_id").is_in(index_ids))
    tickers = sorted(membership.get_column("ticker").unique().to_list())

    existing_tickers: set[str] = set()
    if SECTORS_PATH.exists():
        existing = pl.read_parquet(SECTORS_PATH)
        existing_tickers = set(existing.get_column("ticker").to_list())

    need = [t for t in tickers if t not in existing_tickers]
    print(f"Tickers total={len(tickers)}  already_stored={len(existing_tickers)}  to_download={len(need)}")
    if not need:
        print("Nothing to download.")
        return

    bbg = BloombergConnector()
    bbg.connect()
    if not bbg.connected:
        print("Bloomberg not available.")
        return

    new_df = extract_sectors(bbg, need)
    bbg.disconnect()

    if SECTORS_PATH.exists():
        old = pl.read_parquet(SECTORS_PATH).filter(~pl.col("ticker").is_in(need))
        combined = pl.concat([old, new_df], how="diagonal").sort("ticker")
    else:
        combined = new_df.sort("ticker")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(SECTORS_PATH)
    n_with_sector = combined.filter(pl.col("gics_sector").is_not_null()).height
    print(f"sectors.parquet saved: {combined.height} rows, {n_with_sector} with GICS sector")


if __name__ == "__main__":
    main(sys.argv)
