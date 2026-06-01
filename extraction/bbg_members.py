import sys, os
from datetime import datetime

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDEX_CONFIG, DATA_DIR, MEMBERSHIP_PATH, DATA_START, DATA_END


def _month_starts(start: str, end: str):
    """List the first calendar day of each month between start and end."""
    dt_start = datetime.strptime(start, "%Y-%m-%d").date()
    dt_end = datetime.strptime(end, "%Y-%m-%d").date()
    return pl.date_range(dt_start, dt_end, interval="1mo", eager=True).to_list()


def extract_membership(bbg, progress_callback=None, index_ids=None):
    """Extract index membership.

    index_ids: list of index identifiers to extract; None means all indices.
    Rows for the targeted indices replace the corresponding rows in the
    existing parquet file (merge by index_id).
    """
    targets = {k: v for k, v in INDEX_CONFIG.items()
               if index_ids is None or k in index_ids}
    dates = _month_starts(DATA_START, DATA_END)  # month starts
    rows = []
    total = len(dates) * len(targets)
    done = 0

    for index_id, cfg in targets.items():
        bbg_ticker = cfg["bbg_ticker"]
        for dt in dates:
            date_str = dt.strftime("%Y%m%d")
            try:
                df = bbg.bds(
                    bbg_ticker,
                    "INDX_MWEIGHT",
                    overrides={"END_DATE_OVERRIDE": date_str},
                )
                if df is not None and df.height > 0:
                    member_col = df.columns[0]
                    members = (
                        df.get_column(member_col)
                        .drop_nulls()
                        .cast(pl.Utf8)
                        .to_list()
                    )
                    for m in members:
                        ticker = m if " Equity" in m else f"{m} Equity"
                        rows.append({"date": dt, "index_id": index_id, "ticker": ticker})
            except Exception as e:
                print(f"  {index_id} {date_str}: {e}")

            done += 1
            if progress_callback:
                progress_callback(done / total, f"{index_id} {date_str}")

    if not rows:
        print("No membership data retrieved")
        return pl.DataFrame(schema={"date": pl.Datetime, "index_id": pl.Utf8,
                                    "ticker": pl.Utf8})

    new = pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ns"))
    )

    # Merge with the existing file and replace the targeted indices.
    if MEMBERSHIP_PATH.exists():
        existing = pl.read_parquet(MEMBERSHIP_PATH).with_columns(
            pl.col("date").cast(pl.Datetime("ns"))
        )
        existing = existing.filter(~pl.col("index_id").is_in(list(targets.keys())))
        membership = pl.concat([existing, new], how="diagonal").sort(["date", "index_id"])
    else:
        membership = new

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    membership.write_parquet(MEMBERSHIP_PATH)
    n_unique = membership["ticker"].n_unique()
    print(f"membership.parquet saved: {membership.height} rows, {n_unique} unique tickers")
    return membership
