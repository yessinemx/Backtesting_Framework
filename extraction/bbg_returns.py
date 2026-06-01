import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import polars as pl

from config import (DATA_DIR, MEMBERSHIP_PATH, PRICES_PATH, RETURNS_PATH,
                    DATA_START, DATA_END)

DATE_COL = "date"


def _series_frame(ticker, date_vals):
    """Construit un frame Polars [date, <ticker>] à partir d'un dict {date: px}."""
    dates = list(date_vals.keys())
    values = list(date_vals.values())
    return pl.DataFrame({DATE_COL: dates, ticker: values}).with_columns([
        pl.col(DATE_COL).cast(pl.Datetime("ns")),
        pl.col(ticker).cast(pl.Float64),
    ])


def extract_prices(bbg, membership=None,
                   batch_size: int = 40, progress_callback=None,
                   tickers=None):
    """Telecharge PX_LAST et merge avec prices/returns existants.

    tickers : si fourni, ne telecharge que cette liste (les autres tickers
    de la membership sont ignores). Sinon tout ce qui est dans membership.
    Les tickers deja presents dans prices.parquet sont skip.
    """
    if membership is None:
        membership = pl.read_parquet(MEMBERSHIP_PATH)
    elif not isinstance(membership, pl.DataFrame):
        membership = pl.from_pandas(membership)

    if tickers is None:
        candidate = sorted(membership["ticker"].unique().to_list())
    else:
        candidate = sorted(set(tickers))

    # Skip ceux deja dans le parquet existant
    existing_prices = None
    if PRICES_PATH.exists():
        existing_prices = pl.read_parquet(PRICES_PATH)
        # Compat : anciens parquets ecrits par pandas sans nommer l'index
        if DATE_COL not in existing_prices.columns:
            for legacy in ("__index_level_0__", "index", "Date"):
                if legacy in existing_prices.columns:
                    existing_prices = existing_prices.rename({legacy: DATE_COL})
                    break
        existing_prices = existing_prices.with_columns(
            pl.col(DATE_COL).cast(pl.Datetime("ns"))
        )
        already = set(existing_prices.columns) - {DATE_COL}
        new_tickers = [t for t in candidate if t not in already]
    else:
        new_tickers = candidate

    if not new_tickers:
        print(f"All {len(candidate)} tickers already present, nothing to download")
        if progress_callback:
            progress_callback(1.0, "Nothing to download")
        # Recharger returns aussi pour cohérence
        returns = pl.read_parquet(RETURNS_PATH) if RETURNS_PATH.exists() else existing_prices
        return existing_prices, returns

    print(f"Downloading PX_LAST for {len(new_tickers)} new tickers "
          f"(skipped {len(candidate) - len(new_tickers)} already present)")

    dt_start = datetime.strptime(DATA_START, "%Y-%m-%d")
    dt_end = datetime.strptime(DATA_END, "%Y-%m-%d")

    frames = []
    batches = [new_tickers[i:i+batch_size]
               for i in range(0, len(new_tickers), batch_size)]

    for b_idx, batch in enumerate(batches):
        try:
            raw = bbg.bdh(batch, "PX_LAST", dt_start, dt_end)
            if raw and "PX_LAST" in raw:
                for ticker, date_vals in raw["PX_LAST"].items():
                    if date_vals:
                        frames.append(_series_frame(ticker, date_vals))
        except Exception as e:
            print(f"  Batch {b_idx}: {e}")

        if progress_callback:
            progress_callback((b_idx + 1) / len(batches),
                              f"Batch {b_idx+1}/{len(batches)}")

    if not frames:
        print("No price data retrieved")
        empty = pl.DataFrame(schema={DATE_COL: pl.Datetime})
        return existing_prices if existing_prices is not None else empty, empty

    # Concatenation des nouveaux tickers
    new_prices = frames[0]
    for f in frames[1:]:
        new_prices = new_prices.join(f, on=DATE_COL, how="full", coalesce=True)

    # Merge avec l'existant
    if existing_prices is not None:
        prices = existing_prices.join(new_prices, on=DATE_COL, how="full", coalesce=True)
    else:
        prices = new_prices

    prices = prices.sort(DATE_COL)
    value_cols = [c for c in prices.columns if c != DATE_COL]
    prices = prices.with_columns([pl.col(c).forward_fill() for c in value_cols])

    returns = prices.with_columns([
        ((pl.col(c) / pl.col(c).shift(1)) - 1.0).fill_null(0.0).alias(c)
        for c in value_cols
    ])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(PRICES_PATH)
    returns.write_parquet(RETURNS_PATH)

    print(f"prices.parquet  saved: {prices.shape}")
    print(f"returns.parquet saved: {returns.shape}")
    return prices, returns
