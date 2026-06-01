import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import polars as pl

from config import RISKFREE_CONFIG, DATA_DIR, RISKFREE_PATH, DATA_START, DATA_END

DATE_COL = "date"


def _series_frame(ccy, date_vals):
    """Build a Polars frame [date, <ccy>] from a {date: value} mapping."""
    return pl.DataFrame({
        DATE_COL: list(date_vals.keys()),
        ccy: list(date_vals.values()),
    }).with_columns([
        pl.col(DATE_COL).cast(pl.Datetime),
        pl.col(ccy).cast(pl.Float64),
    ]).sort(DATE_COL)


def extract_riskfree(bbg, progress_callback=None):
    dt_start = datetime.strptime(DATA_START, "%Y-%m-%d")
    dt_end = datetime.strptime(DATA_END, "%Y-%m-%d")

    ccy_frames = []
    currencies = list(RISKFREE_CONFIG.keys())

    for i, (ccy, cfg) in enumerate(RISKFREE_CONFIG.items()):
        day_count = cfg["day_count"]
        combined = None  # frame [date, ccy]

        for ticker in cfg["tickers"]:
            try:
                raw = bbg.bdh(ticker, "PX_LAST", dt_start, dt_end)
                if raw and "PX_LAST" in raw and ticker in raw["PX_LAST"]:
                    vals = raw["PX_LAST"][ticker]
                    if not vals:
                        continue
                    f = _series_frame(ccy, vals)
                    if combined is None:
                        combined = f
                    else:
                        # Fill gaps with the next ticker when available.
                        combined = (
                            combined.join(f, on=DATE_COL, how="full",
                                          coalesce=True, suffix="_next")
                            .with_columns(
                                pl.coalesce([pl.col(ccy), pl.col(f"{ccy}_next")]).alias(ccy)
                            )
                            .select([DATE_COL, ccy])
                        )
                    print(f"  {ticker}: {f.height} points")
            except Exception as e:
                print(f"  {ticker}: {e}")

        if combined is None:
            print(f"No data for {ccy}")
            continue

        # Convert annualized percentage rates into simple daily rates and fill gaps.
        combined = combined.sort(DATE_COL).with_columns(
            pl.col(ccy).forward_fill().backward_fill()
        ).with_columns(
            (pl.col(ccy) / 100.0 / day_count).alias(ccy)
        )
        ccy_frames.append(combined)

        if progress_callback:
            progress_callback((i + 1) / len(currencies), f"{ccy} done")

    if not ccy_frames:
        print("No risk-free data retrieved")
        return pl.DataFrame(schema={DATE_COL: pl.Datetime})

    rf = ccy_frames[0]
    for f in ccy_frames[1:]:
        rf = rf.join(f, on=DATE_COL, how="full", coalesce=True)
    rf = rf.sort(DATE_COL)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rf.write_parquet(RISKFREE_PATH)
    print(f"riskfree.parquet saved: {rf.shape}")
    return rf

