"""Polars utilities: normalize wide/long frames, handle pandas index materialization"""
import polars as pl

DATE_COL = "date"
_PANDAS_INDEX = "__index_level_0__"


def normalize_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a wide parquet frame: materialize date column, cast to Datetime, sort"""
    cols = df.columns
    if _PANDAS_INDEX in cols:
        df = df.rename({_PANDAS_INDEX: DATE_COL})
    elif DATE_COL not in cols:
        df = df.rename({cols[0]: DATE_COL})

    if df.schema[DATE_COL] != pl.Datetime:
        df = df.with_columns(pl.col(DATE_COL).cast(pl.Datetime))

    other = [c for c in df.columns if c != DATE_COL]
    return df.select([DATE_COL, *other]).sort(DATE_COL)


def filter_date_range(df: pl.DataFrame, start=None, end=None) -> pl.DataFrame:
    """Restrict a wide frame to an inclusive date range"""
    if start is not None:
        df = df.filter(pl.col(DATE_COL) >= pl.lit(start).str.to_datetime()
                       if isinstance(start, str) else pl.col(DATE_COL) >= start)
    if end is not None:
        df = df.filter(pl.col(DATE_COL) <= pl.lit(end).str.to_datetime()
                       if isinstance(end, str) else pl.col(DATE_COL) <= end)
    return df


def restrict_universe(prices: pl.DataFrame, tickers) -> pl.DataFrame:
    """Keep only the date column plus the requested tickers that are present"""
    keep = [DATE_COL] + [t for t in tickers if t in prices.columns]
    return prices.select(keep)


def to_pandas_wide(df: pl.DataFrame):
    """Convert a wide Polars frame to a pandas DataFrame indexed by date.

    Compatibility bridge for computation modules that still rely on pandas
    index alignment, such as the backtest engine and indicators
    """
    pdf = df.to_pandas()
    if DATE_COL in pdf.columns:
        pdf = pdf.set_index(DATE_COL)
    pdf.index.name = None
    return pdf


def wide_from_pandas(pdf) -> pl.DataFrame:
    """Convert a pandas DataFrame with a date index into a wide Polars frame"""
    tmp = pdf.copy()
    tmp.index.name = DATE_COL
    return pl.from_pandas(tmp.reset_index())
