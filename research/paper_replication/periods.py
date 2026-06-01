"""
Formation / trading period construction.

Replication of Section 4.2 of the paper.

History is split into 252-business-day blocks. A period n is:
    - formation (in-sample): block k
    - trading (out-of-sample): block k+1

As in Table 1, the trading window of one period becomes the formation window
of the next one (consecutive blocks). K blocks -> K-1 periods.

Implemented in Polars: periods store slices (offset, length) into the
date-sorted price frame so the split happens without copying data.
"""
from dataclasses import dataclass
import polars as pl

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Period:
    index: int                       # period number (1-based)
    train_slice: tuple               # (offset, length) in the sorted frame
    trade_slice: tuple
    train_dates: pl.Series           # formation dates
    trade_dates: pl.Series           # trading dates

    @property
    def train_start(self):
        return self.train_dates[0]

    @property
    def train_end(self):
        return self.train_dates[-1]

    @property
    def trade_start(self):
        return self.trade_dates[0]

    @property
    def trade_end(self):
        return self.trade_dates[-1]

    def __repr__(self):
        return (f"Period {self.index} | train "
                f"{self.train_start.date()}\u2192{self.train_end.date()} | "
                f"trade {self.trade_start.date()}\u2192{self.trade_end.date()}")


def build_periods(dates, block_size=TRADING_DAYS_PER_YEAR, max_periods=None):
    """Build the list of periods (formation, trading).

    Parameters
    ----------
    dates : pl.Series | sequence of datetimes
        Business-day calendar (sorted internally).
    block_size : int
        Block size in business days (252 = 1 year).
    max_periods : int | None
        Optional limit on the number of periods.

    Returns
    -------
    list[Period]
    """
    if not isinstance(dates, pl.Series):
        dates = pl.Series("date", list(dates))
    dates = dates.sort()
    n = dates.len()
    n_blocks = n // block_size
    if n_blocks < 2:
        return []

    periods = []
    for k in range(n_blocks - 1):
        train_off = k * block_size
        trade_off = (k + 1) * block_size
        periods.append(Period(
            index=k + 1,
            train_slice=(train_off, block_size),
            trade_slice=(trade_off, block_size),
            train_dates=dates.slice(train_off, block_size),
            trade_dates=dates.slice(trade_off, block_size),
        ))
        if max_periods and len(periods) >= max_periods:
            break
    return periods
