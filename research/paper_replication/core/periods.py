"""Sequential formation/trading period construction (252-day blocks)."""
from dataclasses import dataclass
import polars as pl

from config import config_paper as research_config

TRADING_DAYS_PER_YEAR = research_config.TRADING_DAYS_PER_YEAR


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


def build_periods(dates, block_size=TRADING_DAYS_PER_YEAR, max_periods=None,
                  paper_periods=None):
    """Build the list of periods (formation, trading).

    Parameters
    ----------
    dates : pl.Series | sequence of datetimes
        Business-day calendar (sorted internally).
    block_size : int
        Block size in business days (252 = 1 year). Ignored when
        `paper_periods` is provided.
    max_periods : int | None
        Optional limit on the number of periods.
    paper_periods : list[tuple[str, str, str, str]] | None
        If given, each entry is (train_start, train_end, trade_start, trade_end)
        as YYYY-MM-DD strings. Periods are sliced by exact calendar matching
        against the available trading-day calendar (paper Table 1).

    Returns
    -------
    list[Period]
    """
    if not isinstance(dates, pl.Series):
        dates = pl.Series("date", list(dates))
    dates = dates.sort()
    n = dates.len()

    if paper_periods:
        return _build_from_calendar(dates, paper_periods, max_periods)

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


def _build_from_calendar(dates, paper_periods, max_periods):
    """Slice the calendar by explicit (train_start, train_end, trade_start, trade_end) tuples.

    Each boundary is snapped to the nearest available trading day on/after the
    requested date (for starts) or on/before the requested date (for ends).
    """
    from datetime import datetime
    arr = dates.to_list()

    def to_dt(s):
        return datetime.strptime(s, "%Y-%m-%d")

    def find_ge(target):
        for i, d in enumerate(arr):
            if d >= target:
                return i
        return -1

    def find_le(target):
        last = -1
        for i, d in enumerate(arr):
            if d <= target:
                last = i
            else:
                break
        return last

    periods = []
    for k, (tr_s, tr_e, td_s, td_e) in enumerate(paper_periods, start=1):
        i0 = find_ge(to_dt(tr_s))
        i1 = find_le(to_dt(tr_e))
        j0 = find_ge(to_dt(td_s))
        j1 = find_le(to_dt(td_e))
        if min(i0, i1, j0, j1) < 0 or i1 < i0 or j1 < j0:
            continue
        train_len = i1 - i0 + 1
        trade_len = j1 - j0 + 1
        periods.append(Period(
            index=k,
            train_slice=(i0, train_len),
            trade_slice=(j0, trade_len),
            train_dates=dates.slice(i0, train_len),
            trade_dates=dates.slice(j0, trade_len),
        ))
        if max_periods and len(periods) >= max_periods:
            break
    return periods
