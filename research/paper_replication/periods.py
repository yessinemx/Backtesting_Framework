"""
Découpage en périodes de formation / trading
============================================
Réplication de la Section 4.2 du papier.

L'historique est découpé en blocs de 252 jours ouvrés. Une période n est :
  - formation (in-sample)    : bloc k
  - trading   (out-of-sample): bloc k+1

Comme dans la Table 1, la fenêtre de trading d'une période devient la
formation de la suivante (blocs consécutifs). K blocs -> K-1 périodes.

Travaille en Polars : les périodes stockent des *slices* (offset, longueur)
dans le frame de prix trié par date, pour un découpage sans copie.
"""
from dataclasses import dataclass
import polars as pl

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Period:
    index: int                       # numéro de période (1-based)
    train_slice: tuple               # (offset, length) dans le frame trié
    trade_slice: tuple
    train_dates: pl.Series           # dates de formation
    trade_dates: pl.Series           # dates de trading

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
    """Construit la liste des périodes (formation, trading).

    Parameters
    ----------
    dates : pl.Series | séquence de datetimes
        Calendrier des jours ouvrés (sera trié).
    block_size : int
        Taille d'un bloc en jours ouvrés (252 = 1 an).
    max_periods : int | None
        Limite optionnelle du nombre de périodes.

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
