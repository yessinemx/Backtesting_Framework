"""
Performance metrics for the pairs portfolio.

Replication of Section 5 (result tables) from the paper.

Pairs are aggregated into an equal-weighted portfolio. Mean return covers all
selected pairs, including inactive ones (P&L = 0):
    R̄_n = Σ_ij R_ij / N_n   (eq. A2)

The portfolio daily return series is aggregated in Polars
(group_by date -> mean), then the ratios are computed in NumPy.
"""
from dataclasses import dataclass, asdict
import numpy as np
import polars as pl

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PairsReport:
    method: str                 # "distance" | "cointegration"
    variant: str                # "standard" | "wavelet"
    n_pairs: int
    n_active: int
    mean_return: float
    std_return: float
    sharpe: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    pct_positive: float
    n_full: int
    n_partial: int
    n_non: int
    n_inactive: int

    def as_dict(self):
        return asdict(self)


def _skewness(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0
    m, s = x.mean(), x.std(ddof=0)
    return 0.0 if s == 0 else float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return 0.0
    m, s = x.mean(), x.std(ddof=0)
    return 0.0 if s == 0 else float(np.mean(((x - m) / s) ** 4) - 3.0)


def _max_drawdown(cum):
    if len(cum) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cum)
    return float(((cum - running_max) / running_max).min())


def _portfolio_daily(pair_results):
    """Daily portfolio series computed as the mean pair return by date."""
    frames = []
    for pr in pair_results:
        if pr.dates is None or len(pr.daily_returns) == 0:
            continue
        frames.append(pl.DataFrame({
            "date": pr.dates,
            "ret": np.asarray(pr.daily_returns, dtype=float),
        }))
    if not frames:
        return np.zeros(0)
    allf = pl.concat(frames, how="vertical")
    agg = allf.group_by("date").agg(pl.col("ret").mean()).sort("date")
    return agg.get_column("ret").to_numpy()


def aggregate_metrics(pair_results, method, variant, n_pairs=None):
    """Aggregate a list of PairResult objects into a PairsReport."""
    if n_pairs is None:
        n_pairs = len(pair_results)

    pnls = np.array([pr.total_pnl for pr in pair_results], dtype=float)
    active = [pr for pr in pair_results if pr.active]
    n_active = len(active)

    mean_return = float(pnls.mean()) if len(pnls) > 0 else 0.0
    std_return = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0

    port_daily = _portfolio_daily(pair_results)
    if len(port_daily) > 1 and port_daily.std(ddof=1) > 0:
        sharpe = float(port_daily.mean() / port_daily.std(ddof=1)
                       * np.sqrt(TRADING_DAYS_PER_YEAR))
        cum = np.cumprod(1.0 + port_daily)
        max_dd = _max_drawdown(cum)
    else:
        sharpe, max_dd = 0.0, 0.0

    if len(pnls) > 0:
        var_95 = float(np.percentile(pnls, 5))
        tail = pnls[pnls <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
    else:
        var_95 = cvar_95 = 0.0

    pct_positive = (
        float(np.mean([pr.total_pnl > 0 for pr in active])) if active else 0.0
    )

    cats = [pr.category for pr in pair_results]
    return PairsReport(
        method=method, variant=variant,
        n_pairs=n_pairs, n_active=n_active,
        mean_return=mean_return, std_return=std_return, sharpe=sharpe,
        skewness=_skewness(pnls), kurtosis=_kurtosis(pnls),
        max_drawdown=max_dd, var_95=var_95, cvar_95=cvar_95,
        pct_positive=pct_positive,
        n_full=cats.count("full"), n_partial=cats.count("partial"),
        n_non=cats.count("non"), n_inactive=cats.count("inactive"),
    )
