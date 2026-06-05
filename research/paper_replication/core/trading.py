"""Pair trading simulation: open on 2-sigma threshold, close on reversion or period end."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Trade:
    open_idx: int
    close_idx: int
    sign: int
    forced: bool
    pnl: float
    ret_i: float
    ret_j: float


@dataclass
class PairResult:
    i: str
    j: str
    beta: float
    trades: list = field(default_factory=list)
    daily_returns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dates: object = None              # pl.Series aligned with daily_returns
    category: str = "inactive"        # full | partial | non | inactive
    total_pnl: float = 0.0
    total_cost: float = 0.0

    @property
    def active(self):
        return len(self.trades) > 0


def simulate_pair(spec, tc_per_share=0.0, forced_close_only=False):
    """Simulate pair trading from a SpreadSpec.

    Parameters
    ----------
    spec : SpreadSpec
        Contains the threshold, β, the trading spread, and aligned original prices.
    tc_per_share : float
        Transaction cost θ (e.g. 0.001 = 10 bps) per round trip and per
        security (eq. A4). Use 0 for a frictionless run.
    forced_close_only : bool
        If True, do not close when the spread crosses zero; close open trades
        only at the end of the trading period. This reproduces the paper's
        forced-close robustness experiment.

    Returns
    -------
    PairResult
    """
    i, j, beta = spec.i, spec.j, spec.beta
    thr = spec.threshold
    abeta = abs(beta)

    eps = np.asarray(spec.trade_spread, dtype=float)
    Si = np.asarray(spec.trade_si, dtype=float)
    Sj = np.asarray(spec.trade_sj, dtype=float)
    n = len(eps)

    daily = np.zeros(n)
    trades = []

    in_trade = False
    sign = 0
    open_idx = 0

    def close_trade(t_open, t_close, sgn, forced):
        pnl = float(daily[t_open + 1:t_close + 1].sum())
        ret_i = (Si[t_close] - Si[t_open]) / Si[t_open]
        ret_j = (Sj[t_close] - Sj[t_open]) / Sj[t_open]
        trades.append(Trade(
            open_idx=t_open, close_idx=t_close, sign=sgn,
            forced=forced, pnl=pnl, ret_i=float(ret_i), ret_j=float(ret_j),
        ))

    for t in range(n):
        if in_trade:
            r_i = (Si[t] - Si[t - 1]) / Si[open_idx]
            r_j = (Sj[t] - Sj[t - 1]) / Sj[open_idx]
            daily[t] = sign * (-r_i + abeta * r_j)

            if not forced_close_only and (np.sign(eps[t]) != np.sign(eps[open_idx]) or eps[t] == 0):
                close_trade(open_idx, t, sign, forced=False)
                in_trade = False
                sign = 0

        if not in_trade and t < n - 1:
            if eps[t] > thr:
                in_trade, sign, open_idx = True, 1, t
            elif eps[t] < -thr:
                in_trade, sign, open_idx = True, -1, t

    if in_trade:
        close_trade(open_idx, n - 1, sign, forced=True)

    # Convergence category.
    if not trades:
        category = "inactive"
    else:
        has_forced = trades[-1].forced
        full_turns = sum(1 for tr in trades if not tr.forced)
        if full_turns >= 1 and not has_forced:
            category = "full"
        elif full_turns >= 1 and has_forced:
            category = "partial"
        else:
            category = "non"

    # Transaction costs (eq. A4).
    total_cost = 0.0
    if tc_per_share > 0:
        for tr in trades:
            total_cost += (2 * (1 + abeta) + (tr.ret_i + abeta * tr.ret_j)) * tc_per_share
        if n > 0:
            daily[-1] -= total_cost

    total_pnl = sum(tr.pnl for tr in trades) - total_cost

    return PairResult(
        i=i, j=j, beta=beta, trades=trades, daily_returns=daily,
        dates=spec.trade_dates, category=category,
        total_pnl=float(total_pnl), total_cost=float(total_cost),
    )
