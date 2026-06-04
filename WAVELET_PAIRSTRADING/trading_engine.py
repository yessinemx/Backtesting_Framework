"""
trading_engine.py
=================
Implements the pairs trading rules from Eroğlu et al. (2023), Section 4.3
and the return calculation from Appendix A.1.

Trading rules:
- Open long/short when |spread| > 2σ (computed on training period)
- Close when spread crosses zero (first sign change)
- Force-close at end of trading period
- Returns: committed capital formula (eq. A1–A3)

Position logic (paper, p. 1134):
  If ê_{t} > threshold_upper (+2σ):
      Short $1 of S_i, Long $|β| of S_j
  If ê_{t} < threshold_lower (−2σ):
      Long $1 of S_i, Short $|β| of S_j

Daily P&L:
  r_{i,j,t} = sign(spread_open) × [
      − (S_i,t − S_i,t-1) / S_i,t_open
      + |β| × (S_j,t − S_j,t-1) / S_j,t_open
  ]

Transaction costs (Section 5.5.3):
  θ = 10 bps per share per half-turn (paper uses 0.1%)
  Full round-trip cost: (1 + |β| + S_i,close/S_i,open + |β|·S_j,close/S_j,open) × θ
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from spread_estimation import SpreadParams, build_spread


@dataclass
class TradeRecord:
    """Record for a single completed trade."""
    stock_i: str
    stock_j: str
    open_date: object
    close_date: object
    open_idx: int
    close_idx: int
    direction: int          # +1: short i / long j; -1: long i / short j
    beta: float
    gross_return: float
    net_return: float       # after transaction costs
    force_closed: bool      # True if closed at period end


@dataclass
class PairResult:
    """All trades and daily P&L for one pair in one trading period."""
    stock_i: str
    stock_j: str
    total_return: float         # sum of gross returns across all trades
    total_net_return: float
    n_trades: int
    n_full_convergent: int      # fully closed
    n_non_convergent: int       # force-closed
    n_partial_convergent: int   # opened multiple times, some force-closed
    daily_pnl: np.ndarray       # length = trading period
    trades: list = field(default_factory=list)


def _trade_return(
    prices_i: np.ndarray,
    prices_j: np.ndarray,
    open_idx: int,
    close_idx: int,
    direction: int,
    beta: float,
) -> float:
    """
    Compute total return for one trade from open_idx+1 to close_idx (inclusive).
    Uses committed capital formula from Appendix A.1.

    direction = +1  => ê > 0: short $1 of i, long $|β| of j
    direction = -1  => ê < 0: long $1 of i, short $|β| of j
    """
    p_i_open = prices_i[open_idx]
    p_j_open = prices_j[open_idx]

    total = 0.0
    for t in range(open_idx + 1, close_idx + 1):
        di = (prices_i[t] - prices_i[t - 1]) / p_i_open
        dj = (prices_j[t] - prices_j[t - 1]) / p_j_open
        # direction=+1: we lost on short i, gained on long j
        daily = direction * (-di + abs(beta) * dj)
        total += daily
    return total


def _compute_daily_pnl(
    prices_i: np.ndarray,
    prices_j: np.ndarray,
    open_idx: int,
    close_idx: int,
    direction: int,
    beta: float,
    T_trade: int,
) -> np.ndarray:
    """Daily P&L array for one trade (length T_trade, non-zero only during trade)."""
    pnl = np.zeros(T_trade)
    p_i_open = prices_i[open_idx]
    p_j_open = prices_j[open_idx]
    for t in range(open_idx + 1, close_idx + 1):
        di = (prices_i[t] - prices_i[t - 1]) / p_i_open
        dj = (prices_j[t] - prices_j[t - 1]) / p_j_open
        pnl[t] = direction * (-di + abs(beta) * dj)
    return pnl


def _transaction_cost(
    prices_i: np.ndarray,
    prices_j: np.ndarray,
    open_idx: int,
    close_idx: int,
    beta: float,
    theta: float,
) -> float:
    """
    Transaction cost formula from Appendix A.1 eq. A4.
    theta = cost per $ per share (e.g. 0.001 = 10 bps)
    """
    r_i = (prices_i[close_idx] - prices_i[open_idx]) / prices_i[open_idx]
    r_j = (prices_j[close_idx] - prices_j[open_idx]) / prices_j[open_idx]
    cost = (2 * (1 + abs(beta)) + (r_i + abs(beta) * r_j)) * theta
    return max(cost, 0.0)


def trade_pair(
    stock_i: str,
    stock_j: str,
    trade_prices_i: np.ndarray,
    trade_prices_j: np.ndarray,
    spread_series: np.ndarray,
    params: SpreadParams,
    trade_dates: pd.DatetimeIndex,
    theta: float = 0.001,
) -> PairResult:
    """
    Execute the full trading strategy for one pair over the trading period.

    Parameters
    ----------
    trade_prices_i/j : price arrays over trading period (length T_trade)
    spread_series    : pre-computed spread values (same length)
    params           : SpreadParams with threshold (2σ from training)
    trade_dates      : DatetimeIndex for trading period
    theta            : transaction cost per share (default 10 bps = 0.001)
    """
    T = len(spread_series)
    threshold = params.threshold
    beta = params.beta

    # Track state
    in_trade = False
    direction = 0
    open_idx = 0
    trades = []
    daily_pnl = np.zeros(T)
    n_full = 0
    n_force = 0
    active_trade_opens = 0

    for t in range(T):
        eps = spread_series[t]

        if not in_trade:
            if eps > threshold:
                in_trade = True
                direction = 1    # short i, long j
                open_idx = t
                active_trade_opens += 1
            elif eps < -threshold:
                in_trade = True
                direction = -1   # long i, short j
                open_idx = t
                active_trade_opens += 1

        else:
            # Close conditions: sign change OR end of period
            sign_changed = (np.sign(eps) != np.sign(spread_series[open_idx])) or (eps == 0)
            end_of_period = (t == T - 1)

            if sign_changed or end_of_period:
                close_idx = t
                force_closed = end_of_period and not sign_changed

                gross = _trade_return(
                    trade_prices_i, trade_prices_j,
                    open_idx, close_idx, direction, beta
                )
                tc = _transaction_cost(
                    trade_prices_i, trade_prices_j,
                    open_idx, close_idx, beta, theta
                )
                net = gross - tc

                # Daily pnl contribution
                daily_pnl += _compute_daily_pnl(
                    trade_prices_i, trade_prices_j,
                    open_idx, close_idx, direction, beta, T
                )

                trades.append(TradeRecord(
                    stock_i=stock_i,
                    stock_j=stock_j,
                    open_date=trade_dates[open_idx] if open_idx < len(trade_dates) else None,
                    close_date=trade_dates[close_idx] if close_idx < len(trade_dates) else None,
                    open_idx=open_idx,
                    close_idx=close_idx,
                    direction=direction,
                    beta=beta,
                    gross_return=gross,
                    net_return=net,
                    force_closed=force_closed,
                ))

                if force_closed:
                    n_force += 1
                else:
                    n_full += 1

                in_trade = False
                direction = 0

    n_trades = len(trades)
    total_gross = sum(t.gross_return for t in trades)
    total_net = sum(t.net_return for t in trades)

    # Classify: partial convergent = pair with at least one full-turn + one force-close
    # (paper definition: partially convergent = opened, had some closes, still had a non-convergent part)
    n_partial = max(0, n_force) if n_full > 0 else 0
    if n_full > 0 and n_force > 0:
        n_partial = 1

    return PairResult(
        stock_i=stock_i,
        stock_j=stock_j,
        total_return=total_gross,
        total_net_return=total_net,
        n_trades=n_trades,
        n_full_convergent=n_full,
        n_non_convergent=n_force,
        n_partial_convergent=n_partial,
        daily_pnl=daily_pnl,
        trades=trades,
    )


def run_trading_period(
    pairs_params: dict,
    trade_prices: pd.DataFrame,
    trade_prices_filtered: Optional[pd.DataFrame],
    wavelet: bool = False,
    theta: float = 0.001,
) -> list:
    """
    Run all pairs over a single trading period.

    Parameters
    ----------
    pairs_params         : dict (stock_i, stock_j) -> SpreadParams
    trade_prices         : standard prices over trading period
    trade_prices_filtered: wavelet-filtered prices over trading period
    wavelet              : if True, use filtered prices for spread computation
    theta                : transaction cost

    Returns
    -------
    List of PairResult objects
    """
    spread_prices = trade_prices_filtered if (wavelet and trade_prices_filtered is not None) else trade_prices
    trade_dates = trade_prices.index

    results = []
    for (si, sj), params in pairs_params.items():
        if si not in spread_prices.columns or sj not in spread_prices.columns:
            continue

        # Spread uses (filtered or standard) prices but with fixed training coefficients
        s_i_spread = spread_prices[si].values
        s_j_spread = spread_prices[sj].values
        spread = build_spread(s_i_spread, s_j_spread, params.alpha, params.beta)

        # PnL always uses original (unflitered) prices as per paper
        p_i = trade_prices[si].values
        p_j = trade_prices[sj].values

        result = trade_pair(
            si, sj, p_i, p_j, spread, params,
            trade_dates, theta=theta
        )
        results.append(result)

    return results
