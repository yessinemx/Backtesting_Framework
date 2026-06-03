"""
performance.py
==============
Computes performance metrics matching the paper's tables:
- Table 4: basic statistics (mean return, std, skewness, kurtosis)
- Table 5: Sharpe ratio, max drawdown, % positive, VaR 5%, CVaR 5%
- Table 6: trade convergence analysis
- Table 7: proportion × return decomposition
- Table 8: unit root rejection frequency of spreads
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from typing import List
from trading_engine import PairResult


def _annualised_sharpe(daily_returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe from daily P&L series."""
    excess = daily_returns - rf / 252
    if np.std(excess) == 0:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(252)


def _max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative return series."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (1 + peak)
    return float(np.min(drawdown))


def _var_cvar(returns_list: list, level: float = 0.05) -> tuple:
    """VaR and CVaR at given level from list of trade returns."""
    arr = np.array(returns_list)
    if len(arr) == 0:
        return 0.0, 0.0
    var = np.percentile(arr, level * 100)
    cvar = arr[arr <= var].mean() if np.any(arr <= var) else var
    return float(var), float(cvar)


def compute_period_metrics(
    results: List[PairResult],
    n_total_pairs: int,
) -> dict:
    """
    Compute all metrics for one trading period.

    Parameters
    ----------
    results        : list of PairResult from trading_engine
    n_total_pairs  : total number of pairs considered (including inactive)

    Returns
    -------
    dict of scalar metrics
    """
    N = n_total_pairs

    # Trade-level returns
    all_trade_returns = []
    for r in results:
        for t in r.trades:
            all_trade_returns.append(t.gross_return)

    # Pair-level total returns (including pairs with 0 trades = 0)
    pair_returns = [r.total_return for r in results]

    # Daily P&L — average across all pairs (eq. A2)
    if results:
        T_trade = len(results[0].daily_pnl)
        daily_matrix = np.vstack([r.daily_pnl for r in results])
        # Average return per day across ALL pairs (committed capital, incl. inactive)
        daily_avg = daily_matrix.sum(axis=0) / N
    else:
        daily_avg = np.array([0.0])
        T_trade = 0

    cum_returns = np.cumsum(daily_avg)

    # Active pairs
    active = [r for r in results if r.n_trades > 0]
    n_active = len(active)

    # Convergence categories
    n_full = sum(r.n_full_convergent > 0 and r.n_non_convergent == 0 for r in active)
    n_non_conv = sum(r.n_full_convergent == 0 and r.n_non_convergent > 0 for r in active)
    n_partial = sum(r.n_partial_convergent > 0 for r in active)

    # Returns by category
    def cat_returns(cat_filter):
        subset = [r for r in active if cat_filter(r)]
        return [t.gross_return for r in subset for t in r.trades]

    full_ret = cat_returns(lambda r: r.n_full_convergent > 0 and r.n_non_convergent == 0)
    nonconv_ret = cat_returns(lambda r: r.n_full_convergent == 0 and r.n_non_convergent > 0)

    var5, cvar5 = _var_cvar(pair_returns, 0.05)

    mean_return = np.mean(pair_returns) if pair_returns else 0.0
    std_return = np.std(pair_returns, ddof=1) if len(pair_returns) > 1 else 0.0
    skew = float(stats.skew(pair_returns)) if len(pair_returns) > 3 else 0.0
    kurt = float(stats.kurtosis(pair_returns, fisher=False)) if len(pair_returns) > 3 else 0.0

    sharpe = _annualised_sharpe(daily_avg)
    mdd = abs(_max_drawdown(cum_returns))
    pct_positive = np.mean(np.array(all_trade_returns) > 0) * 100 if all_trade_returns else 0.0

    return {
        # Table 4
        "mean_return": mean_return,
        "std_return": std_return,
        "skewness": skew,
        "kurtosis": kurt,
        # Table 5
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "pct_positive": pct_positive,
        "var_5pct": var5,
        "cvar_5pct": cvar5,
        # Table 6
        "n_total_pairs": N,
        "n_active": n_active,
        "pct_active": n_active / N * 100,
        "n_full_convergent": n_full,
        "n_partial_convergent": n_partial,
        "n_non_convergent": n_non_conv,
        "pct_full": n_full / N * 100,
        "pct_partial": n_partial / N * 100,
        "pct_non_conv": n_non_conv / N * 100,
        "mean_ret_full": np.mean(full_ret) * 100 if full_ret else 0.0,
        "mean_ret_nonconv": np.mean(nonconv_ret) * 100 if nonconv_ret else 0.0,
        # Series
        "daily_pnl": daily_avg,
        "cum_returns": cum_returns,
        "n_trades": sum(r.n_trades for r in active),
    }


def spread_stationarity_rate(
    pairs_params: dict,
    trade_prices: pd.DataFrame,
    trade_prices_filtered: pd.DataFrame = None,
    wavelet: bool = False,
    adf_sig: float = 0.05,
) -> float:
    """
    Compute fraction of trading-period spreads that are stationary (ADF test).
    Reproduces Table 8.
    """
    from spread_estimation import build_spread

    spread_prices = trade_prices_filtered if (wavelet and trade_prices_filtered is not None) else trade_prices
    rejections = 0
    total = 0

    for (si, sj), params in pairs_params.items():
        if si not in spread_prices.columns or sj not in spread_prices.columns:
            continue
        s_i = spread_prices[si].values
        s_j = spread_prices[sj].values
        spread = build_spread(s_i, s_j, params.alpha, params.beta)
        try:
            pval = adfuller(spread, regression="c", autolag="BIC")[1]
            if pval < adf_sig:
                rejections += 1
        except Exception:
            pass
        total += 1

    return rejections / total if total > 0 else 0.0


def summarise_results(
    period_metrics: list,
    label: str = "Method",
) -> pd.DataFrame:
    """
    Aggregate metrics across periods — matches paper's Min/Max/Mean columns.

    Parameters
    ----------
    period_metrics : list of dicts from compute_period_metrics
    label          : row label
    """
    keys = [
        "mean_return", "std_return", "skewness", "kurtosis",
        "sharpe_ratio", "max_drawdown", "pct_positive", "var_5pct", "cvar_5pct",
    ]
    rows = []
    for k in keys:
        vals = [m[k] for m in period_metrics]
        rows.append({
            "metric": k,
            "label": label,
            "min": np.min(vals),
            "max": np.max(vals),
            "mean": np.mean(vals),
        })
    return pd.DataFrame(rows)


def print_table4(coint_metrics: list, md_metrics: list):
    """Print Table 4 style summary."""
    print("\n=== TABLE 4: Basic Statistics ===")
    print(f"{'':35s} {'Cointegration':>35s} {'Min.Distance':>35s}")
    print(f"{'':35s} {'Min':>10s} {'Max':>10s} {'Mean':>10s}  {'Min':>10s} {'Max':>10s} {'Mean':>10s}")

    for key, fmt, label in [
        ("mean_return", "{:+.2%}", "Return"),
        ("std_return", "{:.4f}", "Std. Dev."),
        ("skewness", "{:+.4f}", "Skewness"),
        ("kurtosis", "{:.4f}", "Kurtosis"),
        ("sharpe_ratio", "{:+.4f}", "Sharpe Ratio"),
        ("max_drawdown", "{:.2%}", "Max. Drawdown"),
        ("pct_positive", "{:.2f}%", "% Positive"),
        ("var_5pct", "{:.2%}", "VaR (5%)"),
        ("cvar_5pct", "{:.2%}", "CVaR (5%)"),
    ]:
        c_vals = [m[key] for m in coint_metrics]
        m_vals = [m[key] for m in md_metrics]
        print(
            f"{label:35s} "
            f"{fmt.format(min(c_vals)):>10s} {fmt.format(max(c_vals)):>10s} {fmt.format(np.mean(c_vals)):>10s}  "
            f"{fmt.format(min(m_vals)):>10s} {fmt.format(max(m_vals)):>10s} {fmt.format(np.mean(m_vals)):>10s}"
        )
