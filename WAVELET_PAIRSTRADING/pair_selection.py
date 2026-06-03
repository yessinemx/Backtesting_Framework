"""
pair_selection.py
=================
Implements the two pair-selection methods from Eroğlu et al. (2023):

1. Minimum Distance Method (Gatev et al. 2006) — Section 2.1
   - Normalized prices: S̃_{i,t} = S_{i,t} / S_{i,t0}
   - Sort pairs by mean squared distance D_{i,j} = (1/T) Σ(S̃_i − S̃_j)²
   - Keep top `n_pairs` (default 1000 as in paper)

2. Cointegration Method (Vidyamurthy 2004) — Section 2.2
   - Johansen trace test (5% level, intercept in cointegrating relation,
     no deterministic trend in levels, lag selected by SIC per paper Section 2.2)
   - Keep all cointegrated pairs
   - Also returns the Johansen cointegrating vector (β) used in spread_estimation
"""

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings


# ── 1. Minimum Distance ──────────────────────────────────────────────────────

def minimum_distance_pairs(
    prices: pd.DataFrame,
    n_pairs: int = 1000,
) -> pd.DataFrame:
    """
    Select pairs by minimum sum-of-squared-distances on normalised prices.

    Parameters
    ----------
    prices  : DataFrame (T × N) of price levels (training period)
    n_pairs : number of top pairs to return (paper uses 1000)

    Returns
    -------
    DataFrame with columns ['stock_i', 'stock_j', 'distance']
    sorted ascending by distance (best pairs first).
    """
    # Normalise: divide by initial price (eq. 1 in paper)
    p0 = prices.iloc[0]
    norm = prices / p0  # S̃_{i,t} = S_{i,t} / S_{i,t0}

    tickers = list(prices.columns)
    norm_arr = norm.values  # T × N

    records = []
    for i, j in combinations(range(len(tickers)), 2):
        diff = norm_arr[:, i] - norm_arr[:, j]
        d = np.mean(diff ** 2)
        records.append((tickers[i], tickers[j], d))

    pairs_df = pd.DataFrame(records, columns=["stock_i", "stock_j", "distance"])
    pairs_df = pairs_df.sort_values("distance").head(n_pairs).reset_index(drop=True)
    return pairs_df


# ── 2. Cointegration (Johansen) ──────────────────────────────────────────────

def _optimal_lag_sic(s1: np.ndarray, s2: np.ndarray, max_lag: int = 10) -> int:
    """
    Select VAR lag order by Schwarz Information Criterion (SIC/BIC).
    Paper Section 2.2: 'we choose the optimal lag length using Schwarz
    Information Criterion'.
    """
    data = np.column_stack([s1, s2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(data)
        # select_order returns an object; .bic gives the lag minimising BIC
        sel = model.select_order(maxlags=min(max_lag, len(data) // 10))
    lag = sel.bic
    return max(int(lag), 1)  # at least 1


def _johansen_test(
    s1: np.ndarray,
    s2: np.ndarray,
    det_order: int = 0,
    sig_level: int = 1,
) -> tuple:
    """
    Run Johansen trace test on (s1, s2).

    Paper spec (Section 2.2):
    - det_order = 0 : intercept in cointegrating relation, no deterministic
                      trend in the levels of the data
    - sig_level = 1 : 5% significance level
    - Lag selected by SIC

    Returns
    -------
    (is_cointegrated: bool, beta: float | None)
        beta is the normalised cointegrating coefficient such that
        S_i − beta * S_j is stationary (β from Johansen eigenvector,
        normalised on the first element).
        Returns None if not cointegrated.
    """
    try:
        lag = _optimal_lag_sic(s1, s2)
        data = np.column_stack([s1, s2])
        result = coint_johansen(data, det_order=det_order, k_ar_diff=lag)

        # Trace statistic for H0: r=0
        trace_stat  = result.lr1[0]
        critical_val = result.cvt[0, sig_level]

        if trace_stat <= critical_val:
            return False, None

        # Extract cointegrating vector, normalise on first element (s1)
        # evec[:,0] is the eigenvector associated with the largest eigenvalue
        evec = result.evec[:, 0]
        beta = -evec[1] / evec[0]   # so that evec[0]*s1 + evec[1]*s2 ~ I(0)
                                     # => s1 - beta*s2 ~ I(0)
        return True, float(beta)

    except Exception:
        return False, None


def cointegration_pairs(
    prices: pd.DataFrame,
    det_order: int = 0,
    sig_level: int = 1,
    max_pairs: int = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Select all cointegrated pairs via Johansen trace test with SIC lag selection.

    Parameters
    ----------
    prices    : DataFrame (T × N) of price levels (training period)
    det_order : 0 = intercept in coint. relation, no trend in levels (paper)
    sig_level : 1 = 5% significance (paper)
    max_pairs : optional cap (None = keep all)
    verbose   : print progress

    Returns
    -------
    DataFrame with columns ['stock_i', 'stock_j', 'beta_johansen']
        beta_johansen : cointegrating coefficient from Johansen eigenvector,
                        passed to spread_estimation to use instead of OLS beta.
    """
    tickers = list(prices.columns)
    N = len(tickers)
    total_pairs = N * (N - 1) // 2

    records = []
    checked = 0
    for i, j in combinations(range(N), 2):
        s1 = prices.iloc[:, i].values
        s2 = prices.iloc[:, j].values
        is_coint, beta = _johansen_test(s1, s2, det_order=det_order, sig_level=sig_level)
        if is_coint:
            records.append((tickers[i], tickers[j], beta))
        checked += 1
        if verbose and checked % 1000 == 0:
            print(f"  Johansen: {checked}/{total_pairs} checked, "
                  f"{len(records)} cointegrated")

    pairs_df = pd.DataFrame(records, columns=["stock_i", "stock_j", "beta_johansen"])
    if max_pairs is not None:
        pairs_df = pairs_df.head(max_pairs)
    return pairs_df.reset_index(drop=True)
