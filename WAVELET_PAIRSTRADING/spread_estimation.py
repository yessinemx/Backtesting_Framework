"""
spread_estimation.py
====================
Estimates spread coefficients (α, β) and constructs spreads.
Implements both standard and wavelet-filtered versions per Section 4.2.

Standard spread  : ê_{t} = S_i,t − α̂ − β̂·S_j,t
Wavelet spread   : ê^w_{t} = Ṽ_i,t − α̂^w − β̂^w·Ṽ_j,t

Two estimation regimes (paper Section 2.1–2.2 + Section 4.2):

- Minimum Distance : β estimated by OLS on price levels.
  The paper acknowledges this produces a spurious regression but accepts it
  because results are evaluated over a large number of pairs (p.1134).

- Cointegration : β comes from the Johansen cointegrating eigenvector
  (already stored in the pairs DataFrame as 'beta_johansen').
  α is then estimated as the OLS intercept of (S_i − β·S_j) on a constant,
  i.e. α̂ = mean(S_i − β·S_j) over the training period.

In both cases:
- σ = std of the training-period spread (with that α, β)
- Threshold = 2σ (paper Section 4.3)
- Coefficients are FIXED for the entire trading period
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpreadParams:
    """Estimated parameters for a single pair."""
    stock_i:   str
    stock_j:   str
    alpha:     float
    beta:      float
    sigma:     float      # std of spread in training period
    threshold: float      # = 2 * sigma


# ── OLS (used for MD and for wavelet re-estimation on filtered prices) ────────

def _ols(s_i: np.ndarray, s_j: np.ndarray) -> tuple:
    """
    OLS: S_i = α + β·S_j + ε
    Returns (alpha, beta, residual_std).
    """
    X = np.column_stack([np.ones(len(s_j)), s_j])
    coeffs, _, _, _ = np.linalg.lstsq(X, s_i, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    sigma = float(np.std(s_i - alpha - beta * s_j, ddof=1))
    return alpha, beta, sigma


# ── Johansen-anchored (used for cointegration method, standard prices) ────────

def _coint_params(s_i: np.ndarray, s_j: np.ndarray, beta_johansen: float) -> tuple:
    """
    Given the Johansen cointegrating coefficient β, estimate α as the mean
    of (S_i − β·S_j) over the training period.
    σ = std of that spread.
    """
    spread = s_i - beta_johansen * s_j
    alpha  = float(np.mean(spread))
    sigma  = float(np.std(spread - alpha, ddof=1))
    return alpha, beta_johansen, sigma


# ── Public API ────────────────────────────────────────────────────────────────

def build_spread(
    s_i: np.ndarray,
    s_j: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Spread: ε_t = S_i,t − α − β·S_j,t"""
    return s_i - alpha - beta * s_j


def estimate_all_pairs(
    pairs: pd.DataFrame,
    train_prices: pd.DataFrame,
    train_prices_filtered: Optional[pd.DataFrame] = None,
    wavelet: bool = False,
) -> dict:
    """
    Estimate (α, β, σ, threshold) for every pair.

    Estimation logic:
    ┌─────────────┬──────────────┬───────────────────────────────────────────┐
    │ wavelet     │ pairs has    │ How β is estimated                        │
    │             │ beta_johansen│                                           │
    ├─────────────┼──────────────┼───────────────────────────────────────────┤
    │ False       │ No           │ OLS on standard prices (MD method)        │
    │ False       │ Yes          │ Johansen β + OLS α on standard prices     │
    │ True        │ No           │ OLS on wavelet-filtered prices (MD+wav)   │
    │ True        │ Yes          │ OLS on wavelet-filtered prices (CI+wav)   │
    │             │              │ (re-estimates both α^w and β^w on Ṽ)      │
    └─────────────┴──────────────┴───────────────────────────────────────────┘

    Note on wavelet+cointegration:
        The paper (Section 4.2) re-estimates the cointegrating relation on the
        filtered prices Ṽ_{i} and Ṽ_{j}, obtaining (α̂^w, β̂^w) by OLS.
        It does NOT reuse the Johansen β on filtered prices.

    Parameters
    ----------
    pairs                 : DataFrame with 'stock_i', 'stock_j', and optionally
                            'beta_johansen' (present for cointegration pairs)
    train_prices          : standard prices over training period
    train_prices_filtered : wavelet-filtered prices (required when wavelet=True)
    wavelet               : True → estimate on filtered prices

    Returns
    -------
    dict: (stock_i, stock_j) -> SpreadParams
    """
    prices = (
        train_prices_filtered
        if (wavelet and train_prices_filtered is not None)
        else train_prices
    )
    has_johansen = "beta_johansen" in pairs.columns

    params = {}
    for _, row in pairs.iterrows():
        si, sj = row["stock_i"], row["stock_j"]
        if si not in prices.columns or sj not in prices.columns:
            continue

        s_i = prices[si].values
        s_j = prices[sj].values

        if wavelet:
            # Always OLS on filtered prices (paper Section 4.2)
            alpha, beta, sigma = _ols(s_i, s_j)

        elif has_johansen and pd.notna(row["beta_johansen"]):
            # Cointegration method, standard prices:
            # use Johansen β, estimate α as mean of spread
            alpha, beta, sigma = _coint_params(s_i, s_j, float(row["beta_johansen"]))

        else:
            # MD method, standard prices: plain OLS
            alpha, beta, sigma = _ols(s_i, s_j)

        params[(si, sj)] = SpreadParams(
            stock_i=si,
            stock_j=sj,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            threshold=2.0 * sigma,
        )
    return params
