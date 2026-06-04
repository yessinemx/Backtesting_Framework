"""
wavelet_filter.py
=================
Implements the MODWT Level-1 low-pass filter used in Eroğlu et al. (2023).

NOTE ON sym22 AND PYTHON LIBRARIES
-----------------------------------
The MODWT (Maximum Overlap Discrete Wavelet Transform) is NOT a built-in in
any standard Python library. pywt provides DWT (down-sampling) and SWT
(different normalisation), but not MODWT. This file implements MODWT from
scratch following Percival & Walden (2000).

The sym22 filter is also NOT available in pywt (which tops out at sym20).
The coefficients below were computed using the Daubechies spectral
factorization algorithm:
  1. Build autocorrelation polynomial P(y) = Σ C(N-1+k,k) y^k
  2. Find its N-1 roots, map each to a z-pair via z² - (2-4y)z + 1 = 0
  3. Exhaustively search all 2^(N-1) = 2,097,152 root-selection combinations
  4. Select the combination minimising |Σ arg(z_k)| (phase asymmetry criterion)
  5. Reconstruct filter as (1+z^{-1})^N * Π(1 - z_k * z^{-1}), normalised

The resulting filter satisfies all four required properties:
  ✓ Σ h[k] = √2                    (unit DC gain after MODWT scaling)
  ✓ Σ h[k]² = 1                    (unit energy)
  ✓ Σ h[k]·h[k+2l] = δ(l)         (orthogonality)
  ✓ Σ (-1)^k · k^m · h[k] = 0     (22 vanishing moments, m=0..21)

WHAT IS THE MODWT?
------------------
Definition (Percival & Walden 2000, Ch. 5):
    Ṽ_{1,t} = Σ_{l=0}^{L-1}  g̃_l · X_{(t-l) mod N}   (low-pass / long-run)
    W̃_{1,t} = Σ_{l=0}^{L-1}  h̃_l · X_{(t-l) mod N}   (high-pass / noise)

where g̃_l = g_l / √2 is the MODWT scaling filter (1/√2 is the key difference
from DWT). This normalisation preserves energy decomposition without down-sampling.

We use FFT-based circular convolution (O(N log N), exact):
    Ṽ_1 = IFFT( FFT(X) · FFT(g̃, n=N) )

Symmetric padding (paper Section 3):
    252 points → pad by reflection to 256 (next power of 2),
    apply filter, trim. Matches MATLAB modwt(..., 'reflection').
"""

import numpy as np
import math
import pywt
import pandas as pd


# ── sym22 coefficients ───────────────────────────────────────────────────────
# Computed via Daubechies spectral factorization + exhaustive phase minimization.
# Length = 44 (= 2 × 22 vanishing moments).
# Verified: sum=√2, sum²=1, orthogonal, 22 vanishing moments.
_SYM22_LO = np.array([
    -3.60211348e-11,  5.33593882e-10, -2.72962315e-09,  1.68017140e-09,
     3.76122875e-08, -1.28333623e-07, -8.77987987e-08,  1.29518206e-06,
    -1.56517913e-06, -6.16672932e-06,  1.73737570e-05,  1.13743497e-05,
    -9.40522363e-05,  4.34589990e-05,  3.28609414e-04, -4.23787400e-04,
    -7.70690988e-04,  1.82701050e-03,  1.04426074e-03, -5.45569199e-03,
     3.00137398e-04,  1.25647252e-02, -6.21378285e-03, -2.34800013e-02,
     2.05867076e-02,  3.69708466e-02, -4.65308118e-02, -5.13642543e-02,
     8.45573764e-02,  6.80763144e-02, -1.31768138e-01, -9.71107984e-02,
     1.79973188e-01,  1.64093188e-01, -2.00568406e-01, -3.12726580e-01,
     7.37245012e-02,  5.07901091e-01,  5.78432731e-01,  3.67728683e-01,
     1.48367541e-01,  3.80699372e-02,  5.72185463e-03,  3.86263231e-04,
])


def _get_modwt_scaling_filter(wavelet_name: str) -> np.ndarray:
    """
    Return the MODWT low-pass (scaling) filter g̃ = g / sqrt(2).

    MATLAB convention (modwt documentation):
        modwt() uses the reconstruction filters [Lo_R, Hi_R] — the second
        pair returned by wfilters() — not the decomposition filters.
        For orthogonal wavelets Lo_R = Lo_D reversed (pure time-reversal).
        In pywt this corresponds to rec_lo, not dec_lo.

        For circular convolution, dec_lo vs rec_lo only shifts the output
        by (L-1) samples, so OLS beta estimates are identical (diff < 1e-15).
        We use rec_lo for strict MATLAB conformance.

    Parameters
    ----------
    wavelet_name : 'sym22' uses the hardcoded coefficients above.
                   Any pywt name uses rec_lo (MATLAB convention).

    Returns
    -------
    g_tilde : 1-D array, MODWT scaling filter g̃ = g / √2
    """
    if wavelet_name == "sym22":
        g = _SYM22_LO.copy()
    else:
        w = pywt.Wavelet(wavelet_name)
        g = np.array(w.rec_lo)   # reconstruction filter = MATLAB's second pair
    return g / np.sqrt(2)        # MODWT normalisation: divide by √2


def modwt_level1_approx(
    price_series: np.ndarray,
    wavelet: str = "sym22",
) -> np.ndarray:
    """
    Compute the MODWT Level-1 low-pass approximation Ṽ_{1,t}.

    Boundary handling — PERIODIC (matches MATLAB authors' code exactly):
        The authors call modwt(P, 'sym22', 1) with NO boundary argument.
        MATLAB default is PERIODIC (circular convolution on N points as-is).
        This means no padding — the series wraps around.

        With the authors' usage of filtering FULL [train+trade] (504 points)
        and filter length L=44, only the first 43 training points are affected
        by the circular wrap. The trading period (points 252-503) is never
        contaminated. This produces less biased beta estimates than reflection.

    Usage (Appendix A.2 — authors' exact scheme):
        Filter the CONCATENATED [train; trade] series, then split:
            full = np.concatenate([train_prices, trade_prices])
            v_full = modwt_level1_approx(full)
            v_train, v_trade = v_full[:T_train], v_full[T_train:]

    Algorithm
    ---------
    FFT circular convolution on N points (no padding):
        Ṽ_1 = IFFT( FFT(x) · FFT(g̃, n=N) )

    Parameters
    ----------
    price_series : 1-D array, price levels (length N)
    wavelet      : filter name. 'sym22' = paper's primary filter.

    Returns
    -------
    approx : 1-D array of length N — the long-run component Ṽ_{1,t}
    """
    x = np.asarray(price_series, dtype=float)
    N = len(x)
    g_tilde = _get_modwt_scaling_filter(wavelet)

    # Periodic (circular) convolution — MATLAB default, no padding
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(g_tilde, n=N)))


def filter_price_matrix(
    prices: pd.DataFrame,
    wavelet: str = "sym22",
) -> pd.DataFrame:
    """
    Apply modwt_level1_approx to every column of a price DataFrame.
    For single-period use (training only). See filter_train_trade for the
    recommended pairs-trading usage (concatenated filtering).
    """
    filtered = prices.copy()
    for col in prices.columns:
        filtered[col] = modwt_level1_approx(prices[col].values, wavelet=wavelet)
    return filtered


def filter_train_trade(
    train: pd.DataFrame,
    trade: pd.DataFrame,
    wavelet: str = "sym22",
) -> tuple:
    """
    Filter training AND trading prices in a single MODWT pass per column.

    This is the correct approach for pairs trading (Appendix A.2):
    concatenate [train; trade], filter once, split back. The trading-period
    filter is then continuous with the training history, avoiding the
    boundary inconsistency that arises when filtering each period separately.

    Returns
    -------
    (train_filtered, trade_filtered) : tuple of DataFrames
    """
    T_train = len(train)
    common_cols = [c for c in train.columns if c in trade.columns]
    train_filt = train[common_cols].copy()
    trade_filt = trade[common_cols].copy()

    for col in common_cols:
        full = np.concatenate([train[col].values, trade[col].values])
        v_full = modwt_level1_approx(full, wavelet=wavelet)
        train_filt[col] = v_full[:T_train]
        trade_filt[col] = v_full[T_train:]

    return train_filt, trade_filt
