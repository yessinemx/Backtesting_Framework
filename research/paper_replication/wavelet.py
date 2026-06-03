"""
Maximum Overlap Discrete Wavelet Transform (MODWT).

Replication of Section 3 of the paper.

This implementation uses MODWT with no down-sampling, so the filtered series
keeps its original length, which is required for real-time trading signals.
It uses:
    - Symlet filters (symN)
    - Level-1 approximation (long-term component)
    - symmetric extension to handle edge effects

Note on sym22
-------------
The paper uses the `sym22` filter (22 vanishing moments, length 44). PyWavelets
only provides Symlets up to `sym20`. We default to `sym20`, the closest
AVAILABLE member of the SAME family (Symlets are "least-asymmetric", i.e. very
close to linear phase). The paper's Table 16 shows Sharpe ratios are stable
across high-order Symlets: sym20 (~2.57) is close to sym22 (~2.61).

Do NOT substitute db22 here. Although db22 shares the same length (44) and
number of vanishing moments (22), Daubechies wavelets are "extremal phase" and
highly asymmetric: their scaling-filter energy is centred near index 37 of 44
(vs the symmetric midpoint ~21.5), introducing a large frequency-dependent
phase lag that destroys the price-trend extraction the strategy relies on. The
least-asymmetric Symlet phase is exactly what produces the paper's positive
wavelet returns.
"""
import numpy as np
import pywt

# Default family: nearest available Symlet to the paper's sym22 (same family,
# least-asymmetric / near-linear phase). sym20 is the highest Symlet provided
# by PyWavelets.
DEFAULT_WAVELET = "sym20"


def _modwt_filters(wavelet):
    """Return the MODWT filters (low-pass g~, high-pass h~).

    Orthonormal DWT filters are rescaled by 1/sqrt(2) to obtain the MODWT
    filters (Percival & Walden, 2000).
    """
    wavelet_type = getattr(pywt, "Wavelet")
    w = wavelet_type(wavelet)
    g = np.asarray(w.dec_lo, dtype=float)   # low-pass scaling filter
    h = np.asarray(w.dec_hi, dtype=float)   # high-pass wavelet filter
    g_t = g / np.sqrt(2.0)
    h_t = h / np.sqrt(2.0)
    return g_t, h_t


def _mra_level1(x, wavelet):
    """Level-1 MODWT multiresolution analysis (smooth, detail).

    Returns (V1, W1) where V1 is the Level-1 long-run (smooth) component and
    W1 is the Level-1 short-run (detail) component, with V1 + W1 == x.

    Uses the undecimated SWT and its inverse so the reconstruction is
    ZERO-PHASE: the smooth tracks the price trend without the ~L/2 time lag
    introduced by a single causal convolution. The boundary is handled by
    symmetric reflection (the paper's "symmetrization", Section 3.1).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.array([]), np.array([])
    # Symmetric reflection padding (paper Section 3.1 "symmetrization") to tame
    # boundary effects. pywt's SWT-MRA only supports periodization internally,
    # so we reflect-pad ourselves, then crop back to the original support.
    L = pywt.Wavelet(wavelet).dec_len
    pad = L
    # SWT requires the (padded) length to be a multiple of 2 for level 1.
    total = n + 2 * pad
    extra = (-total) % 2
    xp = np.pad(x, (pad, pad + extra), mode="symmetric")
    comps = pywt.mra(xp, wavelet, level=1, transform="swt")
    smooth = np.asarray(comps[0], dtype=float)[pad:pad + n]
    detail = np.asarray(comps[1], dtype=float)[pad:pad + n]
    return smooth, detail


def modwt_smooth(x, wavelet=DEFAULT_WAVELET):
    """Long-term component V_{1,t} (level-1 approximation).

    This is the denoised series used to build the spread. Zero-phase, so the
    smoothed series is time-aligned with the original prices.
    """
    smooth, _ = _mra_level1(x, wavelet)
    return smooth


def modwt_detail(x, wavelet=DEFAULT_WAVELET):
    """Short-term component W_{1,t} (level-1 detail), i.e. filtered noise."""
    _, detail = _mra_level1(x, wavelet)
    return detail


def filter_prices(prices, wavelet=DEFAULT_WAVELET):
    """Filter and denoise a price block column by column.

    Parameters
    ----------
    prices : pl.DataFrame | pl.Series | np.ndarray
        Price data. For a wide DataFrame, the "date" column is preserved.

    Returns
    -------
    Same type as the input, containing the long-term component V_{1,t}.
    """
    import polars as pl

    if isinstance(prices, pl.Series):
        return pl.Series(prices.name, modwt_smooth(prices.to_numpy(), wavelet))

    if isinstance(prices, pl.DataFrame):
        out = {}
        for col in prices.columns:
            if col == "date":
                out[col] = prices.get_column(col)
            else:
                out[col] = modwt_smooth(
                    prices.get_column(col).to_numpy().astype(float), wavelet
                )
        return pl.DataFrame(out)

    return modwt_smooth(np.asarray(prices, dtype=float), wavelet)
