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
only provides Symlets up to `sym20`. The paper itself shows in Table 16 that
Sharpe ratios from sym12 to sym24 are very close and stable: sym20 (~2.57) is
close to sym22 (~2.61). This implementation therefore defaults to `sym20` as
the closest available equivalent, while keeping the family configurable.
"""
import numpy as np
import pywt

# Default family: the closest available option to sym22 in PyWavelets.
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


def _filter_level1(x, filt):
    """Apply a level-1 MODWT filter with symmetric extension.

    Output[t] = sum_l filt[l] * x[t-l], with boundaries handled by symmetric reflection.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    L = filt.size
    if n == 0:
        return np.array([])
    # Symmetric padding at the start of the series, using the filter length.
    xpad = np.pad(x, (L, 0), mode="symmetric")
    conv = np.convolve(xpad, filt)
    return conv[L:L + n]


def modwt_smooth(x, wavelet=DEFAULT_WAVELET):
    """Long-term component V_{1,t} (level-1 approximation).

    This is the denoised series used to build the spread.
    """
    g_t, _ = _modwt_filters(wavelet)
    return _filter_level1(x, g_t)


def modwt_detail(x, wavelet=DEFAULT_WAVELET):
    """Short-term component W_{1,t} (level-1 detail), i.e. filtered noise."""
    _, h_t = _modwt_filters(wavelet)
    return _filter_level1(x, h_t)


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
