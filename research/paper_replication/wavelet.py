"""
Maximum Overlap Discrete Wavelet Transform (MODWT) — level-1 long-run component.

Replication of Section 3 of the paper.

The level-1 MODWT scaling coefficient is

    V_{1,t} = sum_{l=0}^{L-1} g~_l * Z_{t-l}      (paper eq. 3)

i.e. a length-L weighted moving average of the series, where g~ = g / sqrt(2)
is the rescaled orthonormal scaling (low-pass) filter and L the filter length.
MODWT keeps the original sample length (no down-sampling), which is required to
build a real-time trading signal.

Phase alignment (important)
---------------------------
Applied as a plain causal convolution, a length-L symlet delays its output by
the filter group delay (~L/2 samples). For sym20 (L = 40) that is a ~20-day lag,
so a "denoised" price would actually trail the real price by ~20 trading days and
the trading signal would act on stale data. We therefore advance the output by
the group delay so the smooth is **zero-phase** (centred): V_{1,t} uses prices on
both sides of t. This matches the two-sided / "forward-looking" behaviour the
paper itself discusses for sym22 in Section 5.5.4. Edges use symmetric padding
(the "symmetrization" the paper mentions in Section 3.1).

Note on sym22
-------------
The paper uses `sym22` (22 vanishing moments, length 44). PyWavelets only ships
Symlets up to `sym20`. Per the paper's own Table 16 the risk-adjusted results are
essentially flat across sym18–sym24 (sym20 ~ sym22), so `sym20` is used as the
closest available proxy, while the family stays configurable.
"""
import numpy as np
import pywt

# Default family: the closest available option to sym22 in PyWavelets.
DEFAULT_WAVELET = "sym20"


def _modwt_filters(wavelet):
    """Return the MODWT filters (low-pass g~, high-pass h~).

    Orthonormal DWT filters are rescaled by 1/sqrt(2) to obtain the MODWT
    filters (Percival & Walden, 2000). The low-pass filter then has unit DC
    gain (sum g~ = 1), so it preserves the level/scale of the price series.
    """
    w = pywt.Wavelet(wavelet)
    g = np.asarray(w.dec_lo, dtype=float) / np.sqrt(2.0)   # low-pass (scaling)
    h = np.asarray(w.dec_hi, dtype=float) / np.sqrt(2.0)   # high-pass (wavelet)
    return g, h


def _filter_level1(x, filt):
    """Apply a level-1 MODWT filter, zero-phase (centred), symmetric boundaries.

    Output[t] ~= sum_l filt[l] * x[t-l], but advanced by the filter group delay
    (L-1)//2 so the result is aligned with (not lagged behind) the input.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    L = filt.size
    if n == 0:
        return np.array([])
    group_delay = (L - 1) // 2
    xpad = np.pad(x, (L, L), mode="symmetric")
    conv = np.convolve(xpad, filt)
    start = L + group_delay
    return conv[start:start + n]


def modwt_smooth(x, wavelet=DEFAULT_WAVELET):
    """Long-run component V_{1,t} (level-1 approximation), zero-phase.

    This is the denoised series used to build the spread.
    """
    g, _ = _modwt_filters(wavelet)
    return _filter_level1(x, g)


def modwt_detail(x, wavelet=DEFAULT_WAVELET):
    """Short-run component W_{1,t} (level-1 detail), i.e. the filtered-out noise."""
    _, h = _modwt_filters(wavelet)
    return _filter_level1(x, h)


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
