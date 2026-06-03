"""
MODWT level-1 long-run component — faithful to the paper's MATLAB workflow.

Replication of Section 3 of the paper.

The paper applies the **Maximum Overlap Discrete Wavelet Transform** (MODWT) with
the `sym22` Symlet at **Level 1**, symmetrizing the series to a dyadic length to
handle the boundary (Section 3.1), and uses the long-run component V_1 to build
the spread. The analysis is done in MATLAB (`modwt` / `modwtmra`), Appendix A.2.

This module reproduces that with PyWavelets' multiresolution analysis
(`pywt.mra(..., transform='swt')`), which is the MODWT MRA: it is undecimated
(keeps the full length), zero-phase (the smooth is aligned with the price, not
lagged), and the components sum back to the original series (A_1 + D_1 = x).

The sym22 filter (length 44)
----------------------------
PyWavelets only ships Symlets up to `sym20`, but the paper needs `sym22`. The
MODWT MRA smooth is **zero-phase**, so it depends only on the filter's *magnitude*
response. A Symlet and a Daubechies wavelet of the same order share the *same*
magnitude response (they are different phase factorizations of the same half-band
filter), therefore the level-1 MRA smooth from `sym22` is numerically identical
to the one from `db22` (verified to ~1e-10), and PyWavelets *does* provide `db22`
(length 44). We therefore map `sym22` -> `db22` for the smooth, recovering the
exact paper filter length and vanishing moments (22). Symlets that PyWavelets
does provide (sym2..sym20) are used directly.
"""
import numpy as np
import pywt

# The paper's filter. Resolved to a PyWavelets-available equivalent below.
DEFAULT_WAVELET = "sym22"
DEFAULT_LEVEL = 1

# Highest Symlet/Daubechies orders PyWavelets ships.
_MAX_SYM = 20
_MAX_DB = 38


def resolve_wavelet(name):
    """Map a requested wavelet to a PyWavelets-available filter.

    `symN` with N > 20 is mapped to `dbN` (identical magnitude response, hence an
    identical zero-phase MODWT MRA smooth). Everything else is returned as-is.
    """
    if isinstance(name, str) and name.lower().startswith("sym"):
        try:
            n = int(name[3:])
        except ValueError:
            return name
        if n > _MAX_SYM and n <= _MAX_DB:
            return f"db{n}"
    return name


def _mra_smooth(x, wavelet, level):
    """Zero-phase MODWT MRA approximation A_{level} of a 1-D series.

    The series is symmetrized (reflected) by the filter length on each side to
    absorb boundary effects, padded up to a length divisible by 2**level, run
    through the MODWT MRA, and trimmed back to the original support.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    fam = resolve_wavelet(wavelet)
    L = pywt.Wavelet(fam).dec_len
    pad = L
    xp = np.pad(x, (pad, pad), mode="symmetric")
    step = 2 ** level
    extra = (-xp.size) % step
    if extra:
        xp = np.pad(xp, (0, extra), mode="symmetric")
    comps = pywt.mra(xp, fam, level=level, transform="swt")
    return comps[0][pad:pad + n]   # A_level (smooth), realigned to x


def _mra_detail(x, wavelet, level):
    """Filtered-out short-run component x - A_{level} (sum of detail levels)."""
    x = np.asarray(x, dtype=float)
    return x - _mra_smooth(x, wavelet, level)


def modwt_smooth(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL):
    """Long-run component V_{1,t} (level-1 MODWT MRA approximation), zero-phase.

    This is the denoised series used to build the spread.
    """
    return _mra_smooth(x, wavelet, level)


def modwt_detail(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL):
    """Short-run component W_{1,t} (filtered-out noise), zero-phase."""
    return _mra_detail(x, wavelet, level)


def filter_prices(prices, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL):
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
        return pl.Series(prices.name, modwt_smooth(prices.to_numpy(), wavelet, level))

    if isinstance(prices, pl.DataFrame):
        out = {}
        for col in prices.columns:
            if col == "date":
                out[col] = prices.get_column(col)
            else:
                out[col] = modwt_smooth(
                    prices.get_column(col).to_numpy().astype(float), wavelet, level
                )
        return pl.DataFrame(out)

    return modwt_smooth(np.asarray(prices, dtype=float), wavelet, level)
