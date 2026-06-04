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

from config import config_paper as research_config

# Default family: nearest available Symlet to the paper's sym22 (same family,
# least-asymmetric / near-linear phase). sym20 is the highest Symlet provided
# by PyWavelets.
DEFAULT_WAVELET = research_config.DEFAULT_WAVELET


def _modwt_filters(wavelet):
    """Return the MODWT filters (low-pass g~, high-pass h~).

    `symN` with N > 20 is mapped to `sym20` — the closest *actual* Symlet
    PyWavelets ships. For the raw scaling coefficient (used as the trading
    signal) the filter PHASE matters, and Symlets are near-linear-phase like the
    paper's sym22, whereas `dbN` is minimum-phase (lagged) — so `sym20` is the
    right proxy here. Symlets PyWavelets already provides (sym2..sym20) pass
    through unchanged.
    """
    if isinstance(name, str) and name.lower().startswith("sym"):
        try:
            n = int(name[3:])
        except ValueError:
            return name
        if n > _MAX_SYM:
            return f"sym{_MAX_SYM}"
    return name


def _modwt_v1_periodic(x, fam, level=1):
    """Level-1 MODWT scaling V_1 and detail W_1 with CIRCULAR boundary — exactly
    MATLAB's ``modwt`` default: V_1[t] = sum_l g~_l x[(t-l) mod N].

    WARNING — boundary look-ahead. On a concatenated [training; trading] series
    the circular wrap makes the early (training) coefficients depend on the END
    of the trading window, leaking future information into the in-sample
    coefficient/threshold estimation. This is what the authors' MATLAB code does
    (`mainfile_basic_simulations.m`); it reproduces the paper's headline but is
    NOT tradeable. Only level 1 is supported (the paper's setting).
    """
    g = np.asarray(pywt.Wavelet(fam).dec_lo, dtype=float) / np.sqrt(2.0)
    h = np.asarray(pywt.Wavelet(fam).dec_hi, dtype=float) / np.sqrt(2.0)
    n = x.size
    L = g.size
    idx = (np.arange(n)[:, None] - np.arange(L)[None, :]) % n
    xi = x[idx]
    return (xi * g[None, :]).sum(axis=1), (xi * h[None, :]).sum(axis=1)


def _modwt_coeffs(x, wavelet, level, boundary="symmetric"):
    """MODWT level-J scaling V_J and detail W_J COEFFICIENTS (paper eq. 3).

    Eq. (3) defines V_{1,t} = sum_l g~_l Z_{t-l} — the MODWT *scaling
    coefficient*, a length-L weighted average of the prices (long-run component),
    NOT the MRA reconstruction A_1 = x - D_1 (which barely differs from x).

    boundary : "symmetric" | "periodic"
        "symmetric" (default, honest): reflect at the edges; no information
        crosses the train/trade boundary.
        "periodic" (paper-faithful): MATLAB's circular boundary — reproduces the
        paper but injects boundary look-ahead (see ``_modwt_v1_periodic``).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x, x
    fam = resolve_wavelet(wavelet)
    if boundary == "periodic":
        return _modwt_v1_periodic(x, fam, level)
    L = pywt.Wavelet(fam).dec_len
    pad = L
    # SWT requires the (padded) length to be a multiple of 2 for level 1.
    total = n + 2 * pad
    extra = (-total) % 2
    xp = np.pad(x, (pad, pad + extra), mode="symmetric")
    comps = pywt.mra(xp, wavelet, level=1, transform="swt")
    smooth = np.asarray(comps[0], dtype=float)[pad:pad + n]
    detail = np.asarray(comps[1], dtype=float)[pad:pad + n]
    return smooth, detail


def modwt_smooth(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL,
                 boundary="symmetric"):
    """Long-run component V_{1,t} (level-1 MODWT scaling coefficient, eq. 3).

    This is the denoised series used to build the spread. See ``_modwt_coeffs``
    for the ``boundary`` option ("symmetric" honest vs "periodic" paper-faithful).
    """
    return _modwt_coeffs(x, wavelet, level, boundary)[0]


def modwt_detail(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL,
                 boundary="symmetric"):
    """Short-run component W_{1,t} (filtered-out noise, level-1 detail coeff)."""
    return _modwt_coeffs(x, wavelet, level, boundary)[1]


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
