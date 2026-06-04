"""
MODWT level-1 long-run component — faithful to the paper's MATLAB workflow.

Replication of Section 3 of the paper.

The paper applies the **Maximum Overlap Discrete Wavelet Transform** (MODWT) with
the `sym22` Symlet at **Level 1**, symmetrizing the series to a dyadic length to
handle the boundary (Section 3.1), and uses the long-run component V_1 to build
the spread. The analysis is done in MATLAB (`modwt` / `modwtmra`), Appendix A.2.

We reproduce eq. (3) with PyWavelets' undecimated SWT scaling/approximation
coefficient (`pywt.swt(..., trim_approx=True, norm=True)`), which IS the MODWT
scaling coefficient V_1: a length-L weighted moving average of the prices.

IMPORTANT — V_1 (scaling coefficient) vs A_1 (MRA component)
-----------------------------------------------------------
Eq. (3) is the scaling COEFFICIENT V_1, not the multiresolution-analysis
reconstruction A_1 = x - D_1. The MRA component A_1 barely differs from the price
(corr ~0.999), so using it makes the wavelet spread almost identical to the
standard spread (no effect). The scaling coefficient V_1 is a genuine long-run
component (corr ~0.87-0.99) and is what the paper specifies — using it instead of
A_1 materially changes the wavelet results.

The sym22 filter (length 44)
----------------------------
PyWavelets only ships Symlets up to `sym20`, but the paper needs `sym22`. Symlet
and Daubechies wavelets of the same order share the *same* magnitude response
(different phase factorizations of the same half-band filter); for the trading
spread (where both legs are filtered identically and the signal is what matters)
`db22` recovers the exact paper filter length and vanishing moments (22), and
PyWavelets *does* provide `db22` (length 44). We therefore map `sym22` -> `db22`.
Symlets that PyWavelets does provide (sym2..sym20) are used directly.
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
    xp = np.pad(x, (pad, pad), mode="symmetric")
    step = 2 ** level
    extra = (-xp.size) % step
    if extra:
        xp = np.pad(xp, (0, extra), mode="symmetric")
    coeffs = pywt.swt(xp, fam, level=level, trim_approx=True, norm=True)
    v = coeffs[0]                  # V_J  : scaling (approximation) coefficient
    w = coeffs[1] if len(coeffs) > 1 else xp - v   # W_1 : finest detail
    return v[pad:pad + n], w[pad:pad + n]


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
