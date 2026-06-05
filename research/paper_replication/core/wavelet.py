"""Level-1 MODWT (Maximum Overlap Discrete Wavelet Transform), no down-sampling."""

from __future__ import annotations

import numpy as np
import pywt

from config import config_paper as research_config

DEFAULT_LEVEL = 1
DEFAULT_WAVELET = research_config.DEFAULT_WAVELET

_LOW_PASS_OVERRIDES: dict[str, tuple[float, ...]] = {}


def _qmf_from_low_pass(low_pass: np.ndarray) -> np.ndarray:
    """Quadrature mirror high-pass filter h_l = (-1)^l g_{L-1-l}."""
    signs = np.where(np.arange(low_pass.size) % 2 == 0, 1.0, -1.0)
    return signs * low_pass[::-1]


def resolve_wavelet(name: str) -> str:
    """Return a usable wavelet name; raise if sym22 is requested but not bundled."""
    family = str(name).lower()
    if family in _LOW_PASS_OVERRIDES:
        return family
    try:
        pywt.Wavelet(family)
        return family
    except Exception as exc:  # noqa: BLE001
        if family == "sym22":
            raise ValueError(
                "Exact sym22 coefficients are not bundled with PyWavelets in this "
                "environment. Add the 44 MATLAB/Wavelet-Toolbox low-pass "
                "coefficients to _LOW_PASS_OVERRIDES['sym22'], or set "
                "PAIRS_CONFIG['wavelet'] to an installed wavelet such as 'sym20'."
            ) from exc
        raise


def _orthonormal_filters(wavelet: str) -> tuple[np.ndarray, np.ndarray]:
    family = resolve_wavelet(wavelet)
    if family in _LOW_PASS_OVERRIDES:
        low_pass = np.asarray(_LOW_PASS_OVERRIDES[family], dtype=float)
        high_pass = _qmf_from_low_pass(low_pass)
        return low_pass, high_pass
    w = pywt.Wavelet(family)
    return np.asarray(w.dec_lo, dtype=float), np.asarray(w.dec_hi, dtype=float)


def _modwt_filters(wavelet: str) -> tuple[np.ndarray, np.ndarray]:
    low_pass, high_pass = _orthonormal_filters(wavelet)
    scale = np.sqrt(2.0)
    return low_pass / scale, high_pass / scale


def _reflect_indices(indices: np.ndarray, n: int) -> np.ndarray:
    if n == 1:
        return np.zeros_like(indices)
    period = 2 * n - 2
    wrapped = np.mod(indices, period)
    return np.where(wrapped < n, wrapped, period - wrapped)


def _modwt_convolve_level1(x: np.ndarray, filt: np.ndarray, boundary: str) -> np.ndarray:
    n = x.size
    if n == 0:
        return x.copy()
    taps = np.arange(filt.size)
    sample_index = np.arange(n)[:, None] - taps[None, :]
    if boundary == "periodic":
        sample_index = np.mod(sample_index, n)
    elif boundary == "symmetric":
        sample_index = _reflect_indices(sample_index, n)
    else:
        raise ValueError("boundary must be 'symmetric' or 'periodic'")
    return (x[sample_index] * filt[None, :]).sum(axis=1)


def _modwt_coeffs(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL,
                  boundary="symmetric") -> tuple[np.ndarray, np.ndarray]:
    if level != 1:
        raise ValueError("Only level-1 MODWT is implemented for this replication.")
    values = np.asarray(x, dtype=float)
    low_pass, high_pass = _modwt_filters(wavelet)
    smooth = _modwt_convolve_level1(values, low_pass, boundary)
    detail = _modwt_convolve_level1(values, high_pass, boundary)
    return smooth, detail


def modwt_smooth(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL,
                 boundary="symmetric"):
    """Level-1 long-run MODWT coefficient V_1,t."""
    return _modwt_coeffs(x, wavelet, level, boundary)[0]


def modwt_detail(x, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL,
                 boundary="symmetric"):
    """Level-1 short-run MODWT coefficient W_1,t."""
    return _modwt_coeffs(x, wavelet, level, boundary)[1]


def filter_prices(prices, wavelet=DEFAULT_WAVELET):
    """Filter and denoise a price block column by column.

    Parameters
    ----------
    prices : pl.DataFrame | pl.Series | np.ndarray
        Price data. For a wide DataFrame, the "date" column is preserved.

    Returns
    -------
    Same type as the input, containing the long-term component V_{1,t}
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
