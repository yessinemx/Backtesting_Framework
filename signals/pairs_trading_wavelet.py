"""
Pairs Trading with Wavelet Transform (paper-aligned).

This is the live-backtester wrapper around the exact MODWT level-1 smoothing
used in the paper replication (research/paper_replication/core/wavelet.py):

    V_{1,t} = sum_l (g_l / sqrt(2)) * Z_{t-l mod T}

which is the paper's eq. (1) (Eroglu, Yener, Yigit 2023). The default wavelet
family and boundary handling are read from config.config_paper.PAIRS_CONFIG so
that the live strategy and the paper replication stay in sync.

Methodology:
- Formation window of length ``formation_period`` on the asset's log-prices.
- Apply MODWT level-1 smoothing column-by-column to extract the long-run
  component V_{1,t} (denoised prices).
- Pair selection: top-N pairs with the smallest mean squared distance between
  the normalised denoised series (Gatev et al. 2006 + the paper's wavelet
  preprocessing).
- Trading signal at ``date``: z-score the spread (denoised_i - denoised_j) on
  the formation window; go LONG the underperformer / SHORT the outperformer
  when |z| > entry, flatten when |z| < exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import PairsTradingBase

# Paper-aligned MODWT (single source of truth in research/paper_replication).
try:
    from research.paper_replication.core.wavelet import (
        modwt_smooth as _paper_modwt_smooth,
        DEFAULT_WAVELET as _PAPER_DEFAULT_WAVELET,
    )
    _PAPER_BACKEND_OK = True
except Exception:
    _paper_modwt_smooth = None
    _PAPER_DEFAULT_WAVELET = "sym20"
    _PAPER_BACKEND_OK = False

# Fallback for environments where the research package is unavailable.
try:
    import pywt
except ImportError:
    pywt = None


def _fallback_denoise(series, wavelet, level):
    """Soft-thresholded multi-level DWT (legacy behaviour) for the fallback path."""
    if pywt is None or len(series) < 2 ** level:
        return np.asarray(series, dtype=float)
    coeffs = pywt.wavedec(series, wavelet, level=level, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    thr = sigma * np.sqrt(2.0 * np.log(len(series))) if sigma > 0 else 0.0
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(new_coeffs, wavelet, mode="periodization")
    return rec[: len(series)]


def _wavelet_denoise(series, wavelet="sym20", level=1):
    """Apply the paper's MODWT level-1 smoothing (V_{1,t}) when available.

    The ``level`` argument is kept for API compatibility and is forced to 1
    (the paper only uses level-1 MODWT). Falls back to the legacy multi-level
    DWT soft-thresholding if the research package is not importable.
    """
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return arr
    if _PAPER_BACKEND_OK:
        try:
            return _paper_modwt_smooth(arr, wavelet=wavelet)
        except Exception:
            pass
    return _fallback_denoise(arr, wavelet, max(1, int(level)))


class PairsTradingWavelet(PairsTradingBase):
    """Live-backtester pairs trading driven by the paper's MODWT smoothing."""

    def __init__(self, parameters=None):
        # Wavelet-specific defaults (common params handled by PairsTradingBase).
        extra = {
            "wavelet": _PAPER_DEFAULT_WAVELET,   # sym20 by default (matches paper)
            "wavelet_level": 1,                  # MODWT level-1 (paper setting)
        }
        super().__init__("Pairs Trading (Wavelet)", extra_defaults=extra, parameters=parameters)

    def generate_signals(self, prices, date, members):
        fp = int(self.parameters["formation_period"])
        topn = int(self.parameters["top_n_pairs"])
        entry = float(self.parameters["entry_threshold"])
        exit_thr = float(self.parameters["exit_threshold"])
        wavelet = str(self.parameters["wavelet"])
        level = int(self.parameters["wavelet_level"])
        min_hist = int(self.parameters["min_history"])

        cols = [t for t in members if t in prices.columns]
        if len(cols) < 2:
            return {}

        window = prices[cols].loc[:date].tail(fp).dropna(axis=1, how="any")
        if len(window) < max(min_hist, 2 ** max(1, level)):
            return {}

        # Normalized log-prices over the formation window (paper convention).
        log_px = np.log(window.where(window > 0))
        log_px = log_px.dropna(axis=1, how="any")
        if log_px.shape[1] < 2:
            return {}

        norm = (log_px - log_px.iloc[0]).to_numpy()
        denoised = np.apply_along_axis(
            lambda s: _wavelet_denoise(s, wavelet=wavelet, level=level),
            axis=0, arr=norm,
        )

        n_obs, n_assets = denoised.shape
        # Mean Euclidean distance between each pair: d_{ij} = mean_t (x_it - x_jt)^2.
        sq = denoised ** 2
        gram = denoised.T @ denoised
        sumsq = sq.sum(axis=0)
        dist = (sumsq[:, None] + sumsq[None, :] - 2.0 * gram) / n_obs
        np.fill_diagonal(dist, np.inf)

        iu, ju = np.triu_indices(n_assets, k=1)
        flat = dist[iu, ju]
        if flat.size == 0:
            return {}
        k = min(topn, flat.size)
        idx = np.argpartition(flat, k - 1)[:k]
        tickers = log_px.columns.to_numpy()

        signals = {t: 0.0 for t in cols}
        for p in idx:
            i, j = iu[p], ju[p]
            spread = denoised[:, i] - denoised[:, j]
            mu = spread.mean()
            sigma = spread.std(ddof=0)
            if sigma <= 0 or not np.isfinite(sigma):
                continue
            z = (spread[-1] - mu) / sigma
            ti, tj = tickers[i], tickers[j]
            if z > entry:
                # i outperformed -> short i, long j.
                signals[ti] -= 1.0
                signals[tj] += 1.0
            elif z < -entry:
                signals[ti] += 1.0
                signals[tj] -= 1.0
            elif abs(z) < exit_thr:
                # Flat position (exit) -> no additional signal.
                pass

        # Collapse to {-1, 0, +1}.
        out = {}
        for t, v in signals.items():
            if v > 0:
                out[t] = 1
            elif v < 0:
                out[t] = -1
            else:
                out[t] = 0
        return out

    @staticmethod
    def get_parameters_schema():
        schema = PairsTradingBase._common_schema()
        # Wavelet-specific additions (inserted after top_n_pairs for logical order).
        schema["wavelet"] = {
            "type": "str", "default": _PAPER_DEFAULT_WAVELET,
            "options": ["haar", "db2", "db4", "db6", "sym4", "sym8", "sym20"],
            "label": "Wavelet Family (MODWT)",
        }
        schema["wavelet_level"] = {
            "type": "int", "min": 1, "max": 1,
            "default": 1, "label": "MODWT Level (paper = 1)",
        }
        return schema
