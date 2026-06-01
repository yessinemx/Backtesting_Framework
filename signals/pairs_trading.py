"""
Pairs Trading with Wavelet Transform.

Based on Eroglu, Yener, Yigit (2023), "Pairs trading with wavelet transform",
Quantitative Finance. doi:10.1080/14697688.2023.2230249

Methodology:
- Formation period : normalize log-prices, denoise via discrete wavelet
  transform (soft-thresholded detail coefficients), then select the top-N
  pairs with the minimum sum of squared distances (Gatev et al. 2006).
- Trading period : compute the spread on the denoised normalized series,
  z-score it on the formation window, go LONG the underperformer and SHORT
  the outperformer when |z| > entry, flatten the pair when |z| < exit.
"""
import numpy as np
import pandas as pd

from signals import BaseStrategy

try:
    import pywt
except ImportError:
    pywt = None


def _wavelet_denoise(series, wavelet="db4", level=2):
    if pywt is None or len(series) < 2 ** level:
        return series
    coeffs = pywt.wavedec(series, wavelet, level=level, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    thr = sigma * np.sqrt(2.0 * np.log(len(series))) if sigma > 0 else 0.0
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(new_coeffs, wavelet, mode="periodization")
    return rec[: len(series)]


class PairsTradingWavelet(BaseStrategy):

    def __init__(self, parameters=None):
        default = {
            "formation_period": 252,
            "top_n_pairs": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "wavelet": "db4",
            "wavelet_level": 2,
            "min_history": 60,
        }
        params = {**default, **(parameters or {})}
        super().__init__("Pairs Trading (Wavelet)", params)

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
        if len(window) < max(min_hist, 2 ** level):
            return {}

        # log-prices normalisés sur la fenêtre de formation
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
        # distance euclidienne moyenne entre chaque paire
        # d_{ij} = mean_t (x_it - x_jt)^2
        sq = denoised ** 2
        gram = denoised.T @ denoised
        sumsq = sq.sum(axis=0)
        dist = (sumsq[:, None] + sumsq[None, :] - 2.0 * gram) / n_obs
        np.fill_diagonal(dist, np.inf)

        # extrait les top-N paires uniques (i < j)
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
                # i surperforme -> short i, long j
                signals[ti] -= 1.0
                signals[tj] += 1.0
            elif z < -entry:
                signals[ti] += 1.0
                signals[tj] -= 1.0
            elif abs(z) < exit_thr:
                # position neutre (exit) -> rien à ajouter
                pass

        # collapse en {-1, 0, +1}
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
        return {
            "formation_period": {
                "type": "int", "min": 60, "max": 504,
                "default": 252, "label": "Formation Period (days)",
            },
            "top_n_pairs": {
                "type": "int", "min": 1, "max": 100,
                "default": 20, "label": "Top N Pairs",
            },
            "entry_threshold": {
                "type": "float", "min": 0.5, "max": 5.0,
                "default": 2.0, "label": "Entry Z-Score",
            },
            "exit_threshold": {
                "type": "float", "min": 0.0, "max": 2.0,
                "default": 0.5, "label": "Exit Z-Score",
            },
            "wavelet": {
                "type": "str", "default": "db4",
                "options": ["db2", "db4", "db6", "sym4", "haar"],
                "label": "Wavelet Family",
            },
            "wavelet_level": {
                "type": "int", "min": 1, "max": 5,
                "default": 2, "label": "Wavelet Decomposition Level",
            },
            "min_history": {
                "type": "int", "min": 20, "max": 252,
                "default": 60, "label": "Min History (days)",
            },
        }
