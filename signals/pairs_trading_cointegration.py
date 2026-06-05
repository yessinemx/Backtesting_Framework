"""Pairs trading with Engle-Granger cointegration (benchmark strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import coint

from signals.pairs_trading_base import PairsTradingBase


def _ols_beta(x, y):
    """OLS slope/intercept for y = alpha + beta * x.  Returns (alpha, beta)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 0:
        return 0.0, 0.0
    beta = ((x - x_mean) * (y - y_mean)).sum() / denom
    alpha = y_mean - beta * x_mean
    return float(alpha), float(beta)


def _half_life_from_spread(spread):
    """Estimate half-life from AR(1) on the spread, clipped to [2, 252] days."""
    s = np.asarray(spread, dtype=float)
    if s.size < 5:
        return 21.0
    s_lag = s[:-1]
    s_now = s[1:]
    denom = (s_lag * s_lag).sum()
    if denom <= 0:
        return 21.0
    phi = float((s_lag * s_now).sum() / denom)
    if phi <= 0 or phi >= 1:
        return 21.0
    hl = -np.log(2.0) / np.log(phi)
    return float(np.clip(hl, 2.0, 252.0))


class PairsTradingCointegration(PairsTradingBase):

    def __init__(self, parameters=None):
        # Cointegration-specific defaults (common params handled by PairsTradingBase).
        extra = {
            "correlation_top_k": 200,
            "coint_pvalue_max": 0.05,
            "max_holding_mult": 2.0,
            "reselect_every": 5,
        }
        super().__init__("Pairs Trading (Cointegration)", extra_defaults=extra, parameters=parameters)
        # list[(ticker_x, ticker_y, alpha, beta, mu, sigma, half_life)]
        self._pairs_cache = None
        self._reselect_counter = 0
        # Persistent state machine per pair, keyed by (ticker_x, ticker_y):
        #   {"direction": +1/-1, "entry_step": int}
        self._pair_states: dict[tuple[str, str], dict] = {}
        self._step = 0   # counts generate_signals invocations (rebalance index)

    def reset_state(self):
        """Drop cached pairs and open positions (use between independent backtests)."""
        self._pairs_cache = None
        self._reselect_counter = 0
        self._pair_states.clear()
        self._step = 0

    def _select_pairs(self, window):
        """Pre-filter by correlation, then run Engle-Granger on the survivors."""
        top_k = int(self.parameters["correlation_top_k"])
        pmax = float(self.parameters["coint_pvalue_max"])
        topn = int(self.parameters["top_n_pairs"])

        log_px = np.log(window.where(window > 0)).dropna(axis=1, how="any")
        if log_px.shape[1] < 2:
            return []

        # Correlation pre-filter on log-returns.
        rets = log_px.diff().dropna(how="any")
        if len(rets) < 20:
            return []
        corr = rets.corr().to_numpy()
        n = corr.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        abs_corr = np.abs(corr[iu, ju])
        if abs_corr.size == 0:
            return []
        k = min(top_k, abs_corr.size)
        # Highest correlation = best cointegration candidates.
        idx = np.argpartition(abs_corr, -k)[-k:]
        cand_pairs = list(zip(iu[idx], ju[idx]))
        tickers = log_px.columns.to_numpy()

        scored = []
        for i, j in cand_pairs:
            x = log_px.iloc[:, i].to_numpy()
            y = log_px.iloc[:, j].to_numpy()
            try:
                _, pval, _ = coint(x, y)
            except Exception:
                continue
            if not np.isfinite(pval) or pval > pmax:
                continue
            alpha, beta = _ols_beta(x, y)
            if beta == 0.0:
                continue
            spread = y - alpha - beta * x
            mu = float(spread.mean())
            sigma = float(spread.std(ddof=0))
            if sigma <= 0 or not np.isfinite(sigma):
                continue
            half_life = _half_life_from_spread(spread - mu)
            scored.append((pval, tickers[i], tickers[j], alpha, beta, mu, sigma, half_life))

        if not scored:
            return []
        scored.sort(key=lambda t: t[0])
        scored = scored[:topn]
        return [(tx, ty, a, b, m, s, hl) for (_, tx, ty, a, b, m, s, hl) in scored]

    def generate_signals(self, prices, date, members):
        fp = int(self.parameters["formation_period"])
        entry = float(self.parameters["entry_threshold"])
        exit_thr = float(self.parameters["exit_threshold"])
        max_hold_mult = float(self.parameters["max_holding_mult"])
        min_hist = int(self.parameters["min_history"])
        reselect = max(1, int(self.parameters["reselect_every"]))

        cols = [t for t in members if t in prices.columns]
        if len(cols) < 2:
            return {}

        window = prices[cols].loc[:date].tail(fp).dropna(axis=1, how="any")
        if len(window) < min_hist:
            return {}

        # Cached pair selection (re-estimate every `reselect` rebalances).
        if self._pairs_cache is None or self._reselect_counter % reselect == 0:
            self._pairs_cache = self._select_pairs(window)
            # Drop state for pairs that are no longer in the cache.
            current_keys = {(tx, ty) for tx, ty, *_ in (self._pairs_cache or [])}
            self._pair_states = {k: v for k, v in self._pair_states.items() if k in current_keys}
        self._reselect_counter += 1
        self._step += 1

        signals = {t: 0.0 for t in cols}
        if not self._pairs_cache:
            return {t: 0 for t in signals}

        latest_log = np.log(window).iloc[-1]
        for tx, ty, alpha, beta, mu, sigma, half_life in self._pairs_cache:
            if tx not in latest_log.index or ty not in latest_log.index:
                continue
            x_t = float(latest_log[tx])
            y_t = float(latest_log[ty])
            s_t = y_t - alpha - beta * x_t
            z = (s_t - mu) / sigma if sigma > 0 else 0.0

            key = (tx, ty)
            state = self._pair_states.get(key)

            # ---- Exit logic on an open position ------------------------------
            if state is not None:
                holding = self._step - state["entry_step"]
                exit_now = abs(z) < exit_thr or holding > max_hold_mult * half_life
                if exit_now:
                    self._pair_states.pop(key, None)
                    state = None

            # ---- Entry logic if flat -----------------------------------------
            if state is None:
                if z > entry:
                    self._pair_states[key] = {"direction": -1, "entry_step": self._step}
                    state = self._pair_states[key]
                elif z < -entry:
                    self._pair_states[key] = {"direction": +1, "entry_step": self._step}
                    state = self._pair_states[key]

            # ---- Emit persistent leg signals ---------------------------------
            if state is not None:
                d = state["direction"]   # +1 = long Y / short X, -1 = short Y / long X
                signals[ty] += d
                signals[tx] -= d * float(beta)

        # Collapse to {-1, 0, +1}.
        out = {}
        for t, v in signals.items():
            if v > 1e-9:
                out[t] = 1
            elif v < -1e-9:
                out[t] = -1
            else:
                out[t] = 0
        return out

    @staticmethod
    def get_parameters_schema():
        schema = PairsTradingBase._common_schema()
        schema["correlation_top_k"] = {
            "type": "int", "min": 20, "max": 2000,
            "default": 200, "label": "Correlation Pre-filter Top K",
        }
        schema["coint_pvalue_max"] = {
            "type": "float", "min": 0.001, "max": 0.20,
            "default": 0.05, "label": "Cointegration p-value max",
        }
        schema["max_holding_mult"] = {
            "type": "float", "min": 0.5, "max": 10.0,
            "default": 2.0, "label": "Max Holding (\u00d7 half-life)",
        }
        schema["reselect_every"] = {
            "type": "int", "min": 1, "max": 24,
            "default": 5, "label": "Re-select Pairs Every N Rebalances",
        }
        return schema
