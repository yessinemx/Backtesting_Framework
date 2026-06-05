"""
Pairs Trading with Partial Cointegration (Clegg / Kalman, benchmark).

Methodology (adapted from notebook Part 2 - Cointegration Partielle.ipynb):
- State-space model on log-prices :
    X2_t = beta * X1_t + M_t + R_t,                       (observation)
    M_t  = rho * M_{t-1} + eta_t,    eta ~ N(0, sigma_m2) (mean-reverting)
    R_t  = R_{t-1}        + nu_t,    nu  ~ N(0, sigma_r2) (random walk)
    X1_t = X1_{t-1}       + xi_t,    xi  ~ N(0, sigma_x2) (random walk, observed)
- Parameters theta = (beta, rho, sigma_m2, sigma_r2) estimated by MLE on the
  formation window using a steady-state Kalman recursion.
- Mean-reversion quality measured by
    R2_MR = sigma_m2 / (2*sigma_m2 + (1 + rho) * sigma_r2).
- Selection: pre-filter by absolute correlation on log-returns + Engle-Granger
  cointegration, then MLE-fit Clegg's SSM on the survivors and keep the
  ``top_n_pairs`` with the highest R2_MR (>= ``r2_mr_min``).
- Trading signal: the filtered mean-reverting component M_hat acts as the
  spread; z-score on the formation window's M_hat triggers entries when
  |z| > entry and flattens when |z| < exit.

Pair selection is heavy; the strategy caches the chosen pairs and the
filter state and only re-estimates the SSM every ``reselect_every``
rebalances (Kalman filter itself is run on the fresh window each call to
recover the latest M_hat for the signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from statsmodels.tsa.stattools import coint

from signals import PairsTradingBase


# ---------------------------------------------------------------------------
# Clegg state-space helpers (vectorised + steady-state for the MLE).
# ---------------------------------------------------------------------------
def _build_matrices(theta, sigma_x2):
    beta, rho, sigma_m2, sigma_r2 = theta
    H = np.array([[beta, 1.0, 1.0], [1.0, 0.0, 0.0]])
    F = np.array([[1.0, 0.0, 0.0], [0.0, rho, 0.0], [0.0, 0.0, 1.0]])
    Q = np.array([
        [sigma_x2, 0.0, 0.0],
        [0.0, sigma_m2, 0.0],
        [0.0, 0.0, sigma_r2],
    ])
    return H, F, Q


def _kalman_filter(X1, X2, theta):
    T = len(X1)
    sigma_x2 = float(np.var(np.diff(X1)))
    H, F, Q = _build_matrices(theta, sigma_x2)
    Y = np.column_stack((X2, X1))
    I = np.eye(3)
    M_hat = np.zeros(T)
    z = np.array([X1[0], 0.0, 0.0])
    P = np.diag([sigma_x2, 1.0, 1.0])
    for t in range(T):
        z_pred = F @ z
        P_pred = F @ P @ F.T + Q
        nu = Y[t] - H @ z_pred
        S = H @ P_pred @ H.T
        S = 0.5 * (S + S.T)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-15:
            S_inv = np.linalg.pinv(S)
        else:
            S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = P_pred @ H.T @ S_inv
        z = z_pred + K @ nu
        P = (I - K @ H) @ P_pred
        M_hat[t] = z[1]
    return M_hat


def _kalman_loglik_ss(X1, X2, theta, n_riccati=50):
    T = len(X1)
    sigma_x2 = float(np.var(np.diff(X1)))
    H, F, Q = _build_matrices(theta, sigma_x2)
    Y = np.column_stack((X2, X1))
    I = np.eye(3)

    P = np.diag([sigma_x2, 1.0, 1.0])
    for _ in range(n_riccati):
        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T
        S = 0.5 * (S + S.T)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-15:
            return -1e12
        S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = P_pred @ H.T @ S_inv
        P = (I - K @ H) @ P_pred

    P_pred = F @ P @ F.T + Q
    S = H @ P_pred @ H.T
    S = 0.5 * (S + S.T)
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    if abs(det) < 1e-15:
        return -1e12
    S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
    K = P_pred @ H.T @ S_inv
    log_det = np.log(abs(det) + 1e-15)
    const = -0.5 * (log_det + 2 * np.log(2 * np.pi))

    z = np.array([X1[0], 0.0, 0.0])
    HF = H @ F
    loglik = 0.0
    for t in range(T):
        nu = Y[t] - HF @ z
        loglik += const - 0.5 * (nu @ S_inv @ nu)
        z = F @ z + K @ nu
    return loglik


def _neg_loglik(params, X1, X2):
    beta, rho, sigma_m2, sigma_r2 = params
    if sigma_m2 <= 0 or sigma_r2 <= 0 or abs(rho) >= 1:
        return 1e12
    return -_kalman_loglik_ss(X1, X2, params)


def _init_theta(X1, X2):
    # OLS slope as beta init; rho = 0.5, variances = 0.01.
    x = np.asarray(X1, dtype=float)
    y = np.asarray(X2, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    beta0 = float(((x - xm) * (y - y.mean())).sum() / denom) if denom > 0 else 1.0
    return np.array([beta0, 0.5, 0.01, 0.01])


def _fit_pair(X1, X2):
    theta0 = _init_theta(X1, X2)
    bounds = [(None, None), (-0.999, 0.999), (1e-8, None), (1e-8, None)]
    opt = minimize(_neg_loglik, theta0, args=(X1, X2),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 100, "ftol": 1e-7})
    return opt.x, float(-opt.fun), bool(opt.success)


def _r2_mr(theta):
    _, rho, sigma_m2, sigma_r2 = theta
    denom = 2.0 * sigma_m2 + (1.0 + rho) * sigma_r2
    return sigma_m2 / denom if denom > 0 else np.nan


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class PairsTradingPartialCointegration(PairsTradingBase):

    def __init__(self, parameters=None):
        # PartialCointegration-specific defaults (common params handled by base).
        # top_n_pairs overrides the base default of 20 → 10 (heavier MLE fitting).
        extra = {
            "top_n_pairs": 10,
            "correlation_top_k": 100,
            "coint_pvalue_max": 0.10,
            "r2_mr_min": 0.50,
            "max_holding_mult": 2.0,
            "reselect_every": 6,
        }
        super().__init__("Pairs Trading (Partial Cointegration)", extra_defaults=extra, parameters=parameters)
        # list[(ticker_x, ticker_y, theta)]
        self._pairs_cache = None
        self._reselect_counter = 0
        # Persistent state per pair, keyed by (ticker_x, ticker_y):
        #   {"direction": +1/-1, "entry_step": int}
        self._pair_states: dict[tuple[str, str], dict] = {}
        self._step = 0

    def reset_state(self):
        """Drop cached pairs and open positions (use between independent backtests)."""
        self._pairs_cache = None
        self._reselect_counter = 0
        self._pair_states.clear()
        self._step = 0

    def _select_pairs(self, window):
        top_k = int(self.parameters["correlation_top_k"])
        pmax = float(self.parameters["coint_pvalue_max"])
        r2_min = float(self.parameters["r2_mr_min"])
        topn = int(self.parameters["top_n_pairs"])

        log_px = np.log(window.where(window > 0)).dropna(axis=1, how="any")
        if log_px.shape[1] < 2:
            return []
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
        idx = np.argpartition(abs_corr, -k)[-k:]
        tickers = log_px.columns.to_numpy()

        scored = []
        for ii, jj in zip(iu[idx], ju[idx]):
            x = log_px.iloc[:, ii].to_numpy()
            y = log_px.iloc[:, jj].to_numpy()
            try:
                _, pval, _ = coint(x, y)
            except Exception:
                continue
            if not np.isfinite(pval) or pval > pmax:
                continue
            try:
                theta, _ll, _ok = _fit_pair(x, y)
            except Exception:
                continue
            r2 = _r2_mr(theta)
            if not np.isfinite(r2) or r2 < r2_min:
                continue
            scored.append((r2, tickers[ii], tickers[jj], theta))

        if not scored:
            return []
        scored.sort(key=lambda t: -t[0])
        scored = scored[:topn]
        return [(tx, ty, theta) for (_r2, tx, ty, theta) in scored]

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

        # Pair selection (expensive: cached across `reselect` rebalances).
        if self._pairs_cache is None or self._reselect_counter % reselect == 0:
            self._pairs_cache = self._select_pairs(window)
            current_keys = {(tx, ty) for tx, ty, *_ in (self._pairs_cache or [])}
            self._pair_states = {k: v for k, v in self._pair_states.items() if k in current_keys}
        self._reselect_counter += 1
        self._step += 1
        if not self._pairs_cache:
            return {t: 0 for t in cols}

        log_px = np.log(window.where(window > 0)).dropna(axis=1, how="any")
        signals = {t: 0.0 for t in cols}

        for tx, ty, theta in self._pairs_cache:
            if tx not in log_px.columns or ty not in log_px.columns:
                continue
            x = log_px[tx].to_numpy()
            y = log_px[ty].to_numpy()
            beta = float(theta[0])
            rho = float(theta[1])
            try:
                M_hat = _kalman_filter(x, y, theta)
            except Exception:
                continue
            mu = float(M_hat.mean())
            sigma = float(M_hat.std(ddof=0))
            if sigma <= 0 or not np.isfinite(sigma):
                continue
            z = (M_hat[-1] - mu) / sigma

            # Half-life from the AR(1) coefficient rho on M_t.
            if 0 < abs(rho) < 1:
                half_life = float(np.clip(-np.log(2.0) / np.log(abs(rho)), 2.0, 252.0))
            else:
                half_life = 21.0

            key = (tx, ty)
            state = self._pair_states.get(key)

            # ---- Exit logic --------------------------------------------------
            if state is not None:
                holding = self._step - state["entry_step"]
                exit_now = abs(z) < exit_thr or holding > max_hold_mult * half_life
                if exit_now:
                    self._pair_states.pop(key, None)
                    state = None

            # ---- Entry logic -------------------------------------------------
            if state is None:
                if z > entry:
                    self._pair_states[key] = {"direction": -1, "entry_step": self._step}
                    state = self._pair_states[key]
                elif z < -entry:
                    self._pair_states[key] = {"direction": +1, "entry_step": self._step}
                    state = self._pair_states[key]

            # ---- Emit persistent leg signals --------------------------------
            if state is not None:
                d = state["direction"]
                signals[ty] += d
                signals[tx] -= d * beta

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
        # Override top_n_pairs default to 10 (heavier MLE fitting).
        schema["top_n_pairs"]["default"] = 10
        schema["top_n_pairs"]["max"] = 50
        schema["correlation_top_k"] = {
            "type": "int", "min": 20, "max": 1000,
            "default": 100, "label": "Correlation Pre-filter Top K",
        }
        schema["coint_pvalue_max"] = {
            "type": "float", "min": 0.01, "max": 0.20,
            "default": 0.10, "label": "Cointegration p-value max",
        }
        schema["r2_mr_min"] = {
            "type": "float", "min": 0.0, "max": 1.0,
            "default": 0.50, "label": "R\u00b2 Mean-Reversion min",
        }
        schema["max_holding_mult"] = {
            "type": "float", "min": 0.5, "max": 10.0,
            "default": 2.0, "label": "Max Holding (\u00d7 half-life)",
        }
        schema["reselect_every"] = {
            "type": "int", "min": 1, "max": 24,
            "default": 6, "label": "Re-select Pairs Every N Rebalances",
        }
        return schema
