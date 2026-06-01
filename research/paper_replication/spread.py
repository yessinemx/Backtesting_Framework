"""
Construction du spread et des seuils
====================================
Réplication des Sections 4.2 et 4.3 du papier.

Pour une paire (i, j) :
  - Spread standard  : régression OLS  S_i = α + β·S_j sur la formation,
    puis  ε_t = S_i,t - α̂ - β̂·S_j,t  sur le trading (prix originaux).
  - Spread wavelet   : on filtre d'abord les prix (V = MODWT long-terme),
    régression  V_i = α_w + β_w·V_j sur la formation,
    puis  ε_w,t = V_i,t - α̂_w - β̂_w·V_j,t  sur le trading.

Le seuil de trading vaut 2σ (σ = écart-type du spread en formation).

I/O en Polars (frame wide [date, tickers...]). Le SpreadSpec conserve les
tableaux numpy alignés nécessaires aux P&L : prix ORIGINAUX de la paire sur
la période de trading (toujours utilisés pour le rendement, même en wavelet).
"""
from dataclasses import dataclass
import numpy as np

from research.paper_replication.wavelet import modwt_smooth, DEFAULT_WAVELET

DATE_COL = "date"


@dataclass
class SpreadSpec:
    i: str
    j: str
    alpha: float
    beta: float
    sigma: float                 # écart-type du spread en formation
    threshold: float             # 2σ
    train_spread: np.ndarray
    trade_spread: np.ndarray
    trade_si: np.ndarray         # prix ORIGINAUX i (trading) pour les P&L
    trade_sj: np.ndarray         # prix ORIGINAUX j (trading) pour les P&L
    trade_dates: object          # pl.Series des dates de trading alignées
    use_wavelet: bool


def _ols_alpha_beta(y, x):
    """OLS simple y = α + β·x → (α, β)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x.mean()
    ym = y.mean()
    var = np.mean((x - xm) ** 2)
    if var == 0:
        return ym, 0.0
    beta = np.mean((x - xm) * (y - ym)) / var
    alpha = ym - beta * xm
    return float(alpha), float(beta)


def _pair_arrays(prices, i, j):
    """Extrait (dates, S_i, S_j) sans valeur manquante depuis un frame wide."""
    sub = prices.select([DATE_COL, i, j]).drop_nulls()
    si = sub.get_column(i).to_numpy().astype(float)
    sj = sub.get_column(j).to_numpy().astype(float)
    return sub.get_column(DATE_COL), si, sj


def build_spread(i, j, train_prices, trade_prices, use_wavelet=False,
                 n_sigma=2.0, wavelet=DEFAULT_WAVELET):
    """Construit le spread d'une paire sur formation + trading.

    Parameters
    ----------
    train_prices, trade_prices : pl.DataFrame
        Prix ORIGINAUX wide [date, tickers...] de chaque période.
    use_wavelet : bool
        Si True, (α_w, β_w) et le spread sont estimés sur les prix filtrés
        MODWT ; les P&L (ailleurs) restent calculés sur les prix originaux.

    Returns
    -------
    SpreadSpec | None
    """
    _, si_tr, sj_tr = _pair_arrays(train_prices, i, j)
    td_dates, si_td, sj_td = _pair_arrays(trade_prices, i, j)
    if len(si_tr) < 30 or len(si_td) < 5:
        return None

    if use_wavelet:
        # filtrage MODWT recalculé sur chaque fenêtre (cf. A.2)
        x_tr_i = modwt_smooth(si_tr, wavelet)
        x_tr_j = modwt_smooth(sj_tr, wavelet)
        x_td_i = modwt_smooth(si_td, wavelet)
        x_td_j = modwt_smooth(sj_td, wavelet)
    else:
        x_tr_i, x_tr_j = si_tr, sj_tr
        x_td_i, x_td_j = si_td, sj_td

    alpha, beta = _ols_alpha_beta(x_tr_i, x_tr_j)

    train_spread = x_tr_i - alpha - beta * x_tr_j
    trade_spread = x_td_i - alpha - beta * x_td_j

    sigma = float(np.std(train_spread, ddof=1))
    if not np.isfinite(sigma) or sigma == 0:
        return None

    return SpreadSpec(
        i=i, j=j, alpha=alpha, beta=beta,
        sigma=sigma, threshold=n_sigma * sigma,
        train_spread=train_spread, trade_spread=trade_spread,
        trade_si=si_td, trade_sj=sj_td, trade_dates=td_dates,
        use_wavelet=use_wavelet,
    )
