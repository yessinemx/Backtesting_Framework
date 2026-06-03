"""
Spread and threshold construction.

Replication of Sections 4.2 and 4.3 of the paper.

For a pair (i, j):
    - Standard spread: OLS regression S_i = α + β·S_j on the formation sample,
        then ε_t = S_i,t - α̂ - β̂·S_j,t on the trading sample using original prices.
    - Wavelet spread: first filter prices (V = long-term MODWT component),
        regress V_i = α_w + β_w·V_j on the formation sample,
        then ε_w,t = V_i,t - α̂_w - β̂_w·V_j,t on the trading sample.

The trading threshold is 2σ, where σ is the standard deviation of the
formation spread.

Inputs and outputs use Polars wide frames [date, tickers...]. SpreadSpec keeps
the aligned NumPy arrays needed for P&L, including the original trading-period
prices, which remain the basis for return calculation even in the wavelet case.
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
    sigma: float                 # formation spread standard deviation
    threshold: float             # 2σ
    train_spread: np.ndarray
    trade_spread: np.ndarray
    trade_si: np.ndarray         # original trading prices for i used in P&L
    trade_sj: np.ndarray         # original trading prices for j used in P&L
    trade_dates: object          # aligned pl.Series of trading dates
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
    """Extract (dates, S_i, S_j) without missing values from a wide frame."""
    sub = prices.select([DATE_COL, i, j]).drop_nulls()
    si = sub.get_column(i).to_numpy().astype(float)
    sj = sub.get_column(j).to_numpy().astype(float)
    return sub.get_column(DATE_COL), si, sj


def build_spread(i, j, train_prices, trade_prices, use_wavelet=False,
                 n_sigma=2.0, wavelet=DEFAULT_WAVELET):
    """Build the spread for one pair across formation and trading.

    Parameters
    ----------
    train_prices, trade_prices : pl.DataFrame
        Original wide price frames [date, tickers...] for each period.
    use_wavelet : bool
        If True, estimate (α_w, β_w) and the spread on MODWT-filtered prices.
        P&L is still computed elsewhere on the original prices.

    Returns
    -------
    SpreadSpec | None
    """
    _, si_tr, sj_tr = _pair_arrays(train_prices, i, j)
    td_dates, si_td, sj_td = _pair_arrays(trade_prices, i, j)
    if len(si_tr) < 30 or len(si_td) < 5:
        return None

    if use_wavelet:
        # Recompute MODWT filtering on each window (see Appendix A.2).
        x_tr_i = modwt_smooth(si_tr, wavelet)
        x_tr_j = modwt_smooth(sj_tr, wavelet)
        x_td_i = modwt_smooth(si_td, wavelet)
        x_td_j = modwt_smooth(sj_td, wavelet)
    else:
        x_tr_i, x_tr_j = si_tr, sj_tr
        x_td_i, x_td_j = si_td, sj_td

    alpha, beta = _ols_alpha_beta(x_tr_i, x_tr_j)

    # Defensive cap: drop pairs with explosive |beta| caused by extreme price-level
    # mismatches (e.g. BRK.A-class shares vs ordinary equities). Paper Sec. 4.2
    # notes the spurious-regression issue but its 415-stock SPX universe naturally
    # avoids such pairings; we apply an explicit cap here.
    if not np.isfinite(beta) or abs(beta) > 10.0:
        return None

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
