"""
Spread and threshold construction.

Replication of Sections 4.2 and 4.3 of the paper.

For a pair (i, j):
    - Standard spread: OLS regression S_i = α + β·S_j on the formation sample,
        then ε_t = S_i,t - α̂ - β̂·S_j,t on the trading sample.
    - Wavelet spread: first filter prices (V = long-term MODWT component),
        regress V_i = α_w + β_w·V_j on the formation sample,
        then ε_w,t = V_i,t - α̂_w - β̂_w·V_j,t on the trading sample.

The trading threshold is 2σ, where σ is the standard deviation of the
formation spread.

Per-window normalization
------------------------
Prices are divided by their formation-start value before the spread is built
(both legs start at 1.0). This is invariant to the split-adjustment reference
date: with prices back-adjusted for splits that happened after the sample (e.g.
a stock whose later splits deflate its in-sample price toward zero), a raw-price
OLS would otherwise produce extreme β and unbounded P&L. Normalizing recovers the
balanced, dollar-neutral pairs trade the paper intends (β ≈ O(1)). Returns
(ΔS/S_to) are scale-invariant, so the trade P&L is unaffected by the rescaling
itself — only the degenerate β explosions are removed.

Wavelet filtering spans the full series
---------------------------------------
For the wavelet variant the formation and trading windows are concatenated and
filtered once, then split, so the trading-period smooth is continuous with the
formation history (cf. the recursive real-time scheme in Appendix A.2) instead of
being filtered as an isolated block.

Inputs and outputs use Polars wide frames [date, tickers...]. SpreadSpec keeps
the aligned NumPy arrays needed for P&L.
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
    trade_si: np.ndarray         # (normalized) trading prices for i used in P&L
    trade_sj: np.ndarray         # (normalized) trading prices for j used in P&L
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
                 n_sigma=2.0, wavelet=DEFAULT_WAVELET, normalize=True,
                 boundary="symmetric"):
    """Build the spread for one pair across formation and trading.

    Parameters
    ----------
    train_prices, trade_prices : pl.DataFrame
        Original wide price frames [date, tickers...] for each period.
    use_wavelet : bool
        If True, estimate (α_w, β_w) and the spread on MODWT-filtered prices.
    normalize : bool
        If True, divide each leg by its formation-start price (see module docstring).
    boundary : "symmetric" | "periodic"
        MODWT boundary used when ``use_wavelet`` is True. "symmetric" is honest;
        "periodic" reproduces the paper's MATLAB result but leaks trading-period
        data into the in-sample estimate (see wavelet module docstring).

    Returns
    -------
    SpreadSpec | None
    """
    _, si_tr, sj_tr = _pair_arrays(train_prices, i, j)
    td_dates, si_td, sj_td = _pair_arrays(trade_prices, i, j)
    if len(si_tr) < 30 or len(si_td) < 5:
        return None

    # Per-formation-window normalization (split-adjustment invariant).
    if normalize:
        bi, bj = si_tr[0], sj_tr[0]
        if bi == 0 or bj == 0 or not (np.isfinite(bi) and np.isfinite(bj)):
            return None
    else:
        bi = bj = 1.0
    ni_tr, nj_tr = si_tr / bi, sj_tr / bj
    ni_td, nj_td = si_td / bi, sj_td / bj

    if use_wavelet:
        # Filter the full (formation + trading) series once, then split.
        nt = len(ni_tr)
        fi = modwt_smooth(np.concatenate([ni_tr, ni_td]), wavelet, boundary=boundary)
        fj = modwt_smooth(np.concatenate([nj_tr, nj_td]), wavelet, boundary=boundary)
        x_tr_i, x_tr_j = fi[:nt], fj[:nt]
        x_td_i, x_td_j = fi[nt:], fj[nt:]
    else:
        x_tr_i, x_tr_j = ni_tr, nj_tr
        x_td_i, x_td_j = ni_td, nj_td

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
        trade_si=ni_td, trade_sj=nj_td, trade_dates=td_dates,
        use_wavelet=use_wavelet,
    )
