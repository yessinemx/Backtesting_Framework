"""
Asset-pricing tests of the wavelet pairs-trading returns.

Replication of Section 5.4 (Tables 13-15, Figure 9) of the paper.

The paper regresses the daily wavelet pairs-trading returns on common risk
factors and checks whether the intercept (alpha / abnormal return) is positive
and significant under four models:
    FFPI  : Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA)
    FFMR  : Mkt-RF, SMB, HML, MOM, STrev
    q     : Hou-Xue-Zhang q-factors (Mkt-RF, R_ME, R_IA, R_ROE, R_EG)
    ICAPM : Petkova (Mkt-RF, SMB, HML, DIV, TERM, DEF, TBILL)
Standard errors are Newey-West (1987) HAC.

Data availability
-----------------
The full FFPI/FFMR/q/ICAPM factors are external datasets (Ken French's library,
global-q.org, FRED). They are not in the local data folder, so this module:
  * always computes the **market model** (CAPM) alpha from local data — the
    market factor is the equal-weight S&P 500 return in excess of the USD
    risk-free rate; this directly answers the paper's core question (is there
    abnormal return beyond market exposure?);
  * runs any of the four full models when a factor file is supplied via
    `load_factor_file()` (date-indexed CSV/parquet with the factor columns).

Everything uses the daily portfolio return series produced by `figures.collect_run`.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import config_paper as research_config

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

TRADING_DAYS = research_config.TRADING_DAYS_PER_YEAR


@dataclass
class AlphaResult:
    model: str
    alpha_daily: float
    alpha_annual_pct: float          # alpha * 252 * 100
    t_alpha: float
    r2: float
    n: int
    betas: dict = field(default_factory=dict)
    tstats: dict = field(default_factory=dict)

    @property
    def significant(self):
        return abs(self.t_alpha) > 1.96


def _nw_lags(n):
    """Newey-West automatic bandwidth (Newey & West 1994 plug-in rule)."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def factor_regression(y, factors, model_name="model", lags=None):
    """Newey-West OLS of daily returns `y` on `factors` (DataFrame). Intercept=alpha."""
    if sm is None:
        raise ImportError("statsmodels is required for asset-pricing regressions.")
    # Align y and factors on their (date) index; rename to avoid clashes.
    y = y.rename("y") if isinstance(y, pd.Series) else pd.Series(np.asarray(y), name="y")
    df = pd.concat([y, factors], axis=1).dropna()
    if df.shape[0] < 30:
        return None
    Y = df["y"].to_numpy()
    X = sm.add_constant(df[factors.columns].to_numpy())
    lags = _nw_lags(len(Y)) if lags is None else lags
    fit = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    names = ["const"] + list(factors.columns)
    betas = {names[k]: float(fit.params[k]) for k in range(1, len(names))}
    tstats = {names[k]: float(fit.tvalues[k]) for k in range(1, len(names))}
    a = float(fit.params[0])
    return AlphaResult(
        model=model_name, alpha_daily=a, alpha_annual_pct=a * TRADING_DAYS * 100,
        t_alpha=float(fit.tvalues[0]), r2=float(fit.rsquared), n=len(Y),
        betas=betas, tstats=tstats,
    )


def market_excess(bench_index, riskfree_usd):
    """Daily market excess return: equal-weight S&P 500 return minus USD risk-free.

    Parameters
    ----------
    bench_index : pd.Series   (EW member daily return, date-indexed)
    riskfree_usd : pd.Series  (daily USD risk-free rate, date-indexed)
    """
    rf = riskfree_usd.reindex(bench_index.index).ffill().fillna(0.0)
    return (bench_index - rf).rename("Mkt-RF")


def run_market_model(daily_returns, mkt_excess, label=""):
    """Market-model (CAPM) alpha for a daily return series."""
    factors = mkt_excess.to_frame("Mkt-RF")
    res = factor_regression(daily_returns, factors, model_name=f"Market model {label}".strip())
    return res


def run_full_model(daily_returns, factor_df, model_name):
    """Run one of the paper's multi-factor models from a supplied factor frame."""
    return factor_regression(daily_returns, factor_df, model_name=model_name)


def load_factor_file(path, columns=None):
    """Load an external factor file (CSV/parquet), date-indexed, daily.

    The file must have a date column ('date' or first column) plus one column per
    factor (e.g. Mkt-RF, SMB, HML, RMW, CMA). Values are daily decimal returns.
    """
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def alpha_table(results):
    """Tidy DataFrame from a list of AlphaResult (skips None)."""
    rows = []
    for r in results:
        if r is None:
            continue
        rows.append({
            "model": r.model, "alpha_daily": r.alpha_daily,
            "alpha_annual_%": round(r.alpha_annual_pct, 2),
            "t(alpha)": round(r.t_alpha, 2), "significant_5%": r.significant,
            "R2": round(r.r2, 3), "n": r.n,
        })
    return pd.DataFrame(rows)


def fig9_yearly_alpha(daily_returns, mkt_excess, method=""):
    """Figure 9 - yearly evolution of the (annualized) market-model alpha."""
    s = pd.Series(daily_returns).dropna()
    s.index = pd.to_datetime(s.index)
    fig = go.Figure()
    rows = []
    for year, grp in s.groupby(s.index.year):
        mx = mkt_excess.reindex(grp.index)
        res = factor_regression(grp, mx.to_frame("Mkt-RF"), model_name=str(year))
        if res is not None:
            rows.append((year, res.alpha_annual_pct, res.t_alpha))
    if rows:
        years = [r[0] for r in rows]
        alphas = [r[1] for r in rows]
        colors = ["#2ca02c" if abs(r[2]) > 1.96 else "#9aa0a6" for r in rows]
        fig.add_trace(go.Bar(x=years, y=alphas, marker_color=colors,
                             name="annualized alpha (%)"))
        fig.add_hline(y=0, line=dict(color="#475569", width=1))
    fig.update_layout(title=f"Figure 9 - Yearly market-model alpha ({method}); "
                            f"green = significant at 5%",
                      xaxis_title="Year", yaxis_title="Annualized alpha (%)",
                      template="plotly_white", height=420)
    return fig
