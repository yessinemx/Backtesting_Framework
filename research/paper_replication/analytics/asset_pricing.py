"""Asset-pricing tests for the paper replication.

The author MATLAB files run Fama-French, q-factor and Petkova ICAPM regressions
on daily wavelet pairs-trading returns. This module mirrors those specifications
when a readable local factor file is present. CSV/parquet files are preferred;
simple MAT matrices are supported; MATLAB MCOS/dataset objects are reported with
a conversion hint because SciPy cannot decode them reliably without MATLAB.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import config_paper as research_config

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

TRADING_DAYS = research_config.TRADING_DAYS_PER_YEAR

MATLAB_FM_COLUMNS = (
    "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "STrev", "LTrev", "RF",
    "MKT", "F11", "F12", "F13", "F14", "TERM", "DEF", "DIV", "TBILL",
)

FAMA_FRENCH_SPECS = OrderedDict({
    "FF five factor I": (1, 2, 3, 4, 5),
    "FF five factor II": (1, 2, 3, 6, 8),
    "FF five factor III": (1, 2, 3, 6, 7),
    "FF seven factor I": (1, 2, 3, 4, 5, 6, 8),
    "FF seven factor II": (1, 2, 3, 4, 5, 6, 7),
    "FF eight factor": (1, 2, 3, 4, 5, 6, 7, 8),
})

Q_FACTOR_ALIASES = (
    ("R_MKT", "R_MKT_RF", "Mkt-RF", "MKT_RF", "Q_MKT"),
    ("R_ME", "ME", "Q_ME"),
    ("R_IA", "IA", "Q_IA"),
    ("R_ROE", "ROE", "Q_ROE"),
    ("R_EG", "EG", "Q_EG"),
)

PETKOVA_POSITIONS = (10, 18, 15, 17, 16, 3, 2)
# Alias-based fallback: same Petkova ICAPM variables as the MATLAB script, but
# resolved by canonical/common column names so the regression still works when
# the CSV stores columns in a different order than the original FM table.
PETKOVA_ALIASES = (
    ("R_MKT", "MKT", "Mkt", "Market"),
    ("TBILL", "DGS1MO", "RF1MO", "TBill1M"),
    ("TERM", "Termspread", "Term_spread", "TermSpread"),
    ("DIV", "Divyield", "DivYield", "Div_yield", "DividendYield"),
    ("DEF", "DefSpread", "defspread", "Def_spread"),
    ("HML", "Hml"),
    ("SMB", "Smb"),
)
RF_POSITION = 9


class FactorDataUnavailable(RuntimeError):
    """Raised when a factor file exists but cannot be decoded safely."""


@dataclass
class AlphaResult:
    model: str
    alpha_daily: float
    alpha_annual_pct: float          # alpha * 252 * 100
    t_alpha: float
    r2: float
    adj_r2: float
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
        t_alpha=float(fit.tvalues[0]), r2=float(fit.rsquared),
        adj_r2=float(fit.rsquared_adj), n=len(Y),
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


def _matlab_datenum_to_timestamp(value):
    ordinal = int(value)
    fraction = float(value) % 1.0
    return datetime.fromordinal(ordinal) + timedelta(days=fraction) - timedelta(days=366)


def _generic_factor_columns(n_cols):
    names = list(MATLAB_FM_COLUMNS[:n_cols])
    if len(names) < n_cols:
        names.extend(f"factor_{idx}" for idx in range(len(names) + 1, n_cols + 1))
    return names


def _normalise_factor_units(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    max_abs = df[numeric_cols].abs().max(skipna=True).max(skipna=True)
    if pd.notna(max_abs) and max_abs > 1.0:
        df = df.copy()
        df[numeric_cols] = df[numeric_cols] / 100.0
    return df


def _coerce_factor_frame(df):
    df = pd.DataFrame(df).copy()
    date_col = "date" if "date" in df.columns else "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return _normalise_factor_units(df)


def _load_mat_factor_file(path):
    try:
        import scipy.io as sio
        from scipy.io.matlab import MatlabOpaque
    except ImportError as exc:  # pragma: no cover
        raise FactorDataUnavailable("scipy is required to read MAT factor files.") from exc

    data = sio.loadmat(path, simplify_cells=True, struct_as_record=False, squeeze_me=True)
    if any(isinstance(value, MatlabOpaque) for value in data.values()):
        raise FactorDataUnavailable(
            "FMdata.mat contains a MATLAB MCOS/dataset object. Export the FM table "
            "from MATLAB to research/data/factors.csv or FMdata.csv/parquet."
        )
    candidate = None
    for name in ("FM", "factors", "factor_data"):
        if name in data:
            candidate = data[name]
            break
    if candidate is None:
        arrays = [v for k, v in data.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        candidate = arrays[0] if arrays else None
    if candidate is None:
        raise FactorDataUnavailable("No factor matrix was found in the MAT file.")
    if isinstance(candidate, dict):
        return _coerce_factor_frame(candidate)
    arr = np.asarray(candidate)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise FactorDataUnavailable("The MAT factor object is not a two-dimensional date/factor matrix.")
    dates = [_matlab_datenum_to_timestamp(value) for value in arr[:, 0]]
    frame = pd.DataFrame(arr[:, 1:], columns=_generic_factor_columns(arr.shape[1] - 1))
    frame.insert(0, "date", dates)
    return _coerce_factor_frame(frame)


def load_factor_file(path, columns=None):
    """Load a local factor file, date-indexed and daily.

    CSV/parquet files must have a date column ('date'/'Date' or first column).
    Values may be daily decimals or daily percent returns; percent-looking data
    are divided by 100. Simple MAT matrices are supported as date + factors.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix == ".mat":
        df = _load_mat_factor_file(path)
    else:
        raise ValueError(f"Unsupported factor file format: {path}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df = _coerce_factor_frame(df)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def load_available_factor_data(paths=None):
    """Return the first readable factor DataFrame from configured local paths."""
    errors = []
    candidates = paths or getattr(research_config, "PAPER_FACTOR_CANDIDATE_PATHS", ())
    for path in candidates:
        path = Path(path)
        if not path.exists():
            continue
        try:
            return load_factor_file(path), str(path), None
        except FactorDataUnavailable as exc:
            errors.append(f"{path.name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    note = "; ".join(errors) if errors else "No local factor file was found."
    return None, None, note


def _resolve_positions(df, positions):
    cols = list(df.columns)
    if len(cols) < max(positions):
        return None
    return [cols[pos - 1] for pos in positions]


def _resolve_aliases(df, alias_groups):
    resolved = []
    lower_map = {str(col).lower().replace("_", "").replace("-", ""): col for col in df.columns}
    for aliases in alias_groups:
        found = None
        for alias in aliases:
            key = alias.lower().replace("_", "").replace("-", "")
            if key in lower_map:
                found = lower_map[key]
                break
        if found is None:
            return None
        resolved.append(found)
    return resolved


def _risk_free_series(factor_df):
    rf_col = _resolve_aliases(factor_df, (("RF", "RiskFree", "Risk_Free", "TBILL_RF"),))
    if rf_col:
        return factor_df[rf_col[0]]
    rf_pos = _resolve_positions(factor_df, (RF_POSITION,))
    return factor_df[rf_pos[0]] if rf_pos else None


def _excess_returns(daily_returns, factor_df):
    y = daily_returns.rename("y") if isinstance(daily_returns, pd.Series) else pd.Series(daily_returns, name="y")
    rf = _risk_free_series(factor_df)
    if rf is None:
        return y
    frame = pd.concat([y, rf.rename("rf")], axis=1).dropna()
    return (frame["y"] - frame["rf"]).rename("y")


def _result_row(result, family, method, variant, source, tc_per_share=None):
    row = {
        "factor_family": family,
        "method": method,
        "variant": variant,
        "model": result.model,
        "alpha_daily": result.alpha_daily,
        "alpha_annual_%": round(result.alpha_annual_pct, 2),
        "t(alpha)": round(result.t_alpha, 2),
        "significant_5%": result.significant,
        "R2": round(result.r2, 3),
        "adj_R2": round(result.adj_r2, 3),
        "n": result.n,
        "factor_source": source,
    }
    if tc_per_share is not None:
        row["tc_per_share"] = tc_per_share
    for name, value in result.betas.items():
        key = str(name).replace(" ", "_").replace("/", "_")
        row[f"beta_{key}"] = round(value, 6)
        row[f"t_{key}"] = round(result.tstats.get(name, np.nan), 2)
    return row


def run_fama_french_models(daily_returns, factor_df, method="", variant="", source="", tc_per_share=None):
    rows = []
    y = _excess_returns(daily_returns, factor_df)
    for model_name, positions in FAMA_FRENCH_SPECS.items():
        cols = _resolve_positions(factor_df, positions)
        if cols is None:
            continue
        result = factor_regression(y, factor_df[cols], model_name=model_name)
        if result is not None:
            rows.append(_result_row(result, "fama_french", method, variant, source, tc_per_share))
    return rows


def run_q_factor_model(daily_returns, factor_df, method="", variant="", source="", tc_per_share=None):
    cols = _resolve_aliases(factor_df, Q_FACTOR_ALIASES)
    if cols is None:
        return []
    result = factor_regression(_excess_returns(daily_returns, factor_df), factor_df[cols], model_name="Hou-Xue-Zhang q-factor")
    if result is None:
        return []
    return [_result_row(result, "q_factor", method, variant, source, tc_per_share)]


def run_petkova_model(daily_returns, factor_df, method="", variant="", source="", tc_per_share=None):
    cols = _resolve_aliases(factor_df, PETKOVA_ALIASES)
    if cols is None:
        cols = _resolve_positions(factor_df, PETKOVA_POSITIONS)
    rf = _risk_free_series(factor_df)
    if cols is None or rf is None:
        return []
    y = daily_returns.rename("y") if isinstance(daily_returns, pd.Series) else pd.Series(daily_returns, name="y")
    df = pd.concat([y, rf.rename("rf"), factor_df[cols]], axis=1).dropna()
    if df.shape[0] < 35:
        return []
    y_col, rf_col = "y", "rf"
    factor_cols = [col for col in df.columns if col not in {y_col, rf_col}]
    values = df[factor_cols].to_numpy(dtype=float)
    demeaned = values - values.mean(axis=0)
    design = np.column_stack([np.ones(len(demeaned) - 1), demeaned[:-1]])
    coeffs = np.linalg.lstsq(design, demeaned[1:], rcond=None)[0]
    residuals = demeaned[1:] - design @ coeffs
    cov = np.cov(residuals, rowvar=False)
    try:
        inv_chol = np.linalg.inv(np.linalg.cholesky(cov))
    except np.linalg.LinAlgError:
        inv_chol = np.linalg.pinv(cov)
    scale = np.sqrt(np.var(values[:, 0], ddof=1))
    innovations = residuals @ inv_chol * scale
    petkova = pd.DataFrame(index=df.index[1:])
    petkova[str(factor_cols[0])] = df[factor_cols[0]].iloc[1:].to_numpy(dtype=float)
    for idx, col in enumerate(factor_cols[1:], start=1):
        petkova[f"innovation_{col}"] = innovations[:, idx]
    y_excess = (df[y_col].iloc[1:] - df[rf_col].iloc[1:]).rename("y")
    result = factor_regression(y_excess, petkova, model_name="ICAPM Petkova full sample")
    if result is None:
        return []
    return [_result_row(result, "petkova_icapm", method, variant, source, tc_per_share)]


def run_paper_factor_models(daily_returns, factor_df, method="", variant="", source="", tc_per_share=None):
    rows = []
    rows.extend(run_fama_french_models(daily_returns, factor_df, method, variant, source, tc_per_share))
    rows.extend(run_q_factor_model(daily_returns, factor_df, method, variant, source, tc_per_share))
    rows.extend(run_petkova_model(daily_returns, factor_df, method, variant, source, tc_per_share))
    return rows


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
            "R2": round(r.r2, 3), "adj_R2": round(r.adj_r2, 3), "n": r.n,
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
