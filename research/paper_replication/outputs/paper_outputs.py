"""Paper-numbered tables and figure filenames for the replication outputs."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import polars as pl

from config import config_paper as research_config
from research.paper_replication.outputs.output_writer import clear_outputs, save_figure, save_table
from research.paper_replication.core.wavelet import modwt_smooth


def _as_polars(frame):
    if frame is None:
        return pl.DataFrame()
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame.reset_index(drop=True))
    return pl.DataFrame(frame)


def _placeholder_table(reason: str) -> pl.DataFrame:
    return pl.DataFrame([{"status": "not_reproduced", "reason": reason}])


def _safe_float_scalar(value, default=0.0) -> float:
    if pd.isna(value):
        return float(default)
    return float(np.real(np.asarray(value).item()))


def build_table1(prices, periods) -> pl.DataFrame:
    rows = []
    for period in periods:
        rows.append({
            "period": period.index,
            "formation_start": str(period.train_start.date()),
            "formation_end": str(period.train_end.date()),
            "trading_start": str(period.trade_start.date()),
            "trading_end": str(period.trade_end.date()),
            "formation_days": int(period.train_slice[1]),
            "trading_days": int(period.trade_slice[1]),
            "universe_pool": max(prices.width - 1, 0),
        })
    return pl.DataFrame(rows)


def build_table2(prices) -> pl.DataFrame:
    cols = [col for col in prices.columns if col != "date"]
    if not cols:
        return pl.DataFrame()
    returns = prices.select([pl.col(col).pct_change().alias(col) for col in cols])
    values = returns.to_numpy().ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return pl.DataFrame()
    series = pd.Series(values, dtype="float64")
    skew = series.skew()
    kurt = series.kurt()
    return pl.DataFrame([{
        "n_observations": int(values.size),
        "mean_daily_return": float(values.mean()),
        "std_daily_return": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "skewness": _safe_float_scalar(skew),
        "kurtosis": _safe_float_scalar(kurt),
        "min_daily_return": float(values.min()),
        "median_daily_return": float(np.median(values)),
        "max_daily_return": float(values.max()),
    }])


def build_table3(runs) -> pl.DataFrame:
    data = runs.get("cointegration")
    if data is None or data.selection_stats.empty:
        return pl.DataFrame()
    return _as_polars(data.selection_stats.assign(method="cointegration"))


def build_table4(report) -> pl.DataFrame:
    rows = []
    for row in report["comparison"].to_dicts():
        rows.extend([
            {
                "method": row["method"],
                "variant": "standard",
                "replication_mean_return_pct": row["repl_std_return_%"],
                "paper_mean_return_pct": row["paper_std_return_%"],
                "replication_sharpe": row["repl_std_sharpe"],
                "paper_sharpe": row["paper_std_sharpe"],
                "return_gap_pct": row["repl_std_return_%"] - row["paper_std_return_%"],
                "sharpe_gap": row["repl_std_sharpe"] - row["paper_std_sharpe"],
            },
            {
                "method": row["method"],
                "variant": "wavelet",
                "replication_mean_return_pct": row["repl_wav_return_%(honest)"],
                "paper_mean_return_pct": row["paper_wav_return_%"],
                "replication_sharpe": row["repl_wav_sharpe(honest)"],
                "paper_sharpe": row["paper_wav_sharpe"],
                "return_gap_pct": row["repl_wav_return_%(honest)"] - row["paper_wav_return_%"],
                "sharpe_gap": row["repl_wav_sharpe(honest)"] - row["paper_wav_sharpe"],
            },
            {
                "method": row["method"],
                "variant": "wavelet_pf",
                "replication_mean_return_pct": row["repl_wav_return_%(paper)"],
                "paper_mean_return_pct": row["paper_wav_return_%"],
                "replication_sharpe": row["repl_wav_sharpe(paper)"],
                "paper_sharpe": row["paper_wav_sharpe"],
                "return_gap_pct": None if row["repl_wav_return_%(paper)"] is None else row["repl_wav_return_%(paper)"] - row["paper_wav_return_%"],
                "sharpe_gap": None if row["repl_wav_sharpe(paper)"] is None else row["repl_wav_sharpe(paper)"] - row["paper_wav_sharpe"],
            },
        ])
    return pl.DataFrame(rows)


def build_table5(report) -> pl.DataFrame:
    rows = []
    for method, summary in report["summaries"].items():
        for row in summary.to_dicts():
            rows.append({
                "method": method,
                "variant": row["variant"],
                "max_drawdown": row["max_drawdown"],
                "var_95": row.get("var_95"),
                "cvar_95": row["cvar_95"],
                "pct_positive": row["pct_positive"],
            })
    return pl.DataFrame(rows)


def build_table6(runs) -> pl.DataFrame:
    rows = []
    for method, data in runs.items():
        if data.cats.empty:
            continue
        cats = data.cats.groupby("variant", as_index=False).mean(numeric_only=True)
        trade = data.trade_stats.groupby("variant", as_index=False).mean(numeric_only=True)
        merged = cats.merge(trade, on="variant", how="left")
        for row in merged.to_dict(orient="records"):
            rows.append({"method": method, **row})
    return _as_polars(pd.DataFrame(rows))


def build_table7(runs) -> pl.DataFrame:
    rows = []
    metrics = ("full_prop", "partial_prop", "non_prop", "full_ret", "partial_ret", "non_ret")
    for method, data in runs.items():
        if data.cats.empty:
            continue
        pivot = data.cats.pivot(index="period", columns="variant", values=list(metrics))
        for period in sorted(data.cats["period"].unique()):
            row = {"method": method, "period": int(period)}
            for metric in metrics:
                std_key = (metric, "standard")
                wav_key = (metric, "wavelet")
                if std_key in pivot.columns and wav_key in pivot.columns:
                    row[f"{metric}_wavelet_minus_standard"] = float(
                        pivot.loc[period, wav_key] - pivot.loc[period, std_key]
                    )
                else:
                    row[f"{metric}_wavelet_minus_standard"] = float("nan")
            rows.append(row)
    return _as_polars(pd.DataFrame(rows))


def build_table8(runs) -> pl.DataFrame:
    rows = []
    for method, data in runs.items():
        if data.unit_root.empty:
            continue
        grouped = data.unit_root.groupby("variant", as_index=False).agg(
            rejection_rate_pct=(
                "rejected_5pct",
                lambda s: float(100.0 * pd.Series(s).dropna().mean()) if pd.Series(s).dropna().size else float("nan"),
            ),
            mean_pvalue=(
                "pvalue",
                lambda s: float(pd.Series(s).dropna().mean()) if pd.Series(s).dropna().size else float("nan"),
            ),
            n_spreads=("pvalue", lambda s: int(pd.Series(s).dropna().size)),
        )
        for row in grouped.to_dict(orient="records"):
            rows.append({"method": method, **row})
    return _as_polars(pd.DataFrame(rows))


def build_table9(runs) -> pl.DataFrame:
    rows = []
    for method, data in runs.items():
        if data.spread_stats.empty:
            continue
        grouped = data.spread_stats.groupby("variant", as_index=False).mean(numeric_only=True)
        for row in grouped.to_dict(orient="records"):
            rows.append({"method": method, **row})
    return _as_polars(pd.DataFrame(rows))


def build_table10(prices, periods, n_components=5) -> pl.DataFrame:
    rows = []
    for period in periods:
        trade_prices = prices.slice(*period.trade_slice)
        cols = [c for c in trade_prices.columns if c != "date"]
        if not cols:
            continue
        returns = trade_prices.select([pl.col(c).pct_change().alias(c) for c in cols]).drop_nulls()
        if returns.height < 3 or returns.width < 2:
            continue
        x = returns.to_numpy().astype(float)
        x = x[:, np.isfinite(x).all(axis=0)]
        if x.shape[1] < 2:
            continue
        x = x - x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, ddof=1)
        keep = std > 0
        x = x[:, keep] / std[keep]
        if x.shape[1] < 2:
            continue
        _, singular_values, _ = np.linalg.svd(x, full_matrices=False)
        explained = singular_values ** 2
        shares = explained / explained.sum()
        cumulative = np.cumsum(shares)
        for component, (share, cum_share) in enumerate(zip(shares[:n_components], cumulative[:n_components]), start=1):
            rows.append({
                "period": period.index,
                "component": component,
                "variance_explained_pct": float(100.0 * share),
                "cumulative_variance_explained_pct": float(100.0 * cum_share),
                "n_assets": int(x.shape[1]),
            })
    return _as_polars(pd.DataFrame(rows))


def _ols_alpha_beta_np(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    var = np.mean((x - xm) ** 2)
    beta = 0.0 if var == 0 else np.mean((x - xm) * (y - ym)) / var
    return float(ym - beta * xm), float(beta)


def _rw(rng, n, sigma=1.0):
    return np.cumsum(rng.normal(scale=sigma, size=n))


def build_table12(n_replications=120) -> pl.DataFrame:
    rng = np.random.default_rng(20230619)
    t_train = research_config.TRADING_DAYS_PER_YEAR
    n_total = 2 * t_train
    beta_values = (1.0, 2.0, 4.0)
    sigma2_values = (1.581, 7.906, 15.811, 31.623)
    rows = []
    for beta_true in beta_values:
        for sigma2_2 in sigma2_values:
            for sigma2_1 in sigma2_values:
                for integration_order in (0, 1):
                    beta_std, beta_wav, msfe_std, msfe_wav = [], [], [], []
                    for _ in range(n_replications):
                        latent_s2 = _rw(rng, n_total)
                        latent_error = (
                            rng.normal(size=n_total)
                            if integration_order == 0 else _rw(rng, n_total)
                        )
                        latent_s1 = beta_true * latent_s2 + latent_error
                        obs_s1 = latent_s1 + rng.normal(scale=np.sqrt(sigma2_1), size=n_total)
                        obs_s2 = latent_s2 + rng.normal(scale=np.sqrt(sigma2_2), size=n_total)

                        alpha_s, beta_s = _ols_alpha_beta_np(obs_s1[:t_train], obs_s2[:t_train])
                        smooth_s1 = modwt_smooth(obs_s1, wavelet=research_config.DEFAULT_WAVELET, boundary="periodic")
                        smooth_s2 = modwt_smooth(obs_s2, wavelet=research_config.DEFAULT_WAVELET, boundary="periodic")
                        alpha_w, beta_w = _ols_alpha_beta_np(smooth_s1[:t_train], smooth_s2[:t_train])

                        test_s1 = obs_s1[t_train:]
                        test_s2 = obs_s2[t_train:]
                        beta_std.append(beta_s)
                        beta_wav.append(beta_w)
                        msfe_std.append(float(np.mean((test_s1 - alpha_s - beta_s * test_s2) ** 2)))
                        msfe_wav.append(float(np.mean((test_s1 - alpha_w - beta_w * test_s2) ** 2)))
                    rows.append({
                        "beta_true": beta_true,
                        "sigma2_2": sigma2_2,
                        "sigma2_1": sigma2_1,
                        "error_integration": f"I({integration_order})",
                        "beta_hat_standard": float(np.mean(beta_std)),
                        "beta_hat_wavelet": float(np.mean(beta_wav)),
                        "msfe_standard": float(np.mean(msfe_std)),
                        "msfe_wavelet": float(np.mean(msfe_wav)),
                        "n_replications": n_replications,
                    })
    return _as_polars(pd.DataFrame(rows))


def build_table11(runs) -> pl.DataFrame:
    rows = []
    for method, data in runs.items():
        for variant, series in (
            ("standard", data.std_daily),
            ("wavelet", data.wav_daily),
            ("wavelet_pf", data.pf_daily),
        ):
            if series is None or getattr(series, "empty", True):
                continue
            s = pd.Series(series).dropna()
            if s.empty:
                continue
            s.index = pd.to_datetime(s.index)
            for year, grp in s.groupby(pd.DatetimeIndex(s.index).year):
                std = grp.std(ddof=1)
                sharpe = float(grp.mean() / std * np.sqrt(research_config.TRADING_DAYS_PER_YEAR)) if len(grp) > 1 and std > 0 else 0.0
                rows.append({
                    "method": method,
                    "variant": variant,
                    "year": int(year),
                    "annualized_sharpe": sharpe,
                })
    return _as_polars(pd.DataFrame(rows))


def build_table15(alpha_table) -> pl.DataFrame:
    if alpha_table is None or alpha_table.empty:
        return pl.DataFrame()
    return _as_polars(alpha_table)


def build_table16(wavelet_sweeps) -> pl.DataFrame:
    rows = []
    for method, df in wavelet_sweeps.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["method"] = method
        rows.append(tmp)
    return _as_polars(pd.concat(rows, ignore_index=True)) if rows else pl.DataFrame()


def build_table17(horizon_sweeps) -> pl.DataFrame:
    rows = []
    for method, df in horizon_sweeps.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["method"] = method
        rows.append(tmp)
    return _as_polars(pd.concat(rows, ignore_index=True)) if rows else pl.DataFrame()


def _concat_run_frames(runs, attr):
    frames = []
    for method, data in runs.items():
        frame = getattr(data, attr, None)
        if frame is None or frame.empty:
            continue
        tmp = frame.copy()
        tmp["method"] = method
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_table13(alpha_table) -> pl.DataFrame:
    if alpha_table is None or alpha_table.empty:
        return pl.DataFrame()
    frame = alpha_table.copy()
    if "factor_family" in frame.columns:
        ff = frame[frame["factor_family"].eq("fama_french")]
        if not ff.empty:
            return _as_polars(ff)
        status = frame[frame["factor_family"].eq("factor_data_status")]
        if not status.empty:
            return _as_polars(status)
    frame["factor_model"] = "local_market_model_proxy"
    frame["note"] = "Readable Fama-French factors were not found; this uses the local equal-weight market factor."
    return _as_polars(frame)


def build_table14(alpha_table) -> pl.DataFrame:
    if alpha_table is None or alpha_table.empty or "factor_family" not in alpha_table.columns:
        return pl.DataFrame()
    frame = alpha_table[alpha_table["factor_family"].isin(["q_factor", "petkova_icapm", "factor_data_status"])].copy()
    return _as_polars(frame)


def build_table18(runs) -> pl.DataFrame:
    frame = _concat_run_frames(runs, "transaction_stats")
    if frame.empty:
        return pl.DataFrame()
    grouped = frame.groupby(["method", "variant", "tc_per_share"], as_index=False).agg(
        mean_return=("mean_return", "mean"),
        sharpe=("sharpe", "mean"),
        pct_positive=("pct_positive", "mean"),
        total_cost=("total_cost", "sum"),
    )
    grouped["mean_return_pct"] = 100.0 * grouped["mean_return"]
    return _as_polars(grouped)


def build_table19(runs) -> pl.DataFrame:
    frame = _concat_run_frames(runs, "transaction_stats")
    if frame.empty:
        return pl.DataFrame()
    grouped = frame.groupby(["method", "variant", "tc_per_share"], as_index=False).agg(
        average_number_of_trades=("n_trades", "mean"),
        average_active_pairs=("n_active", "mean"),
    )
    return _as_polars(grouped)


def build_table20(runs) -> pl.DataFrame:
    frame = _concat_run_frames(runs, "transaction_stats")
    if frame.empty:
        return pl.DataFrame()
    grouped = frame.groupby(["method", "variant", "tc_per_share", "period"], as_index=False).agg(
        number_of_trades=("n_trades", "mean"),
        active_pairs=("n_active", "mean"),
    )
    return _as_polars(grouped)


def build_table21(runs, alpha_table) -> pl.DataFrame:
    if alpha_table is not None and not alpha_table.empty and "tc_per_share" in alpha_table.columns:
        frame = alpha_table.copy()
        frame = frame[pd.to_numeric(frame["tc_per_share"], errors="coerce").fillna(0.0).gt(0.0)]
        if not frame.empty:
            return _as_polars(frame)
    tc = build_table18(runs)
    if tc.is_empty():
        return build_table13(alpha_table)
    return tc.with_columns(
        pl.lit("local_market_model_proxy_after_transaction_costs").alias("factor_model")
    )


def build_table22(runs) -> pl.DataFrame:
    frame = _concat_run_frames(runs, "forced_close_stats")
    if frame.empty:
        return pl.DataFrame()
    grouped = frame.groupby(["method", "variant"], as_index=False).agg(
        mean_return=("mean_return", "mean"),
        sharpe=("sharpe", "mean"),
        pct_positive=("pct_positive", "mean"),
        average_number_of_trades=("n_trades", "mean"),
    )
    grouped["mean_return_pct"] = 100.0 * grouped["mean_return"]
    return _as_polars(grouped)


# ---------------------------------------------------------------------------
# Rigorous paper validation
# ---------------------------------------------------------------------------

# Absolute tolerances calibrated against the magnitude of paper headline numbers.
# Returns are reported in percentage points (e.g. 11.82 = 11.82% mean per period),
# Sharpe ratios are unit-less. PASS/WARN/FAIL bands.
VALIDATION_TOLERANCES = {
    "mean_return_pct": {"pass": 1.0, "warn": 3.0},
    "sharpe":          {"pass": 0.30, "warn": 0.80},
}

VALIDATION_FIELD_MAP = {
    "standard": {
        "return": ("repl_std_return_%", "paper_std_return_%"),
        "sharpe": ("repl_std_sharpe", "paper_std_sharpe"),
    },
    "wavelet_honest": {
        "return": ("repl_wav_return_%(honest)", "paper_wav_return_%"),
        "sharpe": ("repl_wav_sharpe(honest)", "paper_wav_sharpe"),
    },
    "wavelet_paper": {
        "return": ("repl_wav_return_%(paper)", "paper_wav_return_%"),
        "sharpe": ("repl_wav_sharpe(paper)", "paper_wav_sharpe"),
    },
}


def _status_from_delta(metric_key: str, abs_delta: float) -> str:
    band = VALIDATION_TOLERANCES[metric_key]
    if abs_delta <= band["pass"]:
        return "PASS"
    if abs_delta <= band["warn"]:
        return "WARN"
    return "FAIL"


def _fmt_value(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return f"{float(value):.4f}"
    return str(value)


def _validation_row(method, variant, metric_name, metric_key, paper, repl):
    if paper is None or repl is None:
        return {
            "category": "headline",
            "method": method,
            "variant": variant,
            "metric": metric_name,
            "paper_value": _fmt_value(paper),
            "replication_value": _fmt_value(repl),
            "abs_delta": None,
            "rel_delta_pct": None,
            "tolerance_pass": VALIDATION_TOLERANCES[metric_key]["pass"],
            "tolerance_warn": VALIDATION_TOLERANCES[metric_key]["warn"],
            "status": "MISSING",
            "comment": "Replication did not produce this variant in this run.",
        }
    paper_f = float(paper)
    repl_f = float(repl)
    abs_delta = repl_f - paper_f
    abs_d = abs(abs_delta)
    rel = (abs_d / abs(paper_f) * 100.0) if abs(paper_f) > 1e-12 else float("nan")
    return {
        "category": "headline",
        "method": method,
        "variant": variant,
        "metric": metric_name,
        "paper_value": _fmt_value(paper_f),
        "replication_value": _fmt_value(repl_f),
        "abs_delta": round(abs_delta, 4),
        "rel_delta_pct": None if np.isnan(rel) else round(rel, 2),
        "tolerance_pass": VALIDATION_TOLERANCES[metric_key]["pass"],
        "tolerance_warn": VALIDATION_TOLERANCES[metric_key]["warn"],
        "status": _status_from_delta(metric_key, abs_d),
        "comment": "",
    }


def _structural_row(metric, paper_value, repl_value, status, comment=""):
    abs_delta = None
    rel = None
    try:
        if paper_value is not None and repl_value is not None:
            pv = float(paper_value)
            rv = float(repl_value)
            abs_delta = rv - pv
            rel = abs(abs_delta) / abs(pv) * 100.0 if abs(pv) > 1e-12 else None
    except (TypeError, ValueError):
        abs_delta = None
        rel = None
    return {
        "category": "structural",
        "method": "-",
        "variant": "-",
        "metric": metric,
        "paper_value": _fmt_value(paper_value),
        "replication_value": _fmt_value(repl_value),
        "abs_delta": None if abs_delta is None else round(abs_delta, 4),
        "rel_delta_pct": None if rel is None else round(rel, 2),
        "tolerance_pass": None,
        "tolerance_warn": None,
        "status": status,
        "comment": comment,
    }


def _factor_row(alpha_table) -> dict:
    if alpha_table is None or getattr(alpha_table, "empty", True):
        return _structural_row(
            "factor_regressions_available",
            paper_value="FF/q/Petkova (Tables 13-14)",
            repl_value="none",
            status="FAIL",
            comment="No alpha table produced.",
        )
    if "factor_family" not in alpha_table.columns:
        return _structural_row(
            "factor_regressions_available",
            paper_value="FF/q/Petkova (Tables 13-14)",
            repl_value="market_proxy_only",
            status="WARN",
            comment="Only the local market-model proxy is available; no factor_family column.",
        )
    families = set(alpha_table["factor_family"].dropna().unique().tolist())
    has_ff = "fama_french" in families
    has_q = "q_factor" in families
    has_pet = "petkova_icapm" in families
    only_status = families.issubset({"factor_data_status"})
    if has_ff and has_q and has_pet:
        status = "PASS"
        comment = "Fama-French, q-factor, and Petkova ICAPM regressions all present."
    elif (has_ff or has_q or has_pet):
        status = "WARN"
        present = sorted(f for f in ("fama_french", "q_factor", "petkova_icapm") if f in families)
        comment = "Partial factor coverage: " + ", ".join(present)
    elif only_status:
        status = "FAIL"
        comment = "Factor file not readable; commit research/data/factors.csv or run research/paper_replication/bootstrap/fetch_factors.py."
    else:
        status = "WARN"
        comment = "Unrecognised factor families: " + ", ".join(sorted(families))
    return _structural_row(
        "factor_regressions_available",
        paper_value="FF/q/Petkova (Tables 13-14)",
        repl_value=", ".join(sorted(families)) if families else "none",
        status=status,
        comment=comment,
    )


def build_paper_validation(report) -> pl.DataFrame:
    """Compare replication outputs to the paper headline numbers and structural facts.

    Rows fall in two categories:
        * ``headline`` — per-method/variant comparison of mean return (%) and Sharpe ratio
          against Tables 4 and 5, with PASS/WARN/FAIL bands.
        * ``structural`` — sanity checks on universe size, number of trading periods,
          and presence of Fama-French / q-factor / Petkova regressions.
    """
    rows = []
    comparison = report.get("comparison")
    if comparison is not None and comparison.height:
        for row in comparison.to_dicts():
            method = row["method"]
            for variant, fields in VALIDATION_FIELD_MAP.items():
                ret_repl_key, ret_paper_key = fields["return"]
                sr_repl_key, sr_paper_key = fields["sharpe"]
                rows.append(_validation_row(
                    method, variant, "mean_return_pct", "mean_return_pct",
                    paper=row.get(ret_paper_key),
                    repl=row.get(ret_repl_key),
                ))
                rows.append(_validation_row(
                    method, variant, "sharpe", "sharpe",
                    paper=row.get(sr_paper_key),
                    repl=row.get(sr_repl_key),
                ))

    # Structural checks
    universe_pool = int(report.get("universe_pool", 0) or 0)
    paper_universe = int(research_config.PAPER_REFERENCE_UNIVERSE_SIZE)
    if paper_universe > 0:
        rel = abs(universe_pool - paper_universe) / paper_universe
        if rel <= 0.05:
            status = "PASS"
        elif rel <= 0.20:
            status = "WARN"
        else:
            status = "FAIL"
        rows.append(_structural_row(
            "universe_pool_size",
            paper_value=paper_universe,
            repl_value=universe_pool,
            status=status,
            comment="Paper uses 415 S&P 500 names; local pool drawn from point-in-time SPX membership.",
        ))

    n_periods = int(report.get("n_periods", 0) or 0)
    paper_periods = len(research_config.PAIRS_CONFIG.get("paper_periods") or [])
    if paper_periods:
        if n_periods == paper_periods:
            status = "PASS"
        elif n_periods >= max(1, paper_periods - 1):
            status = "WARN"
        else:
            status = "FAIL"
        rows.append(_structural_row(
            "n_trading_periods",
            paper_value=paper_periods,
            repl_value=n_periods,
            status=status,
            comment="Paper runs 7 yearly formation/trading blocks 2010-2018.",
        ))

    rows.append(_factor_row(report.get("alpha_table")))

    df = _as_polars(pd.DataFrame(rows))
    if not df.is_empty():
        df = df.with_columns(
            pl.col("status").cast(pl.Utf8),
        )
    return df


def _validation_status_counts(table: pl.DataFrame) -> dict:
    if table.is_empty() or "status" not in table.columns:
        return {}
    counts = table.group_by("status").len().to_dicts()
    return {row["status"]: int(row["len"]) for row in counts}


def build_paper_tables(report):
    diagnostics = report.get("figure_diagnostics", {})
    runs = diagnostics.get("runs", {})

    tables = OrderedDict({
        "table1_description_of_the_data_sample": build_table1(report["prices"], report["periods"]),
        "table2_basic_statistics_for_the_return_series": build_table2(report["prices"]),
        "table3_number_of_cointegrated_pairs": build_table3(runs),
        "table4_summary_of_results_for_basic_statistics": build_table4(report),
        "table5_downside_risk_measures_of_the_trading_periods": build_table5(report),
        "table6_trade_convergence_and_profit_analysis": build_table6(runs),
        "table7_proportion_and_return_differences": build_table7(runs),
        "table8_wavelet_and_standard_spread_unit_root_test_results": build_table8(runs),
        "table9_standard_error_comparison_of_wavelet_and_standard_spread": build_table9(runs),
        "table10_marginal_variance_explained_of_principal_components": build_table10(report["prices"], report["periods"]),
        "table11_evolution_of_annual_sharpe_ratios": build_table11(runs),
        "table12_simulation_results_with_high_frequency_contamination": build_table12(),
        "table13_fama_french_five_factor_models": build_table13(report.get("alpha_table")),
        "table14_q_factor_and_icapm_petkova_models": build_table14(report.get("alpha_table")),
        "table15_annualized_abnormal_returns": build_table15(report.get("alpha_table")),
        "table16_sharpe_ratios_under_different_wavelet_classes": build_table16(diagnostics.get("wavelet_sweeps", {})),
        "table17_pairs_trading_returns_at_different_trading_period_spans": build_table17(diagnostics.get("horizon_sweeps", {})),
        "table18_key_statistics_no_transaction_cost_vs_transaction_cost": build_table18(runs),
        "table19_average_number_of_trades_real_vs_transaction_cost": build_table19(runs),
        "table20_yearly_evolution_of_average_trades": build_table20(runs),
        "table21_annualized_abnormal_returns_transaction_cost": build_table21(runs, report.get("alpha_table")),
        "table22_profits_from_standard_pairs_trading_when_trades_are_forced_closed": build_table22(runs),
        "paper_validation_summary": build_paper_validation(report),
    })

    for name, reason in research_config.UNAVAILABLE_TABLES.items():
        existing = tables.get(name)
        if existing is None or _as_polars(existing).is_empty():
            tables[name] = _placeholder_table(reason)
    return tables


def save_paper_outputs(report):
    """Clear existing outputs and save paper-numbered tables and figures."""
    clear_outputs()

    written_tables = {}
    for name, table in build_paper_tables(report).items():
        frame = _as_polars(table)
        if frame.is_empty():
            continue
        written_tables[name] = save_table(frame, name)

    written_figures = {}
    for internal_name, fig in report.get("figures", {}).items():
        output_name = research_config.FIGURE_FILE_NAMES.get(internal_name)
        if output_name is None:
            continue
        written_figures[output_name] = save_figure(fig, output_name, formats=("png",))

    return {"tables": written_tables, "figures": written_figures}