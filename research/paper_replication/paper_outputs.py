"""Paper-numbered tables and figure filenames for the replication outputs."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import polars as pl

from config import config_paper as research_config
from research.paper_replication.output_writer import clear_outputs, save_figure, save_table


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
                "replication_mean_return_pct": row["repl_wav_return_%"],
                "paper_mean_return_pct": row["paper_wav_return_%"],
                "replication_sharpe": row["repl_wav_sharpe"],
                "paper_sharpe": row["paper_wav_sharpe"],
                "return_gap_pct": row["repl_wav_return_%"] - row["paper_wav_return_%"],
                "sharpe_gap": row["repl_wav_sharpe"] - row["paper_wav_sharpe"],
            },
            {
                "method": row["method"],
                "variant": "opt",
                "replication_mean_return_pct": row["opt_return_%(lookahead)"],
                "paper_mean_return_pct": None,
                "replication_sharpe": row["opt_sharpe(lookahead)"],
                "paper_sharpe": None,
                "return_gap_pct": None,
                "sharpe_gap": None,
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


def build_table11(runs) -> pl.DataFrame:
    rows = []
    for method, data in runs.items():
        for variant, series in (
            ("standard", data.std_daily),
            ("wavelet", data.wav_daily),
            ("opt", data.opt_daily),
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
        "table11_evolution_of_annual_sharpe_ratios": build_table11(runs),
        "table15_annualized_abnormal_returns": build_table15(report.get("alpha_table")),
        "table16_sharpe_ratios_under_different_wavelet_classes": build_table16(diagnostics.get("wavelet_sweeps", {})),
        "table17_pairs_trading_returns_at_different_trading_period_spans": build_table17(diagnostics.get("horizon_sweeps", {})),
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