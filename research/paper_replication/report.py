"""
Shared paper-replication report builder.

Single entry point used by both the CLI driver (`research/replicate_paper.py`)
and the Streamlit app (Step 7). It runs the point-in-time SPX replication for the
requested methods, assembles the standard-vs-wavelet summary, the paper-comparison
table, and (optionally) every reproducible figure.
"""
from __future__ import annotations

import polars as pl

from config import config_paper as research_config
from loaders import load_prices, members_asof
from research.paper_replication.periods import build_periods
from research.paper_replication.pipeline import run_period
from research.paper_replication import figures as paper_figures


def run_method(method, prices, periods, params, include_opt=True):
    """Run one selection method across all periods (point-in-time SPX universe)."""
    params = dict(params)
    params["method"] = method
    index_id = params.get("index_id", research_config.PAIRS_CONFIG["index_id"])
    rows = []
    for period in periods:
        universe = members_asof(period.train_start, index_id=index_id)
        res = run_period(period, prices, params, universe=universe,
                         include_opt=include_opt)
        if res is None:
            continue
        reps = [res.standard, res.wavelet] + ([res.opt] if res.opt is not None else [])
        for rep in reps:
            row = {k: getattr(rep, k) for k in research_config.REPORT_METRIC_COLUMNS}
            row["period"] = res.period_index
            row["trade_end"] = str(res.trade_end)
            row["variant"] = rep.variant
            rows.append(row)
    return pl.DataFrame(rows)


def summarize(df):
    return (
        df.group_by("variant")
        .agg([pl.col(c).mean().alias(c) for c in research_config.REPORT_METRIC_COLUMNS])
        .sort("variant", descending=True)  # standard first, then wavelet
    )


def build_report(source="data", methods=research_config.DEFAULT_METHODS,
                 params=None, with_figures=True, sweeps=True,
                 save_figures=False,
                 start=research_config.REPORT_START_DATE,
                 end=research_config.REPORT_END_DATE):
    """Run the replication and return summaries, comparison, by-period, figures.

    Returns
    -------
    dict with keys:
        "summaries"  : {method: pl.DataFrame}
        "by_period"  : {method: pl.DataFrame}
        "comparison" : pl.DataFrame (paper vs replication)
        "figures"    : {name: plotly.Figure}  (empty if with_figures=False)
        "params", "n_periods", "universe_pool"
    """
    if params is None:
        params = dict(research_config.PAIRS_CONFIG)
    params = dict(params)
    params["index_id"] = params.get("index_id", research_config.PAIRS_CONFIG["index_id"])
    params["tc_per_share"] = research_config.HEADLINE_TC_PER_SHARE

    prices = load_prices(source=source, index_id=params["index_id"], start=start, end=end)
    periods = build_periods(
        prices.get_column("date"),
        block_size=params["block_size"],
        max_periods=params.get("max_periods"),
        paper_periods=params.get("paper_periods"),
    )

    summaries, by_period, comparison_rows = {}, {}, []
    for method in methods:
        df = run_method(method, prices, periods, params)
        summaries[method] = summarize(df)
        by_period[method] = df.sort(["period", "variant"])

        std = summaries[method].filter(pl.col("variant") == "standard")
        wav = summaries[method].filter(pl.col("variant") == "wavelet")
        opt = summaries[method].filter(pl.col("variant") == "opt")
        p = research_config.PAPER_COMPARISON_TARGETS[method]
        comparison_rows.append({
            "method": method,
            "repl_std_return_%": round(std["mean_return"][0] * 100, 2),
            "repl_wav_return_%": round(wav["mean_return"][0] * 100, 2),
            "opt_return_%(lookahead)": round(opt["mean_return"][0] * 100, 2) if opt.height else None,
            "paper_std_return_%": p["std_ret"],
            "paper_wav_return_%": p["wav_ret"],
            "repl_std_sharpe": round(std["sharpe"][0], 2),
            "repl_wav_sharpe": round(wav["sharpe"][0], 2),
            "opt_sharpe(lookahead)": round(opt["sharpe"][0], 2) if opt.height else None,
            "paper_std_sharpe": p["std_sr"],
            "paper_wav_sharpe": p["wav_sr"],
        })

    figures, figure_diagnostics = {}, {}
    alpha_table = None
    if with_figures:
        figures, figure_diagnostics = paper_figures.generate_all(
            prices, periods, params, methods=methods, sweeps=sweeps, save=save_figures
        )
        alpha_table = figure_diagnostics.get("alpha_table")

    return {
        "summaries": summaries,
        "by_period": by_period,
        "comparison": pl.DataFrame(comparison_rows),
        "figures": figures,
        "alpha_table": alpha_table,
        "figure_diagnostics": figure_diagnostics,
        "params": params,
        "n_periods": len(periods),
        "universe_pool": prices.width - 1,
        "prices": prices,
        "periods": periods,
    }
