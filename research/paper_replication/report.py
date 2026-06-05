"""Build paper-replication report: run periods, aggregate summaries, generate figures."""
from __future__ import annotations

import polars as pl

from config import config_paper as research_config
from loaders import load_prices, members_asof
from research.paper_replication.core.periods import build_periods
from research.paper_replication.core.pipeline import run_period
from research.paper_replication.outputs import figures as paper_figures


# Paper headline numbers (Tables 4 & 5, before transaction costs).
PAPER = {
    "distance":      {"std_ret": -0.55, "wav_ret": 11.82, "std_sr": -0.21, "wav_sr": 3.69},
    "cointegration": {"std_ret": -1.81, "wav_ret": 9.66,  "std_sr": -0.40, "wav_sr": 2.82},
}

METRIC_COLS = ["mean_return", "sharpe", "skewness", "kurtosis", "max_drawdown",
               "cvar_95", "pct_positive", "n_full", "n_partial", "n_non", "n_pairs"]


def run_method(method, prices, periods, params, include_paper_faithful=True, source="data"):
    """Run one selection method across all periods (point-in-time SPX universe)."""
    params = dict(params)
    params["method"] = method
    index_id = params.get("index_id", research_config.PAIRS_CONFIG["index_id"])
    rows = []
    for period in periods:
        if period.index % 5 == 0 or period.index == 1:
            print(f"  [run_method/{method}] period {period.index}/{len(periods)}", flush=True)
        universe = members_asof(period.train_start, index_id=index_id, source=source)
        res = run_period(period, prices, params, universe=universe,
                         include_paper_faithful=include_paper_faithful)
        if res is None:
            continue
        reps = [res.standard, res.wavelet]
        if res.paper_faithful is not None:
            reps.append(res.paper_faithful)
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
                 end=research_config.REPORT_END_DATE,
                 tc_sweep=None):
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

    print(f"[build_report] {len(periods)} periods, {prices.width-1} tickers, methods={methods}", flush=True)
    summaries, by_period, comparison_rows = {}, {}, []
    for method in methods:
        print(f"[build_report] running method={method}...", flush=True)
        df = run_method(method, prices, periods, params, source=source)
        summaries[method] = summarize(df)
        by_period[method] = df.sort(["period", "variant"])

        std = summaries[method].filter(pl.col("variant") == "standard")
        wav = summaries[method].filter(pl.col("variant") == "wavelet")
        pf = summaries[method].filter(pl.col("variant") == "wavelet_pf")
        p = research_config.PAPER_COMPARISON_TARGETS[method]
        comparison_rows.append({
            "method": method,
            "repl_std_return_%": round(std["mean_return"][0] * 100, 2),
            "repl_wav_return_%(honest)": round(wav["mean_return"][0] * 100, 2),
            "repl_wav_return_%(paper)": round(pf["mean_return"][0] * 100, 2) if pf.height else None,
            "paper_std_return_%": p["std_ret"],
            "paper_wav_return_%": p["wav_ret"],
            "repl_std_sharpe": round(std["sharpe"][0], 2),
            "repl_wav_sharpe(honest)": round(wav["sharpe"][0], 2),
            "repl_wav_sharpe(paper)": round(pf["sharpe"][0], 2) if pf.height else None,
            "paper_std_sharpe": p["std_sr"],
            "paper_wav_sharpe": p["wav_sr"],
        })

    print("[build_report] methods done — generating figures...", flush=True)
    figures, figure_diagnostics = {}, {}
    alpha_table = None
    if with_figures:
        figures, figure_diagnostics = paper_figures.generate_all(
            prices, periods, params, methods=methods, sweeps=sweeps, save=save_figures,
            tc_scenarios=tc_sweep,
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
