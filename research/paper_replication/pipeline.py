"""
Wavelet pairs trading paper replication orchestrator.

For each period (formation, trading):
    1. Select pairs on the formation sample (distance or cointegration).
    2. Build both the standard and wavelet spreads.
    3. Simulate trading on the trading window with 2σ thresholds.
    4. Aggregate metrics by variant.

Data is loaded through the loaders package in Polars. Replication tables and
figures are written to research/outputs/tables and research/outputs/figures.
"""
from dataclasses import dataclass
from typing import Any
import polars as pl

from loaders import load_prices
from research import config as research_config
from research.paper_replication.periods import build_periods
from research.paper_replication.selection import select_pairs
from research.paper_replication.spread import build_spread
from research.paper_replication.trading import simulate_pair
from research.paper_replication.metrics import aggregate_metrics, PairsReport


@dataclass
class PipelineResult:
    standard: PairsReport
    wavelet: PairsReport
    period_index: int
    train_start: object
    trade_end: object


def _run_variant(pairs, train_prices, trade_prices, use_wavelet, params):
    """Build spreads and simulate trading for one variant."""
    results = []
    for i, j in pairs:
        spec = build_spread(
            i, j, train_prices, trade_prices,
            use_wavelet=use_wavelet,
            n_sigma=params["threshold_sigma"],
            wavelet=params["wavelet"],
        )
        if spec is None:
            continue
        results.append(simulate_pair(spec, tc_per_share=params["tc_per_share"]))
    return results


def run_period(period, prices, params):
    """Run the pipeline on a single formation/trading period."""
    train_prices = prices.slice(*period.train_slice)
    trade_prices = prices.slice(*period.trade_slice)

    pairs = select_pairs(
        params["method"], train_prices,
        top_n=params["top_n"],
        candidate_pool=params["candidate_pool"],
        k_ar_diff=params["k_ar_diff"],
    )
    if not pairs:
        return None

    std_results = _run_variant(pairs, train_prices, trade_prices, False, params)
    wav_results = _run_variant(pairs, train_prices, trade_prices, True, params)

    return PipelineResult(
        standard=aggregate_metrics(std_results, params["method"], "standard",
                                   n_pairs=len(pairs)),
        wavelet=aggregate_metrics(wav_results, params["method"], "wavelet",
                                  n_pairs=len(pairs)),
        period_index=period.index,
        train_start=period.train_start,
        trade_end=period.trade_end,
    )


def run_pipeline(params=None, source="data", verbose=True, write_outputs=True):
    """Run the full pipeline across all periods.

    Parameters
    ----------
    params : dict | None
        Parameter dictionary. Defaults to research.config.PAIRS_CONFIG.
    source : "data" | "bloomberg"
        Price source loaded via the loaders package.
    write_outputs : bool
        If True, write tables and figures to research/outputs/.

    Returns
    -------
    dict : {"periods", "summary", "by_period", "params"}
    """
    if params is None:
        params = dict(research_config.PAIRS_CONFIG)

    prices = load_prices(source=source, index_id=params.get("index_id"))
    periods = build_periods(
        prices.get_column("date"),
        block_size=params["block_size"],
        max_periods=params.get("max_periods"),
    )
    if verbose:
        print(f"Universe: {prices.width - 1} securities, {len(periods)} periods")

    period_results = []
    for period in periods:
        if verbose:
            print(f"  → {period} ...", flush=True)
        res = run_period(period, prices, params)
        if res is not None:
            period_results.append(res)
            if verbose:
                print(
                    f"     standard R̄={res.standard.mean_return:+.4f} "
                    f"Sharpe={res.standard.sharpe:+.2f} | "
                    f"wavelet R̄={res.wavelet.mean_return:+.4f} "
                    f"Sharpe={res.wavelet.sharpe:+.2f}"
                )

    by_period = _build_period_table(period_results)
    summary = _build_summary(period_results)

    result = {
        "periods": period_results,
        "summary": summary,
        "by_period": by_period,
        "params": params,
    }

    if write_outputs:
        _write_outputs(result)

    return result


def _build_period_table(period_results) -> pl.DataFrame:
    rows = []
    for pr in period_results:
        for rep in (pr.standard, pr.wavelet):
            row = rep.as_dict()
            row["period"] = pr.period_index
            row["trade_end"] = pr.trade_end
            rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


_METRIC_COLS = [
    "mean_return", "std_return", "sharpe", "skewness", "kurtosis",
    "max_drawdown", "var_95", "cvar_95", "pct_positive",
    "n_pairs", "n_active", "n_full", "n_partial", "n_non", "n_inactive",
]


def _build_summary(period_results) -> pl.DataFrame:
    """Average metrics across all periods, split by variant."""
    if not period_results:
        return pl.DataFrame()
    rows = []
    for variant in ("standard", "wavelet"):
        reports = [getattr(pr, variant) for pr in period_results]
        row: dict[str, Any] = {"variant": variant}
        for col in _METRIC_COLS:
            vals = [getattr(r, col) for r in reports]
            row[col] = float(sum(vals) / len(vals))
        rows.append(row)
    return pl.DataFrame(rows)


def _write_outputs(result):
    """Write replication tables and figures to research/outputs/."""
    from research.paper_replication.output_writer import save_table, save_figure
    import plotly.graph_objects as go

    summary = result["summary"]
    by_period = result["by_period"]
    if summary.is_empty():
        return

    save_table(summary, "table_summary_standard_vs_wavelet")
    save_table(by_period, "table_metrics_by_period")

    variants = summary.get_column("variant").to_list()

    # Figure 1: mean return by variant.
    fig_ret = go.Figure(go.Bar(
        x=variants, y=summary.get_column("mean_return").to_list(),
        marker_color=["#888", "#2ca02c"],
    ))
    fig_ret.update_layout(title="Mean return per pair: standard vs wavelet",
                          yaxis_title="Mean return", template="plotly_white")
    save_figure(fig_ret, "fig_mean_return")

    # Figure 2: annualized Sharpe by variant.
    fig_sr = go.Figure(go.Bar(
        x=variants, y=summary.get_column("sharpe").to_list(),
        marker_color=["#888", "#1f77b4"],
    ))
    fig_sr.update_layout(title="Annualized Sharpe: standard vs wavelet",
                         yaxis_title="Sharpe", template="plotly_white")
    save_figure(fig_sr, "fig_sharpe")

    # Figure 3: convergence categories for the wavelet variant.
    wav = summary.filter(pl.col("variant") == "wavelet")
    fig_cat = go.Figure(go.Bar(
        x=["full", "partial", "non", "inactive"],
        y=[wav["n_full"][0], wav["n_partial"][0],
           wav["n_non"][0], wav["n_inactive"][0]],
        marker_color="#9467bd",
    ))
    fig_cat.update_layout(title="Convergence categories (wavelet, average per period)",
                          yaxis_title="Number of pairs", template="plotly_white")
    save_figure(fig_cat, "fig_convergence_categories")

    # Figure 4: mean return by period.
    if not by_period.is_empty():
        wbp = by_period.filter(pl.col("variant") == "wavelet").sort("period")
        sbp = by_period.filter(pl.col("variant") == "standard").sort("period")
        fig_ts = go.Figure()
        fig_ts.add_scatter(x=sbp.get_column("period").to_list(),
                           y=sbp.get_column("mean_return").to_list(),
                           mode="lines+markers", name="standard",
                           line=dict(color="#888"))
        fig_ts.add_scatter(x=wbp.get_column("period").to_list(),
                           y=wbp.get_column("mean_return").to_list(),
                           mode="lines+markers", name="wavelet",
                           line=dict(color="#2ca02c"))
        fig_ts.update_layout(title="Mean return by period",
                     xaxis_title="Period", yaxis_title="Mean return",
                             template="plotly_white")
        save_figure(fig_ts, "fig_mean_return_by_period")


if __name__ == "__main__":
    result = run_pipeline()
    print("\n=== Standard vs wavelet summary (average across periods) ===")
    print(result["summary"])
