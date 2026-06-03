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

from loaders import load_prices, load_membership
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


def _point_in_time_members(membership, start, end):
    """Tickers that were continuous index members across [start, end].

    `membership` is a long frame [date, index_id, ticker] of monthly
    point-in-time constituents (Bloomberg INDX_MWEIGHT). A ticker qualifies
    for a period only if it appears in EVERY monthly snapshot whose date falls
    inside the formation+trading window. This reproduces the paper's universe
    rule (Section 4.1): a stock enters period n only if it was an index member
    across both formation year n and trading year n+1 ("two consecutive years").
    """
    if membership is None or membership.height == 0:
        return None
    window = membership.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    n_snapshots = window.get_column("date").n_unique()
    if n_snapshots == 0:
        return None
    counts = window.group_by("ticker").agg(
        pl.col("date").n_unique().alias("n")
    )
    eligible = counts.filter(pl.col("n") == n_snapshots)
    return set(eligible.get_column("ticker").to_list())


def _restrict_to_traded_universe(train_prices, trade_prices):
    """Keep only tickers with no missing prices across BOTH train and trade.

    Mirrors the paper's universe rule (Section 4.1): a stock enters period n
    only if it traded across both formation year n and trading year n+1.
    """
    date_col = "date"
    train_cols = [c for c in train_prices.columns if c != date_col]
    trade_cols = [c for c in trade_prices.columns if c != date_col]
    common = [c for c in train_cols if c in trade_cols]
    if not common:
        return train_prices, trade_prices
    null_tr = train_prices.select([pl.col(c).null_count().alias(c) for c in common])
    null_td = trade_prices.select([pl.col(c).null_count().alias(c) for c in common])
    keep = [c for c in common if null_tr[c][0] == 0 and null_td[c][0] == 0]
    return (
        train_prices.select([date_col, *keep]),
        trade_prices.select([date_col, *keep]),
    )


def _drop_non_trading_days(prices, threshold=0.90):
    """Drop Bloomberg forward-fill rows that correspond to NYSE holidays.

    Bloomberg's NON_TRADING_WEEKDAYS fill carries the prior price forward on
    holidays. We detect these by looking for rows where the fraction of
    tickers with an exactly-unchanged price relative to the previous row
    exceeds `threshold`. The result keeps only real NYSE trading days,
    so 252 rows = 1 trading year, matching the paper's calendar.
    """
    date_col = "date"
    cols = [c for c in prices.columns if c != date_col]
    if not cols or prices.height < 2:
        return prices
    # Per-cell flag: True if price equals prior row's price (and both non-null).
    unchanged = prices.select([
        ((pl.col(c) == pl.col(c).shift(1)) & pl.col(c).is_not_null()).alias(c)
        for c in cols
    ])
    valid = prices.select([pl.col(c).is_not_null().alias(c) for c in cols])
    n_unchanged = unchanged.sum_horizontal()
    n_valid = valid.sum_horizontal()
    # Keep row if fewer than `threshold` of valid tickers are unchanged.
    keep_mask = (n_unchanged < threshold * n_valid) | (n_valid == 0)
    # Always keep the first row (no previous to compare to).
    keep_mask = pl.Series([True] + keep_mask.to_list()[1:])
    return prices.filter(keep_mask)


def run_period(period, prices, params, membership=None):
    """Run the pipeline on a single formation/trading period."""
    train_prices = prices.slice(*period.train_slice)
    trade_prices = prices.slice(*period.trade_slice)
    train_prices, trade_prices = _restrict_to_traded_universe(train_prices, trade_prices)

    # Point-in-time S&P 500 membership: keep only stocks that were actual
    # index members across this period's formation+trading window.
    members = _point_in_time_members(membership, period.train_start, period.trade_end)
    if members is not None:
        date_col = "date"
        train_prices = train_prices.select(
            [c for c in train_prices.columns if c == date_col or c in members]
        )
        trade_prices = trade_prices.select(
            [c for c in trade_prices.columns if c == date_col or c in members]
        )

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

    prices_path = None
    if params.get("raw_prices") and source == "data":
        import config as global_config
        prices_path = global_config.RAW_PRICES_PATH
        if not prices_path.exists():
            raise FileNotFoundError(
                f"raw_prices=True but {prices_path} is missing. "
                "Run `py extraction/refresh_raw.py SPX` on a Bloomberg terminal "
                "to download raw unadjusted close prices, or set "
                "PAIRS_CONFIG['raw_prices']=False to use adjusted prices."
            )

    prices = load_prices(
        source=source,
        index_id=params.get("index_id"),
        start=params.get("start_date"),
        end=params.get("end_date"),
        prices_path=prices_path,
    )
    prices = _drop_non_trading_days(prices)
    membership = load_membership(source=source, index_id=params.get("index_id"))
    periods = build_periods(
        prices.get_column("date"),
        block_size=params["block_size"],
        max_periods=params.get("max_periods"),
        paper_periods=params.get("paper_periods"),
    )
    if verbose:
        print(
            f"Universe: {prices.width - 1} securities, "
            f"{prices.height} rows, {len(periods)} periods"
        )

    period_results = []
    for period in periods:
        if verbose:
            print(f"  → {period} ...", flush=True)
        res = run_period(period, prices, params, membership=membership)
        if res is not None:
            period_results.append(res)
            if verbose:
                n_kept = res.standard.n_pairs
                print(
                    f"     pairs={n_kept} | "
                    f"standard R\u0304={res.standard.mean_return:+.4f} "
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
