"""
Faithful replication driver for "Pairs trading with wavelet transform"
(Eroglu, Yener & Yigit, 2023, Quantitative Finance).

Universe : S&P 500 (SPX), point-in-time membership per formation window.
Window   : 2010-03-05 -> 2018-03-15  ->  8 blocks of 252 days = 7 periods (Table 1).
Methods  : minimum distance and cointegration (Johansen).
Spread   : standard vs sym wavelet (zero-phase level-1 MODWT), 2 sigma threshold,
           per-window price normalization (split-adjustment invariant).

Outputs (research/outputs/):
    tables/  summary_<method>.csv, by_period_<method>.csv, paper_comparison.csv
    figures/ mean_return_by_period_<method>.html, summary_<method>.html

Run:
    python research/replicate_paper.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
import plotly.graph_objects as go

from loaders import load_prices, members_asof
from research import config as research_config
from research.paper_replication.periods import build_periods
from research.paper_replication.pipeline import run_period
from research.paper_replication.output_writer import save_table, save_figure
from research.paper_replication import figures as paper_figures

START, END = "2010-03-05", "2018-03-15"

# Paper headline numbers (Tables 4 & 5, before transaction costs).
PAPER = {
    "distance":      {"std_ret": -0.55, "wav_ret": 11.82, "std_sr": -0.21, "wav_sr": 3.69},
    "cointegration": {"std_ret": -1.81, "wav_ret": 9.66,  "std_sr": -0.40, "wav_sr": 2.82},
}

METRIC_COLS = ["mean_return", "sharpe", "skewness", "kurtosis", "max_drawdown",
               "cvar_95", "pct_positive", "n_full", "n_partial", "n_non", "n_pairs"]


def run_method(method: str, prices: pl.DataFrame, periods, params: dict):
    params = dict(params)
    params["method"] = method
    rows = []
    for period in periods:
        universe = members_asof(period.train_start, index_id="SPX")
        res = run_period(period, prices, params, universe=universe)
        if res is None:
            continue
        for rep in (res.standard, res.wavelet):
            row = {k: getattr(rep, k) for k in METRIC_COLS}
            row["period"] = res.period_index
            row["trade_end"] = str(res.trade_end.date())
            row["variant"] = rep.variant
            rows.append(row)
    return pl.DataFrame(rows)


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("variant")
        .agg([pl.col(c).mean().alias(c) for c in METRIC_COLS])
        .sort("variant", descending=True)  # standard first, then wavelet
    )


def main():
    params = dict(research_config.PAIRS_CONFIG)
    params["index_id"] = "SPX"
    params["tc_per_share"] = 0.0  # headline = before transaction costs (Table 4)

    print(f"Loading SPX prices {START} -> {END} ...")
    prices = load_prices(source="data", index_id="SPX", start=START, end=END)
    periods = build_periods(prices.get_column("date"), block_size=params["block_size"])
    print(f"Universe pool: {prices.width - 1} tickers | {prices.height} days | {len(periods)} periods\n")

    comparison_rows = []
    for method in ("distance", "cointegration"):
        df = run_method(method, prices, periods, params)
        summary = summarize(df)
        save_table(summary, f"summary_{method}")
        save_table(df.sort(["period", "variant"]), f"by_period_{method}")

        std = summary.filter(pl.col("variant") == "standard")
        wav = summary.filter(pl.col("variant") == "wavelet")
        p = PAPER[method]
        print(f"{'='*78}\n  {method.upper()}  (averaged over {df['period'].n_unique()} periods, before costs)\n{'='*78}")
        print(f"  {'metric':<16}{'standard':>12}{'wavelet':>12}   | {'paper std':>10}{'paper wav':>10}")
        print(f"  {'mean return':<16}{std['mean_return'][0]*100:>11.2f}%{wav['mean_return'][0]*100:>11.2f}%"
              f"   | {p['std_ret']:>9.2f}%{p['wav_ret']:>9.2f}%")
        print(f"  {'Sharpe (ann.)':<16}{std['sharpe'][0]:>12.2f}{wav['sharpe'][0]:>12.2f}"
              f"   | {p['std_sr']:>10.2f}{p['wav_sr']:>10.2f}")
        print(f"  {'skewness':<16}{std['skewness'][0]:>12.2f}{wav['skewness'][0]:>12.2f}")
        print(f"  {'% positive':<16}{std['pct_positive'][0]*100:>11.1f}%{wav['pct_positive'][0]*100:>11.1f}%")
        print(f"  {'max drawdown':<16}{std['max_drawdown'][0]*100:>11.2f}%{wav['max_drawdown'][0]*100:>11.2f}%")
        print(f"  avg pairs/period: {wav['n_pairs'][0]:.0f}   "
              f"(full {wav['n_full'][0]:.0f} / partial {wav['n_partial'][0]:.0f} / non {wav['n_non'][0]:.0f})\n")

        comparison_rows.append({
            "method": method,
            "repl_std_return_%": round(std["mean_return"][0]*100, 2),
            "repl_wav_return_%": round(wav["mean_return"][0]*100, 2),
            "paper_std_return_%": p["std_ret"],
            "paper_wav_return_%": p["wav_ret"],
            "repl_std_sharpe": round(std["sharpe"][0], 2),
            "repl_wav_sharpe": round(wav["sharpe"][0], 2),
            "paper_std_sharpe": p["std_sr"],
            "paper_wav_sharpe": p["wav_sr"],
        })

        # Figures.
        sbp = df.filter(pl.col("variant") == "standard").sort("period")
        wbp = df.filter(pl.col("variant") == "wavelet").sort("period")
        fig = go.Figure()
        fig.add_scatter(x=sbp["period"].to_list(), y=[v*100 for v in sbp["mean_return"]],
                        mode="lines+markers", name="standard", line=dict(color="#888"))
        fig.add_scatter(x=wbp["period"].to_list(), y=[v*100 for v in wbp["mean_return"]],
                        mode="lines+markers", name="wavelet", line=dict(color="#2ca02c"))
        fig.update_layout(title=f"Mean return by period - {method}",
                          xaxis_title="Period", yaxis_title="Mean return (%)",
                          template="plotly_white")
        save_figure(fig, f"mean_return_by_period_{method}")

    comp = pl.DataFrame(comparison_rows)
    save_table(comp, "paper_comparison")
    print(f"{'='*78}\n  PAPER vs REPLICATION (saved to research/outputs/tables/paper_comparison.csv)\n{'='*78}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(comp)

    # Reproduce the paper's figures (Fig 1-11) and the asset-pricing alphas (Sec 5.4).
    print(f"\n{'='*78}\n  Generating paper figures + asset-pricing alphas ...\n{'='*78}")
    figs, alpha_table = paper_figures.generate_all(
        prices, periods, params, methods=("distance", "cointegration"),
        sweeps=True, save=True)
    print(f"  {len(figs)} figures written.")
    if alpha_table is not None and not alpha_table.empty:
        save_table(pl.from_pandas(alpha_table), "asset_pricing_alphas")
        print("\n  Asset-pricing (market model) alphas:")
        print(alpha_table.to_string(index=False))

    print(f"\nTables  -> {research_config.TABLES_DIR}")
    print(f"Figures -> {research_config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
