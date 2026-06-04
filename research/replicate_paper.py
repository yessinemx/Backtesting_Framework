"""
Faithful replication driver for "Pairs trading with wavelet transform"
(Eroglu, Yener & Yigit, 2023, Quantitative Finance).

Universe : S&P 500 (SPX), point-in-time membership per formation window.
Window   : 2010-03-05 -> 2018-03-15  ->  8 blocks of 252 days = 7 periods (Table 1).
Methods  : minimum distance and cointegration (Johansen).
Spread   : standard vs sym22 wavelet (MODWT scaling coefficient V_1, eq. 3), 2 sigma
           threshold from the formation window, per-window normalization.
Variants : standard, wavelet, and "Opt" (look-ahead benchmark, beta fit on the
           trading window — Table 11 last row; NOT tradeable).

All tables and figures are written to research/outputs/.

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

from config import config_paper as research_config
from research.paper_replication.paper_outputs import save_paper_outputs
from research.paper_replication.report import build_report


def main():
    print("Running point-in-time S&P 500 replication (this includes the look-ahead Opt "
          "benchmark and all figures; the wavelet-class/horizon sweeps make it ~5 min)...\n")
    params = dict(research_config.PAIRS_CONFIG)
    params["tc_per_share"] = research_config.HEADLINE_TC_PER_SHARE

    report = build_report(
        source="paper_data", methods=research_config.DEFAULT_METHODS,
        params=params, with_figures=True, sweeps=True, save_figures=False,
    )
    written = save_paper_outputs(report)

    print(f"Universe pool: {report['universe_pool']} tickers | {report['n_periods']} periods\n")

    # Console summary.
    for method, summary in report["summaries"].items():
        print(f"{'='*70}\n  {method.upper()}  (avg over {report['n_periods']} periods, before costs)\n{'='*70}")
        with pl.Config(tbl_cols=8, tbl_width_chars=160, float_precision=4):
            print(summary.select(["variant", "mean_return", "sharpe", "pct_positive",
                                  "n_full", "n_partial", "n_non"]))
        print()

    print(f"{'='*70}\n  PAPER vs REPLICATION (+ Opt look-ahead benchmark)\n{'='*70}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=240, float_precision=2):
        print(report["comparison"])

    if report["alpha_table"] is not None and not report["alpha_table"].empty:
        print(f"\n{'='*70}\n  Asset-pricing (market-model) alphas, Section 5.4\n{'='*70}")
        print(report["alpha_table"].to_string(index=False))

    n_tables = len(written["tables"])
    n_figures = len(written["figures"])
    print(f"\n{n_figures} static figures + {n_tables} numbered tables written.")
    print(f"Tables  -> {research_config.TABLES_DIR}")
    print(f"Figures -> {research_config.FIGURES_DIR}")
    print("\nNote: 'Opt' is the paper's hypothetical look-ahead upper bound "
          "(beta fit on the trading period). It is NOT a tradeable strategy.")


if __name__ == "__main__":
    main()
