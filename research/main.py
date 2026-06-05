"""Unified entry point for the wavelet pairs-trading paper replication.

Two run modes share the same module:

* **Default (lightweight)** — backwards-compatible CLI used by the Streamlit app
  (Step 7) and the GitHub Actions smoke test. It runs `run_pipeline` for one
  method/wavelet combination and prints a one-line summary. CLI flags let the
  caller override every parameter from `config.config_paper.PAIRS_CONFIG`.

* **Full replication** (``--full``) — reproduces every paper-numbered table and
  figure end-to-end via `build_report` + `save_paper_outputs`. This is the
  long-running path that exercises both methods, both wavelet variants, the
  sweeps, and the asset-pricing regressions. Outputs land in
  `research/outputs/{tables,figures}/`.

Examples
--------

    py research/main.py --max-periods 1 --no-write-outputs --quiet   # smoke
    py research/main.py --full                                       # full run
    py research/main.py --full --no-sweeps                           # faster
    py research/main.py --full --tc-sweep 0.0 0.001 0.005            # Tables 18-21
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Force UTF-8 on stdout/stderr so Polars Unicode box-drawing characters do not
# crash under Windows' default cp1252 console when output is piped or tee'd.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from config import config_paper as research_config
from research.paper_replication import run_pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the wavelet pairs-trading replication. Default mode is a single "
            "method/wavelet smoke run; pass --full for the full paper reproduction."
        )
    )
    parser.add_argument("--source", choices=["data", "bloomberg", "paper_data"],
                        default="data", help="Price source used by loaders (default: data).")
    parser.add_argument("--index-id", default=research_config.PAIRS_CONFIG.get("index_id"),
                        help="Universe id (SPX, NDX, SX5E, ...).")
    parser.add_argument("--method", choices=["distance", "cointegration"],
                        default=research_config.PAIRS_CONFIG["method"],
                        help="Pair-selection method (single-method mode).")
    parser.add_argument("--wavelet", default=research_config.PAIRS_CONFIG["wavelet"],
                        help="Wavelet family.")
    parser.add_argument("--top-n", type=int, default=research_config.PAIRS_CONFIG["top_n"],
                        help="Number of selected pairs (distance ranking).")
    parser.add_argument("--candidate-pool", type=int,
                        default=research_config.PAIRS_CONFIG["candidate_pool"],
                        help="Pre-filter pool size before cointegration.")
    parser.add_argument("--block-size", type=int,
                        default=research_config.PAIRS_CONFIG["block_size"],
                        help="Formation/trading block size (business days).")
    parser.add_argument("--threshold-sigma", type=float,
                        default=research_config.PAIRS_CONFIG["threshold_sigma"],
                        help="Entry threshold in spread standard deviations.")
    parser.add_argument("--tc-per-share", type=float,
                        default=research_config.PAIRS_CONFIG["tc_per_share"],
                        help="Transaction cost per share / round-trip proxy.")
    parser.add_argument("--max-periods", type=int,
                        default=research_config.PAIRS_CONFIG.get("max_periods"),
                        help="Optional cap on the number of rolling periods.")
    parser.add_argument("--start-date", default=research_config.PAIRS_CONFIG.get("start_date"),
                        help="First date of the price sample, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=research_config.PAIRS_CONFIG.get("end_date"),
                        help="Last date of the price sample, YYYY-MM-DD.")
    parser.add_argument("--no-write-outputs", action="store_true",
                        help="Run without writing tables/figures (single-method mode).")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable per-period progress logs.")

    full = parser.add_argument_group("Full paper replication (--full)")
    full.add_argument("--full", action="store_true",
                      help="Run the full multi-method replication and save all paper tables/figures.")
    full.add_argument("--no-sweeps", action="store_true",
                      help="Skip the wavelet-class and trading-horizon sweeps (faster).")
    full.add_argument("--tc-sweep", type=float, nargs="+", metavar="TC",
                      help="List of tc_per_share values to run for Tables 18-21 (e.g. 0.0 0.001 0.005).")
    return parser


def _build_params(args: argparse.Namespace) -> dict:
    params = dict(research_config.PAIRS_CONFIG)
    params.update({
        "index_id": args.index_id,
        "method": args.method,
        "wavelet": args.wavelet,
        "top_n": args.top_n,
        "candidate_pool": args.candidate_pool,
        "block_size": args.block_size,
        "threshold_sigma": args.threshold_sigma,
        "tc_per_share": args.tc_per_share,
        "max_periods": args.max_periods,
        "start_date": args.start_date,
        "end_date": args.end_date,
    })
    return params


# ---------------------------------------------------------------------------
# Single-method (lightweight) mode
# ---------------------------------------------------------------------------
def _print_summary(summary: pl.DataFrame, params: dict, write_outputs: bool) -> None:
    print("\n=== Pairs replication summary ===")
    print(f"method={params['method']} wavelet={params['wavelet']} index_id={params['index_id']}")
    if summary.is_empty():
        print("No periods were produced. Check data coverage and configuration.")
        return
    print(summary)
    if write_outputs:
        print(f"\nTables  : {research_config.TABLES_DIR}")
        print(f"Figures : {research_config.FIGURES_DIR}")


def _run_single_method(args: argparse.Namespace) -> None:
    params = _build_params(args)
    result = run_pipeline(
        params=params, source=args.source, verbose=not args.quiet,
        write_outputs=not args.no_write_outputs,
    )
    _print_summary(result["summary"], params, write_outputs=not args.no_write_outputs)


# ---------------------------------------------------------------------------
# Full paper replication
# ---------------------------------------------------------------------------
def _run_full(args: argparse.Namespace) -> None:
    # Local imports keep --help / single-method runs lightweight.
    from research.paper_replication.report import build_report
    from research.paper_replication.outputs.paper_outputs import save_paper_outputs

    print("Running full point-in-time S&P 500 replication "
          "(both methods, all variants, all figures)...\n")

    params = _build_params(args)
    sweeps = not args.no_sweeps
    tc_values: Sequence[float] | None = args.tc_sweep

    report = build_report(
        source=args.source if args.source != "data" else "paper_data",
        methods=research_config.DEFAULT_METHODS,
        params=params, with_figures=True, sweeps=sweeps, save_figures=False,
        tc_sweep=tc_values,
    )
    written = save_paper_outputs(report)

    print(f"Universe pool: {report['universe_pool']} tickers | "
          f"{report['n_periods']} periods\n")

    for method, summary in report["summaries"].items():
        print(f"{'='*70}")
        print(f"  {method.upper()}  (avg over {report['n_periods']} periods, before costs)")
        print(f"{'='*70}")
        with pl.Config(tbl_cols=8, tbl_width_chars=160, float_precision=4):
            print(summary.select(["variant", "mean_return", "sharpe", "pct_positive",
                                  "n_full", "n_partial", "n_non"]))
        print()

    print(f"{'='*70}\n  PAPER vs REPLICATION  (Tables 4 & 5)\n{'='*70}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=240, float_precision=2):
        print(report["comparison"])

    if report["alpha_table"] is not None and not report["alpha_table"].empty:
        print(f"\n{'='*70}\n  Asset-pricing alphas (Tables 13/14/15)\n{'='*70}")
        print(report["alpha_table"].head(20).to_string(index=False))

    n_tables = len(written["tables"])
    n_figures = len(written["figures"])
    print(f"\n{n_figures} static figures + {n_tables} numbered tables written.")
    print(f"Tables  -> {research_config.TABLES_DIR}")
    print(f"Figures -> {research_config.FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.full:
        _run_full(args)
    else:
        _run_single_method(args)


if __name__ == "__main__":
    main()
