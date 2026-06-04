"""Standalone runner for the wavelet pairs-trading paper replication."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config_paper as research_config
from research.paper_replication import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone replication pipeline for the paper "
            "'Pairs trading with wavelet transform'."
        )
    )
    parser.add_argument(
        "--source",
        choices=["data", "bloomberg"],
        default="data",
        help="Price source used by loaders (default: data).",
    )
    parser.add_argument(
        "--index-id",
        default=research_config.PAIRS_CONFIG.get("index_id"),
        help="Restrict the universe to one index id, e.g. SPX, NDX, SX5E.",
    )
    parser.add_argument(
        "--method",
        choices=["distance", "cointegration"],
        default=research_config.PAIRS_CONFIG["method"],
        help="Pair-selection method.",
    )
    parser.add_argument(
        "--wavelet",
        default=research_config.PAIRS_CONFIG["wavelet"],
        help="Wavelet family used by the replication pipeline.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=research_config.PAIRS_CONFIG["top_n"],
        help="Number of selected pairs for the distance ranking.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=research_config.PAIRS_CONFIG["candidate_pool"],
        help="Pre-filter pool size before the cointegration test.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=research_config.PAIRS_CONFIG["block_size"],
        help="Formation/trading block size in business days.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=research_config.PAIRS_CONFIG["threshold_sigma"],
        help="Entry threshold expressed in spread standard deviations.",
    )
    parser.add_argument(
        "--tc-per-share",
        type=float,
        default=research_config.PAIRS_CONFIG["tc_per_share"],
        help="Transaction cost per share / round trip proxy used in the paper.",
    )
    parser.add_argument(
        "--max-periods",
        type=int,
        default=research_config.PAIRS_CONFIG.get("max_periods"),
        help="Optional cap on the number of rolling periods.",
    )
    parser.add_argument(
        "--start-date",
        default=research_config.PAIRS_CONFIG.get("start_date"),
        help="First date (inclusive) of the price sample, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=research_config.PAIRS_CONFIG.get("end_date"),
        help="Last date (inclusive) of the price sample, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--no-write-outputs",
        action="store_true",
        help="Run the replication without writing tables and figures to research/outputs/.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-period progress logs.",
    )
    return parser


def _build_params(args: argparse.Namespace) -> dict:
    params = dict(research_config.PAIRS_CONFIG)
    params.update(
        {
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
        }
    )
    return params


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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    params = _build_params(args)
    result = run_pipeline(
        params=params,
        source=args.source,
        verbose=not args.quiet,
        write_outputs=not args.no_write_outputs,
    )
    _print_summary(result["summary"], params, write_outputs=not args.no_write_outputs)


if __name__ == "__main__":
    main()