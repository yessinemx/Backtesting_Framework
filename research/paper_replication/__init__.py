"""
Paper replication package for the wavelet pairs-trading study.

Public API:
    - run_pipeline: run the full pipeline across all periods.
    - run_period: run a single formation/trading period.
    - build_periods: split the calendar into 252-day periods.
    - select_pairs: select pairs by minimum distance or cointegration.
    - build_spread: build standard or wavelet spreads with 2σ thresholds.
    - simulate_pair: simulate trading for a single pair.
    - aggregate_metrics: portfolio-level metrics.
    - output_writer: write tables and figures into research/outputs/.
"""
from research.paper_replication.wavelet import (
    modwt_smooth,
    modwt_detail,
    filter_prices,
    DEFAULT_WAVELET,
)
from research.paper_replication.periods import build_periods, Period
from research.paper_replication.selection import (
    select_pairs,
    select_min_distance,
    select_cointegration,
)
from research.paper_replication.spread import build_spread, SpreadSpec
from research.paper_replication.trading import simulate_pair, PairResult, Trade
from research.paper_replication.metrics import aggregate_metrics, PairsReport
from research.paper_replication.output_writer import (
        save_table,
        save_figure,
        ensure_dirs,
        clear_outputs,
)
from research.paper_replication.pipeline import run_pipeline, run_period, PipelineResult

__all__ = [
    "modwt_smooth",
    "modwt_detail",
    "filter_prices",
    "DEFAULT_WAVELET",
    "build_periods",
    "Period",
    "select_pairs",
    "select_min_distance",
    "select_cointegration",
    "build_spread",
    "SpreadSpec",
    "simulate_pair",
    "PairResult",
    "Trade",
    "aggregate_metrics",
    "PairsReport",
    "save_table",
    "save_figure",
    "ensure_dirs",
    "clear_outputs",
    "run_pipeline",
    "run_period",
    "PipelineResult",
]
