"""
Paper replication package for the wavelet pairs-trading study.

Layout
------
- core/       : selection, spread, trading, wavelet, periods, pipeline, metrics
- analytics/  : asset_pricing (FF / q / Petkova-ICAPM regressions)
- outputs/    : figures, paper_outputs (numbered tables), png_export, output_writer
- bootstrap/  : bootstrap_data.py (Bloomberg dataset extraction)
- report.py   : end-to-end report builder used by main.py --full and Step 7
- __init__.py : flat public API re-exports (this file)

Public API:
    - run_pipeline / run_period : full pipeline / single period
    - build_periods             : split the calendar into 252-day periods
    - select_pairs              : minimum-distance or cointegration selection
    - build_spread              : standard or wavelet spreads with 2 sigma thresholds
    - simulate_pair             : simulate trading for a single pair
    - aggregate_metrics         : portfolio-level metrics
    - save_paper_outputs        : dump all paper tables/figures to disk
"""
from research.paper_replication.core.wavelet import (
    modwt_smooth,
    modwt_detail,
    filter_prices,
    DEFAULT_WAVELET,
)
from research.paper_replication.core.periods import build_periods, Period
from research.paper_replication.core.selection import (
    select_pairs,
    select_min_distance,
    select_cointegration,
)
from research.paper_replication.core.spread import build_spread, SpreadSpec
from research.paper_replication.core.trading import simulate_pair, PairResult, Trade
from research.paper_replication.core.metrics import aggregate_metrics, PairsReport
from research.paper_replication.core.pipeline import (
    run_pipeline,
    run_period,
    PipelineResult,
)
from research.paper_replication.outputs import output_writer, figures, paper_outputs
from research.paper_replication.outputs.output_writer import (
    save_table,
    save_figure,
    ensure_dirs,
    clear_outputs,
)
from research.paper_replication.outputs.paper_outputs import save_paper_outputs
from research.paper_replication.analytics import asset_pricing

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
    "save_paper_outputs",
    "run_pipeline",
    "run_period",
    "PipelineResult",
    "output_writer",
    "figures",
    "paper_outputs",
    "asset_pricing",
]
