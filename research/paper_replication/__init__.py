"""
Paper replication package for the wavelet pairs-trading study.

API publique :
    - run_pipeline      : exécute le pipeline complet (toutes périodes).
    - run_period        : exécute une seule période (formation, trading).
    - build_periods     : découpe le calendrier en périodes 252 jours.
    - select_pairs      : sélection distance minimale / cointégration.
    - build_spread      : spread standard ou wavelet + seuils 2σ.
    - simulate_pair     : simulation du trading d'une paire.
    - aggregate_metrics : métriques de portefeuille.
    - output_writer     : écriture des tables et figures sous research/outputs/.
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
