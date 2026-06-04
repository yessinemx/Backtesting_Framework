"""Quick comparison: replication vs paper (no figures, no sweeps)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl
from config import config_paper as cfg
from research.paper_replication.report import build_report

report = build_report(
    source="paper_data",
    methods=cfg.DEFAULT_METHODS,
    with_figures=False,
    sweeps=False,
)

print(f"\nUniverse pool: {report['universe_pool']} tickers | {report['n_periods']} periods\n")

for method, summary in report["summaries"].items():
    print(f"{'='*60}")
    print(f"  {method.upper()} (avg over {report['n_periods']} periods)")
    print(f"{'='*60}")
    with pl.Config(tbl_cols=8, tbl_width_chars=160, float_precision=4):
        print(summary.select(["variant", "mean_return", "sharpe", "pct_positive",
                               "n_full", "n_partial", "n_non"]))
    print()

print(f"{'='*60}")
print("  PAPER vs REPLICATION")
print(f"{'='*60}")
with pl.Config(tbl_cols=-1, tbl_width_chars=240, float_precision=2):
    print(report["comparison"])
