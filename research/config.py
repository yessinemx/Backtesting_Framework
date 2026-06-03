from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = Path(__file__).resolve().parent
DOCS_DIR = RESEARCH_DIR / "docs"
OUTPUTS_DIR = RESEARCH_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Fixed parameters for the paper replication workflow.
PAIRS_CONFIG = {
    "wavelet": "sym22",
    "block_size": 252,
    "method": "distance",
    "top_n": 1000,
    "candidate_pool": 2000,
    "k_ar_diff": 1,
    "threshold_sigma": 2.0,
    "tc_per_share": 0.001,
    "index_id": None,
    "max_periods": None,
}