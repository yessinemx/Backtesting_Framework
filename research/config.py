from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = Path(__file__).resolve().parent
DOCS_DIR = RESEARCH_DIR / "docs"
OUTPUTS_DIR = RESEARCH_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Fixed parameters for the paper replication workflow.
PAIRS_CONFIG = {
    # Eroglu/Yener/Yigit (2023) "Pairs trading with wavelet transform"
    # Paper specifies sym22 (length-44, 22 vanishing moments, least-asymmetric/
    # near-linear phase). PyWavelets implements Symlets only up to sym20, the
    # highest available member of the SAME least-asymmetric family. Table 16 of
    # the paper shows results are stable across high-order Symlets (sym20 ~ sym22).
    # NOTE: db22 is NOT an acceptable substitute - it is extremal-phase /
    # highly asymmetric and produces a large phase lag that breaks the strategy.
    "wavelet": "sym20",
    "block_size": 252,           # 252 NYSE trading days = 1 financial year
    "method": "distance",        # "distance" or "cointegration"
    "top_n": 1000,               # min-distance: top-1000 pairs
    "candidate_pool": 2000,      # cointegration pre-filter pool
    "k_ar_diff": 1,              # Johansen lag
    "threshold_sigma": 2.0,      # 2-sigma trade trigger
    "tc_per_share": 0.001,       # 10 bps half-turn per share (Sec. 5.5.3)
    "index_id": "SPX",           # paper universe = S&P 500
    "max_beta": 10.0,            # defensive cap on |beta| to avoid scale-extreme pairs
    # Use RAW (unadjusted) close prices from data/prices_raw.parquet instead of
    # the total-return-adjusted prices.parquet. Total-return-adjusted series are
    # too smooth for level-1 MODWT filtering to correct the regression
    # coefficients, so the paper's wavelet effect cannot appear. Requires running
    # `py extraction/refresh_raw.py SPX` first to populate the raw file.
    "raw_prices": True,
    # Paper Table 1: 7 periods anchored to exact calendar dates.
    # Each tuple = (train_start, train_end, trade_start, trade_end), YYYY-MM-DD.
    "paper_periods": [
        ("2010-03-05", "2011-03-03", "2011-03-04", "2012-03-05"),
        ("2011-03-04", "2012-03-05", "2012-03-06", "2013-03-08"),
        ("2012-03-06", "2013-03-08", "2013-03-11", "2014-03-10"),
        ("2013-03-11", "2014-03-10", "2014-03-11", "2015-03-12"),
        ("2014-03-11", "2015-03-12", "2015-03-13", "2016-03-15"),
        ("2015-03-13", "2016-03-15", "2016-03-16", "2017-03-15"),
        ("2016-03-16", "2017-03-15", "2017-03-16", "2018-03-15"),
    ],
    # Padding for data load (slightly before period-1 train start, slightly after period-7 trade end).
    "start_date": "2010-03-01",
    "end_date":   "2018-03-31",
    "max_periods": None,         # use all 7 hardcoded periods
}