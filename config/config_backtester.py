from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

PRICES_PATH = DATA_DIR / "prices.parquet"
RETURNS_PATH = DATA_DIR / "returns.parquet"
RAW_PRICES_PATH = DATA_DIR / "prices_raw.parquet"
RAW_RETURNS_PATH = DATA_DIR / "returns_raw.parquet"
MEMBERSHIP_PATH = DATA_DIR / "membership.parquet"
RISKFREE_PATH = DATA_DIR / "riskfree.parquet"

# Transaction costs (one-way, in bps)
TRANSACTION_COST_BPS = 2

# Annual stock borrow cost for short positions (in bps)
SHORT_BORROW_BPS = 50

# Time periods
DATA_START = "2000-01-01"
DATA_END = "2025-12-31"

IS_START = "2000-01-01"
IS_END = "2010-12-31"

OOS_START = "2011-01-01"
OOS_END = "2025-12-31"

# Index universe
INDEX_CONFIG = {
    "SPX": {
        "bbg_ticker": "SPX Index",
        "name": "S&P 500",
        "currency": "USD",
    },
    "NDX": {
        "bbg_ticker": "NDX Index",
        "name": "Nasdaq 100",
        "currency": "USD",
    },
    "UKX": {
        "bbg_ticker": "UKX Index",
        "name": "FTSE 100",
        "currency": "GBP",
    },
    "SX5E": {
        "bbg_ticker": "SX5E Index",
        "name": "Euro Stoxx 50",
        "currency": "EUR",
    },
}

# Risk-free rate tickers
RISKFREE_CONFIG = {
    "USD": {
        "tickers": ["SOFRRATE Index", "FEDL01 Index"],
        "day_count": 360,
    },
    "EUR": {
        "tickers": ["ESTRON Index", "EONIA Index"],
        "day_count": 360,
    },
    "GBP": {
        "tickers": ["SONIO/N Index"],
        "day_count": 365,
    },
}

INDEX_CURRENCY = {k: v["currency"] for k, v in INDEX_CONFIG.items()}

# Rebalance frequencies
REBALANCE_FREQS = {
    "1M": 1,
    "3M": 3,
    "6M": 6,
    "12M": 12,
}

# Parameter grids for in-sample grid search
MA_PARAM_GRID = {
    "fast_window": [10, 20, 30, 40, 50],
    "slow_window": [50, 100, 150, 200],
    "signal_threshold": [0.00, 0.01, 0.02],
}

MOMENTUM_PARAM_GRID = {
    "lookback_period": [63, 126, 189, 252],
    "top_n": [5, 10, 15, 20, 30],
    "skip_recent": [0, 21],
}

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "PRICES_PATH",
    "RETURNS_PATH",
    "RAW_PRICES_PATH",
    "RAW_RETURNS_PATH",
    "MEMBERSHIP_PATH",
    "RISKFREE_PATH",
    "TRANSACTION_COST_BPS",
    "SHORT_BORROW_BPS",
    "DATA_START",
    "DATA_END",
    "IS_START",
    "IS_END",
    "OOS_START",
    "OOS_END",
    "INDEX_CONFIG",
    "RISKFREE_CONFIG",
    "INDEX_CURRENCY",
    "REBALANCE_FREQS",
    "MA_PARAM_GRID",
    "MOMENTUM_PARAM_GRID",
]