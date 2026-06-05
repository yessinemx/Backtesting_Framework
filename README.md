# StratArb X ⚡

Quantitative returns-based backtesting framework with Bloomberg extraction, a Streamlit interface, and a dedicated research area for the wavelet pairs-trading paper replication (Eroglu, Yener & Yigit, 2023).

## Overview

The repository has two scopes that share loaders and data but keep separate parameters and outputs:

- **Backtester** — generic multi-strategy engine accessible through the Streamlit app
- **Paper replication** — full reproduction of the wavelet pairs-trading paper, driven by `research/main.py`

Configuration is split accordingly:
- `config/config_backtester.py` — generic backtester settings
- `config/config_paper.py` — fixed paper replication parameters

## Structure

```
Backtesting_Framework/
├── config/                   # Centralized configuration package
├── app/                      # Streamlit application (StratArb X)
│   ├── main.py               # Entry point: py -m streamlit run app/main.py
│   ├── sidebar.py            # Step navigation
│   ├── registry.py           # Strategy / allocator registry
│   └── steps/                # 7-step wizard
├── allocation/               # Equal-weight and risk-parity allocators
├── extraction/               # Bloomberg extraction scripts + API wrapper
├── indicators/               # Performance and risk metrics
├── loaders/                  # Polars / parquet data loaders
├── optimization/             # Walk-forward grid search
├── portfolio/                # Returns-based backtest engine
├── signals/                  # Strategy implementations
│   ├── pairs_trading_base.py          # Abstract base (common params + interface)
│   ├── pairs_trading_wavelet.py       # MODWT level-1 (paper-aligned, main strategy)
│   ├── pairs_trading_cointegration.py # Engle-Granger benchmark
│   ├── pairs_trading_partial_cointegration.py  # Clegg/Kalman SSM benchmark
│   ├── moving_average.py
│   └── momentum.py
├── visualisation/            # Plotly charts for the backtester UI
├── research/                 # Paper replication
│   ├── main.py               # CLI (single-method or --full replication)
│   ├── paper_replication/    # Library: core/, analytics/, outputs/
│   ├── slides/               # Beamer presentation (Université Paris-Dauphine M272)
│   ├── data/                 # Paper data slice (gitignored)
│   ├── docs/                 # Source paper PDF
│   └── outputs/              # Generated tables + figures (gitignored)
├── tests/                    # 31 unit tests
├── data/                     # Shared parquet data
├── requirements.txt
└── README.md
```

## Installation

```bash
py -m pip install -r requirements.txt
```

Optional Bloomberg API:

```bash
py -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

## Usage

**Streamlit app:**

```bash
py -m streamlit run app/main.py
```

**Paper replication — full run:**

```bash
py research/main.py --full
```

**Paper replication — smoke test:**

```bash
py research/main.py --max-periods 1 --no-write-outputs --quiet
```

## Tests

```bash
py -m unittest discover -s tests
```

31 tests covering the backtest engine, strategy logic, paper pipeline, and output writer.

## Streamlit Workflow

The app follows a 6-step wizard:

| Step | Name             | Description                                      |
|------|------------------|--------------------------------------------------|
| 1    | Data Status      | Dataset checks and Bloomberg extraction          |
| 2    | Index Selection  | Universe and date range                          |
| 3    | Strategy & Alloc | Strategy and allocator selection                 |
| 4    | Parameters       | Parameter tuning, walk-forward grid search, OOS date picker |
| 5    | Execution        | Backtest run                                     |
| 6    | Results          | Metrics and charts                               |

**Pairs Trading mode** (Step 3): selecting "Pairs Trading" activates composite mode — the wavelet strategy (main) and the two cointegration benchmarks run simultaneously, with a comparison chart in Step 6.

## Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| Moving Average Crossover | `MovingAverageCrossover` | SMA short/long crossover |
| Momentum | `MomentumStrategy` | Cross-sectional momentum signal |
| Pairs Trading (Wavelet) | `PairsTradingWavelet` | **Minimum distance** on MODWT-smoothed prices (sym20 level-1) |
| Pairs Trading (Cointegration) | `PairsTradingCointegration` | Engle-Granger cointegration benchmark |
| Pairs Trading (Partial Cointegration) | `PairsTradingPartialCointegration` | Clegg/Kalman SSM benchmark |

> **Note on pair selection**: the Wavelet strategy uses the **minimum distance** method (Gatev et al. 2006) applied to MODWT-denoised log-prices, not cointegration. The cointegration-based methods are the two benchmark strategies.

All pairs trading classes inherit from `PairsTradingBase` which holds five shared parameters (see below).

### Parameter reference

#### Moving Average Crossover
| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 20 | Short MA window (days) |
| `slow_window` | 50 | Long MA window (days) |
| `signal_threshold` | 0.0 | Dead band: signal only if `fast > slow × (1 + thr)` |

#### Momentum
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_period` | 252 | Return lookback window (days) |
| `top_n` | 10 | Number of long positions (top performers) |
| `skip_recent` | 21 | Days skipped at the end to avoid short-term reversal |

#### Pairs Trading — common parameters (all variants)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `formation_period` | 252 | Rolling window (days) for pair selection and spread calibration |
| `top_n_pairs` | 20 | Number of pairs traded simultaneously |
| `entry_threshold` | 2.0 | Z-score to open a position |
| `exit_threshold` | 0.5 | Z-score to close a position (mean reversion) |
| `min_history` | 60 | Minimum observations required before generating signals |

#### Pairs Trading (Wavelet) — additional parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet` | `sym20` | Wavelet family used for MODWT smoothing (paper default) |
| `wavelet_level` | 1 | MODWT decomposition level (fixed at 1, per paper) |

#### Pairs Trading (Cointegration) — additional parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `correlation_top_k` | 200 | Pre-filter: keep top-K pairs by absolute return correlation before running the EG test |
| `coint_pvalue_max` | 0.05 | Maximum p-value for the Engle-Granger cointegration test |
| `max_holding_mult` | 2.0 | Maximum holding duration as a multiple of the estimated half-life |
| `reselect_every` | 5 | Re-run pair selection every N rebalances (cached in between) |

#### Pairs Trading (Partial Cointegration) — additional parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `correlation_top_k` | 100 | Pre-filter by correlation before running EG + Kalman MLE |
| `coint_pvalue_max` | 0.05 | EG test pre-filter threshold |
| `r2_mr_min` | 0.1 | Minimum mean-reversion quality $R^2_{MR}$ from the Clegg SSM |
| `max_holding_mult` | 2.0 | Maximum holding duration as a multiple of the estimated half-life |
| `reselect_every` | 5 | Re-run pair selection every N rebalances |

> **Why Partial Cointegration is slow**: pair selection runs a Kalman MLE (`scipy.optimize.minimize`, L-BFGS-B) per candidate pair. Reduce `correlation_top_k` (e.g. 50) and `top_n_pairs` (e.g. 5) to speed it up.

## Backtester Principles

- P&L computed from daily returns
- weights drift between rebalances; the day's return is applied before rebalancing
- point-in-time membership to limit survivorship bias
- real daily risk-free rates by currency
- stateful strategies (cointegration) are reset at the start of each `run()`

## Data and Artifacts

- `data/` — shared parquet files (versioned)
- `research/data/` — SPX paper slice 2010-03-05 to 2018-03-15 (gitignored)
- `research/outputs/` — generated tables (CSV) and figures (gitignored)

To regenerate the paper data from shared parquets: `research/prepare_paper_data.py`.
