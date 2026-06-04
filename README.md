# Backtesting Framework

Quantitative returns-based backtesting framework with Bloomberg extraction, a Streamlit interface, and a separate research area for the wavelet pairs-trading paper replication.

## Overview

The repository is organized around two distinct scopes:

- `config/config_backtester.py`: global configuration for the generic backtester
- `config/config_paper.py`: fixed configuration for the paper replication workflow

The standard backtester and the paper replication share loaders and data, but keep separate parameters and outputs.

## Structure

```
Backtesting_Framework/
├── config/                           # Centralized configuration package
│   ├── __init__.py                   # Public generic config entrypoint (`import config`)
│   ├── config_backtester.py          # Generic backtester settings
│   └── config_paper.py               # Fixed paper replication settings
├── app/                              # Streamlit application
│   ├── main.py                       # Entry point : py -m streamlit run app/main.py
│   ├── sidebar.py                    # Sidebar navigation
│   ├── registry.py                   # Strategy / allocator registry
│   ├── data.py                       # Cached loaders for the app
│   ├── styles.py                     # CSS
│   └── steps/                        # Wizard: data -> backtest -> paper replication
├── allocation/                       # Allocation methods
├── extraction/                       # Bloomberg extraction + API wrapper
│   ├── bloomberg_api.py
│   ├── bbg_members.py
│   ├── bbg_returns.py
│   └── bbg_riskfree.py
├── indicators/                       # Performance / risk indicators
├── loaders/                          # Polars / Bloomberg / parquet loading
├── optimization/                     # Walk-forward grid search
├── portfolio/                        # Backtest engine
├── signals/                          # Framework strategies
│   ├── moving_average.py
│   ├── momentum.py
│   └── pairs_trading.py
├── visualisation/                    # Backtester figures
├── research/                         # Research and paper replication
│   ├── data/                         # Paper-specific data slice and reference tables
│   │   ├── spx_membership_paper.parquet
│   │   ├── spx_prices_adjusted_paper.parquet
│   │   ├── spx_prices_raw_paper.parquet
│   │   ├── spx_415_candidate_tickers.csv
│   │   ├── spx_prices_adjusted_paper_415.parquet
│   │   ├── spx_prices_raw_paper_415.parquet
│   │   ├── paper_periods_reference.csv
│   │   ├── paper_table2_reference.csv
│   │   ├── paper_universe_counts.csv
│   │   └── paper_universe_415_counts.csv
│   ├── main.py                       # Main paper replication CLI
│   ├── docs/                         # Source documentation / paper
│   ├── notebooks/                    # Research notebooks
│   │   └── table_viewer.ipynb        # Notebook to inspect all generated CSV tables
│   ├── outputs/                      # Generated tables and figures
│   ├── prepare_paper_data.py         # Builds the paper-specific data slice under research/data
│   ├── derive_paper_universe_415.py  # Derives a deterministic candidate fixed 415-ticker universe
│   └── paper_replication/            # Paper replication code
├── data/                             # Shared parquet data
├── requirements.txt
└── README.md
```

## Installation

On Windows:

```bash
py -m pip install -r requirements.txt
```

Optional Bloomberg API:

```bash
py -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

## Lancement

Streamlit application:

```bash
py -m streamlit run app/main.py
```

Paper replication smoke test:

```bash
py research/main.py --index-id SPX --max-periods 1 --no-write-outputs --quiet
```

## Tests et CI

Run the full test suite locally:

```bash
py -m unittest discover -s tests
```

GitHub Actions CI:

- workflow in [.github/workflows/tests.yml](.github/workflows/tests.yml)
- runs on `push` and `pull_request` to `main`
- validates the `unittest` suite and the `research/main.py --help` CLI startup

## Workflow Streamlit

The application follows a 7-step workflow:

1. `Data Status`: dataset checks and conditional Bloomberg extraction
2. `Index Selection`: universe selection
3. `Strategy & Alloc`: strategy / allocator selection
4. `Parameters`: parameter setup and optional grid search
5. `Execution`: backtest execution
6. `Results`: metrics and visualizations
7. `Paper Replication`: separate execution path for the paper replication

## Configurations

Global backtester configuration in [config/config_backtester.py](config/config_backtester.py):

- shared data paths
- index universes and currencies
- risk-free rates
- rebalance frequencies
- standard backtester parameter grids

Local research configuration in [config/config_paper.py](config/config_paper.py):

- `research/data` paths for the paper-specific dataset slice
- `research/docs` and `research/outputs` paths
- fixed paper replication parameters
- table / figure output settings
- notebook path for table review

## Data And Artifacts

The files in [data](data) and the PDF in [research/docs](research/docs) remain versioned so the repository stays reproducible without an external bootstrap step. Generated run artifacts should not be committed:

- [research/data](research/data) isolates the 2010-03-05 to 2018-03-15 SPX paper slice and paper reference tables
- [research/outputs](research/outputs) is reserved for generated outputs
- `research/outputs/tables` contains CSV only
- `research/outputs/tables` and `research/outputs/figures` are ignored by git
- data can be regenerated through the app `Data Status` step or via the Bloomberg extraction scripts

To rebuild the paper-specific data folder from the shared parquet files, run [research/prepare_paper_data.py](research/prepare_paper_data.py).

To review generated tables quickly, open [research/notebooks/table_viewer.ipynb](research/notebooks/table_viewer.ipynb).

## Available Strategies

- `Moving Average Crossover`
- `Momentum`
- `Pairs Trading (Wavelet)` in the framework

The full paper replication remains isolated in [research/paper_replication](research/paper_replication) to avoid mixing research logic with the generic backtest engine.

## Backtester Principles

- P&L is computed from daily returns
- weights drift between rebalances
- the day's return is applied before rebalancing
- point-in-time membership limits survivorship bias
- real daily risk-free rates are used by currency
- signals are computed on an expanding window
