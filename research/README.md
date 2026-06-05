# Paper Replication — Wavelet Pairs Trading

Local replication of *“Pairs trading with wavelet transform”* (Eroglu, Yener & Yigit, 2023, **Quantitative Finance**) using Bloomberg point-in-time S&P 500 data.

## Layout

```
research/
├── README.md                          # this file
├── main.py                            # unified CLI (single-method + --full replication)
├── paper_replication/                 # library package
│   ├── __init__.py                    # flat public API re-exports
│   ├── report.py                      # end-to-end report builder (CLI + Streamlit)
│   ├── core/                          # per-period engine and primitives
│   │   ├── pipeline.py                # per-period orchestrator
│   │   ├── selection.py               # distance / cointegration pair selection
│   │   ├── spread.py                  # standard vs MODWT wavelet spread
│   │   ├── trading.py                 # threshold trading + transaction costs
│   │   ├── wavelet.py                 # MODWT (sym20/sym22 default)
│   │   ├── metrics.py                 # PairsReport aggregator
│   │   └── periods.py                 # formation / trading window builder
│   ├── analytics/
│   │   └── asset_pricing.py           # FF / q-factor / Petkova regressions
│   ├── outputs/                       # rendering and persistence
│   │   ├── figures.py                 # paper figures 1-12
│   │   ├── paper_outputs.py           # 22 tables + paper_validation_summary
│   │   ├── output_writer.py           # CSV / PNG persistence
│   │   └── png_export.py              # plotly -> matplotlib renderer (paper style)
│   └── bootstrap/
│       └── bootstrap_data.py          # Bloomberg one-shot loader for research/data/
├── slides/                            # Beamer presentation (compile to PDF)
│   ├── presentation.tex               # paper + strategy + results
│   └── compile.ps1                    # one-line pdflatex helper
├── data/                              # paper data slice + factors.csv (tracked CSVs)
├── outputs/                           # tables/, figures/ (gitignored)
└── docs/                              # paper PDF + design notes
```

## End-to-end run

```powershell
py research/main.py --full
```

Useful flags:

```powershell
py research/main.py --full --no-sweeps                      # faster (skips Tables 16, 17)
py research/main.py --full --tc-sweep 0.0 0.001 0.005       # populates Tables 18-21
py research/main.py                                         # single-method smoke run
```

This produces, under `research/outputs/`:

- `tables/table1_…csv` through `table22_…csv` (paper-numbered)
- `tables/paper_validation_summary.csv` (PASS / WARN / FAIL vs paper headline numbers)
- `tables/asset_pricing_alphas.csv`
- `figures/figure1_…png` through `figure12_…png` (matplotlib, serif, paper-style)

## Data bootstrap

The paper data slice (`research/data/*.parquet`) is generated from Bloomberg:

```powershell
py research/paper_replication/bootstrap/bootstrap_data.py
```

Requires a working Bloomberg session (`blpapi` installed and Bloomberg Terminal running).

## Factor data for Tables 13 / 14 / 21

`research/data/factors.csv` ships with the repo (committed slice from the
original MATLAB `FMdata.mat`, includes Petkova ICAPM state variables). The
asset-pricing loader picks it up automatically via
`PAPER_FACTOR_CANDIDATE_PATHS` in `config/config_paper.py`.

If you need to refresh it (e.g. extend the sample), regenerate FF + q only via:

```powershell
py research/paper_replication/bootstrap/fetch_factors.py
```

This downloads daily factors from Kenneth French's data library and global-q.org.
Petkova state variables (TERM, DEF, DIV, TBILL) are not refetched — keep the
committed CSV's columns for Tables 14 (Petkova) and 21.

## Rigorous validation

`paper_validation_summary.csv` contains:

| category   | rows | check                                                            |
| ---------- | ---- | ---------------------------------------------------------------- |
| headline   | 12   | mean_return_% and Sharpe vs paper Tables 4-5, per method/variant |
| structural | 3    | universe pool size, number of periods, factor regressions found  |

Tolerances (calibrated to the paper magnitudes):

| metric            | PASS    | WARN    |
| ----------------- | ------- | ------- |
| `mean_return_pct` | ±1.0 pp | ±3.0 pp |
| `sharpe`          | ±0.30   | ±0.80   |

`FAIL` is anything beyond `WARN`. `MISSING` is emitted when a variant was not
produced by the current run.

## Currently missing (vs. the 23-table paper)

All 22 numbered tables and 12 figures are now reproduced end-to-end. Some require
opt-in flags:

| Item            | Status  | Why                                                          |
| --------------- | ------- | ------------------------------------------------------------ |
| Tables 13/14/21 | opt-in  | need a factor file (see *Factor data* above)                 |
| Tables 16-17    | default | reproduced when `--no-sweeps` is **not** passed              |
| Tables 18-21    | opt-in  | reproduced with `--tc-sweep 0.0 0.001 0.005`                 |
| Tables 10/12    | default | PCA + Monte-Carlo simulation, pure Python (no MATLAB needed) |
| Table 22        | default | forced-close standard pairs, pure Python (no MATLAB needed)  |

Paper Table 23 (latency-adjusted Sharpe) is not in the published replication
appendix and is intentionally not reproduced.

## Tests

```powershell
py -m pytest -q
```
