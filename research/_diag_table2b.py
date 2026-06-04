"""Diagnostic: test different interpretations of Table 2 vs paper.

Tests:
  A) Per-ticker stats, equal-weight sector average (current approach)
  B) Equal-weight SECTOR PORTFOLIO returns, then portfolio stats
  C) Per-ticker cross-sectional std averaged over time
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl
import numpy as np
from config import config_paper as cfg

PAPER_TOTAL = {"n": 415, "mean": 15.91, "std": 16.00, "skew": -0.3924, "kurt": 8.0039}
PAPER_SECTORS = {
    "Communication Services":  (15, 19.25, 16.90, -0.4116, 5.9928),
    "Consumer Discretionary":  (51, 17.71, 17.97, -0.3172, 6.2322),
    "Consumer Staples":        (26, 13.16, 12.24, -0.3128, 5.3612),
    "Financials":              (60, 14.91, 20.59, -0.3279, 7.9465),
    "Health Care":             (55, 20.01, 16.23, -0.4824, 6.0463),
}  # subset for brevity

prices = pl.read_parquet(cfg.PAPER_PRICES_ADJUSTED_415_PATH)
universe = pl.read_csv(cfg.PAPER_UNIVERSE_415_TICKERS_PATH)
date_col = "date"
tickers = [c for c in prices.columns if c != date_col]

# --- Build returns frame ---
rets = prices.with_columns([
    ((pl.col(c) / pl.col(c).shift(1)) - 1.0).alias(c) for c in tickers
]).slice(1)

# Sector map
sector_map = {r["ticker"]: r["gics_sector"] for r in universe.to_dicts()}
sectors = sorted(set(sector_map.values()))

# ---- METHOD A: per-ticker stats averaged within sector (CURRENT approach) ----
print("\n=== METHOD A: per-ticker stats, sector average (current) ===")
A_rows = []
for t in tickers:
    s = rets.get_column(t).drop_nulls()
    if s.len() < 20:
        continue
    A_rows.append({
        "ticker": t,
        "sector": sector_map.get(t, "Unknown"),
        "mean": float(s.mean()) * 252 * 100,
        "std":  float(s.std()) * (252 ** 0.5) * 100,
        "skew": float(s.skew()),
        "kurt": float(s.kurtosis()),
    })
A = pl.DataFrame(A_rows)
total_A = A.select([pl.col("mean").mean(), pl.col("std").mean(),
                    pl.col("skew").mean(), pl.col("kurt").mean()])
print(f"TOTAL:  mean={float(total_A['mean'][0]):.2f}%  std={float(total_A['std'][0]):.2f}%  "
      f"skew={float(total_A['skew'][0]):.4f}  kurt={float(total_A['kurt'][0]):.4f}")
print(f"PAPER:  mean={PAPER_TOTAL['mean']:.2f}%  std={PAPER_TOTAL['std']:.2f}%  "
      f"skew={PAPER_TOTAL['skew']:.4f}  kurt={PAPER_TOTAL['kurt']:.4f}")

# ---- METHOD B: equal-weight sector portfolio returns ----
print("\n=== METHOD B: equal-weight sector portfolio returns ===")
B_sector_stats = []
for sec in sectors:
    sec_tickers = [t for t in tickers if sector_map.get(t) == sec and t in rets.columns]
    if len(sec_tickers) < 2:
        continue
    # Build equal-weight portfolio: mean across tickers at each day
    port_ret = rets.select(sec_tickers).mean_horizontal().drop_nulls()
    B_sector_stats.append({
        "sector": sec,
        "n": len(sec_tickers),
        "mean": float(port_ret.mean()) * 252 * 100,
        "std":  float(port_ret.std()) * (252 ** 0.5) * 100,
        "skew": float(port_ret.skew()),
        "kurt": float(port_ret.kurtosis()),
    })
B = pl.DataFrame(B_sector_stats)

# Print for selected sectors
for sec, (pn, pm, ps, psk, pku) in PAPER_SECTORS.items():
    row = B.filter(pl.col("sector") == sec)
    if row.height == 0:
        continue
    r = row.to_dicts()[0]
    print(f"{sec:<30}  local({r['n']:>3}): mean={r['mean']:6.2f}% std={r['std']:6.2f}% skew={r['skew']:7.4f} kurt={r['kurt']:7.4f}")
    print(f"{'':30}  paper({pn:>3}): mean={pm:6.2f}% std={ps:6.2f}% skew={psk:7.4f} kurt={pku:7.4f}")

# Global equal-weight portfolio
all_port = rets.select(tickers).mean_horizontal().drop_nulls()
print(f"\nGlobal EW portfolio (415 stocks):")
print(f"  mean={float(all_port.mean())*252*100:.2f}%  std={float(all_port.std())*(252**0.5)*100:.2f}%  "
      f"skew={float(all_port.skew()):.4f}  kurt={float(all_port.kurtosis()):.4f}")
print(f"PAPER:  mean={PAPER_TOTAL['mean']:.2f}%  std={PAPER_TOTAL['std']:.2f}%  "
      f"skew={PAPER_TOTAL['skew']:.4f}  kurt={PAPER_TOTAL['kurt']:.4f}")

# ---- METHOD C: time-average of cross-sectional std per sector ----
print("\n=== METHOD C: time-average of cross-sectional std (dispersion) ===")
for sec, (pn, pm, ps, psk, pku) in PAPER_SECTORS.items():
    sec_tickers = [t for t in tickers if sector_map.get(t) == sec and t in rets.columns]
    if len(sec_tickers) < 2:
        continue
    sec_rets = rets.select(sec_tickers).to_numpy()
    # Cross-sectional mean and std at each time step
    cs_mean = np.nanmean(sec_rets, axis=1)
    cs_std  = np.nanstd(sec_rets, axis=1, ddof=1)
    # Annualise
    mean_ann = np.nanmean(cs_mean) * 252 * 100
    std_ann  = np.nanmean(cs_std) * (252 ** 0.5) * 100
    print(f"{sec:<30} CS: mean={mean_ann:6.2f}% std={std_ann:6.2f}%  |  paper: mean={pm:.2f}% std={ps:.2f}%")
