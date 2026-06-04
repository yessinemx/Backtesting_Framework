"""Diagnostic: compare stats with/without dead tickers and per price source."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl
from config import config_paper as cfg

adj = pl.read_parquet(cfg.PAPER_PRICES_ADJUSTED_415_PATH)
raw = pl.read_parquet(cfg.PAPER_PRICES_RAW_415_PATH)
tickers = [c for c in adj.columns if c != "date"]
live_tickers = [t for t in tickers if not t[0].isdigit()]
dead_tickers = [t for t in tickers if t[0].isdigit()]
print(f"Total: {len(tickers)}  |  Dead: {len(dead_tickers)}  |  Live: {len(live_tickers)}")

def compute_stats(prices, ticker_list):
    rows = []
    for t in ticker_list:
        if t not in prices.columns:
            continue
        s = prices[t].drop_nulls()
        if s.len() < 20:
            continue
        r = (s / s.shift(1) - 1).drop_nulls()
        rows.append({
            "ticker": t,
            "mean_ann": float(r.mean()) * 252 * 100,
            "std_ann":  float(r.std()) * (252 ** 0.5) * 100,
            "skew": float(r.skew()),
            "kurt": float(r.kurtosis()),
            "max_ret": float(r.max()) * 100,
        })
    return pl.DataFrame(rows)

def print_summary(label, df):
    print(f"\n{label} (n={df.height})")
    print(f"  mean: {float(df['mean_ann'].mean()):6.2f}%   paper: 15.91%")
    print(f"  std:  {float(df['std_ann'].mean()):6.2f}%   paper: 16.00%")
    print(f"  skew: {float(df['skew'].mean()):7.4f}    paper: -0.3924")
    print(f"  kurt: {float(df['kurt'].mean()):7.4f}    paper:  8.0039")
    pcts = [float(df["std_ann"].quantile(q)) for q in [0.25, 0.5, 0.75, 0.9]]
    print(f"  std p25/p50/p75/p90: {pcts[0]:.1f}% / {pcts[1]:.1f}% / {pcts[2]:.1f}% / {pcts[3]:.1f}%")

print_summary("ADJ all 415",  compute_stats(adj, tickers))
print_summary("ADJ live 388", compute_stats(adj, live_tickers))
print_summary("RAW live 388", compute_stats(raw, live_tickers))

# Show dead tickers stats
dead_stats = compute_stats(adj, dead_tickers)
print(f"\nDead tickers stats (n={dead_stats.height})")
print(dead_stats.sort("std_ann", descending=True).select(["ticker", "mean_ann", "std_ann", "skew", "max_ret"]))
