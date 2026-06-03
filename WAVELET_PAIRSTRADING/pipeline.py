"""
pipeline.py
===========
Main pipeline replicating Eroğlu, Yener & Yiğit (2023).

Runs 4 strategies over 7 training/trading periods:
  1. Minimum Distance   — standard prices
  2. Minimum Distance   — sym22 wavelet filtered prices
  3. Cointegration      — standard prices
  4. Cointegration      — sym22 wavelet filtered prices

Usage
-----
    python pipeline.py --prices path/to/sp500_prices.csv

CSV format expected:
    - First column: date (parsed as datetime)
    - Remaining columns: one per ticker (price levels, adjusted for dividends/splits)
    - 415 tickers, March 5 2010 – March 15 2018

Alternatively, call run_pipeline() programmatically.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from wavelet_filter import filter_price_matrix
from pair_selection import minimum_distance_pairs, cointegration_pairs
from spread_estimation import estimate_all_pairs
from trading_engine import run_trading_period
from performance import compute_period_metrics, summarise_results, print_table4, spread_stationarity_rate


# ── Period definitions (Table 1 of paper) ───────────────────────────────────
# Each period: 252 trading days
# We build these from the actual data calendar.
N_PERIODS = 7
PERIOD_LENGTH = 252  # trading days


def split_periods(prices: pd.DataFrame, n: int = N_PERIODS, length: int = PERIOD_LENGTH):
    """
    Split price DataFrame into n training+trading period pairs.
    Each period = `length` days. Training period i → Trading period i.
    Returns list of (train_prices, trade_prices) DataFrames.
    """
    periods = []
    for i in range(n):
        t0 = i * length
        t1 = t0 + length        # end of training
        t2 = t1 + length        # end of trading
        if t2 > len(prices):
            print(f"  Warning: period {i+1} trading end exceeds data length, truncating.")
            t2 = len(prices)
        train = prices.iloc[t0:t1]
        trade = prices.iloc[t1:t2]
        if len(train) < 50 or len(trade) < 20:
            break
        periods.append((train, trade))
        print(f"  Period {i+1}: Train {train.index[0].date()} – {train.index[-1].date()} | "
              f"Trade {trade.index[0].date()} – {trade.index[-1].date()} "
              f"({len(trade)} days)")
    return periods


def run_pipeline(
    prices: pd.DataFrame,
    wavelet_name: str = "sym22",
    n_md_pairs: int = 1000,
    theta: float = 0.001,
    run_cointegration: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full replication pipeline.

    Parameters
    ----------
    prices          : DataFrame of adjusted price levels (T × N)
    wavelet_name    : wavelet filter ('sym22' default, 'sym20' if sym22 unavailable)
    n_md_pairs      : number of pairs for minimum distance method (paper: 1000)
    theta           : transaction cost per share (paper: 10 bps = 0.001)
    run_cointegration : set False to skip Johansen tests (fast debug mode)
    verbose         : print progress

    Returns
    -------
    dict with keys:
        'md_standard', 'md_wavelet', 'coint_standard', 'coint_wavelet'
        Each maps to a list of period metric dicts.
    """
    print(f"\n{'='*60}")
    print(f"Pairs Trading with Wavelet Transform — Pipeline")
    print(f"Wavelet: {wavelet_name} | MD pairs: {n_md_pairs} | θ: {theta*100:.1f}bps")
    print(f"{'='*60}\n")

    # ── Split into periods ──────────────────────────────────────────────────
    print("Splitting data into training/trading periods...")
    periods = split_periods(prices)
    print(f"  → {len(periods)} periods created\n")

    results = {
        "md_standard": [],
        "md_wavelet": [],
        "coint_standard": [],
        "coint_wavelet": [],
    }

    stationarity = {
        "md_standard": [], "md_wavelet": [],
        "coint_standard": [], "coint_wavelet": [],
    }

    for period_idx, (train, trade) in enumerate(periods):
        period_num = period_idx + 1
        print(f"\n{'─'*50}")
        print(f"Period {period_num}/{len(periods)}")
        print(f"{'─'*50}")

        # ── Wavelet filter training AND trading prices ──────────────────────
        print(f"  Applying {wavelet_name} MODWT Level-1 filter...")
        train_filt = filter_price_matrix(train, wavelet=wavelet_name)
        trade_filt = filter_price_matrix(trade, wavelet=wavelet_name)

        # ── Available tickers (need both train & trade) ─────────────────────
        common_tickers = list(set(train.columns) & set(trade.columns))
        train = train[common_tickers]
        trade = trade[common_tickers]
        train_filt = train_filt[common_tickers]
        trade_filt = trade_filt[common_tickers]
        N = len(common_tickers)
        print(f"  {N} tickers in this period")

        # ══ MINIMUM DISTANCE ════════════════════════════════════════════════
        print(f"\n  [MD] Selecting top {n_md_pairs} pairs by distance...")
        md_pairs = minimum_distance_pairs(train, n_pairs=n_md_pairs)
        print(f"  [MD] {len(md_pairs)} pairs selected")

        for method_key, use_wavelet in [("md_standard", False), ("md_wavelet", True)]:
            tag = "wavelet" if use_wavelet else "standard"
            print(f"  [MD-{tag}] Estimating parameters...")

            params = estimate_all_pairs(
                md_pairs, train, train_filt, wavelet=use_wavelet
            )

            print(f"  [MD-{tag}] Trading...")
            period_results = run_trading_period(
                params, trade, trade_filt, wavelet=use_wavelet, theta=theta
            )

            n_total = len(md_pairs)
            metrics = compute_period_metrics(period_results, n_total)
            results[method_key].append(metrics)

            # Stationarity of trading-period spreads (Table 8)
            stat_rate = spread_stationarity_rate(
                params, trade, trade_filt, wavelet=use_wavelet
            )
            stationarity[method_key].append(stat_rate)

            print(f"  [MD-{tag}] Return: {metrics['mean_return']:+.2%}, "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f}, "
                  f"Stationary spreads: {stat_rate:.1%}")

        # ══ COINTEGRATION ════════════════════════════════════════════════════
        if run_cointegration:
            print(f"\n  [COINT] Johansen tests on {N*(N-1)//2} pairs (slow)...")
            coint_pairs = cointegration_pairs(train, verbose=verbose)
            print(f"  [COINT] {len(coint_pairs)} cointegrated pairs found")

            if len(coint_pairs) == 0:
                print("  [COINT] No cointegrated pairs found, skipping period.")
                results["coint_standard"].append({})
                results["coint_wavelet"].append({})
                continue

            for method_key, use_wavelet in [("coint_standard", False), ("coint_wavelet", True)]:
                tag = "wavelet" if use_wavelet else "standard"
                print(f"  [COINT-{tag}] Estimating parameters...")

                params = estimate_all_pairs(
                    coint_pairs, train, train_filt, wavelet=use_wavelet
                )

                print(f"  [COINT-{tag}] Trading...")
                period_results = run_trading_period(
                    params, trade, trade_filt, wavelet=use_wavelet, theta=theta
                )

                n_total = len(coint_pairs)
                metrics = compute_period_metrics(period_results, n_total)
                results[method_key].append(metrics)

                stat_rate = spread_stationarity_rate(
                    params, trade, trade_filt, wavelet=use_wavelet
                )
                stationarity[method_key].append(stat_rate)

                print(f"  [COINT-{tag}] Return: {metrics['mean_return']:+.2%}, "
                      f"Sharpe: {metrics['sharpe_ratio']:.3f}, "
                      f"Stationary spreads: {stat_rate:.1%}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (Mean across periods)")
    print(f"{'='*60}")

    for key in results:
        valid = [m for m in results[key] if m]
        if valid:
            mean_ret = np.mean([m["mean_return"] for m in valid])
            mean_sharpe = np.mean([m["sharpe_ratio"] for m in valid])
            print(f"  {key:20s}: Return = {mean_ret:+.2%}, Sharpe = {mean_sharpe:.3f}")

    # Print Table 4 & 5 style output
    if results["coint_standard"] and any(results["coint_standard"]):
        print_table4(
            [m for m in results["coint_standard"] if m],
            [m for m in results["md_standard"] if m],
        )
        print_table4(
            [m for m in results["coint_wavelet"] if m],
            [m for m in results["md_wavelet"] if m],
        )

    results["_stationarity"] = stationarity
    return results


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pairs Trading with Wavelet Transform — Eroğlu et al. (2023)"
    )
    parser.add_argument(
        "--prices", type=str, required=True,
        help="Path to CSV of adjusted price levels (dates as first column)"
    )
    parser.add_argument(
        "--wavelet", type=str, default="sym22",
        help="Wavelet filter name (default: sym22; use sym20 if sym22 unavailable)"
    )
    parser.add_argument(
        "--n_pairs", type=int, default=1000,
        help="Number of pairs for minimum distance method (default: 1000)"
    )
    parser.add_argument(
        "--theta", type=float, default=0.001,
        help="Transaction cost per share (default: 0.001 = 10 bps)"
    )
    parser.add_argument(
        "--no_coint", action="store_true",
        help="Skip cointegration (faster, for debugging)"
    )
    parser.add_argument(
        "--output", type=str, default="results.pkl",
        help="Output file for results dict (pickle)"
    )
    args = parser.parse_args()

    # Load prices
    print(f"Loading prices from {args.prices}...")
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="any")  # drop tickers with any NaN
    print(f"Loaded: {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} – {prices.index[-1].date()}")

    results = run_pipeline(
        prices,
        wavelet_name=args.wavelet,
        n_md_pairs=args.n_pairs,
        theta=args.theta,
        run_cointegration=not args.no_coint,
    )

    # Save
    import pickle
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
