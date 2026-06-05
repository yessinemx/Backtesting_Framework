"""Regenerate only Figures 2 & 3 (paper-style example pair) without rerunning
the whole replication pipeline.

Runs the distance method on the FIRST period only, picks the most active /
profitable wavelet pair as the example, and saves fig02_example_spread.png and
fig03_example_returns.png to research/outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from config import config_paper as research_config
from loaders import load_prices, members_asof
from research.paper_replication.core.periods import build_periods
from research.paper_replication.core.selection import select_pairs
from research.paper_replication.core.spread import build_spread
from research.paper_replication.core.trading import simulate_pair
from research.paper_replication.core.wavelet import DEFAULT_WAVELET
from research.paper_replication.outputs.figures import (
    fig1_pyramid, fig2_example_spread, fig3_example_returns,
)
from research.paper_replication.outputs.png_export import plotly_to_png


def main():
    params = dict(research_config.PAIRS_CONFIG)
    n_sigma = params["threshold_sigma"]
    wavelet = params.get("wavelet", DEFAULT_WAVELET)
    index_id = params["index_id"]

    print(f"[regen] loading prices (index={index_id})...", flush=True)
    prices = load_prices(
        source="paper_data", start=params["start_date"], end=params["end_date"],
        index_id=index_id,
    )
    paper_periods = params.get("paper_periods")
    if paper_periods:
        paper_periods = paper_periods[:1]
    periods = build_periods(
        prices.get_column("date"),
        block_size=params["block_size"],
        max_periods=1,
        paper_periods=paper_periods,
    )
    if not periods:
        print("[regen] no periods built; aborting.", flush=True)
        return 1
    period = periods[0]
    print(f"[regen] period 1: train {period.train_start}->{period.train_end} | "
          f"trade {period.trade_start}->{period.trade_end}", flush=True)

    universe = members_asof(period.train_start, index_id=index_id)
    keep = ["date"] + [t for t in universe if t in prices.columns]
    pp = prices.select(keep)
    train_p = pp.slice(*period.train_slice)
    trade_p = pp.slice(*period.trade_slice)

    print(f"[regen] selecting distance pairs (top_n={params['top_n']})...", flush=True)
    pairs = select_pairs(
        "distance", train_p, top_n=params["top_n"],
        candidate_pool=params["candidate_pool"], k_ar_diff=params["k_ar_diff"],
    )
    if not pairs:
        print("[regen] no pairs returned; aborting.", flush=True)
        return 1
    print(f"[regen] simulating {len(pairs)} pairs to find the example...", flush=True)

    best_i, best_j, best_score = None, None, -float("inf")
    best_std, best_wav = None, None
    for i, j in pairs:
        s_std = build_spread(i, j, train_p, trade_p, use_wavelet=False,
                             n_sigma=n_sigma, wavelet=wavelet)
        s_wav = build_spread(i, j, train_p, trade_p, use_wavelet=True,
                             n_sigma=n_sigma, wavelet=wavelet, boundary="periodic")
        if s_std is None or s_wav is None:
            continue
        r_wav = simulate_pair(s_wav)
        # Score: many trades + profitable wavelet variant.
        score = len(r_wav.trades) * 10 + r_wav.total_pnl
        if score > best_score:
            r_std = simulate_pair(s_std)
            best_score = score
            best_i, best_j = i, j
            best_std, best_wav = (s_std, r_std), (s_wav, r_wav)

    if best_i is None:
        print("[regen] could not build any pair; aborting.", flush=True)
        return 1

    s_std, r_std = best_std
    s_wav, r_wav = best_wav
    print(f"[regen] example pair: {best_i} / {best_j} "
          f"({len(r_wav.trades)} wavelet trades, pnl={r_wav.total_pnl:.3f})", flush=True)

    example = {
        "pair": f"{best_i} / {best_j}",
        "dates": s_wav.trade_dates.to_list(),
        "std_spread": s_std.trade_spread,
        "wav_spread": s_wav.trade_spread,
        "std_thr": s_std.threshold,
        "wav_thr": s_wav.threshold,
        "std_trades": r_std.trades,
        "wav_trades": r_wav.trades,
        "std_daily": r_std.daily_returns,
        "wav_daily": r_wav.daily_returns,
    }

    out_dir = Path(research_config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover non-canonical names from earlier runs.
    for stale in ("fig01_pyramid.png", "fig02_example_spread.png",
                  "fig03_example_returns.png"):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    fig1 = fig1_pyramid()
    plotly_to_png(fig1, str(out_dir / "figure1_pyramid_algorithm_of_mallat.png"),
                  width=1400, height=820, dpi=150)
    print(f"[regen] wrote {out_dir / 'figure1_pyramid_algorithm_of_mallat.png'}", flush=True)

    fig2 = fig2_example_spread(example)
    plotly_to_png(fig2, str(out_dir / "figure2_example_pair_spread_and_trades.png"),
                  width=1500, height=820, dpi=150)
    print(f"[regen] wrote {out_dir / 'figure2_example_pair_spread_and_trades.png'}", flush=True)

    fig3 = fig3_example_returns(example)
    plotly_to_png(fig3, str(out_dir / "figure3_example_pair_return_series.png"),
                  width=1500, height=640, dpi=150)
    print(f"[regen] wrote {out_dir / 'figure3_example_pair_return_series.png'}", flush=True)

    print("[regen] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
