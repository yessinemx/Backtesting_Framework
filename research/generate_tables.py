"""
Generate the paper's result tables (CSV) for "Pairs trading with wavelet transform".

Tables written to research/outputs/tables/:
    table4_basic_stats.csv        - Return, Std.Dev, Skewness, Kurtosis  (Min/Max/Mean)
    table5_downside_risk.csv      - Sharpe, Max Drawdown, % Positive, VaR, CVaR
    table6_convergence.csv        - % full / partial / non-convergent pairs
    table13_asset_pricing.csv     - market-model annualized alpha (Section 5.4)
    table18_transaction_cost.csv  - key stats before vs after 10 bps cost

Three variants are reported per method: standard, wavelet (honest, symmetric MODWT
boundary), wavelet_pf (paper-faithful, MATLAB's periodic boundary — reproduces the
paper via boundary look-ahead).

Run:  python research/generate_tables.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from loaders import load_prices, members_asof
from research import config as research_config
from research.paper_replication.periods import build_periods
from research.paper_replication.selection import select_pairs
from research.paper_replication.spread import build_spread
from research.paper_replication.trading import simulate_pair
from research.paper_replication.metrics import aggregate_metrics
from research.paper_replication.output_writer import save_table

START, END = "2010-03-05", "2018-03-15"
METHODS = ("distance", "cointegration")
# (label, use_wavelet, boundary)
VARIANTS = [("standard", False, "symmetric"),
            ("wavelet", True, "symmetric"),
            ("wavelet_pf", True, "periodic")]
TC = 0.001   # 10 bps per share / half-turn (paper §5.5.3)


def run_all(method, prices, periods, params):
    """Per period/variant, build the spread once and simulate with & without cost."""
    nocost, withcost = [], []
    for period in periods:
        universe = members_asof(period.train_start, index_id="SPX")
        keep = ["date"] + [t for t in universe if t in prices.columns]
        pp = prices.select(keep)
        train = pp.slice(*period.train_slice)
        trade = pp.slice(*period.trade_slice)
        pairs = select_pairs(method, train, top_n=params["top_n"],
                             candidate_pool=params["candidate_pool"],
                             k_ar_diff=params["k_ar_diff"])
        if not pairs:
            continue
        for vname, use_w, bnd in VARIANTS:
            r0, r1 = [], []
            for i, j in pairs:
                spec = build_spread(i, j, train, trade, use_wavelet=use_w,
                                    n_sigma=params["threshold_sigma"],
                                    wavelet=params["wavelet"], boundary=bnd)
                if spec is None:
                    continue
                r0.append(simulate_pair(spec, tc_per_share=0.0))
                r1.append(simulate_pair(spec, tc_per_share=TC))
            rep0 = aggregate_metrics(r0, method, vname, n_pairs=len(pairs))
            rep1 = aggregate_metrics(r1, method, vname, n_pairs=len(pairs))
            nocost.append({"period": period.index, **rep0.as_dict()})
            withcost.append({"period": period.index, **rep1.as_dict()})
    return pl.DataFrame(nocost), pl.DataFrame(withcost)


def _mmm(df, method_df, variant, col, scale=1.0):
    sub = method_df.filter(pl.col("variant") == variant).get_column(col) * scale
    return round(sub.min(), 4), round(sub.max(), 4), round(sub.mean(), 4)


def _block_table(by_method, rows, title):
    """rows = [(metric_label, column, scale)]; one block per variant per method."""
    out = []
    for variant, _, _ in VARIANTS:
        for label, col, scale in rows:
            row = {"variant": variant, "metric": label}
            for method in METHODS:
                mn, mx, me = _mmm(None, by_method[method], variant, col, scale)
                row[f"{method[:5]}_Min"] = mn
                row[f"{method[:5]}_Max"] = mx
                row[f"{method[:5]}_Mean"] = me
            out.append(row)
    return pl.DataFrame(out)


def _convergence_table(by_method):
    out = []
    for variant, _, _ in VARIANTS:
        for label, num in (("% Full Convergent", "n_full"),
                           ("% Partial Convergent", "n_partial"),
                           ("% Non Convergent", "n_non")):
            row = {"variant": variant, "metric": label}
            for method in METHODS:
                md = by_method[method].filter(pl.col("variant") == variant)
                prop = (md.get_column(num) / md.get_column("n_pairs") * 100.0)
                row[f"{method[:5]}_Min"] = round(prop.min(), 2)
                row[f"{method[:5]}_Max"] = round(prop.max(), 2)
                row[f"{method[:5]}_Mean"] = round(prop.mean(), 2)
            out.append(row)
    return pl.DataFrame(out)


def _tc_table(nocost_by_method, cost_by_method):
    out = []
    for method in METHODS:
        for variant, _, _ in VARIANTS:
            n = nocost_by_method[method].filter(pl.col("variant") == variant)
            c = cost_by_method[method].filter(pl.col("variant") == variant)
            r_no = n.get_column("mean_return").mean() * 100
            r_c = c.get_column("mean_return").mean() * 100
            out.append({
                "method": method, "variant": variant,
                "return_%_noTC": round(r_no, 2),
                "return_%_TC": round(r_c, 2),
                "return_diff_%": round(r_no - r_c, 2),
                "sharpe_noTC": round(n.get_column("sharpe").mean(), 2),
                "sharpe_TC": round(c.get_column("sharpe").mean(), 2),
                "pct_positive_noTC": round(n.get_column("pct_positive").mean() * 100, 1),
                "pct_positive_TC": round(c.get_column("pct_positive").mean() * 100, 1),
                "cvar5_%_TC": round(c.get_column("cvar_95").mean() * 100, 1),
            })
    return pl.DataFrame(out)


def main():
    params = dict(research_config.PAIRS_CONFIG)
    params["index_id"] = "SPX"
    print(f"Loading SPX {START}..{END} and running both methods (3 variants x 2 cost levels)...")
    prices = load_prices(source="data", index_id="SPX", start=START, end=END)
    periods = build_periods(prices.get_column("date"), block_size=params["block_size"])

    nocost, withcost = {}, {}
    for method in METHODS:
        print(f"  {method} ...", flush=True)
        nocost[method], withcost[method] = run_all(method, prices, periods, params)

    # Table 4 - basic statistics
    t4 = _block_table(nocost, [
        ("Return %", "mean_return", 100.0),
        ("Std. Dev.", "std_return", 1.0),
        ("Skewness", "skewness", 1.0),
        ("Kurtosis", "kurtosis", 1.0),
    ], "Basic statistics")
    save_table(t4, "table4_basic_stats")

    # Table 5 - downside risk
    t5 = _block_table(nocost, [
        ("Sharpe (ann.)", "sharpe", 1.0),
        ("Max Drawdown %", "max_drawdown", 100.0),
        ("% Positive", "pct_positive", 100.0),
        ("VaR 5% (%)", "var_95", 100.0),
        ("CVaR 5% (%)", "cvar_95", 100.0),
    ], "Downside risk")
    save_table(t5, "table5_downside_risk")

    # Table 6 - convergence
    t6 = _convergence_table(nocost)
    save_table(t6, "table6_convergence")

    # Table 18 - transaction costs
    t18 = _tc_table(nocost, withcost)
    save_table(t18, "table18_transaction_cost")

    # Table 13 - asset pricing (re-save the market-model alphas under the paper number)
    ap_path = research_config.TABLES_DIR / "asset_pricing_alphas.csv"
    if ap_path.exists():
        save_table(pl.read_csv(ap_path), "table13_asset_pricing")

    print("\n=== Table 4 (basic statistics, before costs) ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(t4)
        print("\n=== Table 6 (convergence %, Mean) ===")
        print(t6.select(["variant", "metric", "dista_Mean", "coint_Mean"]))
        print("\n=== Table 18 (transaction costs) ===")
        print(t18)
    print(f"\nTables written to {research_config.TABLES_DIR}")
    print("Asset-pricing alphas: table13_asset_pricing.csv (run replicate_paper.py / already present).")


if __name__ == "__main__":
    main()
