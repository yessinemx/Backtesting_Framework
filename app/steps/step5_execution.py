"""Step 5 — Execution du backtest."""
import streamlit as st

from config import (
    INDEX_CONFIG, INDEX_CURRENCY,
    REBALANCE_FREQS, OOS_START, OOS_END,
    TRANSACTION_COST_BPS, SHORT_BORROW_BPS,
)
from portfolio.backtest_engine import BacktestEngine
from app.registry import STRATEGIES, ALLOCATORS, PAIRS_BENCHMARKS
from app.data import load_prices, load_returns, load_membership, load_riskfree

_IS_PAIRS = "Pairs Trading"
_PAIRS_LABELS = {
    "main": "Wavelet (Paper)",
    **{k: k for k in PAIRS_BENCHMARKS},
}


def _run_single(strategy_cls, strategy_params, allocator, idx_id, prices, returns,
                membership, rf_daily, rebal_freq, oos_s, oos_e,
                strategy_label, progress_callback=None, params_schedule=None):
    """Run one backtest and return the result."""
    strategy = strategy_cls(strategy_params)
    config = {
        "index_id": idx_id,
        "strategy": strategy_label,
        "allocation": "",
        "strategy_params": strategy_params,
        "allocator_params": {},
        "rebalance_months": REBALANCE_FREQS[rebal_freq],
        "start_date": oos_s,
        "end_date": oos_e,
        "initial_capital": 1_000_000,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "short_borrow_bps": SHORT_BORROW_BPS,
    }
    engine = BacktestEngine(config)
    engine.set_strategy(strategy)
    engine.set_allocator(allocator)
    return engine.run(prices, returns, membership, rf_daily,
                      progress_callback=progress_callback,
                      params_schedule=params_schedule)


def render() -> None:
    idx_id = st.session_state.selected_index
    strategy_name = st.session_state.strategy_name
    allocator_name = st.session_state.allocator_name
    rebal_freq = st.session_state.rebal_freq
    strategy_params = st.session_state.strategy_params
    allocator_params = st.session_state.allocator_params
    benchmark_configs: dict = st.session_state.get("benchmark_configs", {})

    st.markdown(f"""
    <div class="page-header">
        <p class="step-num">Step 5 of 6</p>
        <p class="title">Backtest Execution</p>
        <p class="desc">Running out-of-sample backtest on {INDEX_CONFIG[idx_id]['name']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ctx-bar">
        <div class="ctx-item">Index: <span>{idx_id}</span></div>
        <div class="ctx-item">Strategy: <span>{strategy_name}</span></div>
        <div class="ctx-item">Allocation: <span>{allocator_name}</span></div>
        <div class="ctx-item">Rebalance: <span>{rebal_freq}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="exec-terminal">
        <p class="exec-title">● Execution Console</p>
    </div>
    """, unsafe_allow_html=True)

    bar = st.progress(0)
    status = st.empty()

    try:
        status.text("📥 Loading market data…")
        prices = load_prices()
        returns = load_returns()
        membership = load_membership()
        riskfree = load_riskfree()

        ccy = INDEX_CURRENCY[idx_id]
        rf_daily = riskfree[ccy] if ccy in riskfree.columns else None
        bar.progress(10)

        oos_s = st.session_state.get("oos_start", OOS_START)
        oos_e = st.session_state.get("oos_end",   OOS_END)
        allocator = ALLOCATORS[allocator_name](allocator_params)
        params_schedule = st.session_state.get("wf_schedule")

        # ── Composite Pairs Trading mode ─────────────────────────────────────
        if strategy_name == _IS_PAIRS:
            runs: dict = {}

            # 1) Main: Wavelet (Paper)
            n_runs = 1 + len(benchmark_configs)
            status.text("⚡ Running Wavelet (Paper)…")

            def _prog_main(pct, msg, n=n_runs):
                bar.progress(10 + int(pct * (80 // n)))
                status.text(f"[Wavelet] {msg}")

            runs[_PAIRS_LABELS["main"]] = _run_single(
                STRATEGIES[strategy_name], strategy_params,
                allocator, idx_id, prices, returns, membership, rf_daily,
                rebal_freq, oos_s, oos_e, _PAIRS_LABELS["main"],
                progress_callback=_prog_main,
            )

            # 2) Benchmarks
            for i, (bench_label, bench_params) in enumerate(benchmark_configs.items()):
                status.text(f"⚡ Running {bench_label}…")
                offset = 10 + (80 // n_runs) * (i + 1)

                def _prog_bench(pct, msg, offset=offset, n_runs=n_runs):
                    bar.progress(offset + int(pct * (80 // n_runs)))
                    status.text(f"[{bench_label}] {msg}")

                runs[bench_label] = _run_single(
                    PAIRS_BENCHMARKS[bench_label], bench_params,
                    allocator, idx_id, prices, returns, membership, rf_daily,
                    rebal_freq, oos_s, oos_e, bench_label,
                    progress_callback=_prog_bench,
                )

            st.session_state.result = runs[_PAIRS_LABELS["main"]]
            st.session_state.pairs_results = runs
            bar.progress(100)
            status.text("")

            n_strats = len(runs)
            st.markdown(f"""
            <div class="status-card ok" style="border-left-color:#10b981; text-align:center; padding:1.5rem;">
                <p class="status-value" style="font-size:1.3rem;"><span class="status-ok">●</span> {n_strats} Backtests Complete</p>
                <p style="font-size:0.85rem; color:#94a3b8; margin:0.5rem 0 0 0;">
                    {", ".join(runs.keys())} · {oos_s} → {oos_e}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Single strategy mode ─────────────────────────────────────────────
        else:
            st.session_state.pairs_results = None
            strategy = STRATEGIES[strategy_name](strategy_params)
            config = {
                "index_id": idx_id,
                "strategy": strategy_name,
                "allocation": allocator_name,
                "strategy_params": strategy_params,
                "allocator_params": allocator_params,
                "rebalance_months": REBALANCE_FREQS[rebal_freq],
                "start_date": oos_s,
                "end_date":   oos_e,
                "initial_capital": 1_000_000,
                "transaction_cost_bps": TRANSACTION_COST_BPS,
                "short_borrow_bps": SHORT_BORROW_BPS,
            }
            engine = BacktestEngine(config)
            engine.set_strategy(strategy)
            engine.set_allocator(allocator)

            def prog(pct, msg):
                bar.progress(10 + int(pct * 85))
                status.text(f"⚡ {msg}")

            result = engine.run(prices, returns, membership, rf_daily,
                                progress_callback=prog, params_schedule=params_schedule)

            st.session_state.result = result
            bar.progress(100)
            status.text("")

            n_rebal = len(result.rebalance_dates)
            st.markdown(f"""
            <div class="status-card ok" style="border-left-color:#10b981; text-align:center; padding:1.5rem;">
                <p class="status-value" style="font-size:1.3rem;"><span class="status-ok">●</span> Backtest Complete</p>
                <p style="font-size:0.85rem; color:#94a3b8; margin:0.5rem 0 0 0;">
                    {n_rebal} rebalances · {oos_s} → {oos_e} · Initial capital $1,000,000
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 View Results", type="primary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        if st.button("← Back to Parameters"):
            st.session_state.step = 4
            st.rerun()
