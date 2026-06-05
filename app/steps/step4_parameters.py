"""Step 4 — Parameters: manual setup or walk-forward optimization."""
import pandas as pd
import streamlit as st

from config import (
    INDEX_CONFIG, INDEX_CURRENCY,
    REBALANCE_FREQS, MA_PARAM_GRID, MOMENTUM_PARAM_GRID,
    DATA_START, OOS_START, OOS_END,
)
from optimization.grid_search import GridSearch
from app.registry import STRATEGIES, ALLOCATORS, PAIRS_BENCHMARKS
from app.data import load_prices, load_returns, load_membership, load_riskfree

_IS_PAIRS = "Pairs Trading"


def render() -> None:
    idx_id = st.session_state.selected_index
    strategy_name = st.session_state.strategy_name
    allocator_name = st.session_state.allocator_name
    rebal_freq = st.session_state.rebal_freq

    st.markdown(
        """
        <div class="page-header">
            <p class="step-num">Step 4 of 6</p>
            <p class="title">Parameters Configuration</p>
            <p class="desc">Fine-tune strategy parameters — optionally run in-sample grid search for optimal calibration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="ctx-bar">
        <div class="ctx-item">Index: <span>{INDEX_CONFIG[idx_id]['name']}</span></div>
        <div class="ctx-item">Strategy: <span>{strategy_name}</span></div>
        <div class="ctx-item">Allocation: <span>{allocator_name}</span></div>
        <div class="ctx-item">Rebalance: <span>{rebal_freq}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Optimization</p>', unsafe_allow_html=True)

    def _reset_optim():
        st.session_state["wf_schedule"] = None

    # Walk-Forward optimization is only available for MA and Momentum.
    if strategy_name == _IS_PAIRS:
        optim_mode = "Manual"
        st.info("ℹ️ Walk-Forward optimization is not available for Pairs Trading. Parameters are set manually below.", icon="ℹ️")
    else:
        optim_mode = st.radio(
            "Optimization mode",
            options=["Manual", "Walk-Forward"],
            index=0, horizontal=True, key="optim_mode_radio",
            on_change=_reset_optim,
        )

    is_best: dict = {}

    if optim_mode == "Walk-Forward":
        rebal_n = REBALANCE_FREQS[rebal_freq]
        n_oos_months = len(pd.date_range(OOS_START, OOS_END, freq="ME"))

        col_wf1, col_wf2 = st.columns(2)
        with col_wf1:
            train_years = st.slider(
                "Training window (years)", min_value=2, max_value=10, value=3,
                key="wf_train_years",
                help="Length of the rolling training window used for each grid search.",
            )
        with col_wf2:
            reoptim_freq = st.slider(
                "Re-optimize every N rebalancings", min_value=1, max_value=12, value=3,
                key="wf_reoptim_freq",
                help="1 = every rebalance, 3 = every 3 rebalances with carry-forward in between.",
            )

        n_searches = max(1, (n_oos_months // rebal_n + reoptim_freq - 1) // reoptim_freq)
        st.caption(f"~{n_searches} grid searches — parameters are carried forward between re-optimizations.")

        if st.session_state.get("wf_schedule") is None:
            if st.button("▶️ Launch Walk-Forward", type="primary", use_container_width=True):
                prices = load_prices()
                returns = load_returns()
                membership = load_membership()
                ccy = INDEX_CURRENCY[idx_id]
                riskfree = load_riskfree()
                rf_daily = riskfree[ccy] if ccy in riskfree.columns else None

                gs = GridSearch(prices, returns, membership, rf_daily,
                                idx_id, DATA_START, OOS_END,
                                rebalance_months=REBALANCE_FREQS[rebal_freq])

                alloc_key = "ERC" if "ERC" in allocator_name else "EW"
                grid = MA_PARAM_GRID if strategy_name == "Moving Average Crossover" else MOMENTUM_PARAM_GRID
                strat_key = "MA" if strategy_name == "Moving Average Crossover" else "Momentum"

                bar_wf = st.progress(0)
                status_wf = st.empty()

                def wf_cb(p, msg):
                    bar_wf.progress(int(p * 100))
                    status_wf.text(f"⚡ {msg}")

                schedule = gs.build_params_schedule(
                    strat_key, grid, OOS_START, OOS_END,
                    train_years=train_years, allocation=alloc_key,
                    reoptim_freq=reoptim_freq,
                    progress_callback=wf_cb,
                )
                st.session_state["wf_schedule"] = schedule
                status_wf.empty()
                st.rerun()
        else:
            schedule = st.session_state["wf_schedule"]
            n = len(schedule)
            st.markdown(f"""
            <div class="status-card ok" style="border-left-color: #10b981;">
                <p class="status-value"><span class="status-ok">●</span> Walk-Forward Complete — {n} re-optimizations</p>
            </div>
            """, unsafe_allow_html=True)
            sched_df = pd.DataFrame([{"date": str(d.date()), **p} for d, p in schedule.items()])
            st.dataframe(sched_df, use_container_width=True)

    # ── Strategy parameters ──
    st.markdown('<p class="section-hdr">Strategy Parameters</p>', unsafe_allow_html=True)
    strategy_class = STRATEGIES[strategy_name]
    schema = strategy_class.get_parameters_schema()
    strategy_params: dict = {}
    for pname, pcfg in schema.items():
        default_val = is_best.get(pname, pcfg["default"])
        if pcfg["type"] == "int":
            strategy_params[pname] = st.slider(
                pcfg["label"], pcfg["min"], pcfg["max"],
                int(default_val), key=f"sp_{pname}")
        elif pcfg["type"] == "float":
            strategy_params[pname] = st.slider(
                pcfg["label"], pcfg["min"], pcfg["max"],
                float(default_val), step=0.01, key=f"sp_{pname}")
        elif pcfg["type"] == "bool":
            strategy_params[pname] = st.checkbox(
                pcfg["label"], value=bool(default_val), key=f"sp_{pname}")
        elif pcfg["type"] == "str" and "options" in pcfg:
            opts = list(pcfg["options"])
            idx0 = opts.index(default_val) if default_val in opts else 0
            strategy_params[pname] = st.selectbox(
                pcfg["label"], opts, index=idx0, key=f"sp_{pname}")

    # ── Benchmark strategies (only for Pairs Trading composite mode) ──
    benchmark_configs: dict = {}
    if strategy_name == _IS_PAIRS:
        st.markdown('<p class="section-hdr">Benchmark Strategies</p>', unsafe_allow_html=True)
        st.caption(
            "Choose which benchmark(s) to run alongside the Wavelet strategy. "
            "Results will be compared side-by-side in Step 6."
        )
        for bench_label, bench_class in PAIRS_BENCHMARKS.items():
            enabled = st.checkbox(f"Include **{bench_label}**", value=True,
                                  key=f"bench_enable_{bench_label}")
            if enabled:
                with st.expander(f"{bench_label} parameters", expanded=False):
                    bschema = bench_class.get_parameters_schema()
                    bparams: dict = {}
                    for pname, pcfg in bschema.items():
                        default_val = pcfg["default"]
                        if pcfg["type"] == "int":
                            bparams[pname] = st.slider(
                                pcfg["label"], pcfg["min"], pcfg["max"],
                                int(default_val), key=f"bp_{bench_label}_{pname}")
                        elif pcfg["type"] == "float":
                            bparams[pname] = st.slider(
                                pcfg["label"], pcfg["min"], pcfg["max"],
                                float(default_val), step=0.01, key=f"bp_{bench_label}_{pname}")
                        elif pcfg["type"] == "bool":
                            bparams[pname] = st.checkbox(
                                pcfg["label"], value=bool(default_val),
                                key=f"bp_{bench_label}_{pname}")
                        elif pcfg["type"] == "str" and "options" in pcfg:
                            opts = list(pcfg["options"])
                            idx0 = opts.index(default_val) if default_val in opts else 0
                            bparams[pname] = st.selectbox(
                                pcfg["label"], opts, index=idx0,
                                key=f"bp_{bench_label}_{pname}")
                    benchmark_configs[bench_label] = bparams

    # ── Allocator parameters ──
    st.markdown('<p class="section-hdr">Allocation Parameters</p>', unsafe_allow_html=True)
    allocator_class = ALLOCATORS[allocator_name]
    a_schema = allocator_class.get_parameters_schema()
    allocator_params: dict = {}
    for pname, pcfg in a_schema.items():
        default_val = pcfg["default"]
        if pcfg["type"] == "int":
            allocator_params[pname] = st.slider(
                pcfg["label"], pcfg["min"], pcfg["max"],
                int(default_val), key=f"ap_{pname}")
        elif pcfg["type"] == "float":
            allocator_params[pname] = st.slider(
                pcfg["label"], pcfg["min"], pcfg["max"],
                float(default_val), step=0.01, key=f"ap_{pname}")
        elif pcfg["type"] == "bool":
            allocator_params[pname] = st.checkbox(
                pcfg["label"], value=bool(default_val), key=f"ap_{pname}")

    oos_start = OOS_START
    oos_end = OOS_END

    st.markdown('<p class="section-hdr">Configuration Summary</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="config-panel">
        <div class="summary-row"><span class="key">Index</span><span class="val">{INDEX_CONFIG[idx_id]['name']} ({idx_id})</span></div>
        <div class="summary-row"><span class="key">Strategy</span><span class="val">{strategy_name}</span></div>
        <div class="summary-row"><span class="key">Allocation</span><span class="val">{allocator_name}</span></div>
        <div class="summary-row"><span class="key">Rebalance</span><span class="val">{rebal_freq}</span></div>
        <div class="summary-row"><span class="key">OOS Period</span><span class="val">{oos_start} → {oos_end}</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state["wf_schedule"] = None
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Launch Backtest OOS", type="primary", use_container_width=True):
            st.session_state.strategy_params = strategy_params
            st.session_state.allocator_params = allocator_params
            st.session_state.benchmark_configs = benchmark_configs
            st.session_state.oos_start = str(oos_start)
            st.session_state.oos_end = str(oos_end)
            st.session_state.step = 5
            st.rerun()
