"""Paper replication page for the wavelet pairs-trading study."""
import streamlit as st

import config
from research import config as research_config
from research.paper_replication import run_pipeline


def _default_index_options():
    return [None] + list(config.INDEX_CONFIG.keys())


def render() -> None:
    st.markdown('<p class="section-hdr">Paper Replication</p>', unsafe_allow_html=True)
    st.caption("Dedicated runner for the paper replication workflow, separate from the standard backtest.")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            source = st.selectbox("Source", ["data", "bloomberg"], index=0)
            index_id = st.selectbox("Index", _default_index_options(), index=0)
            method = st.selectbox("Selection Method", ["distance", "cointegration"], index=0)
        with col2:
            wavelet = st.text_input("Wavelet", value=research_config.PAIRS_CONFIG["wavelet"])
            top_n = st.number_input("Top N Pairs", min_value=1, value=int(research_config.PAIRS_CONFIG["top_n"]))
            candidate_pool = st.number_input(
                "Candidate Pool", min_value=1, value=int(research_config.PAIRS_CONFIG["candidate_pool"])
            )
        with col3:
            block_size = st.number_input("Block Size", min_value=21, value=int(research_config.PAIRS_CONFIG["block_size"]))
            threshold_sigma = st.number_input(
                "Threshold Sigma", min_value=0.5, value=float(research_config.PAIRS_CONFIG["threshold_sigma"]), step=0.1
            )
            max_periods = st.number_input("Max Periods", min_value=1, value=1)

        write_outputs = st.checkbox("Write outputs to research/outputs/", value=False)

        run_col1, run_col2 = st.columns(2)
        with run_col1:
            run_smoke = st.button("Run Smoke Test", use_container_width=True)
        with run_col2:
            run_full = st.button("Run Full Replication", use_container_width=True)

    if run_smoke or run_full:
        params = dict(research_config.PAIRS_CONFIG)
        params.update(
            {
                "index_id": index_id,
                "method": method,
                "wavelet": wavelet,
                "top_n": int(top_n),
                "candidate_pool": int(candidate_pool),
                "block_size": int(block_size),
                "threshold_sigma": float(threshold_sigma),
                "max_periods": None if run_full else int(max_periods),
            }
        )
        with st.spinner("Running paper replication..."):
            result = run_pipeline(
                params=params,
                source=source,
                verbose=False,
                write_outputs=write_outputs,
            )
        st.session_state.paper_replication_result = result

    result = st.session_state.get("paper_replication_result")
    if not result:
        st.info("Choose a setup, then run a smoke test or a full replication.")
        return

    summary = result["summary"]
    by_period = result["by_period"]
    params = result["params"]

    st.markdown('<p class="section-hdr">Summary</p>', unsafe_allow_html=True)
    st.write(
        f"method={params['method']} | wavelet={params['wavelet']} | index_id={params['index_id']}"
    )
    st.dataframe(summary.to_pandas(), use_container_width=True)

    if not by_period.is_empty():
        st.markdown('<p class="section-hdr">By Period</p>', unsafe_allow_html=True)
        st.dataframe(by_period.to_pandas(), use_container_width=True)

    st.markdown('<p class="section-hdr">Outputs</p>', unsafe_allow_html=True)
    st.write(f"Tables: {research_config.TABLES_DIR}")
    st.write(f"Figures: {research_config.FIGURES_DIR}")
