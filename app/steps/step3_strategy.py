"""Step 3 — Strategy & Allocation."""
import streamlit as st

from config import INDEX_CONFIG, REBALANCE_FREQS
from app.registry import STRATEGIES, ALLOCATORS


_STRAT_DESC = {
    "Moving Average Crossover": "Long signal when fast MA crosses above slow MA. Trend-following approach.",
    "Momentum": "Cross-sectional ranking by past cumulative returns. Selects top performers.",
    "Pairs Trading": "Market-neutral statistical arbitrage. Runs three variants in parallel: "
        "Wavelet MODWT, Engle-Granger Cointegration and Partial Cointegration. ",
}
_ALLOC_DESC = {
    "Equal Weight (EW)": "Uniform allocation across selected assets at each rebalance.",
    "ERC (Risk Parity)": "Inverse volatility weighting — lower vol assets receive higher weight.",
}


def render() -> None:
    idx_id = st.session_state.selected_index

    st.markdown(
        """
        <div class="page-header">
            <p class="step-num">Step 3 of 6</p>
            <p class="title">Strategy & Allocation</p>
            <p class="desc">Configure the signal generation method and portfolio allocation scheme</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="ctx-bar">
        <div class="ctx-item">Index: <span>{INDEX_CONFIG[idx_id]['name']} ({idx_id})</span></div>
        <div class="ctx-item">Currency: <span>{INDEX_CONFIG[idx_id]['currency']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="config-panel">
            <p class="panel-title">Signal Strategy</p>
        </div>""", unsafe_allow_html=True)
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), label_visibility="collapsed")
        st.caption(_STRAT_DESC.get(strategy_name, ""))

    with col2:
        st.markdown("""
        <div class="config-panel">
            <p class="panel-title">Allocation Method</p>
        </div>""", unsafe_allow_html=True)
        allocator_name = st.selectbox("Allocation", list(ALLOCATORS.keys()), label_visibility="collapsed")
        st.caption(_ALLOC_DESC.get(allocator_name, ""))

    st.markdown('<p class="section-hdr">Rebalance Frequency</p>', unsafe_allow_html=True)
    rebal_label = st.selectbox("Rebalance frequency", list(REBALANCE_FREQS.keys()),
                               index=0, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Continue to Parameters →", type="primary", use_container_width=True):
            st.session_state.strategy_name = strategy_name
            st.session_state.allocator_name = allocator_name
            st.session_state.rebal_freq = rebal_label
            st.session_state.step = 4
            st.rerun()
