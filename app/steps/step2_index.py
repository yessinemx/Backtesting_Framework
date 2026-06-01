"""Step 2 — Index selection: choose the investment universe."""
import streamlit as st

from config import INDEX_CONFIG
from app.data import load_membership, load_prices


def render() -> None:
    st.markdown(
        """
        <div class="page-header">
            <p class="step-num">Step 2 of 6</p>
            <p class="title">Index Selection</p>
            <p class="desc">Choose the benchmark index to define your investment universe</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-hdr">Available Indices</p>', unsafe_allow_html=True)

    membership = load_membership()
    prices = load_prices()

    cols = st.columns(len(INDEX_CONFIG))
    flag_map = {
        "SPX": '<img src="https://flagcdn.com/w80/us.png" style="height:48px;border-radius:4px;">',
        "NDX": '<img src="https://flagcdn.com/w80/us.png" style="height:48px;border-radius:4px;">',
        "UKX": '<img src="https://flagcdn.com/w80/gb.png" style="height:48px;border-radius:4px;">',
        "SX5E": '<img src="https://flagcdn.com/w80/eu.png" style="height:48px;border-radius:4px;">',
    }
    for i, (idx_id, cfg) in enumerate(INDEX_CONFIG.items()):
        n = membership[membership["index_id"] == idx_id]["ticker"].nunique()
        idx_tickers = membership[membership["index_id"] == idx_id]["ticker"].unique()
        avail = len([t for t in idx_tickers if t in prices.columns])
        with cols[i]:
            st.markdown(f"""
            <div class="sel-card">
                <div class="icon">{flag_map.get(idx_id, '📈')}</div>
                <p class="card-ticker">{idx_id}</p>
                <p class="card-title">{cfg['name']}</p>
                <p class="card-sub">{cfg['currency']} · {n} constituents · {avail} w/ prices</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button(f"Select {idx_id}", key=f"sel_{idx_id}", type="primary", use_container_width=True):
                st.session_state.selected_index = idx_id
                st.session_state.step = 3
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Data Status"):
        st.session_state.step = 1
        st.rerun()
