"""Step 1 — Data status: file checks and market overview."""
import polars as pl
import streamlit as st

from config import (
    INDEX_CONFIG,
    PRICES_PATH, RETURNS_PATH, MEMBERSHIP_PATH, RISKFREE_PATH,
)
from app.data import load_prices, load_membership, load_riskfree


def _missing_indices() -> list[str]:
    """Indices declared in INDEX_CONFIG but missing from membership.parquet."""
    if not MEMBERSHIP_PATH.exists():
        return list(INDEX_CONFIG.keys())
    present = set(
        pl.read_parquet(MEMBERSHIP_PATH, columns=["index_id"])
        ["index_id"].unique().to_list()
    )
    return [k for k in INDEX_CONFIG.keys() if k not in present]


def _run_bbg_extraction(index_ids: list[str] | None = None,
                        do_riskfree: bool = True) -> None:
    """Extract only the target `index_ids` (None means all indices)."""
    try:
        from extraction.bloomberg_api import BloombergConnector
        bbg = BloombergConnector()
        bbg.connect()
        if not bbg.connected:
            st.error("Bloomberg is not connected")
            st.stop()

        from extraction.bbg_members import extract_membership
        from extraction.bbg_returns import extract_prices
        from extraction.bbg_riskfree import extract_riskfree

        bar = st.progress(0)
        status = st.empty()
        label = ", ".join(index_ids) if index_ids else "tous les indices"

        status.text(f"Membership ({label})…")
        mem = extract_membership(
            bbg,
            index_ids=index_ids,
            progress_callback=lambda p, m: (bar.progress(int(p * 33)), status.text(f"Members: {m}")),
        )

        # Restrict extraction to tickers from the requested indices.
        target_tickers = None
        if index_ids is not None and mem.height > 0:
            target_tickers = (
                mem.filter(pl.col("index_id").is_in(index_ids))
                ["ticker"].unique().to_list()
            )

        status.text(f"Historical Prices ({label})…")
        extract_prices(
            bbg, mem, tickers=target_tickers,
            progress_callback=lambda p, m: (bar.progress(33 + int(p * 34)), status.text(f"Prices: {m}")),
        )

        if do_riskfree:
            status.text("Risk free rate…")
            extract_riskfree(
                bbg,
                progress_callback=lambda p, m: (bar.progress(67 + int(p * 33)), status.text(f"RF: {m}")),
            )

        bar.progress(100)
        status.text("Extraction completed")
        st.cache_data.clear()
        st.rerun()

    except Exception as e:
        st.error(f"Extraction error: {e}")


def render() -> None:
    # ── Hero Banner ──
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">● SYSTEM ONLINE</div>
            <p class="hero-title">Quantitative Backtesting Engine</p>
            <p class="hero-subtitle">Returns-based systematic strategy framework — Bloomberg data · Grid-search optimization · Point-in-time composition</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    files = {
        "prices.parquet":     PRICES_PATH,
        "returns.parquet":    RETURNS_PATH,
        "membership.parquet": MEMBERSHIP_PATH,
        "riskfree.parquet":   RISKFREE_PATH,
    }
    all_ok = all(path.exists() for path in files.values())

    if all_ok:
        with st.spinner("Loading metadata…"):
            prices = load_prices()
            membership = load_membership()
            load_riskfree()  # warm cache

        st.markdown('<p class="section-hdr">Market Data Overview</p>', unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        date_min = prices.index.min().strftime("%Y-%m-%d")
        date_max = prices.index.max().strftime("%Y-%m-%d")
        with mc1:
            st.markdown(f"""
            <div class="metric-tile">
                <p class="label">Coverage</p>
                <p class="value" style="font-size:1.1rem;">{date_min}</p>
                <p class="sub">→ {date_max}</p>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-tile">
                <p class="label">Tickers</p>
                <p class="value">{prices.shape[1]:,}</p>
                <p class="sub">unique instruments</p>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class="metric-tile">
                <p class="label">Trading Days</p>
                <p class="value">{prices.shape[0]:,}</p>
                <p class="sub">observations</p>
            </div>""", unsafe_allow_html=True)
        with mc4:
            years = (prices.index.max() - prices.index.min()).days / 365.25
            st.markdown(f"""
            <div class="metric-tile">
                <p class="label">History Depth</p>
                <p class="value">{years:.1f}<span style="font-size:0.9rem; color:#64748b;"> yrs</span></p>
                <p class="sub">of market data</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-hdr">Index Universe</p>', unsafe_allow_html=True)
        idx_cols = st.columns(len(INDEX_CONFIG))
        present_ids = set(membership["index_id"].unique())
        for i, (idx_id, cfg) in enumerate(INDEX_CONFIG.items()):
            sub = membership[membership["index_id"] == idx_id]
            n = sub["ticker"].nunique()
            present = idx_id in present_ids and n > 0
            badge = (
                '<span style="color:#10b981">● available</span>' if present
                else '<span style="color:#ef4444">● missing</span>'
            )
            with idx_cols[i]:
                st.markdown(f"""
                <div class="idx-card">
                    <p class="ticker">{idx_id}</p>
                    <p class="name">{cfg['name']}</p>
                    <p class="detail">{cfg['currency']} · {n} tickers · {badge}</p>
                </div>""", unsafe_allow_html=True)

        # Offer a targeted extraction for indices that are still missing.
        missing = _missing_indices()
        if missing:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-card ko"><p class="status-label">Missing data</p>'
                f'<p class="status-value"><span class="status-ko">●</span> {", ".join(missing)}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Extract {', '.join(missing)} from Bloomberg",
                         type="secondary", use_container_width=True):
                _run_bbg_extraction(index_ids=missing, do_riskfree=False)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Proceed to Index Selection →", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    else:
        st.divider()
        st.subheader("🔧 Bloomberg Extraction Required")
        missing = _missing_indices()
        label = ", ".join(missing) if missing else "all indices"
        st.markdown(f"Universe: **{label}**. Bloomberg Terminal must be connected.")
        if st.button(f"Launch extraction ({label})", type="primary"):
            _run_bbg_extraction(index_ids=missing or None)
