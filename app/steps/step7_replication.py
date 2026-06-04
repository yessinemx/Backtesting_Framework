"""Paper replication page: wavelet pairs-trading study, with the paper's figures."""
import streamlit as st

from config import config_paper as research_config
from research.paper_replication.report import build_report


def _show_figs(figures, prefix):
    """Render every figure whose name contains `prefix`, in sorted order."""
    for name in sorted(figures):
        if prefix in name:
            st.plotly_chart(figures[name], use_container_width=True, key=f"fig_{name}")


def render() -> None:
    st.markdown('<p class="section-hdr">Paper Replication — Pairs Trading with Wavelet Transform</p>',
                unsafe_allow_html=True)
    st.caption("Eroğlu, Yener & Yiğit (2023), Quantitative Finance. "
               "Point-in-time S&P 500, 2010–2018 (7 formation/trading periods), before transaction costs.")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            source = st.selectbox("Source", ["data", "bloomberg"], index=0)
        with c2:
            methods = st.multiselect("Methods", list(research_config.DEFAULT_METHODS),
                                     default=list(research_config.DEFAULT_METHODS))
        with c3:
            sweeps = st.checkbox("Include robustness sweeps (Fig 10 & 11) — slower (~5 min)", value=False)
        run = st.button("Run replication", type="primary", use_container_width=True)

    if run:
        if not methods:
            st.warning("Select at least one method.")
            return
        params = dict(research_config.PAIRS_CONFIG)
        params["tc_per_share"] = research_config.HEADLINE_TC_PER_SHARE
        with st.spinner("Running point-in-time replication and building figures..."):
            report = build_report(source=source, methods=tuple(methods),
                                  params=params, with_figures=True, sweeps=sweeps)
        st.session_state.paper_report = report

    report = st.session_state.get("paper_report")
    if not report:
        st.info("Configure the run above, then click **Run replication**.")
        return

    figures = report["figures"]
    comp = report["comparison"]
    st.success(f"Universe pool: {report['universe_pool']} tickers · {report['n_periods']} periods")

    tabs = st.tabs(["📊 Overview", "🔬 Example pair", "📈 Cumulative & Sharpe",
                    "🧩 Trade categories", "🌫️ Noise", "💰 Asset pricing", "🧪 Robustness"])

    with tabs[0]:
        st.markdown('<p class="section-hdr">Paper vs replication</p>', unsafe_allow_html=True)
        st.dataframe(comp.to_pandas(), use_container_width=True)
        st.caption("Replication matches the paper's *scale* and *direction* (wavelet ≥ standard); "
                   "the paper's headline +12% does not fully reproduce on this dataset (≈84–272 names "
                   "with sparse coverage vs the paper's 415). See the response notes for the analysis.")
        for method in report["summaries"]:
            st.markdown(f'<p class="section-hdr">{method} — standard vs wavelet (avg over periods)</p>',
                        unsafe_allow_html=True)
            st.dataframe(report["summaries"][method].to_pandas(), use_container_width=True)
        if "fig01_pyramid" in figures:
            st.plotly_chart(figures["fig01_pyramid"], use_container_width=True, key="fig_pyramid")

    with tabs[1]:
        st.caption("Figure 2 & 3 — a representative pair: spread + 2σ thresholds + trade markers, and its cumulative return.")
        _show_figs(figures, "fig02_")
        _show_figs(figures, "fig03_")

    with tabs[2]:
        st.caption("Figure 4 — cumulative returns (standard / wavelet / index / buy & hold). "
                   "Figure 5 — expanding daily Sharpe ratio.")
        _show_figs(figures, "fig04_")
        _show_figs(figures, "fig05_")

    with tabs[3]:
        st.caption("Figures 6 & 7 — yearly proportions and returns per convergence category (wavelet). "
                   "The bottom chart is the key forensic finding: full-convergence is ~equal for "
                   "standard and wavelet, and only the look-ahead **Opt** reproduces the paper's 12%→32% "
                   "jump — i.e. the jump needs future information that the wavelet cannot supply.")
        _show_figs(figures, "_categories_")
        _show_figs(figures, "fig12_convergence_jump_")

    with tabs[4]:
        st.caption("Figure 8 — variance of the filtered-out noise vs the unfiltered return (negative correlation).")
        _show_figs(figures, "fig08_")

    with tabs[5]:
        st.caption("Section 5.4 — asset-pricing test. Market-model (CAPM) alpha of the daily "
                   "pairs returns with Newey-West HAC errors; the market factor is the equal-weight "
                   "S&P 500 excess return. Full FF5 / q-factor / Petkova models need external factor "
                   "files (Ken French, global-q, FRED).")
        alpha = report.get("alpha_table")
        if alpha is not None and not alpha.empty:
            st.markdown('<p class="section-hdr">Market-model alphas</p>', unsafe_allow_html=True)
            st.dataframe(alpha, use_container_width=True)
        st.markdown('<p class="section-hdr">Figure 9 — yearly alpha</p>', unsafe_allow_html=True)
        _show_figs(figures, "fig09_")

    with tabs[6]:
        if any("fig10_" in n or "fig11_" in n for n in figures):
            st.caption("Figure 10 — returns & Sharpe across wavelet classes. "
                       "Figure 11 — annualized profit by trading horizon.")
            _show_figs(figures, "fig10_")
            _show_figs(figures, "fig11_")
        else:
            st.info("Re-run with **Include robustness sweeps** checked to produce Figures 10 & 11.")

    st.markdown('<p class="section-hdr">Outputs</p>', unsafe_allow_html=True)
    st.write(f"Tables: `{research_config.TABLES_DIR}`  ·  Figures: `{research_config.FIGURES_DIR}`")
    st.caption("Run `python research/replicate_paper.py` to also write all tables and figures to disk.")
