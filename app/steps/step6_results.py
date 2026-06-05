"""Step 6 — Results: KPIs, analytics tabs, and export."""
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    INDEX_CURRENCY, OOS_START, OOS_END,
    TRANSACTION_COST_BPS, SHORT_BORROW_BPS,
)
from indicators.perf import PerformanceIndicators
from indicators.risk import RiskIndicators
from portfolio.port import PortfolioExporter
from visualisation.plots import (
    plot_equity_curve, plot_drawdown, plot_monthly_returns,
    plot_rolling_sharpe, plot_rolling_beta, plot_weights_over_time,
    plot_num_positions, plot_sub_period_bars,
    plot_pairs_comparison, plot_pairs_comparison_drawdown,
)
from app.data import load_returns, load_membership


def _kpi_color(val, invert=False):
    if isinstance(val, str):
        return "kpi-neutral"
    if invert:
        return "kpi-negative" if val < 0 else "kpi-positive"
    return "kpi-positive" if val > 0 else "kpi-negative"


def render() -> None:
    idx_id = st.session_state.get("selected_index", "")
    strategy_name = st.session_state.get("strategy_name", "")
    allocator_name = st.session_state.get("allocator_name", "")

    st.markdown("""
    <div class="page-header">
        <p class="step-num">Step 6 of 6</p>
        <p class="title">Backtest Results</p>
        <p class="desc">Performance analytics, risk metrics, portfolio composition and export</p>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.result
    if result is None:
        st.error("No results available")
        st.stop()

    st.markdown(f"""
    <div class="ctx-bar">
        <div class="ctx-item">Index: <span>{idx_id}</span></div>
        <div class="ctx-item">Strategy: <span>{strategy_name}</span></div>
        <div class="ctx-item">Allocation: <span>{allocator_name}</span></div>
        <div class="ctx-item">TC: <span>{TRANSACTION_COST_BPS} bps</span></div>
        <div class="ctx-item">Borrow: <span>{SHORT_BORROW_BPS} bps/yr</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Equal-weighted market proxy built from index members over the OOS window.
    _mkt_returns = None
    try:
        _all_returns = load_returns()
        _membership = load_membership()
        _oos_s = result.config.get("start_date", OOS_START)
        _oos_e = result.config.get("end_date",   OOS_END)
        _members = _membership[
            (_membership["index_id"] == idx_id) &
            (_membership["date"] <= _oos_e)
        ]["ticker"].unique().tolist()
        _cols = [t for t in _members if t in _all_returns.columns]
        if _cols:
            _mkt_returns = _all_returns.loc[_oos_s:_oos_e, _cols].mean(axis=1)
    except Exception:
        pass

    perf = PerformanceIndicators(result, benchmark_returns=_mkt_returns)
    risk = RiskIndicators(result, benchmark_returns=_mkt_returns)
    pr = perf.generate_report()
    rr = risk.generate_report()

    tc_total = result.total_transaction_costs
    tc_pct = tc_total / result.config.get("initial_capital", 1_000_000) * 100

    kpis = [
        ("Total Return", f"{pr.total_return:+.1f}%", _kpi_color(pr.total_return)),
        ("CAGR", f"{pr.cagr:+.1f}%", _kpi_color(pr.cagr)),
        ("Volatility", f"{rr.annualized_volatility:.1f}%", "kpi-neutral"),
        ("Sharpe", f"{pr.sharpe_ratio:.2f}", _kpi_color(pr.sharpe_ratio)),
        ("Max Drawdown", f"{rr.max_drawdown:.1f}%", "kpi-negative"),
        ("Transaction Costs", f"{tc_pct:.2f}%", "kpi-neutral"),
    ]

    kpi_cols = st.columns(6)
    for i, (label, value, css_class) in enumerate(kpis):
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <p class="kpi-label">{label}</p>
                <p class="kpi-value {css_class}">{value}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pairs Trading composite comparison (shown only in pairs mode) ────────
    pairs_results: dict | None = st.session_state.get("pairs_results")
    if pairs_results and len(pairs_results) > 1:
        st.markdown('<p class="section-hdr">Strategy Comparison — Pairs Trading</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_pairs_comparison(pairs_results),
                        use_container_width=True, key="chart_pairs_cmp")
        st.plotly_chart(plot_pairs_comparison_drawdown(pairs_results),
                        use_container_width=True, key="chart_pairs_dd_cmp")

        # KPI table across all 3 strategies
        rows = []
        for label, res in pairs_results.items():
            r_rets = res.get_returns()
            cum = (1 + r_rets).prod()
            ann = cum ** (252 / max(len(r_rets), 1)) - 1
            vol = r_rets.std() * (252 ** 0.5)
            sr = (r_rets.mean() / r_rets.std() * (252 ** 0.5)) if r_rets.std() > 0 else 0.0
            c = (1 + r_rets).cumprod()
            dd = ((c - c.expanding().max()) / c.expanding().max()).min() * 100
            rows.append({
                "Strategy": label,
                "Total Return %": f"{(cum - 1) * 100:+.1f}",
                "CAGR %": f"{ann * 100:+.1f}",
                "Vol % (ann.)": f"{vol * 100:.1f}",
                "Sharpe": f"{sr:.2f}",
                "Max DD %": f"{dd:.1f}",
            })
        import pandas as _pd
        st.dataframe(_pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)
        st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Performance", "📉 Risk", "🔢 Indicators",
        "⚖️ Weights & Signals", "📅 Sub-Periods", "💾 Export",
    ])

    with tab1:
        st.markdown('<p class="section-hdr">Equity Curve</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_equity_curve(result), use_container_width=True, key="chart_equity")
        st.markdown('<p class="section-hdr">Monthly Returns Heatmap</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_monthly_returns(result), use_container_width=True, key="chart_monthly")

    with tab2:
        st.markdown('<p class="section-hdr">Drawdown</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_drawdown(result), use_container_width=True, key="chart_drawdown")
        st.markdown('<p class="section-hdr">Rolling Sharpe Ratio (252d)</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_rolling_sharpe(result), use_container_width=True, key="chart_rolling_sharpe")
        st.markdown('<p class="section-hdr">Rolling Beta vs Index (252d)</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_rolling_beta(risk), use_container_width=True, key="chart_rolling_beta")
        st.markdown('<p class="section-hdr">Number of Positions</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_num_positions(result), use_container_width=True, key="chart_num_pos")

    with tab3:
        st.markdown('<p class="section-hdr">Comprehensive Indicators</p>', unsafe_allow_html=True)
        col_p, col_r = st.columns(2)

        with col_p:
            perf_dict = perf.summary_dict()
            rows_html = ""
            for k, v in perf_dict.items():
                if v == "":
                    rows_html += f'<tr class="section-row"><td colspan="2">{k}</td></tr>'
                else:
                    rows_html += f'<tr><td>{k}</td><td class="mono">{v}</td></tr>'
            st.markdown(f"""
            <div class="config-panel">
                <p class="panel-title">Performance</p>
                <table class="ind-table">{rows_html}</table>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            risk_dict = risk.summary_dict()
            rows_html = ""
            for k, v in risk_dict.items():
                if v == "":
                    rows_html += f'<tr class="section-row"><td colspan="2">{k}</td></tr>'
                else:
                    rows_html += f'<tr><td>{k}</td><td class="mono">{v}</td></tr>'
            st.markdown(f"""
            <div class="config-panel">
                <p class="panel-title">Risk</p>
                <table class="ind-table">{rows_html}</table>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<p class="section-hdr">Portfolio Weights Over Time</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_weights_over_time(result), use_container_width=True, key="chart_weights")

        dfs = result.to_dataframes()
        if not dfs["weights"].empty:
            st.markdown('<p class="section-hdr">Latest Weights</p>', unsafe_allow_html=True)
            last_date = dfs["weights"]["date"].max()
            last_w = dfs["weights"][dfs["weights"]["date"] == last_date].sort_values(
                "weight", ascending=False)
            st.dataframe(last_w, use_container_width=True)

        if not dfs["signals"].empty:
            st.markdown('<p class="section-hdr">Signal History (Last 200)</p>', unsafe_allow_html=True)
            st.dataframe(dfs["signals"].tail(200), use_container_width=True)

    with tab5:
        st.markdown('<p class="section-hdr">Sub-Period Analysis</p>', unsafe_allow_html=True)

        _oos_s = int(st.session_state.get("oos_start", OOS_START)[:4])
        _oos_e = int(st.session_state.get("oos_end",   OOS_END)[:4])
        sub_periods = {}
        y = _oos_s
        while y <= _oos_e:
            y2 = min(y + 1, _oos_e)
            label = f"{y}-{y2}" if y2 > y else str(y)
            sub_periods[label] = (f"{y}-01-01", f"{y2}-12-31")
            y = y2 + 1

        st.plotly_chart(plot_sub_period_bars(result, periods=sub_periods),
                        use_container_width=True, key="chart_sub_periods")

        rets = result.get_returns()
        rows = []
        for label, (s, e) in sub_periods.items():
            sub = rets.loc[s:e]
            if sub.empty:
                continue
            ann_ret = ((1 + sub).prod() ** (252 / len(sub)) - 1) * 100
            vol = sub.std() * np.sqrt(252) * 100
            sr = (sub.mean() / sub.std() * np.sqrt(252)) if sub.std() > 0 else 0
            cum = (1 + sub).cumprod()
            dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min() * 100
            rows.append({"Period": label, "Ann. Return %": f"{ann_ret:+.1f}",
                         "Vol %": f"{vol:.1f}", "Sharpe": f"{sr:.2f}",
                         "Max DD %": f"{dd:.1f}"})
        if rows:
            st.markdown('<p class="section-hdr">Detailed Breakdown</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab6:
        st.markdown('<p class="section-hdr">Data Export</p>', unsafe_allow_html=True)

        dfs = result.to_dataframes()
        dl_cols = st.columns(len([n for n, d in dfs.items() if d is not None and not d.empty]))
        col_idx = 0
        for name, df in dfs.items():
            if df is not None and not df.empty:
                csv = df.to_csv()
                with dl_cols[col_idx]:
                    st.markdown(f"""
                    <div class="metric-tile" style="margin-bottom:0.5rem;">
                        <p class="label">{name}</p>
                        <p class="value" style="font-size:1rem;">{len(df):,} rows</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        f"📥 {name}.csv", csv,
                        file_name=f"backtest_{name}.csv", mime="text/csv",
                        key=f"dl_{name}", use_container_width=True,
                    )
                col_idx += 1

        st.markdown('<p class="section-hdr">BBU Portfolio Export</p>', unsafe_allow_html=True)
        st.caption("CSV with columns **date, ticker, weight** — weights at each rebalance date.")

        wh = result.weights_history
        if wh:
            summary_df = PortfolioExporter.summary_table(wh)
            st.dataframe(summary_df, use_container_width=True)
            sorted_dates = sorted(wh.keys())
            st.markdown(f"""
            <div class="ctx-bar">
                <div class="ctx-item">Rebalances: <span>{len(sorted_dates)}</span></div>
                <div class="ctx-item">From: <span>{sorted_dates[0].strftime('%Y-%m-%d')}</span></div>
                <div class="ctx-item">To: <span>{sorted_dates[-1].strftime('%Y-%m-%d')}</span></div>
            </div>
            """, unsafe_allow_html=True)

            _strat_abbrev = "MOM" if "Momentum" in strategy_name else "MA"
            _alloc_abbrev = "ERC" if "ERC" in allocator_name else "EW"
            _ptf_name = f"PTF_{idx_id}_{_alloc_abbrev}_{_strat_abbrev}"
            _ccy = INDEX_CURRENCY.get(idx_id, "USD")
            bbu_content = PortfolioExporter.to_bbu(wh, portfolio_name=_ptf_name,
                                                   cash_ccy=_ccy)
            st.download_button("📥 Download BBU CSV", bbu_content,
                               file_name=f"{_ptf_name}.csv",
                               mime="text/csv", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 New Backtest", use_container_width=True):
        st.session_state.step = 1
        st.session_state.result = None
        st.session_state.pairs_results = None
        st.session_state["wf_schedule"] = None
        st.rerun()
