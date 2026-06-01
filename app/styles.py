"""CSS - theme sombre trading-floor + composants visuels."""
import streamlit as st


_CSS = r"""
/* ── Global dark trading-floor theme ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e17 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin-bottom: 0.75rem;
}

/* ── Status cards ── */
.status-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.status-card:hover {
    border-color: #3b82f6;
}
.status-card.ok {
    border-left: 3px solid #10b981;
}
.status-card.ko {
    border-left: 3px solid #ef4444;
}
.status-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
}
.status-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0.25rem 0 0 0;
}
.status-ok { color: #10b981; }
.status-ko { color: #ef4444; }

/* ── Metric tiles ── */
.metric-tile {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
}
.metric-tile .label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.5rem 0;
}
.metric-tile .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}
.metric-tile .sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: #475569;
    margin: 0.25rem 0 0 0;
}

/* ── Index card ── */
.idx-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.idx-card .ticker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #3b82f6;
    font-weight: 600;
}
.idx-card .name {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #e2e8f0;
    font-weight: 600;
    margin: 0.15rem 0;
}
.idx-card .detail {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #64748b;
}

/* ── Risk-free card ── */
.rf-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.rf-card .ccy {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #8b5cf6;
    font-weight: 600;
}
.rf-card .obs {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #e2e8f0;
    margin: 0.15rem 0 0 0;
}

/* ── Section header ── */
.section-hdr {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e293b;
}

/* ── Sidebar step styling ── */
.step-active {
    background: rgba(59, 130, 246, 0.12);
    border-left: 3px solid #3b82f6;
    padding: 0.4rem 0.75rem;
    border-radius: 0 6px 6px 0;
    margin: 0.15rem 0;
    color: #e2e8f0 !important;
    font-weight: 600;
}
.step-done {
    padding: 0.4rem 0.75rem;
    margin: 0.15rem 0;
    color: #10b981 !important;
}
.step-pending {
    padding: 0.4rem 0.75rem;
    margin: 0.15rem 0;
    color: #475569 !important;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
}

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
}
.page-header .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #3b82f6;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.3rem 0;
}
.page-header .title {
    font-family: 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.02em;
}
.page-header .desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}

/* ── Selectable card ── */
.sel-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
    cursor: pointer;
}
.sel-card:hover {
    border-color: #3b82f6;
    transform: translateY(-2px);
}
.sel-card .icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
.sel-card .card-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0.25rem 0;
}
.sel-card .card-ticker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #3b82f6;
    font-weight: 600;
}
.sel-card .card-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.3rem;
}

/* ── Context bar ── */
.ctx-bar {
    background: linear-gradient(90deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08));
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.6rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.ctx-bar .ctx-item {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #94a3b8;
}
.ctx-bar .ctx-item span {
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
    font-weight: 600;
}

/* ── Config panel ── */
.config-panel {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.config-panel .panel-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.75rem 0;
}

/* ── Summary row ── */
.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e293b;
}
.summary-row:last-child { border-bottom: none; }
.summary-row .key {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
}
.summary-row .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
    font-weight: 600;
}

/* ── Execution terminal ── */
.exec-terminal {
    background: linear-gradient(145deg, #0a0e17, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem 0;
}
.exec-terminal .exec-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 1rem 0;
}

/* ── KPI card (results) ── */
.kpi-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.kpi-card .kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.4rem 0;
}
.kpi-card .kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
}
.kpi-positive { color: #10b981; }
.kpi-negative { color: #ef4444; }
.kpi-neutral  { color: #f1f5f9; }

/* ── Indicator table ── */
.ind-table {
    width: 100%;
    border-collapse: collapse;
}
.ind-table th {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #1e3a5f;
}
.ind-table td {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #e2e8f0;
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid rgba(30,58,95,0.4);
}
.ind-table td.mono {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.ind-table tr.section-row td {
    font-weight: 700;
    color: #3b82f6;
    padding-top: 0.8rem;
    border-bottom: 1px solid #1e3a5f;
}

/* ── Nav buttons ── */
.nav-row {
    display: flex;
    justify-content: space-between;
    margin-top: 1.5rem;
}

/* ── Streamlit tabs override ── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
"""


def inject() -> None:
    """A appeler une fois apres st.set_page_config."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
