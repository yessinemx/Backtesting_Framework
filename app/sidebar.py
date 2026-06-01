"""Sidebar with the logo, step list, and reset button."""
import streamlit as st


_STEPS = {
    1: ("📂", "Data Status"),
    2: ("🔍", "Index Selection"),
    3: ("⚙️", "Strategy & Alloc"),
    4: ("🔧", "Parameters"),
    5: ("🚀", "Execution"),
    6: ("📊", "Results"),
    7: ("📚", "Paper Replication"),
}


def render() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                <div style="font-size:1.8rem; margin-bottom:0.2rem;">⚡</div>
                <div style="font-family:'Inter',sans-serif; font-size:1.1rem; font-weight:700; color:#e2e8f0 !important; letter-spacing:-0.02em;">
                    QUANT BACKTEST
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#475569 !important; letter-spacing:0.1em;">
                    SYSTEMATIC STRATEGIES
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        for sn, (icon, label) in _STEPS.items():
            if sn == st.session_state.step:
                st.markdown(f'<div class="step-active">{icon} {label}</div>', unsafe_allow_html=True)
            elif sn < st.session_state.step:
                st.markdown(f'<div class="step-done">✅ {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="step-pending">○ {label}</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("📚 Open Paper Replication", use_container_width=True):
            st.session_state.step = 7
            st.rerun()

        st.divider()
        if st.button("🔄 Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
