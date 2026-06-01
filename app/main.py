"""Entry point Streamlit — execute via `streamlit run app/main.py`."""
import streamlit as st

st.set_page_config(page_title="Backtesting Framework", page_icon="📊", layout="wide")

from app import styles, sidebar
from app.steps import (
    step1_data, step2_index, step3_strategy,
    step4_parameters, step5_execution, step6_results, step7_replication,
)

styles.inject()

if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.result = None

sidebar.render()

_DISPATCH = {
    1: step1_data.render,
    2: step2_index.render,
    3: step3_strategy.render,
    4: step4_parameters.render,
    5: step5_execution.render,
    6: step6_results.render,
    7: step7_replication.render,
}

_DISPATCH[st.session_state.step]()
