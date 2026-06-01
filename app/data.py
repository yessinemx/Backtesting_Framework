"""Polars loaders converted to pandas and cached for Streamlit."""
import streamlit as st

import loaders
from loaders.base import to_pandas_wide


@st.cache_data(show_spinner=False)
def load_prices():
    return to_pandas_wide(loaders.load_prices())


@st.cache_data(show_spinner=False)
def load_returns():
    return to_pandas_wide(loaders.load_returns())


@st.cache_data(show_spinner=False)
def load_membership():
    return loaders.load_membership().to_pandas()


@st.cache_data(show_spinner=False)
def load_riskfree():
    return to_pandas_wide(loaders.load_riskfree())
