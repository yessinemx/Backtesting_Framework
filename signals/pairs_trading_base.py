"""Base class for all pairs trading strategies."""
from __future__ import annotations

from signals.base import BaseStrategy


# Parameters that are identical in every pairs trading variant.
_COMMON_DEFAULTS: dict = {
    "formation_period": 252,
    "top_n_pairs": 20,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5,
    "min_history": 60,
}

# Corresponding Streamlit slider definitions for the common parameters.
_COMMON_SCHEMA: dict = {
    "formation_period": {
        "type": "int", "min": 60, "max": 504,
        "default": 252, "label": "Formation Period (days)",
    },
    "top_n_pairs": {
        "type": "int", "min": 1, "max": 100,
        "default": 20, "label": "Top N Pairs",
    },
    "entry_threshold": {
        "type": "float", "min": 0.5, "max": 5.0,
        "default": 2.0, "label": "Entry Z-Score",
    },
    "exit_threshold": {
        "type": "float", "min": 0.0, "max": 2.0,
        "default": 0.5, "label": "Exit Z-Score",
    },
    "min_history": {
        "type": "int", "min": 20, "max": 252,
        "default": 60, "label": "Min History (days)",
    },
}


class PairsTradingBase(BaseStrategy):
    """Shared interface for pairs trading variants."""

    #: Detected by the Streamlit app to trigger composite (3-strategy) execution.
    IS_PAIRS_STRATEGY: bool = True

    def __init__(self, name: str, extra_defaults: dict | None = None,
                 parameters: dict | None = None):
        """Merge defaults with strategy-specific and user overrides."""
        merged = {**_COMMON_DEFAULTS, **(extra_defaults or {}), **(parameters or {})}
        super().__init__(name, merged)

    @classmethod
    def _common_schema(cls) -> dict:
        """Return a copy of the common parameter schema."""
        return dict(_COMMON_SCHEMA)

    def reset_state(self) -> None:
        """Clear cached pairs and open positions between backtests."""
        pass
