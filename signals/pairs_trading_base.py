"""
Abstract base class for all Pairs Trading strategies.

All three variants (Wavelet, Cointegration, Partial Cointegration) inherit
from ``PairsTradingBase`` which:
  - holds the five parameters shared by every pairs trading strategy;
  - provides ``_common_schema()`` so subclasses can build their schema without
    repeating the same boilerplate;
  - adds the ``IS_PAIRS_STRATEGY = True`` sentinel so the app layer can
    detect composite execution mode;
  - provides a default no-op ``reset_state()`` so the backtest engine's
    auto-reset hook is always safe to call.
"""
from __future__ import annotations

from signals import BaseStrategy


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
    """Shared interface for wavelet, cointegration, and partial-cointegration
    pairs trading strategies.

    Subclasses must implement:
      - ``generate_signals(prices, date, members)``
      - ``get_parameters_schema()`` (staticmethod)

    ``reset_state()`` has a no-op default and should be overridden by stateful
    subclasses that maintain a per-pair position dictionary between calls.
    """

    #: Detected by the Streamlit app to trigger composite (3-strategy) execution.
    IS_PAIRS_STRATEGY: bool = True

    def __init__(self, name: str, extra_defaults: dict | None = None,
                 parameters: dict | None = None):
        """Merge common defaults ← subclass-specific defaults ← user parameters.

        Parameters
        ----------
        name : str
            Strategy display name passed through to ``BaseStrategy``.
        extra_defaults : dict | None
            Strategy-specific default values (override common defaults where
            keys overlap, e.g. ``top_n_pairs`` for Partial Cointegration).
        parameters : dict | None
            User-supplied overrides (highest priority).
        """
        merged = {**_COMMON_DEFAULTS, **(extra_defaults or {}), **(parameters or {})}
        super().__init__(name, merged)

    @classmethod
    def _common_schema(cls) -> dict:
        """Return a *copy* of the common parameter schema dict.

        Subclass ``get_parameters_schema()`` implementations should start with
        ``schema = PairsTradingBase._common_schema()`` and then add or update
        their own entries before returning.
        """
        return dict(_COMMON_SCHEMA)

    def reset_state(self) -> None:
        """Drop cached pairs and open positions between independent backtests.

        The backtest engine calls this at the start of every ``run()``; the
        default is a no-op for backward compatibility with stateless subclasses.
        """
        pass
