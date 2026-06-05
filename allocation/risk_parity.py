"""Risk parity allocator: inverse-volatility weighting, normalized to 1."""
import pandas as pd
import numpy as np
from allocation import BaseAllocator


class RiskParityAllocator(BaseAllocator):

    def __init__(self, parameters=None):
        default = {"lookback_vol": 63}
        params = {**default, **(parameters or {})}
        super().__init__("ERC (Risk Parity)", params)

    def allocate(self, signals, returns, date):
        lb = int(self.parameters["lookback_vol"])

        longs = [t for t, s in signals.items() if s == 1]
        shorts = [t for t, s in signals.items() if s == -1]

        if not longs or not shorts:
            return {}

        def _inv_vol_weights(tickers):
            vols = {}
            for t in tickers:
                if t not in returns.columns:
                    continue
                r = returns[t].loc[:date].dropna()
                if len(r) >= lb:
                    vol = r.iloc[-lb:].std() * np.sqrt(252)
                    if vol > 0:
                        vols[t] = vol
            if not vols:
                # Not enough history, fall back to equal-weighting.
                available = [t for t in tickers if t in returns.columns]
                if not available:
                    return {}
                w = 1.0 / len(available)
                return {t: w for t in available}
            inv = {t: 1.0 / v for t, v in vols.items()}
            total = sum(inv.values())
            # Normalize weights so they sum to 1.
            return {t: iv / total for t, iv in inv.items()}

        weights = {}
        for t, w in _inv_vol_weights(longs).items():
            weights[t] = +w
        for t, w in _inv_vol_weights(shorts).items():
            weights[t] = -w
        return weights

    @staticmethod
    def get_parameters_schema():
        return {
            "lookback_vol": {
                "type": "int", "min": 21, "max": 252,
                "default": 63, "label": "Volatility Lookback (days)",
            },
        }
