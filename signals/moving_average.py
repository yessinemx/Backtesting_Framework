"""MA crossover: long when fast > slow, short when fast < slow."""
import pandas as pd
from signals.base import BaseStrategy


class MovingAverageCrossover(BaseStrategy):

    def __init__(self, parameters=None):
        default = {"fast_window": 20, "slow_window": 50, "signal_threshold": 0.0}
        params = {**default, **(parameters or {})}
        super().__init__("Moving Average Crossover", params)

    def generate_signals(self, prices, date, members):
        fw = int(self.parameters["fast_window"])
        sw = int(self.parameters["slow_window"])
        thr = float(self.parameters["signal_threshold"])
        signals = {}

        for ticker in members:
            if ticker not in prices.columns:
                continue
            px = prices[ticker].loc[:date].dropna()
            if len(px) < sw:
                continue
            fast = px.iloc[-fw:].mean()
            slow = px.iloc[-sw:].mean()
            if fast > slow * (1 + thr):
                signals[ticker] = 1
            elif fast < slow * (1 - thr):
                signals[ticker] = -1
            else:
                signals[ticker] = 0
        return signals

    @staticmethod
    def get_parameters_schema():
        return {
            "fast_window": {
                "type": "int", "min": 5, "max": 100,
                "default": 20, "label": "Fast MA Window",
            },
            "slow_window": {
                "type": "int", "min": 10, "max": 300,
                "default": 50, "label": "Slow MA Window",
            },
            "signal_threshold": {
                "type": "float", "min": 0.0, "max": 0.10,
                "default": 0.0, "label": "Signal Threshold",
            },
        }
