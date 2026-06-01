"""
Momentum strategy.
Ranks tickers by past return, selects the top N as LONG.
"""
import pandas as pd
from signals import BaseStrategy


class MomentumStrategy(BaseStrategy):

    def __init__(self, parameters=None):
        default = {"lookback_period": 252, "top_n": 10, "skip_recent": 21}
        params = {**default, **(parameters or {})}
        super().__init__("Momentum", params)

    def generate_signals(self, prices, date, members):
        lb = int(self.parameters["lookback_period"])
        top = int(self.parameters["top_n"])
        skip = int(self.parameters["skip_recent"])
        scores = {}

        for ticker in members:
            if ticker not in prices.columns:
                continue
            px = prices[ticker].loc[:date].dropna()
            if len(px) < lb:
                continue
            # on évite le dernier mois pour ne pas capter le retournement court terme
            end_idx = -skip if skip > 0 else None
            start_idx = -lb
            if end_idx is not None and abs(end_idx) < len(px):
                mom = px.iloc[end_idx] / px.iloc[start_idx] - 1
            else:
                mom = px.iloc[-1] / px.iloc[start_idx] - 1
            scores[ticker] = mom

        if not scores:
            return {}

        ranked = sorted(scores, key=scores.get, reverse=True)
        n = len(ranked)
        signals = {}
        for i, t in enumerate(ranked):
            if i < top:
                signals[t] = 1
            elif i >= n - top:
                signals[t] = -1
            else:
                signals[t] = 0
        return signals

    @staticmethod
    def get_parameters_schema():
        return {
            "lookback_period": {
                "type": "int", "min": 21, "max": 504,
                "default": 252, "label": "Lookback Period (days)",
            },
            "top_n": {
                "type": "int", "min": 1, "max": 50,
                "default": 10, "label": "Top N Stocks",
            },
            "skip_recent": {
                "type": "int", "min": 0, "max": 63,
                "default": 21, "label": "Skip Recent Days",
            },
        }
