"""Records daily equity, returns, weights, signals, and transaction costs during a backtest."""
import pandas as pd
import numpy as np


class HistoryTracker:

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self._dates = []
        self._values = []
        self._returns = []
        self.weights_history = {}
        self.signals_history = {}
        self.rebalance_dates = []
        self.transaction_costs_history = {}

    def record_day(self, date, portfolio_value: float, daily_return: float):
        self._dates.append(date)
        self._values.append(portfolio_value)
        self._returns.append(daily_return)

    def record_rebalance(self, date, weights, signals, tc_cost=0):
        self.weights_history[date] = weights
        self.signals_history[date] = signals
        self.transaction_costs_history[date] = tc_cost
        if date not in self.rebalance_dates:
            self.rebalance_dates.append(date)

    @property
    def total_transaction_costs(self):
        return sum(self.transaction_costs_history.values())

    def get_equity_curve(self):
        if not self._dates:
            return pd.Series(dtype=float)
        return pd.Series(self._values,
                         index=pd.DatetimeIndex(self._dates),
                         name="equity")

    def get_returns(self):
        if not self._dates:
            return pd.Series(dtype=float)
        return pd.Series(self._returns,
                         index=pd.DatetimeIndex(self._dates),
                         name="return")

    def get_weights_df(self):
        if not self.weights_history:
            return pd.DataFrame()
        rows = []
        for dt, wts in self.weights_history.items():
            for t, w in wts.items():
                rows.append({"date": dt, "ticker": t, "weight": w})
        return pd.DataFrame(rows)

    def get_signals_df(self):
        if not self.signals_history:
            return pd.DataFrame()
        rows = []
        for dt, sigs in self.signals_history.items():
            for t, s in sigs.items():
                rows.append({"date": dt, "ticker": t, "signal": s})
        return pd.DataFrame(rows)
