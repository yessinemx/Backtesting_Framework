"""
Moteur de backtest returns-based avec frais de transaction.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from portfolio.history_tracker import HistoryTracker


@dataclass
class BacktestResult:
    # stocker toutes les sorties du backtest dans un seul objet
    config: dict
    tracker: HistoryTracker = field(default_factory=HistoryTracker)
    riskfree_curve: pd.Series = field(default_factory=pd.Series)
    riskfree_daily: pd.Series = field(default_factory=pd.Series)

    def get_equity_curve(self):
        return self.tracker.get_equity_curve()

    def get_returns(self):
        return self.tracker.get_returns()

    @property
    def equity_curve(self):
        return self.tracker.get_equity_curve()

    @property
    def daily_returns(self):
        return self.tracker.get_returns()

    @property
    def weights_history(self):
        return self.tracker.weights_history

    @property
    def signals_history(self):
        return self.tracker.signals_history

    @property
    def rebalance_dates(self):
        return self.tracker.rebalance_dates

    @property
    def total_transaction_costs(self):
        return self.tracker.total_transaction_costs

    @property
    def transaction_costs_history(self):
        return self.tracker.transaction_costs_history

    def to_dataframes(self):
        return {
            "equity":  self.equity_curve.to_frame("value") if not self.equity_curve.empty else pd.DataFrame(),
            "returns": self.daily_returns.to_frame("return") if not self.daily_returns.empty else pd.DataFrame(),
            "weights":  self.tracker.get_weights_df(),
            "signals":  self.tracker.get_signals_df(),
            "riskfree": self.riskfree_curve.to_frame("rf_cum") if not self.riskfree_curve.empty else pd.DataFrame(),
        }


class BacktestEngine:

    def __init__(self, config):
        self.config = config
        self.initial_capital = config.get("initial_capital", 1_000_000)
        self.rebalance_months = config.get("rebalance_months", 1)
        self.index_id = config.get("index_id", "SX5E")
        self.start = config.get("start_date", "2015-01-01")
        self.end = config.get("end_date", "2024-12-31")
        self.tc_bps = config.get("transaction_cost_bps", 0)
        self.borrow_bps = config.get("short_borrow_bps", 0)
        self.strategy = None
        self.allocator = None

    def set_strategy(self, strategy):
        self.strategy = strategy

    def set_allocator(self, allocator):
        self.allocator = allocator

    def run(self, prices, returns, membership, riskfree_daily=None,
            progress_callback=None, params_schedule=None):

        prices_oos = prices.loc[self.start:self.end]
        returns_oos = returns.loc[self.start:self.end]
        mem = membership[membership["index_id"] == self.index_id].copy()

        trading_dates = prices_oos.index

        # calendrier de rebalancement : dernier jour ouvré de chaque N mois
        month_ends = trading_dates.to_series().groupby(
            trading_dates.to_period("M")
        ).last()
        rebal_dates_raw = month_ends.iloc[::self.rebalance_months].values
        rebal_set = set(pd.DatetimeIndex(rebal_dates_raw))

        if len(trading_dates) > 0 and trading_dates[0] not in rebal_set:
            rebal_set.add(trading_dates[0])

        tc_rate = self.tc_bps / 10_000.0

        result = BacktestResult(
            config=self.config,
            tracker=HistoryTracker(initial_capital=self.initial_capital),
        )
        tracker = result.tracker

        w = {}                                      # poids courants (fixes entre rebals)
        cum_value = float(self.initial_capital)
        total = len(trading_dates)

        for i, date in enumerate(trading_dates):
            # P&L du jour avec les poids courants
            daily_ret = 0.0
            for t, wt in w.items():
                if t in returns_oos.columns:
                    r = returns_oos.at[date, t] if date in returns_oos.index else 0
                    if np.isnan(r):
                        r = 0
                    daily_ret += wt * r

            cum_value *= (1 + daily_ret)

            # coût d'emprunt journalier sur les positions short
            if self.borrow_bps > 0 and w:
                short_notional = sum(-wt for wt in w.values() if wt < 0)
                borrow_cost = short_notional * (self.borrow_bps / 10_000.0) / 252
                if borrow_cost > 0:
                    cum_value *= (1 - borrow_cost)
                    daily_ret = (1 + daily_ret) * (1 - borrow_cost) - 1

            # rebalancement après le P&L du jour
            if date in rebal_set:
                members = self._get_members(mem, date)
                if members and self.strategy and self.allocator:
                    # mise à jour walk-forward si schedule fourni
                    if params_schedule:
                        sched_p = params_schedule.get(date)
                        if sched_p:
                            self.strategy.parameters.update(sched_p)

                    signals = self.strategy.generate_signals(
                        prices.loc[:date], date, members
                    )
                    new_w = self.allocator.allocate(
                        signals, returns.loc[:date], date
                    )

                    # frais TC = tc_rate × turnover
                    turnover = sum(
                        abs(new_w.get(t, 0) - w.get(t, 0))
                        for t in set(list(new_w.keys()) + list(w.keys()))
                    )
                    tc_cost = turnover * tc_rate
                    cum_value *= (1 - tc_cost)
                    daily_ret = (1 + daily_ret) * (1 - tc_cost) - 1

                    tracker.record_rebalance(date, new_w, signals, tc_cost)
                    w = new_w

            tracker.record_day(date, cum_value, daily_ret)

            if progress_callback:
                progress_callback((i + 1) / total, str(date.date()))

        # Risk-free cumulative track
        if riskfree_daily is not None:
            rf_oos = riskfree_daily.reindex(trading_dates).fillna(0)
            result.riskfree_daily = rf_oos
            result.riskfree_curve = (1 + rf_oos).cumprod() * self.initial_capital
        else:
            result.riskfree_daily = pd.Series(0, index=trading_dates)
            result.riskfree_curve = pd.Series(
                self.initial_capital, index=trading_dates
            )

        return result

    @staticmethod
    def _get_members(mem, date):
        available = mem[mem["date"] <= date]
        if available.empty:
            return []
        latest = available["date"].max()
        return available.loc[available["date"] == latest, "ticker"].tolist()
