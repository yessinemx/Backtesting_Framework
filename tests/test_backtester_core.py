import unittest

import pandas as pd

from optimization.grid_search import GridSearch
from portfolio.backtest_engine import BacktestEngine


class _StaticStrategy:
    def __init__(self) -> None:
        self.parameters = {}

    def generate_signals(self, prices, date, members):
        del prices, date, members
        return {"AAA": 1, "BBB": -1}


class _StaticAllocator:
    def allocate(self, signals, returns, date):
        del signals, returns, date
        return {"AAA": 0.5, "BBB": -0.5}


class BacktesterCoreTests(unittest.TestCase):
    def test_backtest_engine_rebalances_and_tracks_costs(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-31"])
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 110.0],
                "BBB": [100.0, 90.0, 90.0],
            },
            index=dates,
        )
        returns = pd.DataFrame(
            {
                "AAA": [0.0, 0.10, 0.0],
                "BBB": [0.0, -0.10, 0.0],
            },
            index=dates,
        )
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[0]],
                "index_id": ["TEST", "TEST"],
                "ticker": ["AAA", "BBB"],
            }
        )
        riskfree = pd.Series([0.001, 0.001, 0.001], index=dates)

        engine = BacktestEngine(
            {
                "initial_capital": 1000.0,
                "rebalance_months": 1,
                "index_id": "TEST",
                "start_date": "2024-01-02",
                "end_date": "2024-01-31",
                "transaction_cost_bps": 10,
            }
        )
        engine.set_strategy(_StaticStrategy())
        engine.set_allocator(_StaticAllocator())

        result = engine.run(prices, returns, membership, riskfree_daily=riskfree)

        self.assertEqual(result.rebalance_dates, [dates[0], dates[-1]])
        self.assertAlmostEqual(result.transaction_costs_history[dates[0]], 0.001)
        self.assertAlmostEqual(result.total_transaction_costs, 0.001)
        self.assertAlmostEqual(result.daily_returns.iloc[0], -0.001)
        self.assertAlmostEqual(result.daily_returns.iloc[1], 0.1)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 1098.9)
        self.assertAlmostEqual(result.riskfree_curve.iloc[0], 1001.0)

    def test_grid_search_run_ma_filters_invalid_window_combos(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100, 101, 102, 103, 104, 105],
                "BBB": [105, 104, 103, 102, 101, 100],
            },
            index=dates,
        )
        returns = prices.pct_change().fillna(0.0)
        membership = pd.DataFrame(
            {
                "date": [dates[0], dates[0]],
                "index_id": ["TEST", "TEST"],
                "ticker": ["AAA", "BBB"],
            }
        )
        riskfree = pd.Series(0.0, index=dates)
        gs = GridSearch(prices, returns, membership, riskfree, "TEST", str(dates[0].date()), str(dates[-1].date()))

        results = gs.run_ma(
            {
                "fast_window": [2, 3],
                "slow_window": [2, 4],
                "signal_threshold": [0.0],
            }
        )

        combos = set(zip(results["fast_window"], results["slow_window"]))
        self.assertEqual(combos, {(2, 4), (3, 4)})
        self.assertTrue((results["fast_window"] < results["slow_window"]).all())


if __name__ == "__main__":
    unittest.main()