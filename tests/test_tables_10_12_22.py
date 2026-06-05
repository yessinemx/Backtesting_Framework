"""Tests for the in-house implementations of paper Tables 10, 12 and 22.

These tables used to be flagged as "not implemented" but now ship pure-Python
replicas. The tests guard against silent regressions:

* Table 10  -- PCA marginal variance explained.
* Table 12  -- Monte-Carlo high-frequency contamination simulation.
* Table 22  -- Forced-close standard-pairs profits.
"""
import unittest
from datetime import date, timedelta
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl

from research.paper_replication.core.periods import Period
from research.paper_replication.outputs.paper_outputs import (
    build_table10,
    build_table12,
    build_table22,
)


def _toy_prices(n_days: int = 60, n_assets: int = 6, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2010, 1, 4)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    returns = rng.normal(0.0, 0.01, size=(n_days, n_assets))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    data = {"date": dates}
    for k in range(n_assets):
        data[f"T{k:02d}"] = prices[:, k]
    return pl.DataFrame(data)


class Table10Tests(unittest.TestCase):
    def test_pca_returns_components_summing_to_100_pct(self) -> None:
        prices = _toy_prices(n_days=80, n_assets=5)
        dates = prices.get_column("date")
        period = Period(index=1, train_slice=(0, 40), trade_slice=(40, 40),
                        train_dates=dates.slice(0, 40), trade_dates=dates.slice(40, 40))
        table = build_table10(prices, [period], n_components=5)
        self.assertFalse(table.is_empty())
        self.assertSetEqual(set(table.columns),
                            {"period", "component", "variance_explained_pct",
                             "cumulative_variance_explained_pct", "n_assets"})
        last = table.filter(pl.col("component") == 5)
        self.assertAlmostEqual(last["cumulative_variance_explained_pct"][0], 100.0, places=2)

    def test_empty_when_no_periods(self) -> None:
        table = build_table10(_toy_prices(), periods=[])
        self.assertTrue(table.is_empty())


class Table12Tests(unittest.TestCase):
    def test_monte_carlo_matrix_shape_and_coverage(self) -> None:
        # 3 beta x 4 sigma2_2 x 4 sigma2_1 x 2 I-orders = 96 rows
        table = build_table12(n_replications=2)
        self.assertEqual(table.height, 96)
        self.assertSetEqual(set(table.columns),
                            {"beta_true", "sigma2_2", "sigma2_1", "error_integration",
                             "beta_hat_standard", "beta_hat_wavelet",
                             "msfe_standard", "msfe_wavelet", "n_replications"})
        self.assertSetEqual(set(table["error_integration"].to_list()), {"I(0)", "I(1)"})
        self.assertSetEqual(set(table["beta_true"].to_list()), {1.0, 2.0, 4.0})

    def test_wavelet_msfe_is_finite(self) -> None:
        table = build_table12(n_replications=2)
        msfe = table["msfe_wavelet"].to_numpy()
        self.assertTrue(np.all(np.isfinite(msfe)))


@dataclass
class _StubRun:
    forced_close_stats: pd.DataFrame


class Table22Tests(unittest.TestCase):
    def test_forced_close_table_aggregates_runs(self) -> None:
        rows_a = pd.DataFrame([
            {"period": 1, "variant": "standard_forced_close", "mean_return": 0.01,
             "sharpe": 0.4, "pct_positive": 0.55, "n_active": 12, "n_trades": 30},
            {"period": 2, "variant": "standard_forced_close", "mean_return": 0.02,
             "sharpe": 0.6, "pct_positive": 0.6, "n_active": 14, "n_trades": 35},
        ])
        rows_b = pd.DataFrame([
            {"period": 1, "variant": "standard_forced_close", "mean_return": -0.005,
             "sharpe": -0.1, "pct_positive": 0.48, "n_active": 11, "n_trades": 28},
        ])
        runs = {"distance": _StubRun(rows_a), "cointegration": _StubRun(rows_b)}
        table = build_table22(runs)
        self.assertFalse(table.is_empty())
        self.assertIn("method", table.columns)
        methods = set(table["method"].to_list())
        self.assertSetEqual(methods, {"distance", "cointegration"})

    def test_empty_runs_yield_empty_table(self) -> None:
        runs = {"distance": _StubRun(pd.DataFrame())}
        table = build_table22(runs)
        self.assertTrue(table.is_empty())


if __name__ == "__main__":
    unittest.main()
