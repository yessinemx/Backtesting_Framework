from unittest.mock import patch
import unittest

import pandas as pd
import numpy as np

from signals.pairs_trading_wavelet import PairsTradingWavelet
from signals.pairs_trading_cointegration import PairsTradingCointegration
from signals.pairs_trading_partial_cointegration import PairsTradingPartialCointegration


class PairsTradingStrategyTests(unittest.TestCase):
    def test_generate_signals_returns_empty_without_enough_history(self) -> None:
        strategy = PairsTradingWavelet({"formation_period": 10, "min_history": 10, "wavelet_level": 2})
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0],
                "BBB": [100.0, 99.0, 98.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )

        signals = strategy.generate_signals(prices, prices.index[-1], ["AAA", "BBB"])

        self.assertEqual(signals, {})

    def test_generate_signals_produces_long_short_pair(self) -> None:
        strategy = PairsTradingWavelet(
            {
                "formation_period": 6,
                "top_n_pairs": 1,
                "entry_threshold": 1.0,
                "exit_threshold": 0.25,
                "wavelet_level": 1,
                "min_history": 6,
            }
        )
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 100.0, 100.0, 100.0, 100.0, 120.0],
                "BBB": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=pd.date_range("2024-01-01", periods=6, freq="B"),
        )

        with patch("signals.pairs_trading_wavelet._wavelet_denoise", side_effect=lambda series, wavelet="db4", level=2: series):
            signals = strategy.generate_signals(prices, prices.index[-1], ["AAA", "BBB"])

        self.assertEqual(signals["AAA"], -1)
        self.assertEqual(signals["BBB"], 1)
        self.assertTrue(set(signals.values()).issubset({-1, 0, 1}))


class CointegrationStrategyTests(unittest.TestCase):
    """Minimal smoke tests for the Engle-Granger cointegration benchmark."""

    def _make_prices(self, n=80):
        rng = np.random.default_rng(0)
        base = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
        prices = pd.DataFrame(
            {"AAA": np.exp(base), "BBB": np.exp(base + rng.normal(0, 0.005, n))},
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        return prices

    def test_returns_empty_without_enough_history(self):
        strategy = PairsTradingCointegration({"formation_period": 60, "min_history": 60})
        prices = self._make_prices(n=10)
        signals = strategy.generate_signals(prices, prices.index[-1], list(prices.columns))
        self.assertEqual(signals, {})

    def test_returns_valid_signal_values(self):
        strategy = PairsTradingCointegration(
            {"formation_period": 60, "min_history": 60, "top_n_pairs": 1,
             "entry_threshold": 0.5, "reselect_every": 1}
        )
        prices = self._make_prices(n=80)
        signals = strategy.generate_signals(prices, prices.index[-1], list(prices.columns))
        self.assertTrue(set(signals.values()).issubset({-1, 0, 1}))

    def test_reset_state_clears_cache(self):
        strategy = PairsTradingCointegration({"formation_period": 60, "min_history": 60})
        prices = self._make_prices(n=80)
        strategy.generate_signals(prices, prices.index[-1], list(prices.columns))
        strategy.reset_state()
        self.assertIsNone(strategy._pairs_cache)
        self.assertEqual(strategy._pair_states, {})


class PartialCointegrationStrategyTests(unittest.TestCase):
    """Minimal smoke tests for the Clegg/Kalman partial cointegration benchmark."""

    def _make_prices(self, n=80):
        rng = np.random.default_rng(1)
        base = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
        prices = pd.DataFrame(
            {"AAA": np.exp(base), "BBB": np.exp(base + rng.normal(0, 0.005, n))},
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        return prices

    def test_returns_empty_without_enough_history(self):
        strategy = PairsTradingPartialCointegration({"formation_period": 60, "min_history": 60})
        prices = self._make_prices(n=10)
        signals = strategy.generate_signals(prices, prices.index[-1], list(prices.columns))
        self.assertEqual(signals, {})

    def test_returns_valid_signal_values(self):
        strategy = PairsTradingPartialCointegration(
            {"formation_period": 60, "min_history": 60, "top_n_pairs": 1,
             "entry_threshold": 0.5, "reselect_every": 1}
        )
        prices = self._make_prices(n=80)
        signals = strategy.generate_signals(prices, prices.index[-1], list(prices.columns))
        self.assertTrue(set(signals.values()).issubset({-1, 0, 1}))

    def test_reset_state_clears_cache(self):
        strategy = PairsTradingPartialCointegration({"formation_period": 60, "min_history": 60})
        prices = self._make_prices(n=80)
        strategy.generate_signals(prices, prices.index[-1], list(prices.columns))
        strategy.reset_state()
        self.assertIsNone(strategy._pairs_cache)
        self.assertEqual(strategy._pair_states, {})


if __name__ == "__main__":
    unittest.main()