from unittest.mock import patch
import unittest

import pandas as pd

from signals.pairs_trading import PairsTradingWavelet


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

        with patch("signals.pairs_trading._wavelet_denoise", side_effect=lambda series, wavelet="db4", level=2: series):
            signals = strategy.generate_signals(prices, prices.index[-1], ["AAA", "BBB"])

        self.assertEqual(signals["AAA"], -1)
        self.assertEqual(signals["BBB"], 1)
        self.assertTrue(set(signals.values()).issubset({-1, 0, 1}))


if __name__ == "__main__":
    unittest.main()