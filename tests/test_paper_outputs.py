import unittest

import pandas as pd
import polars as pl

from research.paper_replication.paper_outputs import build_paper_tables


class PaperOutputsTests(unittest.TestCase):
    def test_computed_tables_are_not_overwritten_by_placeholders(self) -> None:
        report = {
            "prices": pl.DataFrame({"date": []}),
            "periods": [],
            "comparison": pl.DataFrame([]),
            "summaries": {},
            "alpha_table": None,
            "figure_diagnostics": {
                "runs": {},
                "wavelet_sweeps": {
                    "distance": pd.DataFrame(
                        {
                            "wavelet": ["sym20"],
                            "mean_return": [1.23],
                            "sharpe": [0.45],
                        }
                    )
                },
                "horizon_sweeps": {
                    "distance": pd.DataFrame(
                        {
                            "horizon_days": [63],
                            "standard": [2.0],
                            "wavelet": [3.0],
                        }
                    )
                },
            },
        }

        tables = build_paper_tables(report)

        self.assertEqual(
            set(tables["table16_sharpe_ratios_under_different_wavelet_classes"].columns),
            {"wavelet", "mean_return", "sharpe", "method"},
        )
        self.assertEqual(
            set(tables["table17_pairs_trading_returns_at_different_trading_period_spans"].columns),
            {"horizon_days", "standard", "wavelet", "method"},
        )


if __name__ == "__main__":
    unittest.main()
