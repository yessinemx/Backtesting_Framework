import unittest

import pandas as pd
import polars as pl

from research.paper_replication.outputs.paper_outputs import (
    build_paper_tables,
    build_paper_validation,
    build_table5,
)


class PaperOutputsTests(unittest.TestCase):
    def test_build_table5_keeps_var95_when_present(self) -> None:
        report = {
            "summaries": {
                "distance": pl.DataFrame(
                    {
                        "variant": ["standard"],
                        "max_drawdown": [-0.12],
                        "var_95": [-0.03],
                        "cvar_95": [-0.05],
                        "pct_positive": [0.6],
                    }
                )
            }
        }

        table = build_table5(report)

        self.assertEqual(table.columns, ["method", "variant", "max_drawdown", "var_95", "cvar_95", "pct_positive"])
        self.assertEqual(table.to_dicts()[0]["var_95"], -0.03)

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


class PaperValidationTests(unittest.TestCase):
    def _comparison(self):
        return pl.DataFrame(
            [
                {
                    "method": "distance",
                    "repl_std_return_%": -0.55,
                    "repl_wav_return_%(honest)": 11.50,
                    "repl_wav_return_%(paper)": 12.10,
                    "paper_std_return_%": -0.55,
                    "paper_wav_return_%": 11.82,
                    "repl_std_sharpe": -0.20,
                    "repl_wav_sharpe(honest)": 3.55,
                    "repl_wav_sharpe(paper)": 3.80,
                    "paper_std_sharpe": -0.21,
                    "paper_wav_sharpe": 3.69,
                },
                {
                    "method": "cointegration",
                    "repl_std_return_%": -8.0,
                    "repl_wav_return_%(honest)": 9.50,
                    "repl_wav_return_%(paper)": None,
                    "paper_std_return_%": -1.81,
                    "paper_wav_return_%": 9.66,
                    "repl_std_sharpe": -1.50,
                    "repl_wav_sharpe(honest)": 2.79,
                    "repl_wav_sharpe(paper)": None,
                    "paper_std_sharpe": -0.40,
                    "paper_wav_sharpe": 2.82,
                },
            ]
        )

    def test_headline_validation_flags_pass_warn_fail_and_missing(self):
        report = {
            "comparison": self._comparison(),
            "universe_pool": 410,
            "n_periods": 7,
            "alpha_table": pd.DataFrame(
                {
                    "factor_family": ["fama_french", "q_factor", "petkova_icapm"],
                    "model": ["FF five factor I", "q-factor (Hou-Xue-Zhang)", "Petkova ICAPM"],
                    "alpha_annual_pct": [4.0, 3.5, 2.8],
                }
            ),
        }

        table = build_paper_validation(report).to_pandas()
        headline = table[table["category"] == "headline"]

        # Distance / wavelet_honest / mean_return -> within PASS band (|delta|=0.32 <= 1.0)
        row = headline[
            (headline["method"] == "distance")
            & (headline["variant"] == "wavelet_honest")
            & (headline["metric"] == "mean_return_pct")
        ].iloc[0]
        self.assertEqual(row["status"], "PASS")
        self.assertAlmostEqual(row["abs_delta"], -0.32, places=4)

        # Cointegration / standard / mean_return -> FAIL (|delta|=6.19 > 3.0)
        row = headline[
            (headline["method"] == "cointegration")
            & (headline["variant"] == "standard")
            & (headline["metric"] == "mean_return_pct")
        ].iloc[0]
        self.assertEqual(row["status"], "FAIL")

        # Cointegration / standard / sharpe -> FAIL (|delta|=1.10 > 0.80)
        row = headline[
            (headline["method"] == "cointegration")
            & (headline["variant"] == "standard")
            & (headline["metric"] == "sharpe")
        ].iloc[0]
        self.assertEqual(row["status"], "FAIL")

        # Missing variant (cointegration wavelet_paper)
        row = headline[
            (headline["method"] == "cointegration")
            & (headline["variant"] == "wavelet_paper")
            & (headline["metric"] == "mean_return_pct")
        ].iloc[0]
        self.assertEqual(row["status"], "MISSING")

        # Structural rows
        structural = table[table["category"] == "structural"]
        self.assertIn("universe_pool_size", set(structural["metric"]))
        self.assertIn("n_trading_periods", set(structural["metric"]))
        self.assertIn("factor_regressions_available", set(structural["metric"]))
        factor_row = structural[structural["metric"] == "factor_regressions_available"].iloc[0]
        self.assertEqual(factor_row["status"], "PASS")

    def test_factor_row_flags_status_only(self):
        report = {
            "comparison": pl.DataFrame([]),
            "universe_pool": 415,
            "n_periods": 7,
            "alpha_table": pd.DataFrame(
                {"factor_family": ["factor_data_status"], "model": ["status"], "alpha_annual_pct": [0.0]}
            ),
        }
        table = build_paper_validation(report).to_pandas()
        factor_row = table[table["metric"] == "factor_regressions_available"].iloc[0]
        self.assertEqual(factor_row["status"], "FAIL")
        self.assertIn("factors.csv", factor_row["comment"])

    def test_validation_integrated_in_build_paper_tables(self):
        report = {
            "prices": pl.DataFrame({"date": []}),
            "periods": [],
            "comparison": self._comparison(),
            "summaries": {},
            "universe_pool": 415,
            "n_periods": 7,
            "alpha_table": None,
            "figure_diagnostics": {"runs": {}},
        }
        tables = build_paper_tables(report)
        self.assertIn("paper_validation_summary", tables)
        self.assertGreater(tables["paper_validation_summary"].height, 0)


if __name__ == "__main__":
    unittest.main()
