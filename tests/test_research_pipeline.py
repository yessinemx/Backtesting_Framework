from datetime import datetime
from unittest.mock import patch
import unittest

import polars as pl

from research.paper_replication.metrics import PairsReport
from research.paper_replication.pipeline import PipelineResult
import research.paper_replication.pipeline as pipeline_module


def _report(variant: str, mean_return: float, n_pairs: int, n_active: int) -> PairsReport:
    return PairsReport(
        method="distance",
        variant=variant,
        n_pairs=n_pairs,
        n_active=n_active,
        mean_return=mean_return,
        std_return=0.1,
        sharpe=1.2,
        skewness=0.0,
        kurtosis=0.0,
        max_drawdown=-0.05,
        var_95=-0.02,
        cvar_95=-0.03,
        pct_positive=0.5,
        n_full=1,
        n_partial=0,
        n_non=0,
        n_inactive=0,
    )


class ResearchPipelineTests(unittest.TestCase):
    def test_run_pipeline_aggregates_non_empty_periods(self) -> None:
        prices = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "AAA": [100.0, 101.0],
                "BBB": [100.0, 99.0],
            }
        )
        params = {"block_size": 5, "max_periods": 3}
        period_1 = PipelineResult(
            standard=_report("standard", 0.10, 3, 2),
            wavelet=_report("wavelet", 0.20, 3, 2),
            period_index=1,
            train_start=datetime(2024, 1, 1),
            trade_end=datetime(2024, 1, 31),
        )
        period_2 = PipelineResult(
            standard=_report("standard", 0.30, 4, 3),
            wavelet=_report("wavelet", 0.50, 4, 3),
            period_index=3,
            train_start=datetime(2024, 3, 1),
            trade_end=datetime(2024, 3, 31),
        )

        with (
            patch.object(pipeline_module, "load_prices", return_value=prices),
            patch.object(pipeline_module, "build_periods", return_value=[object(), object(), object()]),
            patch.object(pipeline_module, "run_period", side_effect=[period_1, None, period_2]),
            patch.object(pipeline_module, "_write_outputs") as write_outputs,
        ):
            result = pipeline_module.run_pipeline(params=params, source="data", verbose=False, write_outputs=True)

        self.assertEqual(len(result["periods"]), 2)
        self.assertEqual(result["summary"].height, 2)
        self.assertEqual(result["by_period"].height, 4)
        self.assertEqual(set(result["summary"].get_column("variant").to_list()), {"standard", "wavelet"})
        self.assertEqual(result["params"], params)
        write_outputs.assert_called_once_with(result)

    def test_run_pipeline_skips_output_writes_when_disabled(self) -> None:
        prices = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "AAA": [100.0, 101.0],
            }
        )
        params = {"block_size": 5, "max_periods": 1}
        period = PipelineResult(
            standard=_report("standard", 0.10, 1, 1),
            wavelet=_report("wavelet", 0.15, 1, 1),
            period_index=1,
            train_start=datetime(2024, 1, 1),
            trade_end=datetime(2024, 1, 31),
        )

        with (
            patch.object(pipeline_module, "load_prices", return_value=prices),
            patch.object(pipeline_module, "build_periods", return_value=[object()]),
            patch.object(pipeline_module, "run_period", return_value=period),
            patch.object(pipeline_module, "_write_outputs") as write_outputs,
        ):
            result = pipeline_module.run_pipeline(params=params, source="data", verbose=False, write_outputs=False)

        self.assertEqual(len(result["periods"]), 1)
        write_outputs.assert_not_called()


if __name__ == "__main__":
    unittest.main()