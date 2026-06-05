import unittest

import numpy as np
import pandas as pd

from research.paper_replication import asset_pricing as ap
from research.paper_replication import paper_outputs


class AssetPricingFactorTests(unittest.TestCase):
    def test_paper_factor_models_feed_factor_tables(self) -> None:
        index = pd.date_range("2020-01-01", periods=90, freq="B")
        rng = np.random.default_rng(42)
        factors = pd.DataFrame(
            rng.normal(0.0, 0.01, (len(index), len(ap.MATLAB_FM_COLUMNS))),
            index=index,
            columns=list(ap.MATLAB_FM_COLUMNS),
        )
        factors["RF"] = 0.0001
        factors["R_ME"] = rng.normal(0.0, 0.01, len(index))
        factors["R_IA"] = rng.normal(0.0, 0.01, len(index))
        factors["R_ROE"] = rng.normal(0.0, 0.01, len(index))
        factors["R_EG"] = rng.normal(0.0, 0.01, len(index))
        returns = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)

        rows = ap.run_paper_factor_models(
            returns,
            factors,
            method="distance",
            variant="wavelet_after_tc",
            source="synthetic",
            tc_per_share=0.001,
        )
        frame = pd.DataFrame(rows)

        self.assertEqual(set(frame["factor_family"]), {"fama_french", "q_factor", "petkova_icapm"})
        self.assertEqual(len(frame[frame["factor_family"].eq("fama_french")]), 6)
        self.assertFalse(paper_outputs.build_table13(frame).is_empty())
        self.assertFalse(paper_outputs.build_table14(frame).is_empty())
        self.assertFalse(paper_outputs.build_table21({}, frame).is_empty())


if __name__ == "__main__":
    unittest.main()