import unittest

import config
from config import config_paper as research_config
from research.paper_replication import run_pipeline


class ConfigLayoutTests(unittest.TestCase):
    def test_global_config_stays_generic(self) -> None:
        self.assertFalse(hasattr(config, "PAIRS_CONFIG"))
        self.assertTrue(hasattr(config, "INDEX_CONFIG"))
        self.assertTrue(hasattr(config, "REBALANCE_FREQS"))

    def test_research_config_contains_paper_defaults(self) -> None:
        expected_keys = {
            "wavelet",
            "wavelet_boundary",
            "block_size",
            "method",
            "top_n",
            "candidate_pool",
            "k_ar_diff",
            "threshold_sigma",
            "tc_per_share",
            "index_id",
            "max_periods",
            "start_date",
            "end_date",
            "max_beta",
            "paper_periods",
            "raw_prices",
        }
        self.assertEqual(set(research_config.PAIRS_CONFIG), expected_keys)

    def test_research_output_dirs_live_under_research(self) -> None:
        self.assertEqual(research_config.OUTPUTS_DIR.parent, research_config.RESEARCH_DIR)
        self.assertEqual(research_config.TABLES_DIR.parent, research_config.OUTPUTS_DIR)
        self.assertEqual(research_config.FIGURES_DIR.parent, research_config.OUTPUTS_DIR)

    def test_research_config_exposes_fixed_replication_constants(self) -> None:
        self.assertEqual(research_config.TRADING_DAYS_PER_YEAR, 252)
        self.assertEqual(research_config.DEFAULT_METHODS, ("distance", "cointegration"))
        self.assertIn("distance", research_config.PAPER_COMPARISON_TARGETS)
        self.assertIn("fig01_pyramid", research_config.FIGURE_FILE_NAMES)
        self.assertIn("table10_marginal_variance_explained_of_principal_components", research_config.UNAVAILABLE_TABLES)

    def test_paper_replication_api_is_exposed(self) -> None:
        self.assertTrue(callable(run_pipeline))


if __name__ == "__main__":
    unittest.main()