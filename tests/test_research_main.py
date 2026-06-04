import unittest

from config import config_paper as research_config
import research.main as research_main


class ResearchMainTests(unittest.TestCase):
    def test_parser_defaults_follow_research_config(self) -> None:
        parser = research_main._build_parser()

        args = parser.parse_args([])

        self.assertEqual(args.source, "data")
        self.assertEqual(args.method, research_config.PAIRS_CONFIG["method"])
        self.assertEqual(args.wavelet, research_config.PAIRS_CONFIG["wavelet"])
        self.assertEqual(args.top_n, research_config.PAIRS_CONFIG["top_n"])
        self.assertEqual(args.candidate_pool, research_config.PAIRS_CONFIG["candidate_pool"])
        self.assertEqual(args.block_size, research_config.PAIRS_CONFIG["block_size"])
        self.assertEqual(args.threshold_sigma, research_config.PAIRS_CONFIG["threshold_sigma"])
        self.assertEqual(args.tc_per_share, research_config.PAIRS_CONFIG["tc_per_share"])
        self.assertEqual(args.max_periods, research_config.PAIRS_CONFIG["max_periods"])

    def test_build_params_applies_cli_overrides(self) -> None:
        parser = research_main._build_parser()
        args = parser.parse_args(
            [
                "--source",
                "bloomberg",
                "--index-id",
                "SPX",
                "--method",
                "cointegration",
                "--wavelet",
                "db4",
                "--top-n",
                "50",
                "--candidate-pool",
                "120",
                "--block-size",
                "126",
                "--threshold-sigma",
                "1.5",
                "--tc-per-share",
                "0.002",
                "--max-periods",
                "3",
                "--no-write-outputs",
                "--quiet",
            ]
        )

        params = research_main._build_params(args)

        self.assertEqual(args.source, "bloomberg")
        self.assertTrue(args.no_write_outputs)
        self.assertTrue(args.quiet)
        self.assertEqual(params["index_id"], "SPX")
        self.assertEqual(params["method"], "cointegration")
        self.assertEqual(params["wavelet"], "db4")
        self.assertEqual(params["top_n"], 50)
        self.assertEqual(params["candidate_pool"], 120)
        self.assertEqual(params["block_size"], 126)
        self.assertEqual(params["threshold_sigma"], 1.5)
        self.assertEqual(params["tc_per_share"], 0.002)
        self.assertEqual(params["max_periods"], 3)


if __name__ == "__main__":
    unittest.main()