from .config_backtester import ROOT_DIR

RESEARCH_DIR = ROOT_DIR / "research"
DOCS_DIR = RESEARCH_DIR / "docs"
OUTPUTS_DIR = RESEARCH_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
NOTEBOOKS_DIR = RESEARCH_DIR / "notebooks"
TABLE_VIEWER_NOTEBOOK = NOTEBOOKS_DIR / "table_viewer.ipynb"
TABLE_FORMATS = ("csv",)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_METHODS = ("distance", "cointegration")
REPORT_START_DATE = "2010-03-05"
REPORT_END_DATE = "2018-03-15"
HEADLINE_TC_PER_SHARE = 0.0
REPORT_METRIC_COLUMNS = (
    "mean_return",
    "sharpe",
    "skewness",
    "kurtosis",
    "max_drawdown",
    "cvar_95",
    "pct_positive",
    "n_full",
    "n_partial",
    "n_non",
    "n_pairs",
)
PIPELINE_SUMMARY_METRIC_COLUMNS = (
    "mean_return",
    "std_return",
    "sharpe",
    "skewness",
    "kurtosis",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "pct_positive",
    "n_pairs",
    "n_active",
    "n_full",
    "n_partial",
    "n_non",
    "n_inactive",
)
PAPER_COMPARISON_TARGETS = {
    "distance": {"std_ret": -0.55, "wav_ret": 11.82, "std_sr": -0.21, "wav_sr": 3.69},
    "cointegration": {"std_ret": -1.81, "wav_ret": 9.66, "std_sr": -0.40, "wav_sr": 2.82},
}
FIGURE_FILE_NAMES = {
    "fig01_pyramid": "figure1_pyramid_algorithm_of_mallat",
    "fig02_example_spread": "figure2_example_pair_spread_and_trades",
    "fig03_example_returns": "figure3_example_pair_return_series",
    "fig04_cumulative_distance": "figure4_cumulative_excess_returns_distance",
    "fig04_cumulative_cointegration": "figure4_cumulative_excess_returns_cointegration",
    "fig05_daily_sharpe_distance": "figure5_daily_sharpe_ratios_distance",
    "fig05_daily_sharpe_cointegration": "figure5_daily_sharpe_ratios_cointegration",
    "fig06_categories_cointegration": "figure6_yearly_proportions_and_returns_cointegration",
    "fig07_categories_distance": "figure7_yearly_proportions_and_returns_minimum_distance",
    "fig08_noise_corr_distance": "figure8_filtered_noise_correlation_distance",
    "fig08_noise_corr_cointegration": "figure8_filtered_noise_correlation_cointegration",
    "fig09_yearly_alpha_distance": "figure9_yearly_abnormal_returns_distance",
    "fig09_yearly_alpha_cointegration": "figure9_yearly_abnormal_returns_cointegration",
    "fig10_wavelet_classes_distance": "figure10_returns_alternative_wavelet_classes_distance",
    "fig10_wavelet_classes_cointegration": "figure10_returns_alternative_wavelet_classes_cointegration",
    "fig11_horizons_distance": "figure11_profits_different_trading_periods_distance",
    "fig11_horizons_cointegration": "figure11_profits_different_trading_periods_cointegration",
}
UNAVAILABLE_TABLES = {
    "table10_marginal_variance_explained_of_principal_components": "Principal-component variance analysis is not implemented in the local workflow.",
    "table12_simulation_results_with_high_frequency_contamination": "The high-frequency contamination simulation is not implemented in the local workflow.",
    "table13_fama_french_five_factor_models": "The Fama-French factor file is not available in the local dataset.",
    "table14_q_factor_and_icapm_petkova_models": "The q-factor and ICAPM-Petkova factor files are not available in the local dataset.",
    "table16_sharpe_ratios_under_different_wavelet_classes": "The wavelet-class sweep table is not regenerated in the lightweight local refresh path.",
    "table17_pairs_trading_returns_at_different_trading_period_spans": "The trading-horizon sweep table is not regenerated in the lightweight local refresh path.",
    "table18_key_statistics_no_transaction_cost_vs_transaction_cost": "Transaction-cost comparison tables are not generated in the single-pass local workflow.",
    "table19_average_number_of_trades_real_vs_transaction_cost": "Transaction-cost trade-count comparisons are not generated in the single-pass local workflow.",
    "table20_yearly_evolution_of_average_trades": "Yearly trade-count evolution under transaction-cost scenarios is not generated in the single-pass local workflow.",
    "table21_annualized_abnormal_returns_transaction_cost": "Transaction-cost abnormal-return tables need external factor files that are not available locally.",
    "table22_profits_from_standard_pairs_trading_when_trades_are_forced_closed": "The forced-close-only standard-pairs experiment is not implemented in the local workflow.",
}

# Fixed parameters for the paper replication workflow.
PAIRS_CONFIG = {
    "wavelet": "sym20",
    "block_size": 252,
    "method": "distance",
    "top_n": 1000,
    "candidate_pool": 2000,
    "k_ar_diff": 1,
    "threshold_sigma": 2.0,
    "tc_per_share": 0.001,
    "index_id": "SPX",
    "max_beta": 10.0,
    "raw_prices": True,
    "paper_periods": [
        ("2010-03-05", "2011-03-03", "2011-03-04", "2012-03-05"),
        ("2011-03-04", "2012-03-05", "2012-03-06", "2013-03-08"),
        ("2012-03-06", "2013-03-08", "2013-03-11", "2014-03-10"),
        ("2013-03-11", "2014-03-10", "2014-03-11", "2015-03-12"),
        ("2014-03-11", "2015-03-12", "2015-03-13", "2016-03-15"),
        ("2015-03-13", "2016-03-15", "2016-03-16", "2017-03-15"),
        ("2016-03-16", "2017-03-15", "2017-03-16", "2018-03-15"),
    ],
    "start_date": "2010-03-01",
    "end_date": "2018-03-31",
    "max_periods": None,
}

DEFAULT_WAVELET = PAIRS_CONFIG["wavelet"]
