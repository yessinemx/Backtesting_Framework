"""Unit tests for the level-1 MODWT smoother used in the paper replication."""
import unittest

import numpy as np
import polars as pl

from research.paper_replication.core.wavelet import (
    DEFAULT_WAVELET,
    filter_prices,
    modwt_detail,
    modwt_smooth,
    resolve_wavelet,
)


class ModwtSmoothTests(unittest.TestCase):
    def test_output_length_matches_input(self):
        x = np.linspace(1.0, 2.0, 128)
        self.assertEqual(modwt_smooth(x).shape[0], x.shape[0])
        self.assertEqual(modwt_detail(x).shape[0], x.shape[0])

    def test_constant_series_passes_through_low_pass(self):
        # The normalized low-pass filter has unit DC gain, so a constant
        # series is returned unchanged by the smoother.
        x = np.full(64, 3.5)
        np.testing.assert_allclose(modwt_smooth(x), x, atol=1e-9)

    def test_constant_series_has_zero_detail(self):
        # The high-pass filter coefficients sum to zero, so the detail of a
        # constant series is (numerically) zero.
        x = np.full(64, 3.5)
        np.testing.assert_allclose(modwt_detail(x), 0.0, atol=1e-9)

    def test_smoothing_reduces_high_frequency_variance(self):
        rng = np.random.default_rng(0)
        trend = np.linspace(0.0, 1.0, 256)
        noisy = trend + rng.normal(scale=0.3, size=256)
        smooth = modwt_smooth(noisy)
        self.assertLess(float(np.var(smooth)), float(np.var(noisy)))

    def test_periodic_boundary_runs_and_preserves_length(self):
        x = np.sin(np.linspace(0.0, 6.0, 96))
        out = modwt_smooth(x, boundary="periodic")
        self.assertEqual(out.shape[0], x.shape[0])

    def test_invalid_boundary_raises(self):
        with self.assertRaises(ValueError):
            modwt_smooth(np.arange(32.0), boundary="reflect")

    def test_level_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            modwt_smooth(np.arange(32.0), level=2)

    def test_empty_input_returns_empty(self):
        self.assertEqual(modwt_smooth(np.array([])).shape[0], 0)


class ResolveWaveletTests(unittest.TestCase):
    def test_installed_wavelet_resolves(self):
        self.assertEqual(resolve_wavelet("sym20"), "sym20")

    def test_default_wavelet_resolves(self):
        self.assertEqual(resolve_wavelet(DEFAULT_WAVELET), DEFAULT_WAVELET)

    def test_unbundled_sym22_raises_with_hint(self):
        # PyWavelets does not bundle exact sym22 coefficients; the replication
        # falls back to sym20 and must fail loudly if sym22 is requested.
        with self.assertRaises(ValueError):
            resolve_wavelet("sym22")


class FilterPricesTests(unittest.TestCase):
    def test_dataframe_preserves_date_column(self):
        df = pl.DataFrame({
            "date": list(range(40)),
            "AAA": np.linspace(10.0, 12.0, 40),
            "BBB": np.linspace(20.0, 18.0, 40),
        })
        out = filter_prices(df)
        self.assertEqual(out.columns, df.columns)
        self.assertEqual(out.get_column("date").to_list(), df.get_column("date").to_list())

    def test_series_input_returns_series_of_same_length(self):
        s = pl.Series("AAA", np.linspace(10.0, 12.0, 40))
        out = filter_prices(s)
        self.assertIsInstance(out, pl.Series)
        self.assertEqual(out.len(), s.len())


if __name__ == "__main__":
    unittest.main()
