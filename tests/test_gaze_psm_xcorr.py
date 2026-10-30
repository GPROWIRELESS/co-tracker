import unittest
import numpy as np
from tools.gaze_psm_xcorr import xcorr, interpolate


class CrossCorrelationTests(unittest.TestCase):
    def test_known_delay_with_missing_samples(self):
        gaze = np.random.default_rng(7).normal(size=(1000, 2))
        psm = np.full_like(gaze, np.nan)
        psm[12:] = gaze[:-12]
        gaze[100:180] = np.nan
        lag, r, _ = xcorr(gaze, psm, 40, 2, 100)
        self.assertAlmostEqual(lag[np.nanargmax(r)], .3)
        self.assertAlmostEqual(np.nanmax(r), 1.)
        lag, r, _ = xcorr(psm, gaze, 40, 2, 100)
        self.assertAlmostEqual(lag[np.nanargmax(r)], -.3)

    def test_constant_signal_has_no_peak(self):
        _, r, _ = xcorr(np.ones((100, 2)), np.ones((100, 2)), 40, 1, 20)
        self.assertFalse(np.isfinite(r).any())

    def test_long_gap_is_not_interpolated(self):
        result = interpolate(np.array([0., 1.]), np.zeros((2, 2)), np.array([.5]), .1)
        self.assertTrue(np.isnan(result).all())


if __name__ == "__main__":
    unittest.main()
