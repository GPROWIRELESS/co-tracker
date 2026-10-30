"""Synthetic tests for lag sign, spatial interpretation and stage pooling."""
import unittest

import numpy as np

from tools.gaze_psm_core import analyze_curve


def direct_pooled(gaze, psm, ids, lag, within=False):
    """Independent pair enumeration used to check the FFT sufficient statistics."""
    covariance, gv, pv, pairs, occurrences = 0., 0., 0., 0, 0
    distances = []
    for identity in np.unique(ids[ids >= 0]):
        u = np.flatnonzero(ids == identity)
        t = u-lag
        inside = (t >= 0) & (t < len(gaze))
        t, u = t[inside], u[inside]
        valid = np.isfinite(gaze[t]).all(axis=1) & np.isfinite(psm[u]).all(axis=1)
        if within:
            valid &= ids[t] == identity
        a, b = gaze[t[valid]], psm[u[valid]]
        if len(a) < 2:
            continue
        distances.extend(np.linalg.norm(a-b, axis=1))
        ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
        covariance += (ac*bc).sum()
        gv += (ac*ac).sum()
        pv += (bc*bc).sum()
        pairs += len(a)
        occurrences += 1
    r = covariance/np.sqrt(gv*pv) if gv*pv > 0 else np.nan
    return r, pairs, occurrences, np.median(distances) if distances else np.nan


class GazePSMCoreTests(unittest.TestCase):
    def test_positive_and_negative_lag_with_missing_observations(self):
        gaze = np.random.default_rng(4).normal(size=(800, 2))*20
        psm = np.full_like(gaze, np.nan)
        psm[9:] = gaze[:-9]
        gaze[200:250] = np.nan
        result = analyze_curve(gaze, psm, 25, min_overlap=2)
        self.assertAlmostEqual(result["lag_s"][np.nanargmax(result["r"])], .36)
        self.assertAlmostEqual(np.nanmax(result["r"]), 1.)
        reverse = analyze_curve(psm, gaze, 25, min_overlap=2)
        self.assertAlmostEqual(reverse["lag_s"][np.nanargmax(reverse["r"])], -.36)

    def test_spatial_offset_is_not_hidden_by_correlation(self):
        gaze = np.random.default_rng(2).normal(size=(300, 2))*10+1000
        psm = gaze+np.array([200., 0.])
        result = analyze_curve(gaze, psm, 25, max_lag=.2, min_overlap=1,
                               spatial_tolerance=50.)
        zero = np.flatnonzero(result["lag_s"] == 0)[0]
        self.assertAlmostEqual(result["r"][zero], 1.)
        self.assertAlmostEqual(result["distance_median_px"][zero], 200.)
        self.assertEqual(result["proximity"][zero], 0.)

    def test_stage_psm_anchor_permits_pre_stage_gaze(self):
        gaze = np.random.default_rng(1).normal(size=(80, 2))*10
        psm = np.full_like(gaze, np.nan)
        psm[5:] = gaze[:-5]
        ids = np.full(80, -1)
        ids[30:34] = 0
        result = analyze_curve(gaze, psm, 10, max_lag=1, min_overlap=.2,
                               occurrence_ids=ids)
        index = np.flatnonzero(np.isclose(result["lag_s"], .5))[0]
        self.assertAlmostEqual(result["r"][index], 1.)
        self.assertEqual(result["pairs"][index], 4)
        within = analyze_curve(gaze, psm, 10, max_lag=1, min_overlap=.2,
                               occurrence_ids=ids, within_stage=True)
        self.assertTrue(np.isnan(within["r"][index]))
        self.assertEqual(within["pairs"][index], 0)

    def test_fft_pooling_matches_independent_pair_enumeration(self):
        rng = np.random.default_rng(10)
        gaze, psm = rng.normal(size=(2, 120, 2))*10
        ids = np.full(120, -1)
        ids[5:14], ids[30:60], ids[100:119] = 0, 1, 2
        # Large occurrence offsets must not become between-occurrence covariance.
        gaze[30:60] += 500
        psm[30:60] -= 200
        gaze[8], psm[33] = np.nan, np.nan
        for within in (False, True):
            result = analyze_curve(gaze, psm, 10, max_lag=.7, min_overlap=0,
                                   occurrence_ids=ids, within_stage=within,
                                   min_gaze_rms=0, min_psm_rms=0)
            for index, lag in enumerate(range(-7, 8)):
                expected = direct_pooled(gaze, psm, ids, lag, within)
                self.assertAlmostEqual(result["r"][index], expected[0], places=9)
                self.assertEqual(result["pairs"][index], expected[1])
                self.assertEqual(result["occurrences"][index], expected[2])
                self.assertAlmostEqual(result["distance_median_px"][index], expected[3])

    def test_short_occurrences_satisfy_pooled_overlap(self):
        gaze = np.random.default_rng(3).normal(size=(100, 2))*10
        ids = np.repeat(np.arange(10), 10)
        result = analyze_curve(gaze, gaze, 10, max_lag=0, min_overlap=5,
                               occurrence_ids=ids)
        self.assertEqual(result["pairs"][0], 100)
        self.assertEqual(result["occurrences"][0], 10)
        self.assertAlmostEqual(result["r"][0], 1.)

    def test_empty_short_and_stationary_fixed_grid(self):
        for n in (0, 1, 10):
            result = analyze_curve(np.ones((n, 2))*1000, np.ones((n, 2))*1000,
                                   25, max_lag=2, min_overlap=0)
            self.assertEqual(len(result["lag_s"]), 101)
            self.assertFalse(np.isfinite(result["r"]).any())

    def test_compute_spatial_can_be_disabled(self):
        gaze = np.random.default_rng(3).normal(size=(100, 2))*10
        result = analyze_curve(gaze, gaze, 10, min_overlap=1, compute_spatial=False)
        self.assertTrue(np.isnan(result["distance_median_px"]).all())
        self.assertTrue(np.isnan(result["proximity"]).all())


if __name__ == "__main__":
    unittest.main()
