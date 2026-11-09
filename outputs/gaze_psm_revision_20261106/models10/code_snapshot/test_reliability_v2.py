"""Meaningful kernel/regression/missingness/peak identification checks."""
import json
import unittest
import numpy as np
from gaze_psm_core_legacy import analyze_curve
from gaze_psm_reliability_v2 import (occurrence_statistics, aggregate_statistics,
    peak_features, reliability, block_weights)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_legacy_equivalence_missing_and_occurrences(self):
        n = 430
        g = self.rng.normal(size=(n, 2))*30 + [320, 400]
        p = self.rng.normal(size=(n, 2))*25 + [270, 370]
        g[40:55] = np.nan; p[240:250] = np.nan
        ids = np.full(n, -1, dtype=int)
        ids[20:100]=4; ids[180:220]=8; ids[250:400]=9
        for within in (False, True):
            s = occurrence_statistics(g, p, ids, hz=20, max_lag=1, both_in_stage=within)
            actual = aggregate_statistics(s, min_overlap=2)
            expected = analyze_curve(g,p,20,max_lag=1,min_overlap=2,occurrence_ids=ids,
                                     both_in_stage=within,compute_spatial=False)
            for key in ('lag_s','r','pairs','occurrences','gaze_rms','psm_rms'):
                np.testing.assert_allclose(actual[key],expected[key],atol=1e-10,rtol=1e-10,equal_nan=True)

    def test_kernel_against_direct_per_lag_centering(self):
        g=self.rng.normal(size=(100,2))*20; p=self.rng.normal(size=(100,2))*20
        g[8:12]=np.nan; p[19]=np.nan
        ids=np.r_[np.repeat(1,40),np.repeat(-1,10),np.repeat(2,50)]
        s=occurrence_statistics(g,p,ids,10,.5)
        for j,identity in enumerate(s['ids']):
            members=np.flatnonzero(ids==identity)
            for li,lag in enumerate(np.round(s['lag_s']*10).astype(int)):
                gi=members-lag
                keep=(gi>=0)&(gi<len(g)); pi=members[keep];gi=gi[keep]
                good=np.isfinite(g[gi]).all(1)&np.isfinite(p[pi]).all(1)
                gg,pp=g[gi[good]],p[pi[good]]
                if len(gg)<2:
                    expected=np.zeros(4)
                else:
                    gg=gg-gg.mean(0);pp=pp-pp.mean(0)
                    expected=[len(gg),(gg*pp).sum(),(gg*gg).sum(),(pp*pp).sum()]
                np.testing.assert_allclose(s['terms'][j,:,li],expected,atol=1e-8,rtol=1e-10)

    def test_weighted_statistics_equal_repeated_occurrence_terms(self):
        g=self.rng.normal(size=(300,2))*30;p=np.roll(g,2,axis=0)
        ids=np.repeat(np.arange(3),100)
        s=occurrence_statistics(g,p,ids,20,.5)
        weights=np.array([2,0,1])
        result=aggregate_statistics(s,weights,min_overlap=1)
        repeated=dict(s,terms=s['terms'][[0,0,2]],ids=np.array([0,1,2]))
        other=aggregate_statistics(repeated,min_overlap=1)
        for key in ('r','pairs','gaze_rms','psm_rms','occurrences'):
            np.testing.assert_allclose(result[key],other[key],atol=1e-11,equal_nan=True)

    def test_known_delay_signs(self):
        g=self.rng.normal(size=(800,2))*30
        for delay in (-5,0,5):
            p=np.roll(g,delay,axis=0)
            if delay>0:p[:delay]=np.nan
            elif delay<0:p[delay:]=np.nan
            f=peak_features(aggregate_statistics(occurrence_statistics(g,p,None,41,1)))
            self.assertAlmostEqual(f['peak_lag_s'],delay/41)

    def test_unusable_ids_retained_for_paired_sampling(self):
        g=self.rng.normal(size=(200,2))*30;p=g.copy();ids=np.repeat(np.arange(4),50)
        g[:50]=np.nan
        s=occurrence_statistics(g,p,ids,20,.5,both_in_stage=True)
        np.testing.assert_array_equal(s['ids'],np.arange(4))
        self.assertTrue((s['terms'][0]==0).all())

    def test_block_sampling_reproducible_and_fixed_draw_count(self):
        a=block_weights(7,50,42,2);b=block_weights(7,50,42,2)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a.sum(1),np.repeat(7,50))

    def test_multipeak_intervals_are_not_envelope_width(self):
        lags=np.linspace(-2,2,165)
        r=np.maximum(np.exp(-((lags-.7)/.04)**2),np.exp(-((lags+.7)/.04)**2))
        f=peak_features({'r':r,'lag_s':lags})
        self.assertGreaterEqual(f['near_peak_components_0p01'],2)
        self.assertGreater(f['near_peak_span_0p01_s'],1)
        self.assertLess(f['near_peak_contiguous_width_0p01_s'],.1)
        json.dumps(f,allow_nan=False)

    def test_constant_curves_not_reliable_and_single_occ_no_fake_range(self):
        lags=np.linspace(-2,2,165)
        f=peak_features({'r':np.repeat(.5,165),'lag_s':lags})
        self.assertEqual(f['tied_maxima_count'],165)
        self.assertEqual(f['near_peak_contiguous_width_0p01_s'],4)
        g=self.rng.normal(size=(500,2))*30
        s=occurrence_statistics(g,g,None,41,2)
        out=reliability(s,100)
        self.assertFalse(out['summary']['lag_identifiable'])
        self.assertIsNone(out['summary']['bootstrap_range_low_s'])
        self.assertEqual(len(out['peak_samples']),0)
        json.dumps(out['summary'],allow_nan=False)

    def test_paired_resampling_output_deterministic(self):
        g=self.rng.normal(size=(600,2))*30;p=np.roll(g,3,axis=0)
        s=occurrence_statistics(g,p,np.repeat(np.arange(6),100),20,1)
        a=reliability(s,100,12);b=reliability(s,100,12)
        np.testing.assert_array_equal(a['peak_samples'],b['peak_samples'])
        self.assertEqual(a['summary'],b['summary'])
        self.assertTrue(a['summary']['lag_identifiable'])
        self.assertEqual(a['summary']['direction_0p10'],'lead')


if __name__=='__main__':
    unittest.main()
