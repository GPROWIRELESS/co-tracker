"""Tests of mask semantics, temporal labels and the user-directed DLC policy."""
import copy
import json
import unittest
import numpy as np
from scipy.ndimage import uniform_filter1d

from gaze_psm_preprocess_revision import (interpolate_valid, filter_valid,
    prepare_signals, interpolation_provenance, native_diagnostics)


def cache(n=25):
    t = np.arange(n)/41.
    p = np.column_stack([np.arange(n)*2.+20, np.arange(n)*3.+30])
    d = dict(time_s=t.copy(), native_time_s=t.copy(), frame_number=np.arange(n),
             stage=np.where(np.arange(n)<12, "first", "second"), occurrence=(np.arange(n)>=12).astype(int),
             native_stage=np.where(np.arange(n)<12, "first", "second"), native_occurrence=(np.arange(n)>=12).astype(int),
             qc_json=np.array(json.dumps(dict(content_id="synthetic", user="u", task="A"))))
    for side in ("l", "r"):
        d.update({f"native_psm_{side}":p.copy(), f"psm_raw_{side}":p.copy(),
                  f"gaze_raw_{side}":p.copy()+10, f"likelihood_{side}":np.ones(n),
                  f"resolution_{side}":np.array([1280,1024])})
    return d


class PreprocessTests(unittest.TestCase):
    def test_interpolation_does_not_extrapolate_or_cross_long_gap(self):
        t = np.array([0., .05, .3, .35])
        xy = np.column_stack([t, t*2])
        grid = np.array([-.01, .025, .1, .2, .325, .4])
        out = interpolate_valid(t, xy, grid, .1)
        np.testing.assert_array_equal(np.isfinite(out).all(1), [False, True, False, False, True, False])
        np.testing.assert_allclose(out[[1,4]], [[.025,.05],[.325,.65]])

    def test_filter_does_not_bridge_missing_run(self):
        xy = np.vstack([np.ones((4,2)), np.full((2,2),np.nan), np.full((4,2),100.)])
        for method in ("boxcar", "savgol"):
            out, _ = filter_valid(xy, 5, method)
            np.testing.assert_allclose(out[:4], 1)
            np.testing.assert_allclose(out[6:], 100)
            self.assertTrue(np.isnan(out[4:6]).all())

    def test_low_likelihood_and_outside_jump_are_retained(self):
        d = cache()
        d["likelihood_l"][:] = .01
        d["native_psm_l"][10] = [5000., 4000.]
        d["psm_raw_l"] = d["native_psm_l"].copy()
        out = prepare_signals({"gt":d}, {"position_window":1})
        np.testing.assert_array_equal(out["signals"]["position"]["l"]["psm"], d["native_psm_l"])
        self.assertEqual(out["diagnostics"]["l"]["quality_rejected"].sum(), 0)
        self.assertEqual(out["diagnostics"]["l"]["outside_frame"].sum(), 1)
        self.assertEqual(out["diagnostics"]["l"]["low_likelihood"].sum(), 25)

    def test_common_gaze_support_does_not_delete_independent_psm(self):
        gt = cache()
        model = copy.deepcopy(gt)
        model["gaze_raw_l"][8:12] = np.nan
        out = prepare_signals({"gt":gt,"model":model}, {"position_window":1})["signals"]
        p = out["position"]["l"]
        np.testing.assert_array_equal(np.isfinite(p["gazes"]["gt"]), np.isfinite(p["gazes"]["model"]))
        self.assertTrue(p["psm_valid"].all())
        self.assertFalse(p["gaze_valid"][8:12].any())
        self.assertTrue(out["position"]["r"]["gaze_valid"].all())
        self.assertFalse(out["velocity"]["l"]["gaze_valid"][7:12].any())

    def test_legacy_boxcar_and_midpoint_stage(self):
        d = cache()
        out = prepare_signals({"gt":d}, {"velocity_filter":"boxcar", "velocity_window":3})["signals"]
        expected = uniform_filter1d(d["psm_raw_l"], size=3, axis=0, mode="nearest")
        np.testing.assert_array_equal(out["position"]["l"]["psm"], expected)
        np.testing.assert_array_equal(out["velocity"]["l"]["psm"], np.diff(expected, axis=0)*41)
        self.assertEqual(out["velocity"]["l"]["stage"][11], "first")
        self.assertEqual(out["velocity"]["l"]["stage"][12], "second")
        np.testing.assert_allclose(out["velocity"]["l"]["time_s"], (d["time_s"][:-1]+d["time_s"][1:])/2)

    def test_trim_edges_and_short_runs(self):
        xy = np.vstack([np.ones((3,2)), np.full((1,2),np.nan), np.ones((7,2))])
        out, edges = filter_valid(xy,5,"savgol",True)
        self.assertTrue(np.isnan(out[:4]).all())
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(out).all(1)), [6,7,8])
        self.assertEqual(edges.sum(),7)

    def test_centered_savgol_preserves_linear_motion_interior(self):
        d = cache()
        out = prepare_signals({"gt":d}, {})["signals"]["velocity"]["l"]["psm"]
        # Interior filter windows are complete: no artificial timing delay and
        # correct px/s scaling for this known straight-line motion.
        np.testing.assert_allclose(out[2:-2], np.tile([82.,123.],(len(out)-4,1)), atol=1e-9)

    def test_source_timeline_mismatch_rejected(self):
        gt = cache(); model=copy.deepcopy(gt); model["time_s"][3] += .001
        with self.assertRaises(ValueError):
            prepare_signals({"gt":gt,"model":model},{})

    def test_edges_flagged_without_removal(self):
        a = np.column_stack([np.arange(12.), np.arange(12.)])
        out, edge = filter_valid(a, 7, 'savgol', False)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(np.flatnonzero(edge), [0,1,2,9,10,11])
        data = prepare_signals({'gt': cache()}, {'velocity_window':7})
        edges = data['signals']['velocity']['l']['psm_filter_edge']
        np.testing.assert_array_equal(np.flatnonzero(edges), [0,1,2,21,22,23])

    def test_interpolation_reason_partition(self):
        t = np.array([0., .025, .05, .075, .3, .325])
        p = np.column_stack([t,t])
        p[1] = np.nan
        grid = np.array([-.01, 0, .0125, .025, .05, .0625, .1, .2, .31, .4])
        flags = interpolation_provenance(t,p,grid,.1)
        out = interpolate_valid(t,p,grid,.1)
        np.testing.assert_array_equal(np.flatnonzero(flags['observed_exact']), [1,4])
        np.testing.assert_array_equal(np.flatnonzero(flags['filled_nonfinite_hole']), [2,3])
        np.testing.assert_array_equal(np.flatnonzero(flags['ordinary_resample']), [5,8])
        np.testing.assert_array_equal(np.flatnonzero(flags['long_gap']), [6,7])
        np.testing.assert_array_equal(np.flatnonzero(flags['outside_retained_support']), [0,9])
        self.assertFalse(flags['filled_quality_hole'].any())
        classified = sum(flags[k].astype(int) for k in
            ('observed_exact','ordinary_resample','filled_nonfinite_hole'))
        np.testing.assert_array_equal(classified, np.isfinite(out).all(axis=1).astype(int))

    def test_isolated_return_vs_sustained_motion_actual_dt(self):
        t = np.arange(7)*.02
        p = np.column_stack([np.arange(7)*10.,np.zeros(7)])
        p[3,0] += 300
        q = native_diagnostics(t,p,np.ones(7),[1280,1024])
        np.testing.assert_array_equal(np.flatnonzero(q['isolated_jump_candidate']), [3])
        np.testing.assert_allclose(q['native_speed_px_s'][1:3], [500,500])
        fast = np.column_stack([np.arange(7)*100.,np.zeros(7)])
        q = native_diagnostics(t,fast,np.ones(7),[1280,1024])
        self.assertFalse(q['isolated_jump_candidate'].any())
        self.assertFalse(q['quality_rejected'].any())

    def test_velocity_uses_unsmoothed_raw_not_position_filter(self):
        rng = np.random.default_rng(8)
        d = cache(80)
        for s in ('l','r'):
            raw = np.cumsum(rng.normal(size=(80,2)),axis=0)+100
            d['native_psm_'+s] = raw.copy()
            d['psm_raw_'+s] = raw.copy()
            d['gaze_raw_'+s] = raw.copy()+50
        a = prepare_signals({'gt':d}, {'position_window':1,'velocity_window':7})
        b = prepare_signals({'gt':d}, {'position_window':5,'velocity_window':7})
        np.testing.assert_array_equal(a['signals']['velocity']['l']['psm'], b['signals']['velocity']['l']['psm'])
        self.assertFalse(np.array_equal(a['signals']['position']['l']['psm'],b['signals']['position']['l']['psm']))

    def test_prohibited_filter_options_fail_instead_of_silent_ignore(self):
        for config in ({'dlc_likelihood_filter':.5}, {'jump_filter':True}, {'coordinate_range_filter':True}):
            with self.assertRaises(ValueError): prepare_signals({'gt':cache()}, config)

    def test_numerically_matches_existing_model_v2(self):
        import importlib.util
        from pathlib import Path
        existing = Path(r'D:\GitHub_Project\co-tracker\tools\gaze_psm_preprocess_v2.py')
        if not existing.exists(): self.skipTest('Original source unavailable')
        spec = importlib.util.spec_from_file_location('existing_preprocess', existing)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        gt = cache(80)
        rng = np.random.default_rng(7)
        for side in ('l','r'):
            raw = np.cumsum(rng.normal(size=(80,2)),axis=0)+100
            raw[15:20] = np.nan
            gt['native_psm_'+side] = raw
            gt['psm_raw_'+side] = interpolate_valid(gt['native_time_s'],raw,gt['time_s'])
            gt['gaze_raw_'+side] = rng.normal(size=(80,2))*50+100
            gt['gaze_raw_'+side][30:35] = np.nan
        model = copy.deepcopy(gt)
        model['gaze_raw_l'][5] = np.nan
        for config in ({}, {'position_window':1,'velocity_window':7},
                       {'position_window':5,'velocity_window':11,'trim_edges':True},
                       {'velocity_filter':'boxcar','velocity_window':3}):
            a,b = old.prepare_signals({'gt':gt,'model':model}, config), prepare_signals({'gt':gt,'model':model}, config)
            for signal in ('position','velocity'):
                for side in ('l','r'):
                    for key in ('psm','stage','occurrence','time_s','gaze_valid','psm_valid'):
                        np.testing.assert_array_equal(a['signals'][signal][side][key],b['signals'][signal][side][key])
                    for source in ('gt','model'):
                        np.testing.assert_array_equal(a['signals'][signal][side]['gazes'][source],b['signals'][signal][side]['gazes'][source])

    def test_known_lag_sign_survives_symmetric_filters(self):
        import importlib.util
        from pathlib import Path
        core_path = Path(r'D:\GitHub_Project\co-tracker\tools\gaze_psm_core.py')
        if not core_path.exists(): self.skipTest('Core unavailable')
        spec = importlib.util.spec_from_file_location('gaze_core', core_path)
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        rng = np.random.default_rng(13)
        original = np.cumsum(rng.normal(size=(1200,2)),axis=0)*20
        for lag in (-20,0,20):
            p = np.full_like(original,np.nan)
            if lag > 0: p[lag:] = original[:-lag]
            elif lag < 0: p[:lag] = original[-lag:]
            else: p[:] = original
            for method,window in [('boxcar',3),('boxcar',5),('savgol',5),('savgol',7),('savgol',11)]:
                g,_ = filter_valid(original,window,method)
                ps,_ = filter_valid(p,window,method)
                for signal in ('position','velocity'):
                    gg,pp = (g,ps) if signal=='position' else (np.diff(g,axis=0)*41,np.diff(ps,axis=0)*41)
                    result = core.analyze_curve(gg,pp,41,max_lag=1,min_overlap=1,compute_spatial=False)
                    observed = result['lag_s'][np.nanargmax(result['r'])]*41
                    self.assertLessEqual(abs(observed-lag),1, (lag,signal,method,window))


if __name__ == "__main__":
    unittest.main()

