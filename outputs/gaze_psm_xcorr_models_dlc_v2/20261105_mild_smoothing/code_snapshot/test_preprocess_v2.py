"""Tests of mask semantics, temporal labels and the user-directed DLC policy."""
import copy
import json
import unittest
import numpy as np
from scipy.ndimage import uniform_filter1d

from gaze_psm_preprocess_v2 import interpolate_valid, filter_valid, prepare_signals


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


if __name__ == "__main__":
    unittest.main()
