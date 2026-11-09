"""Independent oracles exercise retained v3 core, including unavailable metrics."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from independent_oracles import native_time, smooth, temporal, pixel_area, bootstrap_mean

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT/'code' if (ROOT/'code').is_dir() else ROOT/'code_snapshot'
sys.path.insert(0, str(CODE))
import tl_observed_geometry as measured
from tl_observed_core import summarize
from tl_observed_bootstrap import ParticipantBootstrap
from tl_observed_statistics import participant_summary


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-10, equal_nan=True)


def main():
    rng = np.random.default_rng(783219)
    area_checks = temporal_checks = smooth_checks = weight_checks = 0
    for width, height in ((11, 9), (81, 43), (1280, 1024)):
        count = 24 if width < 100 else 8
        target, left, right = rng.uniform([0, 0], [width, height], (3, count, 2))
        target[:4] = [[0, 0], [width-1e-8, height-1e-8], [.5, .5], [width/2, height/2]]
        left[1] = right[1] = target[1]
        left[2] = right[2]
        for rt, rp in ((0, 0), (.5, 1), (25, 25), (50, 50), (100, 100), (50, 100)):
            areas = measured.areas(target, left, right, width, height, rt, rp, np.ones(count, bool))
            for i in range(count):
                oracle = pixel_area(target[i], left[i], right[i], width, height, rt, rp)
                equal(areas[i], oracle)
                area_checks += 1
    start = 1750430000000000000
    for trial in range(50):
        n = int(rng.integers(1, 100))
        frames = np.cumsum(rng.choice([1, 1, 1, 1, 2], size=n))
        timestamps = start + np.cumsum(rng.choice([0, 16000000, 20000000, 25000000, 100000000, 100000001], size=n))
        native, breaks = native_time(timestamps, frames)
        actual, actual_breaks = measured.times(timestamps, frames)
        equal(native, actual)
        np.testing.assert_array_equal(breaks, actual_breaks)
        weight_checks += 1
        occurrence = np.cumsum(rng.random(n) < .05)
        states = rng.choice([-1, 0, 1, 2, 3], size=n)
        # Runs long enough to test qualified censored dwell and switches.
        states[:min(n, 12)] = np.resize([1,1,1,1,1,1,3,2,2,2,2,2], min(n, 12))
        expected = temporal(states, native, occurrence, breaks)
        actual_temporal = measured.temporal(states, native, occurrence, breaks)
        assert set(expected) == set(actual_temporal)
        for metric in expected:
            equal(expected[metric], actual_temporal[metric])
        temporal_checks += 1
        xy = rng.uniform([0,0], [1280,1024], (n,2))
        xy[rng.random(n) < .3] = np.nan
        expected_xy, expected_fill = smooth(timestamps, xy, occurrence, breaks)
        actual_xy, actual_fill = measured.smooth_gaze(timestamps, xy, occurrence, breaks)
        equal(expected_xy, actual_xy)
        np.testing.assert_array_equal(expected_fill, actual_fill)
        smooth_checks += 1
    boot = ParticipantBootstrap(42, 10000)
    for n in range(12):
        values = rng.normal(size=n)
        expected = bootstrap_mean(values)
        actual = boot.mean(values)
        for metric in expected:
            equal(expected[metric], actual[metric])
    # Unequal trial counts, unequal durations, missing C but valid L_target.
    trials = pd.DataFrame([
        dict(participant='observer1', content_id='watch1', C=.2, L_target=.1, valid_n=5, valid_s=10),
        dict(participant='observer1', content_id='watch2', C=.8, L_target=.9, valid_n=5, valid_s=1000),
        dict(participant='observer2', content_id='watch3', C=.4, L_target=.3, valid_n=5, valid_s=2),
        dict(participant='observer2', content_id='watch4', C=np.nan, L_target=.7, valid_n=5, valid_s=2),
        dict(participant='observer3', content_id='watch5', C=np.nan, L_target=np.nan, valid_n=0, valid_s=0),
    ])
    for key, value in dict(scope='real_all', scenario='main', task='A', stage='whole', source='gaze').items():
        trials[key] = value
    participants, groups = participant_summary(trials, boot)
    p = participants.set_index('participant')
    equal(p.loc['observer1','C'], .5)
    equal(p.loc['observer2','C'], .4)
    equal(p.loc['observer1','L_target'], .5)
    equal(p.loc['observer2','L_target'], .5)
    assert p.loc['observer2','n_trials_C'] == 1
    assert p.loc['observer2','n_trials_L_target'] == 2
    g = groups.set_index('metric')
    equal(g.loc['C','mean'], .45)
    equal(g.loc['L_target','mean'], .5)
    assert g.loc['C','n_participants'] == 2 and g.loc['C','n_trials'] == 3
    assert g.loc['L_target','n_participants'] == 2 and g.loc['L_target','n_trials'] == 4
    for metric in ['C', 'L_target']:
        expected = bootstrap_mean(p[metric])
        for field, value in expected.items():
            equal(g.loc[metric, field], value)
    d = dict(dt=np.array([.1,.1]), occurrence=np.zeros(2,int), breaks=np.array([True,False]))
    no_data = summarize(np.full(2,-1), np.zeros(2,bool), np.full((2,4),np.nan), d)
    assert no_data['valid_s'] == 0 and no_data['valid_n'] == 0
    assert all(np.isnan(no_data[k]) for k in ['C','C_U','L_T','L_P','L_O','L_N','direct_rate','via_rate','dwell_T_ms'])
    only_overlap = summarize(np.full(2,3), np.ones(2,bool), np.tile([.8,0,0,.2],(2,1)), d)
    assert np.isnan(only_overlap['C']) and np.isnan(only_overlap['C_U'])
    equal(only_overlap['L_target'], 1)
    equal(only_overlap['L_gripper'], 1)
    result = dict(status='passed', independent_oracles=True,
        pixel_center_area_cases=area_checks, native_dt_cases=weight_checks,
        smooth_cases=smooth_checks, temporal_censor_and_switch_cases=temporal_checks,
        bootstrap_sample_sizes_checked=list(range(12)), bootstrap_n=10000, bootstrap_seed=42,
        unequal_exposure_equal_record_weighting=True,
        equal_observer_weighting=True, per_metric_missingness_counts=True,
        zero_denominator_NA_and_other_metrics_retained=True,
        empty_empirical_support_not_reported_as_positive_result=True,
        scope='independent synthetic verification of retained numerical methods; separate from empirical input admission')
    (Path(__file__).parent/'independent_core_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
