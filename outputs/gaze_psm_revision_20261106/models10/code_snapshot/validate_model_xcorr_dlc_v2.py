"""Reproducible integration checks for the exploratory gaze/PSM v2 pipeline.

Writes validation artifacts without modifying inputs or prior outputs. Simulation
coverage is diagnostic evidence, not a claim of calibrated empirical intervals.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d

from gaze_psm_reliability_v2 import occurrence_statistics, aggregate_statistics, run_validation_simulations
from gaze_psm_statistics_v2 import group_curves


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(clean(row))


def wilson(successes, total):
    if not total:
        return None, None
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total))/(1+z*z/total)
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return float(max(0, center-radius)), float(min(1, center+radius))


def participant_simulations(count=100, seed=15237, bootstrap=500):
    """Use the actual group-curve function on synthetic participant curves.

    Each participant supplies six occurrences of colored trajectories. The delay
    scenarios include between-person delay variation (SD .15s). The reference
    population delay is the center of that symmetric generating distribution;
    a finite sample's group peak need not equal it. Independent-noise data have
    no unique population lag, so coverage is deliberately undefined.
    """
    rng = np.random.default_rng(seed)
    rows, trial_rows = [], []
    hz, max_lag, occurrences, length_per_occurrence = 20., 1., 6, 140
    total_length = occurrences * length_per_occurrence
    ids = np.full(total_length, -1, dtype=np.int64)
    for k in range(occurrences):
        ids[k*length_per_occurrence+20:k*length_per_occurrence+120] = k
    for npeople in (3, 5):
        for scenario, truth in (('positive_population_delay', .3), ('zero_population_delay', 0.),
                                ('independent_autocorrelated', None)):
            covered, lead, lag, point_errors, widths, boundaries = [], [], [], [], [], []
            for trial in range(count):
                curves = []
                for person in range(npeople):
                    gaze = gaussian_filter1d(rng.normal(size=(total_length, 2)), 2.5, axis=0)*50
                    if truth is None:
                        psm = gaussian_filter1d(rng.normal(size=gaze.shape), 2.5, axis=0)*50
                    else:
                        delay = int(round((truth+rng.normal(0, .15))*hz))
                        psm = np.roll(gaze, delay, axis=0)+rng.normal(size=gaze.shape)*12
                        if delay > 0:
                            psm[:delay] = np.nan
                        elif delay < 0:
                            psm[delay:] = np.nan
                    gaze[rng.random(total_length) < .03] = np.nan
                    stats = occurrence_statistics(gaze, psm, ids, hz, max_lag)
                    curve = aggregate_statistics(stats, min_overlap=2, min_rms=1)
                    curves.append(dict(curve, variant='simulation', task='synthetic', side='l', signal='position',
                                       stage=scenario, mode='PSM in stage', source='gt', user=f'u{person}',
                                       content_id=f'u{person}'))
                summary, _ = group_curves(curves, bootstrap=bootstrap,
                                         seed=int(rng.integers(2**31)), min_users=3)
                result = summary[0]
                low = result['peak_lag_resample_low']; high = result['peak_lag_resample_high']
                peak = result['peak_lag_s']
                islead = bool(np.isfinite(low) and low > .1)
                islag = bool(np.isfinite(high) and high < -.1)
                lead.append(islead); lag.append(islag); widths.append(high-low)
                boundaries.append(result['peak_at_boundary'])
                coverage = None
                if truth is not None:
                    coverage = bool(low <= truth <= high)
                    covered.append(coverage); point_errors.append(abs(peak-truth))
                trial_rows.append(dict(scenario=scenario,n_participants=npeople,trial=trial,
                                       population_reference_delay_s=truth,peak_lag_s=peak,
                                       resample_low_s=low,resample_high_s=high,range_covers_reference=coverage,
                                       range_entirely_leading_0p1=islead,range_entirely_lagging_0p1=islag,
                                       boundary_peak=result['peak_at_boundary']))
            covlow, covhigh = wilson(sum(covered), len(covered))
            false_direction = (np.logical_or(lead, lag) if truth in (0., None) else np.asarray(lag))
            flow, fhigh = wilson(int(false_direction.sum()), count)
            rows.append(dict(scenario=scenario,n_participants=npeople,occurrences_per_participant=occurrences,
                             replicates=count,bootstrap_draws_per_trial=bootstrap,population_reference_delay_s=truth,
                             population_delay_sd_s=.15 if truth is not None else None,
                             range_coverage=float(np.mean(covered)) if covered else None,
                             coverage_wilson_low=covlow,coverage_wilson_high=covhigh,
                             mean_abs_peak_error_s=float(np.mean(point_errors)) if point_errors else None,
                             mean_range_width_s=float(np.mean(widths)),lead_fraction=float(np.mean(lead)),
                             lag_fraction=float(np.mean(lag)),boundary_fraction=float(np.mean(boundaries)),
                             false_direction_fraction=float(np.mean(false_direction)),
                             false_direction_wilson_low=flow,false_direction_wilson_high=fhigh,
                             false_direction_definition='either direction for zero/no unique delay; opposite lag for positive delay',
                             inference_status='diagnostic_only_uncalibrated',
                             coverage_note='No unique true lag for independent colored noise; finite-participant argmax is nonregular.'))
    return rows, trial_rows


def numerical_equal(a, b, path='root'):
    """Require exact float/array equality (including same NaN support)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape:
            raise AssertionError(f'{path}: shape mismatch')
        if aa.dtype.kind in 'fc' or bb.dtype.kind in 'fc':
            if not np.array_equal(aa, bb, equal_nan=True):
                raise AssertionError(f'{path}: numeric array mismatch')
        elif not np.array_equal(aa, bb):
            raise AssertionError(f'{path}: array mismatch')
        return int(aa.size)
    if isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError(f'{path}: keys mismatch')
        return sum(numerical_equal(a[k], b[k], path+'.'+str(k)) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'{path}: length mismatch')
        return sum(numerical_equal(x,y,f'{path}[{j}]') for j,(x,y) in enumerate(zip(a,b)))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return 1
    if a != b:
        raise AssertionError(f'{path}: {a!r} != {b!r}')
    return 1


def worker_invariance(source, output):
    from analyze_model_xcorr_dlc_v2 import analyze_record, BASE
    records = [('User1_1_1','A'),('User1_5_2','C')]
    outputs = []
    file_checks = []
    for workers in (1,2):
        dest = Path(output)/f'worker_{workers}'
        dest.mkdir(parents=True, exist_ok=True)
        jobs = [(str(source),str(dest),cid,task,'primary',dict(BASE),30) for cid,task in records]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs.append(list(pool.map(analyze_record,jobs)))
    values = numerical_equal(outputs[0],outputs[1])
    for cid,_ in records:
        for folder in ('occurrence_statistics','bootstrap_peaks'):
            relative = Path(folder)/'primary'/(cid+'.npz')
            with np.load(Path(output)/'worker_1'/relative, allow_pickle=False) as a:
                with np.load(Path(output)/'worker_2'/relative, allow_pickle=False) as b:
                    if set(a.files) != set(b.files):
                        raise AssertionError(f'{relative}: NPZ keys mismatch')
                    checked = sum(numerical_equal(a[k],b[k],str(relative)+':'+k) for k in a.files)
                    file_checks.append(dict(file=str(relative),arrays=len(a.files),compared_values=checked,exact_match=True))
    return dict(workers=[1,2],records=[c for c,_ in records],bootstrap=30,
                returned_values_compared=values,returned_exact_match=True,files=file_checks,
                path_policy='Output locations excluded; returned data and all persisted arrays compared exactly.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',type=Path,default=Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_models_max_dlc'))
    parser.add_argument('--occurrence-replicates',type=int,default=200)
    parser.add_argument('--participant-replicates',type=int,default=100)
    args=parser.parse_args(argv)
    if args.occurrence_replicates < 1 or args.participant_replicates < 1:
        parser.error('Simulation replicate counts must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    # Worker artifacts stay inside the requested output; no temporary paths are
    # deleted, moved, or formed from source dataset paths.
    tools_dir=Path(__file__).resolve().parent
    summary=dict(status='running',source=str(args.source),output=str(args.output),checks=[],
                 environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
                 simulations_are_calibration=False,
                 code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tools_dir.glob('*.py'))})
    start=time.time(); log=[]
    def execute(name, function):
        tick=time.time()
        try:
            result=function()
            summary['checks'].append(dict(name=name,status='passed',seconds=time.time()-tick,result=result))
            log.append(f'{name}: PASS ({time.time()-tick:.3f}s)')
            print(log[-1],flush=True)
        except Exception as exc:
            trace=traceback.format_exc()
            summary['checks'].append(dict(name=name,status='failed',seconds=time.time()-tick,error=str(exc),traceback=trace))
            log.append(f'{name}: FAIL\n{trace}')
            print(log[-1],flush=True)
        write_json(args.output/'validation_summary.json',summary)
        (args.output/'validation.log').write_text('\n\n'.join(log),encoding='utf-8')

    def tests():
        stream=io.StringIO()
        suite=unittest.TestLoader().discover(str(tools_dir),pattern='test_*v2.py',top_level_dir=str(tools_dir))
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (args.output/'unit_tests.log').write_text(stream.getvalue(),encoding='utf-8')
        if not result.wasSuccessful():
            raise AssertionError(f'{len(result.failures)} failures, {len(result.errors)} errors; see unit_tests.log')
        return dict(tests_run=result.testsRun,failures=0,errors=0,skipped=len(result.skipped))
    execute('all_unit_tests',tests)

    def occurrence_sims():
        rows=run_validation_simulations(count=args.occurrence_replicates)
        write_json(args.output/'occurrence_simulations.json',rows)
        write_csv(args.output/'occurrence_simulations.csv',rows)
        return dict(scenarios=len(rows),replicates_per_scenario=args.occurrence_replicates,
                    interpretation='Observed coverage and errors are diagnostics, not pass/fail thresholds or nominal interval validation.')
    execute('occurrence_simulations',occurrence_sims)

    def participant_sims():
        rows,trials=participant_simulations(count=args.participant_replicates)
        write_json(args.output/'participant_simulations.json',rows)
        write_csv(args.output/'participant_simulations.csv',rows)
        write_csv(args.output/'participant_simulation_trials.csv',trials)
        return dict(scenarios=len(rows),replicates_per_scenario=args.participant_replicates,
                    participant_counts=[3,5],uses_actual_group_curves_function=True,
                    interpretation='Undercoverage/false directions are reported transparently; no claim of calibrated 95% intervals.')
    execute('participant_simulations',participant_sims)
    execute('worker_invariance',lambda:worker_invariance(args.source,args.output/'worker_comparison'))

    failed=any(x['status']=='failed' for x in summary['checks'])
    summary.update(status='failed' if failed else 'passed',elapsed_seconds=time.time()-start,
                   scientific_limit='Passing confirms implementation consistency, not general inferential validity or the correctness of a particular lag.')
    write_json(args.output/'validation_summary.json',summary)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
