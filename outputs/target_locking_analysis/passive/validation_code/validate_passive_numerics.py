"""Reconstruct every passive numerical row directly from authoritative inputs.

This validator imports only independent_oracles, never production code.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from independent_oracles import (STAGES, SCENARIOS, normalize_stage, timeline,
    join_structured, native_time, smooth, temporal, pixel_area, bootstrap_mean)

METRICS = ['L_T','L_P','L_O','L_N','L_target','L_gripper','D','C','B',
           'area_T','area_P','area_O','area_N','C_U','C_excess','E_target',
           'E_gripper','B_excess','direct_rate','via_rate','dwell_T_ms','dwell_P_ms','dwell_O_ms']
PATH_FIELDS = ('gaze_npy_path','source_video_path','stage_path','target_locking_label_path','psm_dlc_csv_path')


def eq(actual, expected, label):
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9, equal_nan=True)
    except AssertionError as exc:
        raise AssertionError(label) from exc


def in_bounds(xy, resolution, closed=False):
    top = (xy <= resolution).all(1) if closed else (xy < resolution).all(1)
    return np.isfinite(xy).all(1) & (xy >= 0).all(1) & top


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_dlc(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        headers = [next(reader) for _ in range(3)]
        records = list(reader)
    indices = np.asarray([int(float(row[0])) for row in records])
    assert np.array_equal(indices, np.arange(len(indices)))
    tracks = []
    for side in ('left_psm','right_psm'):
        columns = []
        for axis in ('x','y','likelihood'):
            matching = [i for i in range(len(headers[1])) if headers[1][i] == side and headers[2][i] == axis]
            assert len(matching) == 1
            columns.append(matching[0])
        tracks.append(np.asarray([[float(r[c]) if r[c] else np.nan for c in columns] for r in records]))
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, default=Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source_hashes = {str(args.mapping): sha(args.mapping)}
    doc = json.loads(args.mapping.read_text(encoding='utf-8-sig'))
    root = Path(doc['data_root'])
    trials = pd.read_csv(args.run/'trial_stage.csv')
    recordings = pd.read_csv(args.run/'recording_audit.csv')
    coverage = pd.read_csv(args.run/'coverage.csv')
    people = pd.read_csv(args.run/'participant_stage_summary.csv')
    groups = pd.read_csv(args.run/'group_stage_summary.csv')
    assert set(trials.analysis_branch) == {'observed_gaze'}
    assert set(trials.source) == {'gaze'} and set(trials.scope) == {'real_all'}
    assert set(trials.scenario) == set(SCENARIOS)
    assert not trials.duplicated(['clip_key','scenario','stage']).any()
    assert not recordings.clip_key.duplicated().any() and not recordings.recording_id.duplicated().any()
    assert not any('decoded_' in col or 'brier' in col.lower() or 'bss' in col.lower() for col in trials)
    assert 'score' not in trials and 'round' not in trials
    expected_keys = {k for k,v in doc['clips'].items() if v['meta'].get('task_name') in STAGES and v['meta'].get('observer_skill') == 'novice' and v['meta'].get('src_skill') == 'novice'}
    assert set(recordings.clip_key) == expected_keys
    mapping_entries_checked = 0
    mapping_file = args.run/'passive_mapping_audit.csv'
    if mapping_file.is_file():
        with mapping_file.open(encoding='utf-8-sig', newline='') as stream:
            mapping_rows = list(csv.DictReader(stream))
        assert len(mapping_rows) == len(doc['clips'])
        assert len({r['clip_key'] for r in mapping_rows}) == len(mapping_rows)
        for audit in mapping_rows:
            original = doc['clips'][audit['clip_key']]
            assert json.loads(audit['raw_entry_json']) == original
            for field in PATH_FIELDS:
                raw = original.get(field)
                missing = raw is None or str(raw).strip().lower() in ('','nan','null','none')
                resolved = None if missing else Path(raw)
                if resolved is not None and not resolved.is_absolute(): resolved = root/resolved
                assert audit[field+'_missing'].lower() == str(missing).lower()
                expected_exists = resolved is not None and resolved.is_file()
                assert audit[field+'_exists'].lower() == str(expected_exists).lower()
                if resolved is not None:
                    assert Path(audit[field+'_resolved']).resolve() == resolved.resolve()
            mapping_entries_checked += 1
    all_checked = 0
    raw_frames = valid_raw_frames = 0
    area_pixel_checks = 0
    temporal_metric_checks = 0
    endpoints = 0
    max_area_error = 0.
    records_out = []
    for record in recordings.itertuples(index=False):
        key = record.clip_key
        meta = doc['clips'][key]['meta']
        assert str(record.participant) == f"User{meta['observer']}"
        assert record.source_content_id == meta['content_id']
        assert record.task == meta['task_name']
        assert str(record.content_id) == key
        assert bool(record.accepted)
        for field in PATH_FIELDS:
            original = doc['clips'][key].get(field)
            assert original is not None and str(original).strip().lower() not in ('','nan','none','null')
            expected = Path(original)
            if not expected.is_absolute(): expected = root/expected
            actual = Path(getattr(record, field))
            assert actual.resolve() == expected.resolve(), (key, field)
            source_hashes.setdefault(str(actual), sha(actual))
        for field in ('video_ts_path','source_video_ts_path'):
            path = Path(getattr(record, field))
            source_hashes.setdefault(str(path), sha(path))
        archive = np.load(args.run/'frames'/f'{record.recording_id}.npz', allow_pickle=False)
        f = {name: archive[name] for name in archive.files}
        archive.close()
        frame, stamp = timeline(record.video_ts_path)
        eq(f['frame_number'], frame, key+'/frame')
        eq(f['timestamp_ns'], stamp, key+'/timestamp')
        weights, breaks = native_time(stamp, frame)
        eq(f['dt_s'], weights, key+'/dt')
        gaze_raw = np.load(record.gaze_npy_path, allow_pickle=False)
        stage_raw = np.load(record.stage_path, allow_pickle=False)
        target_raw = np.load(record.target_locking_label_path, allow_pickle=False)
        selected = []
        for role, raw in [('gaze',gaze_raw),('stage',stage_raw),('target',target_raw)]:
            found, indices = join_structured(raw, frame)
            assert found.all()
            selected.append(indices)
            eq(f[role+'_npy_row'], indices, key+'/'+role+'_row')
        gi, si, ti = selected
        resolution = f['resolution']
        norm = np.column_stack([gaze_raw['gaze_x'][gi], gaze_raw['gaze_y'][gi]]).astype(float)
        raw_ok = in_bounds(norm, np.array([1,1]), closed=True)
        gaze = norm*resolution
        gaze[~raw_ok] = np.nan
        eq(f['gaze_normalized'], norm, key+'/raw_normalized')
        eq(f['gaze_xy'], gaze, key+'/gaze_px')
        np.testing.assert_array_equal(f['gaze_coordinate_valid'], raw_ok)
        endpoints += int((raw_ok & ((norm == 0)|(norm == 1)).any(1)).sum())
        stage_field = 'stage' if 'stage' in stage_raw.dtype.names else 'event'
        stages = normalize_stage(stage_raw[stage_field][si])
        np.testing.assert_array_equal(f['stage'], stages)
        included = np.isin(stages, STAGES[record.task])
        np.testing.assert_array_equal(f['included'], included)
        occurrence = np.zeros(len(frame), int)
        current = -1
        previous = None
        for i, value in enumerate(stages):
            if previous != value: current += 1
            occurrence[i] = current
            previous = value
        np.testing.assert_array_equal(f['occurrence'], occurrence)
        identity = np.zeros(len(frame), bool)
        identity_field = next((s for s in ('target_id','target_identity','point_1_id') if s in target_raw.dtype.names), None)
        if identity_field:
            identity[1:] = target_raw[identity_field][ti][1:] != target_raw[identity_field][ti][:-1]
        np.testing.assert_array_equal(f['identity_change'], identity)
        for i in range(len(frame)):
            if identity[i] or not included[i] or (i and (not included[i-1] or occurrence[i] != occurrence[i-1])):
                breaks[i] = True
        np.testing.assert_array_equal(f['break_before'], breaks)
        slots = np.full((len(frame),5,2), np.nan)
        for slot in range(5):
            x, y = f'point_{slot+1}_x', f'point_{slot+1}_y'
            assert (x in target_raw.dtype.names) == (y in target_raw.dtype.names)
            if x in target_raw.dtype.names:
                slots[:,slot] = np.column_stack([target_raw[x][ti],target_raw[y][ti]])
        slot_ok = np.isfinite(slots).all(2) & (slots >= 0).all(2) & (slots < resolution).all(2)
        count = slot_ok.sum(1)
        center = np.full((len(frame),2),np.nan)
        for i in np.flatnonzero(count): center[i] = slots[i,slot_ok[i]].mean(0)
        bits = (slot_ok*(2**np.arange(5))).sum(1)
        eq(f['target_slots'], slots, key+'/slots')
        eq(f['target_xy'], center, key+'/target_mean')
        eq(f['target_count'], count, key+'/slot_count')
        eq(f['target_slotmask'], bits, key+'/slot_bits')
        sf, st = timeline(record.source_video_ts_path)
        source_index = {int(v): i for i,v in enumerate(sf)}
        local = np.asarray([source_index[int(v)] for v in frame], int)
        eq(f['source_video_local_index'], local, key+'/source_local')
        eq(f['dlc_local_index'], local, key+'/dlc_local')
        eq(st[local], stamp, key+'/inherited_source_clock')
        dl, dr = read_dlc(record.psm_dlc_csv_path)
        assert len(dl) == len(dr) == len(sf)
        left, right = dl[local,:2], dr[local,:2]
        eq(f['left_xy'], left, key+'/left')
        eq(f['right_xy'], right, key+'/right')
        eq(f['likelihood'], np.column_stack([dl[local,2],dr[local,2]]), key+'/likelihood_audit')
        both = in_bounds(left,resolution) & in_bounds(right,resolution)
        geometry = included & (count > 0) & both
        smooth_xy, fill = smooth(stamp, gaze, occurrence, breaks)
        eq(f['smooth_xy'], smooth_xy, key+'/independent_smooth')
        np.testing.assert_array_equal(f['interpolated'], fill)
        record_trials = trials[trials.clip_key.eq(key)]
        record_coverage = coverage[coverage.clip_key.eq(key)]
        for scenario, (rt,rp) in SCENARIOS.items():
            xy = smooth_xy if scenario == 'smooth' else gaze
            valid = geometry & in_bounds(xy,resolution,closed=True)
            np.testing.assert_array_equal(f['valid_real_'+scenario], valid)
            target_hit = ((xy-center)**2).sum(1) <= rt*rt
            psm_hit = (((xy-left)**2).sum(1) <= rp*rp) | (((xy-right)**2).sum(1) <= rp*rp)
            states = np.where(valid,target_hit.astype(int)+2*psm_hit.astype(int),-1)
            np.testing.assert_array_equal(f['state_gaze_'+scenario], states)
            area = f['area_'+scenario]
            assert np.isnan(area[~geometry]).all()
            eq(area[geometry].sum(1),1,key+'/'+scenario+'/area_conservation')
            assert ((area[geometry] >= 0)&(area[geometry] <= 1)).all()
            for name in STAGES[record.task]:
                options = np.flatnonzero(geometry & (stages == name))
                if not len(options): continue
                i = options[len(options)//2]
                expected = pixel_area(center[i],left[i],right[i],int(resolution[0]),int(resolution[1]),rt,rp)
                max_area_error = max(max_area_error,float(np.abs(area[i]-expected).max()))
                eq(area[i],expected,key+'/'+scenario+'/pixel_enumeration')
                area_pixel_checks += 1
            stage_seconds = []
            for stage in ('whole',*STAGES[record.task]):
                selection = included if stage == 'whole' else stages == stage
                mask = valid & selection
                row = record_trials[record_trials.scenario.eq(scenario)&record_trials.stage.eq(stage)]
                cov = record_coverage[record_coverage.scenario.eq(scenario)&record_coverage.stage.eq(stage)]
                assert len(row) == len(cov) == 1
                row, cov = row.iloc[0],cov.iloc[0]
                seconds = weights[mask].sum()
                for dest in (row,cov):
                    eq(dest.valid_n, mask.sum(), key+'/valid_n')
                    eq(dest.valid_s, seconds, key+'/valid_s')
                if stage != 'whole': stage_seconds.append(seconds)
                share = np.full(4,np.nan) if seconds == 0 else np.bincount(states[mask],weights=weights[mask],minlength=4)/seconds
                mean_area = np.full(4,np.nan) if seconds == 0 else (area[mask]*weights[mask,None]).sum(0)/seconds
                nn,tt,pp,oo = share
                an,at,ap,ao = mean_area
                c = tt/(tt+pp) if tt+pp > 0 else np.nan
                cu = at/(at+ap) if at+ap > 0 else np.nan
                metrics = dict(L_N=nn,L_T=tt,L_P=pp,L_O=oo,L_target=tt+oo,L_gripper=pp+oo,D=tt+pp,C=c,B=tt-pp,
                    area_N=an,area_T=at,area_P=ap,area_O=ao,C_U=cu,C_excess=c-cu,E_target=tt+oo-at-ao,E_gripper=pp+oo-ap-ao,B_excess=tt-pp-at+ap)
                expected_time = temporal(np.where(mask,states,-1),weights,occurrence,breaks)
                for metric,value in {**metrics,**expected_time}.items():
                    eq(row[metric],value,key+'/'+scenario+'/'+stage+'/'+metric)
                temporal_metric_checks += len(expected_time)
                if seconds > 0: eq(share.sum(),1,key+'/occupancy_conservation')
                if not np.isfinite(c): assert isinstance(row.C_reason,str) and row.C_reason
                all_checked += 1
            whole = record_trials[record_trials.scenario.eq(scenario)&record_trials.stage.eq('whole')].iloc[0]
            eq(whole.valid_s,sum(stage_seconds),key+'/'+scenario+'/whole_additivity')
            assert not valid[stages == 'drop ring'].any()
        raw_frames += len(frame)
        valid_raw_frames += int(raw_ok.sum())
        records_out.append(dict(recording_id=record.recording_id,clip_key=key,raw_npy_rows=len(frame),
            coordinate_valid_rows=int(raw_ok.sum()),independently_checked_trial_stage_rows=25,
            original_frame_joins_verified=True,source_gaze_used=False))
        print(record.recording_id,'independent raw/smooth/occupancy/temporal checks passed',flush=True)
    group_keys = ['scope','scenario','task','stage','source']
    person_keys = group_keys+['participant']
    expected_people = {}
    for key, subset in trials.groupby(person_keys, sort=True, dropna=False):
        expected_people[key] = dict(means={m:float(subset[m].mean()) for m in METRICS},
            counts={m:int(subset[m].count()) for m in METRICS},trials_total=subset.content_id.nunique(),
            trials_valid=subset[subset.valid_n > 0].content_id.nunique())
    assert len(people) == len(expected_people)
    for row in people.itertuples(index=False):
        key = tuple(getattr(row,k) for k in person_keys)
        expected = expected_people[key]
        for metric in METRICS:
            eq(getattr(row,metric),expected['means'][metric],str(key)+'/'+metric)
            eq(getattr(row,'n_trials_'+metric),expected['counts'][metric],str(key)+'/count/'+metric)
        eq(row.trials_total,expected['trials_total'],str(key)+'/total')
        eq(row.trials_valid,expected['trials_valid'],str(key)+'/valid')
    bootstrap_checks = 0
    for row in groups.itertuples(index=False):
        key = tuple(getattr(row,k) for k in group_keys)
        relevant = [v for k,v in sorted(expected_people.items()) if k[:-1] == key]
        values = np.asarray([r['means'][row.metric] for r in relevant])
        finite = values[np.isfinite(values)]
        eq(row.mean, finite.mean() if len(finite) else np.nan, str(key)+'/group/'+row.metric)
        eq(row.n_participants,len(finite),str(key)+'/n_participants')
        eq(row.n_trials,sum(r['counts'][row.metric] for r in relevant),str(key)+'/n_trials')
        for field,value in bootstrap_mean(values).items():
            eq(getattr(row,field),value,str(key)+'/bootstrap/'+row.metric+'/'+field)
        bootstrap_checks += 1
    modified = [p for p,h in source_hashes.items() if sha(p) != h]
    assert not modified
    report = dict(status='passed', canonical_native_frame_analysis=True,
        physical_presentation_clock_not_established=True,
        raw_npy_rows_reconstructed=raw_frames, raw_coordinate_valid_rows=valid_raw_frames,
        inclusive_endpoint_rows_retained=endpoints, recordings_checked=len(recordings),
        trial_stage_rows_independently_reconstructed=all_checked,
        smooth_reconstructed_independently_for_all_records=True,
        native_dt_reconstructed_before_stage_splits=True,
        region_pixel_enumeration_cases=area_pixel_checks,max_area_absolute_error=max_area_error,
        temporal_metric_comparisons=temporal_metric_checks,
        whole_equals_sum_included_stage_exposure=True,drop_ring_included_seconds=0,
        zero_denominator_NA_verified=True,participant_rows_checked=len(people),
        group_and_bootstrap_rows_checked=bootstrap_checks,bootstrap_n=10000,bootstrap_seed=42,
        observer_and_recording_equal_weights_verified=True,per_metric_counts_verified=True,
        authoritative_JSON_and_original_Frame_Number_joins=True,
        mapping_audit_entries_compared_to_authoritative_JSON=mapping_entries_checked,
        source_gaze_identity_score_round_not_used=True,
        input_files_unchanged=not modified,input_files_checked=len(source_hashes),
        source_sha256=source_hashes,records=records_out)
    output = args.output or args.run/'independent_validation.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','records')},indent=2))


if __name__ == '__main__':
    main()
