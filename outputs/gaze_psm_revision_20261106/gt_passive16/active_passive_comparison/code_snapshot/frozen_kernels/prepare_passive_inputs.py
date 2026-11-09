"""Build passive caches for the unchanged frozen GT84 numerical modules."""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

import cv2
import numpy as np
from gaze_psm_preprocess_revision import interpolate_valid

WORK = Path(__file__).resolve().parent.parent
DATA = Path(r'D:\Dataset\dvss')
LINKS = Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json')
INDEX = Path(r'C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\dvss_pytorch\Index\index.json')
BASELINE = Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106\gt84')
OLD_INPUTS = Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\gaze_psm_revision_20261106_work\inputs\gt84')
INPUTS = WORK / 'inputs'
EMPTY_STAGES = ('', 'none', 'nan', 'null')
STAGES = {
    'A': ('reach ring', 'grab ring', 'transfer ring', 'place ring', 'drop ring'),
    'C': ('reach string', 'wrap around psm', 'threading string through tower', 'pull through and cinch'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(mode_rows(csv.DictReader(stream)))


def timeline(path):
    full = rows(path)
    kept = [row for row in full if row.get('IsDelete', '0').strip() in ('', '0', '0.0')]
    frames = np.array([int(row['Frame_Number']) for row in kept], dtype=np.int64)
    ns = np.array([int(Decimal(row['Timestamp(ns)'])) for row in kept], dtype=np.int64)
    assert len(frames) >= 2 and len(np.unique(frames)) == len(frames), path
    assert np.all(np.diff(ns) > 0), path
    return frames, ns, len(full) - len(kept)


def take_frames(array, frames, path):
    stored = array['Frame_Number'].astype(np.int64)
    assert len(np.unique(stored)) == len(stored), path
    mapping = {int(frame): index for index, frame in enumerate(stored)}
    assert all(int(frame) in mapping for frame in frames), path
    selected = np.array([mapping[int(frame)] for frame in frames], dtype=np.int64)
    return array[selected], selected


def read_dlc(path, expected):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        headers = [next(reader) for _ in range(3)]
        assert [row[0] for row in headers] == ['scorer', 'bodyparts', 'coords'], path
        bodyparts = headers[1]
        coords = headers[2]
        values = [row for row in reader if row]
    assert len(values) == expected, (path, len(values), expected)
    assert [int(row[0]) for row in values] == list(range(expected)), path
    points, likelihood = {}, {}
    for side, bodypart in [('l', 'left_psm'), ('r', 'right_psm')]:
        columns = {}
        for coordinate in ('x', 'y', 'likelihood'):
            found = [i for i in range(1, len(bodyparts)) if bodyparts[i].strip() == bodypart and coords[i].strip() == coordinate]
            assert len(found) == 1, (path, bodypart, coordinate)
            columns[coordinate] = found[0]
        xyz = np.array([[float(row[columns[c]]) if row[columns[c]].strip() else np.nan
                         for c in ('x', 'y', 'likelihood')] for row in values], dtype=float)
        points[side] = xyz[:, :2].copy()
        points[side][~np.isfinite(points[side]).all(axis=1)] = np.nan
        likelihood[side] = xyz[:, 2].copy()
    return points, likelihood


def video_size(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return size, n, fps


def main():
    INPUTS.mkdir(exist_ok=True)
    (INPUTS / 'cache').mkdir(exist_ok=True)
    (INPUTS / 'frame_mapping').mkdir(exist_ok=True)
    manifest = normalize_modes(json.loads(LINKS.read_text(encoding='utf-8-sig')))
    baseline_ids = {row['content_id'] for row in normalize_modes(json.loads((BASELINE / 'run_manifest.json').read_text(encoding='utf-8-sig')))['records']}
    source_files = {}

    def freeze(path, role):
        path = Path(path)
        if path.is_file():
            key = str(path)
            if key not in source_files:
                source_files[key] = dict(path=key, sha256=sha(path), role=role)

    freeze(LINKS, 'passive_link_manifest')
    freeze(INDEX, 'metadata_index')
    inventory, records, policies = [], [], []
    population = Counter()
    for clip_key, entry in manifest['clips'].items():
        meta = entry['meta']
        gaze_path = DATA / entry['gaze_npy_path']
        recording_id = gaze_path.stem.removesuffix('_gaze')
        observer = f"User{meta['observer']}"
        source_id = meta['content_id']
        source_user = f"User{meta['user']}"
        task = meta['task_name']
        task_resolution = 'index.json'
        if source_id == 'User2_1_1':
            task = 'B'
            task_resolution = 'independent score and passive assignment audit, see input_audit'
        population[task] += 1
        skill_ok = meta['observer_skill'] == meta['src_skill'] == 'novice'
        task_ok = task in STAGES
        reason = 'excluded_intermediate_observer_or_source' if not skill_ok else 'excluded_task_outside_A_C' if not task_ok else ''
        source_base = DATA / 'Active_Data' / source_user
        passive_base = gaze_path.parent.parent
        passive_video = passive_base / 'stimuli' / f'{recording_id}.mp4'
        passive_clock = passive_base / 'csv' / f'{recording_id}_video_ts.csv'
        source_video = source_base / 'stimuli' / f'{source_id}.mp4'
        source_clock = source_base / 'csv' / f'{source_id}_video_ts.csv'
        source_npy = source_base / 'npy' / f'{source_id}_gaze.npy'
        dlc_path = source_base / 'sam' / source_id / f'{source_id}_DLC.csv'
        row = dict(recording_id=recording_id, observer_user=observer, observer_skill=meta['observer_skill'],
                   source_content_id=source_id, source_user=source_user, src_skill=meta['src_skill'],
                   seq=meta['seq'], task=task, index_task=meta['task_name'], task_number=meta['task'],
                   split=meta['split'], task_resolution=task_resolution,
                   gaze_npy_path=str(gaze_path), passive_video_path=str(passive_video),
                   passive_video_ts_path=str(passive_clock), source_video_path=str(source_video),
                   source_video_ts_path=str(source_clock), stage_event_npy_path=str(source_npy),
                   target_label_npy_path=str(source_npy), dlc_csv_path=str(dlc_path),
                   stage_link_present=entry['stage_path'] != 'nan', dlc_link_present=entry['psm_dlc_csv_path'] != 'nan',
                   gt84_source_intersection=source_id in baseline_ids, included=not reason, exclusion_reason=reason)
        freeze(gaze_path, 'passive_gaze_npy_inventory')
        inventory.append(row)
        if reason:
            continue
        for path, role in [(passive_video, 'passive_video'), (passive_clock, 'passive_video_ts'),
                           (source_video, 'source_video'), (source_clock, 'source_video_ts'),
                           (source_npy, 'source_stage_npy'), (dlc_path, 'source_dlc_csv')]:
            assert path.is_file(), (recording_id, role, path)
            freeze(path, role)
        passive_frames, passive_ns, passive_deleted = timeline(passive_clock)
        source_frames, source_ns, source_deleted = timeline(source_clock)
        frame_to_dlc = {int(frame): index for index, frame in enumerate(source_frames)}
        assert all(int(frame) in frame_to_dlc for frame in passive_frames), recording_id
        dlc_indices = np.array([frame_to_dlc[int(frame)] for frame in passive_frames], dtype=np.int64)
        assert np.all(np.diff(dlc_indices) == 1), recording_id
        assert np.array_equal(passive_frames, source_frames[dlc_indices]), recording_id
        native_time = (passive_ns - passive_ns[0]) / 1e9
        source_elapsed = (source_ns[dlc_indices] - source_ns[dlc_indices[0]]) / 1e9
        assert np.array_equal(native_time, source_elapsed), recording_id
        assert np.array_equal(passive_ns, source_ns[dlc_indices]), recording_id
        gaze_full = np.load(gaze_path, allow_pickle=False)
        gaze, gaze_rows = take_frames(gaze_full, passive_frames, gaze_path)
        source_full = np.load(source_npy, allow_pickle=False)
        annotations, source_rows = take_frames(source_full, source_frames, source_npy)
        names = annotations.dtype.names
        stage_field = 'stage' if 'stage' in names else 'event'
        assert stage_field in names, source_npy
        labels = np.array([' '.join(str(value).strip().lower().split()) for value in annotations[stage_field]], dtype='U128')
        assert set(labels).issubset(set(STAGES[task]) | set(EMPTY_STAGES)), (source_id, set(labels))
        occurrence = np.r_[0, np.cumsum(labels[1:] != labels[:-1])].astype(np.int64)
        occurrence[np.isin(labels, EMPTY_STAGES)] = -1
        labels, occurrence = labels[dlc_indices], occurrence[dlc_indices]
        source_resolution, source_video_frames, source_fps = video_size(source_video)
        passive_resolution, passive_video_frames, passive_fps = video_size(passive_video)
        assert source_resolution == passive_resolution, recording_id
        assert passive_video_frames == len(passive_frames), recording_id
        assert source_video_frames == len(source_frames), recording_id
        raw_norm = np.column_stack((gaze['gaze_x'], gaze['gaze_y'])).astype(float)
        coordinate_valid = np.isfinite(raw_norm).all(axis=1) & (raw_norm >= 0).all(axis=1) & (raw_norm <= 1).all(axis=1)
        accepted = raw_norm.copy()
        accepted[~coordinate_valid] = np.nan
        gaze_px = accepted * source_resolution
        transform = np.diag([source_resolution[0], source_resolution[1], 1.])
        grid = np.arange(int(np.floor(native_time[-1] * 41.)) + 1) / 41.
        grid_index = np.searchsorted(native_time, grid, side='right') - 1
        grid_index = grid_index.clip(0, len(native_time) - 1)
        grid_stage, grid_occurrence = labels[grid_index].copy(), occurrence[grid_index].copy()
        stale = grid - native_time[grid_index] > .1
        grid_stage[stale], grid_occurrence[stale] = 'none', -1
        points, likelihood = read_dlc(dlc_path, len(source_frames))
        record = dict(content_id=recording_id, recording_id=recording_id, user=observer, observer_user=observer,
                      observer_skill=meta['observer_skill'], source_content_id=source_id, source_user=source_user,
                      src_skill=meta['src_skill'], seq=int(meta['seq']), task=task, task_number=int(meta['task']), split=meta['split'])
        audit = dict(record, npy_path=str(gaze_path), npy_sha256=sha(gaze_path),
                     video_ts_path=str(passive_clock), video_ts_sha256=sha(passive_clock),
                     dlc_path=str(dlc_path), dlc_sha256=sha(dlc_path), source_stage_npy_path=str(source_npy),
                     native_rows=len(gaze), coordinate_valid_rows=int(coordinate_valid.sum()),
                     coordinate_invalid_rows=int((~coordinate_valid).sum()), valid_column_used_for_selection=False,
                     accepted_rows_by_stored_valid_code={str(int(code)): int(np.sum(coordinate_valid & (gaze['valid'] == code))) for code in np.unique(gaze['valid'])},
                     timestamps_source='precise passive video_ts.csv joined on original Frame_Number, equal mapped source video_ts',
                     npy_timestamp_nonincreasing=int(np.sum(np.diff(gaze['Timestamp(ns)'].astype(np.int64)) <= 0)),
                     npy_video_timestamp_mismatches=int(np.sum(gaze['Timestamp(ns)'] != passive_ns)),
                     grid_hz=41., max_interpolation_gap_s=.1, display_mapping='verified identical full-frame image, normalized gaze times source width/height',
                     passive_resolution=passive_resolution, source_resolution=source_resolution,
                     passive_nominal_fps=passive_fps, source_nominal_fps=source_fps,
                     passive_deleted_video_ts_rows=passive_deleted, source_deleted_video_ts_rows=source_deleted,
                     source_crop_first_local_frame=int(dlc_indices[0]), source_crop_last_local_frame=int(dlc_indices[-1]),
                     source_frame_to_dlc_index='original frame key joined to retained source video_ts row index',
                     source_gaze_npy_used_as_gaze=False, original_frame_timestamp_stage_psm_alignment=True)
        cache = dict(time_s=grid, stage=grid_stage, occurrence=grid_occurrence,
                     native_time_s=native_time, native_stage=labels, native_occurrence=occurrence,
                     frame_number=passive_frames, native_source_frame_number=source_frames[dlc_indices],
                     native_gaze_norm=accepted, native_gaze_norm_original=raw_norm, native_gaze_px=gaze_px,
                     native_gaze_coordinate_valid=coordinate_valid, native_gaze_valid_code_diagnostic=gaze['valid'],
                     native_gaze_npy_timestamp_ns=gaze['Timestamp(ns)'], native_video_timestamp_ns=passive_ns,
                     source_video_timestamp_ns=source_ns[dlc_indices], source_dlc_local_index=dlc_indices,
                     native_playback_frame_index=np.arange(len(passive_frames)), display_transform_matrix=transform)
        source_cache_equal = True
        with np.load(OLD_INPUTS / 'cache' / f'{source_id}.npz', allow_pickle=False) as baseline_cache:
            for field, actual in [('frame_number', source_frames), ('native_stage', labels), ('native_occurrence', occurrence), ('native_time_s', native_time), ('time_s', grid)]:
                assert np.array_equal(actual, baseline_cache[field]), (recording_id, field)
            for side in ('l', 'r'):
                native_psm = points[side][dlc_indices]
                native_likelihood = likelihood[side][dlc_indices]
                cache['resolution_' + side] = np.array(source_resolution)
                cache['gaze_raw_' + side] = interpolate_valid(native_time, gaze_px, grid, .1)
                cache['psm_raw_' + side] = interpolate_valid(native_time, native_psm, grid, .1)
                cache['native_psm_' + side] = native_psm
                cache['likelihood_' + side] = native_likelihood
                for field in ('native_psm_' + side, 'likelihood_' + side, 'psm_raw_' + side):
                    assert np.allclose(cache[field], baseline_cache[field], rtol=0, atol=0, equal_nan=True), (recording_id, field)
        audit['source_PSM in stage_grid_equal_frozen_GT84_input'] = source_cache_equal
        cache['qc_json'] = np.asarray(json.dumps(dict(record, **{key: value for key, value in audit.items() if key not in record})))
        cache['fingerprint'] = np.asarray(hashlib.sha256((sha(gaze_path) + sha(passive_clock) + sha(source_npy) + sha(dlc_path)).encode()).hexdigest())
        np.savez_compressed(INPUTS / 'cache' / f'{recording_id}.npz', **cache)
        np.savez_compressed(INPUTS / 'frame_mapping' / f'{recording_id}.npz',
                            original_frame_number=passive_frames, gaze_npy_row_index=gaze_rows,
                            playback_local_index=np.arange(len(passive_frames)), source_retained_row_index=dlc_indices,
                            source_npy_row_index=source_rows[dlc_indices], dlc_local_index=dlc_indices,
                            passive_timestamp_ns=passive_ns, source_timestamp_ns=source_ns[dlc_indices],
                            native_elapsed_s=native_time, grid_source_row_index=grid_index,
                            native_stage=labels, native_occurrence=occurrence, display_transform_matrix=transform)
        row.update(frame_mapping_file=f'input_mapping/{recording_id}.npz', coordinate_transform='normalized x*width,y*height in verified full-frame source image',
                   native_rows=len(gaze), coordinate_valid_rows=int(coordinate_valid.sum()),
                   first_original_frame=int(passive_frames[0]), last_original_frame=int(passive_frames[-1]),
                   duration_s=float(native_time[-1]), stage_field=stage_field, mapping_status='verified')
        records.append(record)
        policies.append(audit)
    config = dict(baseline_version='gt84-fixed-v2.1-npy-20261106', data_root=str(DATA),
                  source_manifest=str(LINKS), gaze_validity='finite normalized x/y in inclusive [0,1], valid diagnostic only',
                  user_identity='observer_user', tasks=['A', 'C'], source_psm_policy='all finite x/y, no quality filtering',
                  native_clock='precise video_ts matched by original Frame_Number', grid_hz=41., max_gap_s=.1)
    save(INPUTS / 'config.json', config)
    save(INPUTS / 'records.json', records)
    save(INPUTS / 'input_policy_audit.json', dict(**config, records=policies, source_files=list(source_files.values())))
    save(INPUTS / 'input_record_inventory.json', inventory)
    with (INPUTS / 'quality.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(dict(records=len(records), observers=len({r['user'] for r in records}), sources=len({r['source_content_id'] for r in records}),
                          task_counts=dict(Counter(r['task'] for r in records)), population_task_counts=dict(population), input_files=len(source_files)), indent=2))


if __name__ == '__main__':
    main()
