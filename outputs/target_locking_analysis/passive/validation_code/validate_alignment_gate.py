"""Independently verify current mapping and unresolved presentation semantics.

Does not import mapping_review or any Target Locking input/measurement helper.
Original inputs are read only. No proposed replacement clock is generated.
"""
import argparse
import bisect
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

LINKS = Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json')
OUT = Path(__file__).resolve().parent
PATH_FIELDS = ('gaze_npy_path', 'source_video_path', 'stage_path',
               'target_locking_label_path', 'psm_dlc_csv_path')
COMMIT = 'e9a153a5e5e284e929272cf256cfeca07b43fa5f'
NOTEBOOK = 'Dataloader/generate_gaze_npy.ipynb'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--output', type=Path, default=OUT/'independent_alignment_validation.json')
    args = parser.parse_args()
    ledger = {}

    def track(path):
        path = Path(path).resolve()
        ledger.setdefault(str(path), sha(path))
        return path

    manifest = json.loads(track(LINKS).read_text(encoding='utf-8-sig'))
    base = Path(manifest['data_root'])
    rows = []
    path_audit = []
    clip_keys = set()
    candidate_keys = []
    for key, entry in manifest['clips'].items():
        assert key not in clip_keys
        clip_keys.add(key)
        meta = entry['meta']
        paths = {}
        for field in PATH_FIELDS:
            value = entry.get(field)
            if value is None or str(value).strip().lower() in ('', 'nan', 'null', 'none'):
                path = None
            else:
                path = Path(value)
                if not path.is_absolute():
                    path = base/path
            paths[field] = path
            path_audit.append(dict(clip_key=key, field=field, original=value,
                                   resolved=str(path) if path else None,
                                   exists=path.is_file() if path else False))
        selected = meta.get('task_name') in ('A', 'C') and meta.get('observer_skill') == 'novice' and meta.get('src_skill') == 'novice'
        if not selected:
            continue
        candidate_keys.append(key)
        assert all(p and p.is_file() for p in paths.values()), key
        for path in paths.values():
            track(path)
        gaze_path = paths['gaze_npy_path']
        recording = gaze_path.stem.removesuffix('_gaze')
        # These non-annotation candidates follow the located historical constructor.
        # Their exact content is then checked; they never replace a JSON label link.
        passive_clock = track(gaze_path.parent.parent/'csv'/f'{recording}_video_ts.csv')
        raw_gaze = track(gaze_path.parent.parent/'csv'/f'{recording}_gaze.csv')
        source_video = paths['source_video_path']
        source_clock = track(source_video.parent.parent/'csv'/f'{source_video.stem}_video_ts.csv')
        passive_all = read_csv(passive_clock)
        source_all = read_csv(source_clock)
        retained = lambda records: [r for r in records if r.get('IsDelete', '0').strip() in ('', '0', '0.0')]
        passive_rows, source_rows = map(retained, (passive_all, source_all))
        pf = np.asarray([int(Decimal(r['Frame_Number'])) for r in passive_rows], np.int64)
        pt = np.asarray([int(Decimal(r['Timestamp(ns)'])) for r in passive_rows], np.int64)
        sf = np.asarray([int(Decimal(r['Frame_Number'])) for r in source_rows], np.int64)
        st = np.asarray([int(Decimal(r['Timestamp(ns)'])) for r in source_rows], np.int64)
        source_index = {int(frame): i for i, frame in enumerate(sf)}
        assert len(source_index) == len(sf)
        mapped = np.asarray([source_index[int(frame)] for frame in pf])
        assert len(np.unique(pf)) == len(pf)
        raw = read_csv(raw_gaze)
        tick_field = next(f for f in raw[0] if f.startswith('TIMETICK('))
        freq = int(tick_field.removeprefix('TIMETICK(f=').removesuffix(')'))
        assert freq == 10000000
        ticks = [int(Decimal(r[tick_field])) for r in raw]
        origin = ticks[0]
        transformed = [int(pt[0])+(t-origin)*100 for t in ticks]
        order = sorted(range(len(raw)), key=lambda i: transformed[i])
        sorted_t = [transformed[i] for i in order]
        selected_rows = []
        for timestamp in pt:
            insertion = bisect.bisect_left(sorted_t, int(timestamp))
            left = max(0, min(len(raw)-1, insertion-1))
            right = max(0, min(len(raw)-1, insertion))
            chosen = right if abs(sorted_t[right]-int(timestamp)) < abs(sorted_t[left]-int(timestamp)) else left
            selected_rows.append(order[chosen])
        selected_rows = np.asarray(selected_rows, np.int64)
        npy = np.load(gaze_path, allow_pickle=False)
        reconstructed = np.asarray([[float(r['BPOGX']), float(r['BPOGY'])] for r in raw], np.float32)
        gaze = np.column_stack([npy['gaze_x'], npy['gaze_y']])
        selected_gaze = np.clip(reconstructed[selected_rows], 0, 1)
        valid = np.asarray([int(Decimal(r['BPOGV'])) for r in raw], np.uint8)
        exact = np.array_equal(gaze, selected_gaze, equal_nan=True) and np.array_equal(npy['valid'], valid[selected_rows]) and np.array_equal(pf, npy['Frame_Number']) and np.array_equal(pt, npy['Timestamp(ns)'])
        vid = np.asarray([int(Decimal(r['VID_FRAME'])) for r in raw], np.int64)[selected_rows]
        delta = vid-np.arange(len(pf))
        coordinate_valid = np.isfinite(gaze).all(1) & (gaze >= 0).all(1) & (gaze <= 1).all(1)
        compared = dict(clip_key=key, recording_id=recording, observer=meta['observer'],
                        source_operator=meta['user'], source_content_id=meta['content_id'],
                        task=meta['task_name'], rows=len(npy),
                        constructor_reproduced=bool(exact),
                        passive_clock_equals_source_clock=bool(np.array_equal(pt, st[mapped])),
                        selected_raw_frame_conflicts=int((delta != 0).sum()),
                        selected_raw_frame_delta_min=int(delta.min()),
                        selected_raw_frame_delta_max=int(delta.max()),
                        coordinate_valid=int(coordinate_valid.sum()),
                        valid_code_filter_used=False,
                        gaze_origin='JSON gaze_npy_path only',
                        canonical_historical_mapping_verified=True,
                        physical_playback_mapping_verified=False,
                        raw_VID_FRAME_is_physical_display_counter='not_established',
                        admission_decision='outside_this_evidence_audit',
                        limitation='canonical_source_elapsed_mapping_reproduced; exported_VID_FRAME_association_differs; physical_presentation_semantics_not_established')
        assert exact, recording
        assert compared['passive_clock_equals_source_clock'], recording
        assert compared['selected_raw_frame_conflicts'] > 0, recording
        rows.append(compared)
    notebook = subprocess.check_output(['git', '-c', 'safe.directory=D:/GitHub_Project/DVSS_Pytorch',
        'show', f'{COMMIT}:{NOTEBOOK}'], cwd=r'D:\GitHub_Project\DVSS_Pytorch')
    nb = json.loads(notebook)
    code = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c.get('cell_type') == 'code')
    (OUT/'independent_historical_constructor.txt').write_text(code, encoding='utf-8')
    assert 'TIMETICK' in code and 'BPOGX' in code
    conflicts = [path for path, before in ledger.items() if sha(path) != before]
    assert not conflicts
    comparisons = {}
    if args.audit_dir:
        other = json.loads((args.audit_dir/'mapping_detail.json').read_text(encoding='utf-8'))
        lookup = {r['clip_key']: r for r in other}
        assert set(lookup) == set(candidate_keys)
        for row in rows:
            expected = lookup[row['clip_key']]
            for mine, theirs in [('constructor_reproduced', 'constructor_exact'),
                                 ('selected_raw_frame_conflicts', 'selected_raw_vid_frame_conflicts'),
                                 ('selected_raw_frame_delta_min', 'selected_raw_vid_frame_delta_min'),
                                 ('selected_raw_frame_delta_max', 'selected_raw_vid_frame_delta_max')]:
                assert row[mine] == expected[theirs], (row['recording_id'], mine)
        comparisons['independent_mapping_audit_reproduced'] = True
    report = dict(status='canonical_mapping_reproduced; physical_presentation_semantics_unresolved',
        total_json_records=len(clip_keys), selected_novice_AC=len(rows),
        selected_observers=len(set(r['observer'] for r in rows)),
        selected_source_videos=len(set(r['source_content_id'] for r in rows)),
        raw_npy_rows_reconstructed=sum(r['rows'] for r in rows),
        all_constructors_reproduced=all(r['constructor_reproduced'] for r in rows),
        all_available_clocks_equal_source_acquisition_clock=all(r['passive_clock_equals_source_clock'] for r in rows),
        recordings_with_raw_playback_conflicts=sum(r['selected_raw_frame_conflicts'] > 0 for r in rows),
        conflicting_rows=sum(r['selected_raw_frame_conflicts'] for r in rows),
        canonical_mapping_verified_recordings=len(rows),
        raw_VID_FRAME_physical_authority='not_established_by_available_vendor_or_project_sources',
        empirical_occupancy_validation='not_performed_by_this_mapping_audit',
        empirical_area_validation='not_performed_by_this_mapping_audit',
        empirical_temporal_validation='not_performed_by_this_mapping_audit',
        source_files_unchanged=not conflicts, source_file_count=len(ledger),
        source_sha256=ledger, path_values_checked=len(path_audit),
        historical_constructor=dict(repo=r'D:\GitHub_Project\DVSS_Pytorch', commit=COMMIT,
             git_path=NOTEBOOK, sha256=hashlib.sha256(notebook).hexdigest(), executed=False),
        comparisons=comparisons, records=rows)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT/'independent_path_resolution.json').write_text(json.dumps(path_audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k,v in report.items() if k not in ('source_sha256','records')}, indent=2))


if __name__ == '__main__':
    main()
