"""Rebuild all available DLC caches from current local inputs and compare arrays."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
import ast, csv, hashlib, json, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import uniform_filter1d

PROJECT = Path(r'D:\GitHub_Project\co-tracker')
DATA = Path(r'D:\Dataset\dvss\Active_Data')
OLD = PROJECT/'outputs/gaze_psm_xcorr_dlc'
OUT = Path(__file__).parent
FRESH = OUT/'current_data'
TEST = {'User1_1_1','User7_1_1','User9_7_1','User11_6_1','User12_3_1',
        'User1_5_2','User5_3_2','User6_1_2','User7_4_1','User12_5_2'}
FFPROBE = shutil.which('ffprobe')
assert FFPROBE


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def video_metadata(path):
    result = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)['streams'][0]
    rate = stream['avg_frame_rate']
    if rate in ('0/0', '0/1'):
        rate = stream['r_frame_rate']
    return int(stream['width']), int(stream['height']), float(Fraction(rate))


def legacy():
    path = PROJECT/'tools/gaze_psm_pipeline_dlc.py'
    names = {'read_rows','number','timestamp_ns','status_code','normalized_stage','timeline','stage_frames',
             'interpolate_valid','smooth_valid','dlc_path','read_dlc','source_files','fingerprint','extract'}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(np=np, csv=csv, Path=Path, Decimal=Decimal, InvalidOperation=InvalidOperation,
               hashlib=hashlib, json=json, os=os, tempfile=tempfile, uniform_filter1d=uniform_filter1d,
               CACHE_VERSION=5, PSM_SOURCE='dlc_left_right_xy', EMPTY_STAGES={'','none','nan','null'},
               video_metadata=video_metadata)
    exec(compile(ast.Module(body=nodes,type_ignores=[]), str(path),'exec'), env)
    return SimpleNamespace(**env)


LEG = legacy()
CONFIG = json.loads((OLD/'config.json').read_text())
CONFIG.update(data_root=str(DATA), output=str(FRESH), phase='all', reuse_cache=False,
              video_metadata_reader='ffprobe; dimensions independently read from current source video; fps is QC-only')
ARGS = SimpleNamespace(**dict(CONFIG, data_root=DATA, output=FRESH))


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape != b.shape:
        return dict(equal=False, old_shape=str(a.shape), current_shape=str(b.shape), max_abs_difference=None, differing_elements=None)
    if a.dtype.kind in 'iufcb' and b.dtype.kind in 'iufcb':
        eq=(a==b)|(np.isnan(a)&np.isnan(b))
        valid=np.isfinite(a)&np.isfinite(b)
        difference=float(np.max(abs(a[valid]-b[valid]))) if valid.any() else 0.
    else:
        eq=a==b;difference=None
    return dict(equal=bool(eq.all()), old_shape=str(a.shape), current_shape=str(b.shape),
                max_abs_difference=difference, differing_elements=int(np.size(eq)-np.count_nonzero(eq)))


def job(path):
    with np.load(path,allow_pickle=False) as z:
        old={k:z[k] for k in z.files}
    qc=json.loads(str(old['qc_json']))
    cid=qc['content_id']
    record={'content_id':cid,'Task Name':qc['task']}
    inputs=[]
    for source in LEG.source_files(record,ARGS):
        stat=source.stat()
        inputs.append(dict(content_id=cid, is_test_video=cid in TEST, path=str(source), bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns, sha256=sha(source) if source.suffix.lower()=='.csv' else '',
                           check='content_sha256' if source.suffix.lower()=='.csv' else 'video_metadata'))
    fresh,newqc=LEG.extract(record,ARGS)
    fields=[]
    for field in sorted(set(old)|set(fresh)):
        if field in {'fingerprint','qc_json'}:
            continue
        assert field in old and field in fresh, (cid,field)
        fields.append(dict(content_id=cid,task=qc['task'],is_test_video=cid in TEST,field=field,
                           **compare(old[field],fresh[field])))
    qcdifferences={k:dict(old=qc.get(k),current=newqc.get(k)) for k in set(qc)|set(newqc)
                   if k!='dlc_csv' and qc.get(k)!=newqc.get(k)}
    details=dict(content_id=cid,task=qc['task'],is_test_video=cid in TEST,old_cache=str(path),
                 old_cache_sha256=sha(path),rebuilt_cache=str(FRESH/'cache'/path.name),
                 rebuilt_cache_sha256=sha(FRESH/'cache'/path.name),all_analysis_arrays_equal=all(f['equal'] for f in fields),
                 compared_array_fields=len(fields),frames=len(fresh['frame_number']),grid_samples=len(fresh['time_s']),
                 old_dlc_csv=qc['dlc_csv'],current_dlc_csv=newqc['dlc_csv'],
                 fingerprint_equal=str(old['fingerprint'])==str(fresh['fingerprint']),
                 metadata_differences=json.dumps(qcdifferences,sort_keys=True))
    return fields,inputs,details,newqc


def main():
    FRESH.mkdir(parents=True,exist_ok=True)
    (FRESH/'config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    paths=sorted((OLD/'cache').glob('*.npz'))
    fields=[];inputs=[];records=[];quality=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(ff,ii,rr,qq) in enumerate(pool.map(job,paths),1):
            fields.extend(ff);inputs.extend(ii);records.append(rr);quality.append(qq)
            print(f'{i}/{len(paths)} {rr["content_id"]}: all {rr["compared_array_fields"]} arrays equal={rr["all_analysis_arrays_equal"]}',flush=True)
    write_csv(OUT/'cache_current_field_comparison.csv',fields)
    write_csv(OUT/'current_input_manifest.csv',inputs)
    write_csv(OUT/'cache_current_record_comparison.csv',records)
    (FRESH/'quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    test=[r for r in records if r['is_test_video']]
    summary=dict(current_data_root=str(DATA),legacy_cache_root=str(OLD/'cache'),n_cached_videos=len(paths),
                 n_test_videos=len(test),all_analysis_arrays_equal=all(r['all_analysis_arrays_equal'] for r in records),
                 all_test_arrays_equal=all(r['all_analysis_arrays_equal'] for r in test),
                 compared_fields=len(fields),mismatching_fields=[f for f in fields if not f['equal']],
                 total_native_frames=sum(r['frames'] for r in records),test_native_frames=sum(r['frames'] for r in test),
                 current_csv_files_hashed=sum(bool(r['sha256']) for r in inputs),
                 fingerprint_note='Old Linux paths/metadata differ from current Windows paths; fingerprint changes are expected. Content arrays are compared directly.',
                 video_note='Current MP4 width/height/fps independently read with ffprobe. No old source-video content hash exists; this audit establishes analysis-input equivalence, not video bitwise identity.',
                 pipeline_sha256=sha(PROJECT/'tools/gaze_psm_pipeline_dlc.py'))
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
