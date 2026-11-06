"""Read-only comparison of current original DLC CSVs and legacy native caches.

Only this script's report directory is written. No xcorr is recalculated.
"""
import ast
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(r'D:\Dataset\dvss\Active_Data')
PROJECT = Path(r'D:\GitHub_Project\co-tracker')
CACHE = PROJECT/'outputs/gaze_psm_xcorr_dlc/cache'
REPORT = Path(__file__).parent
TEST_IDS = ['User1_1_1','User7_1_1','User9_7_1','User11_6_1','User12_3_1',
            'User1_5_2','User5_3_2','User6_1_2','User7_4_1','User12_5_2']


def sha256(path):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024): hasher.update(chunk)
    return hasher.hexdigest()


def compare(current, cached):
    shape_equal=current.shape==cached.shape
    if not shape_equal:
        return {'shape_equal':False,'current_shape':current.shape,'cache_shape':cached.shape,'exact_equal':False}
    finite=np.isfinite(current)&np.isfinite(cached)
    absolute=np.abs(current[finite]-cached[finite])
    return {'shape_equal':True,'shape':current.shape,'exact_equal':bool(np.array_equal(current,cached,equal_nan=True)),
            'nan_mask_equal':bool(np.array_equal(np.isnan(current),np.isnan(cached))),
            'finite_mask_equal':bool(np.array_equal(np.isfinite(current),np.isfinite(cached))),
            'current_nan_values':int(np.isnan(current).sum()),'cache_nan_values':int(np.isnan(cached).sum()),
            'max_abs_difference':float(np.max(absolute)) if len(absolute) else None,
            'differing_finite_values':int(np.sum(current[finite]!=cached[finite])),
            'n_values':current.size}


def csv_out(path, rows):
    columns=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def main():
    source=PROJECT/'tools/gaze_psm_pipeline_dlc.py'
    functions=['read_rows','number','timestamp_ns','timeline','read_dlc']
    tree=ast.parse(source.read_text(encoding='utf-8'))
    subset=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in functions],type_ignores=[])
    namespace={'csv':csv,'Path':Path,'np':np,'Decimal':Decimal,'InvalidOperation':InvalidOperation}
    exec(compile(ast.fix_missing_locations(subset),str(source),'exec'),namespace)
    roots=[Path(r'D:\Dataset'),PROJECT,Path(r'C:\Users\GT502\Desktop\JMRR2027')]
    scans=[];current_paths={}
    for root in roots:
        result=subprocess.run(['rg','--files','-uuu',str(root)],capture_output=True,text=True,encoding='utf-8')
        files=[Path(line) for line in result.stdout.splitlines()]
        csvs=[p for p in files if p.suffix.lower()=='.csv']
        named=[p for p in csvs if 'dlc' in p.name.lower() or 'deeplabcut' in p.name.lower()]
        recognized=[];read_errors=[]
        for path in csvs:
            try:
                with path.open(encoding='utf-8-sig',errors='replace',newline='') as stream:
                    reader=csv.reader(stream);header=[next(reader,[]) for _ in range(3)]
                if [row[0] if row else '' for row in header]==['scorer','bodyparts','coords']:
                    recognized.append(path)
            except Exception as exc:read_errors.append({'path':str(path),'error':str(exc)})
        scans.append({'root':str(root),'file_count':len(files),'csv_count':len(csvs),'filename_dlc_csv_count':len(named),
                      'standard_dlc_header_csv_count':len(recognized),'recognized_paths':[str(p) for p in recognized],
                      'read_errors':read_errors,'rg_exit_code':result.returncode,'rg_stderr':result.stderr})
        if root==roots[0]: current_paths={p.name.removesuffix('_DLC.csv'):p for p in recognized}
    cache_paths={p.stem:p for p in CACHE.glob('*.npz')}
    records=[];flat=[];input_inventory=[]
    for cid in sorted(set(current_paths)|set(cache_paths)):
        row={'content_id':cid,'current_dlc_exists':cid in current_paths,'cache_exists':cid in cache_paths,'is_test_video':cid in TEST_IDS}
        if not row['current_dlc_exists'] or not row['cache_exists']:
            row['status']='missing_input';records.append(row);continue
        csv_path=current_paths[cid];cache_path=cache_paths[cid];base=ROOT/cid.split('_')[0]
        vts=base/'csv'/f'{cid}_video_ts.csv'
        row.update(dlc_csv=str(csv_path),cache_npz=str(cache_path),video_ts_csv=str(vts))
        before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in [csv_path,cache_path,vts]}
        try:
            frames,ns,n_deleted=namespace['timeline'](base,cid)
            points,likelihood=namespace['read_dlc'](csv_path,expected_frames=len(frames))
            current={'frame_number':frames,'native_time_s':(ns-int(ns[0]))/1e9,
                     'native_psm_l':points['l'],'native_psm_r':points['r'],
                     'likelihood_l':likelihood['l'],'likelihood_r':likelihood['r']}
            with np.load(cache_path,allow_pickle=False) as saved:
                metrics={name:compare(value,saved[name]) for name,value in current.items()}
                qc=json.loads(str(saved['qc_json']))
            row.update(n_retained_frames=len(frames),deleted_rows=n_deleted,
                       start_timestamp_ns=int(ns[0]),end_timestamp_ns=int(ns[-1]),
                       original_cache_dlc_path=qc.get('dlc_csv'),source=qc.get('psm_source'),
                       dlc_likelihood_filter=qc.get('dlc_likelihood_filter'),comparisons=metrics,
                       status='exact_match' if all(v['exact_equal'] for v in metrics.values()) else 'mismatch')
            for name,metric in metrics.items():flat.append({'content_id':cid,'field':name,**metric})
            for path in [csv_path,vts]:
                input_inventory.append({'content_id':cid,'role':'dlc_csv' if path==csv_path else 'video_ts_csv',
                                        'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),
                                        'last_write_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()})
        except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        row['input_stat_unchanged']=all((p.stat().st_size,p.stat().st_mtime_ns)==stat for p,stat in before.items())
        records.append(row)
    tests=[]
    for cid in TEST_IDS:
        base=ROOT/cid.split('_')[0]
        paths={'gaze':base/'csv'/f'{cid}_gaze.csv','stage':base/'npy'/f'{cid}_gaze.csv',
               'video_ts':base/'csv'/f'{cid}_video_ts.csv','mp4':base/'stimuli'/f'{cid}.mp4',
               'dlc':base/'sam'/cid/f'{cid}_DLC.csv'}
        row={'content_id':cid}
        for role,path in paths.items():
            row[role+'_path']=str(path);row[role+'_exists']=path.is_file()
            row[role+'_bytes']=path.stat().st_size if path.is_file() else None
        with paths['stage'].open(encoding='utf-8-sig',newline='') as stream:
            row['stage_header']=next(csv.reader(stream))
        tests.append(row)
    summary={'audit_utc':datetime.now(timezone.utc).isoformat(),'read_only_source_audit':True,
             'method':'Original read_dlc and timeline function AST bodies; exact per-value comparison, equal_nan=True; no xcorr computed.',
             'pipeline_source':str(source),'pipeline_sha256':sha256(source),'functions':functions,
             'scanned_roots':scans,'dlc_file_count':len(current_paths),'cache_count':len(cache_paths),
             'record_count':len(records),'exact_match_count':sum(r['status']=='exact_match' for r in records),
             'missing_or_error_records':[r for r in records if r['status']!='exact_match'],
             'total_retained_frames':sum(r.get('n_retained_frames',0) for r in records),
             'all_input_stats_unchanged':all(r.get('input_stat_unchanged',False) for r in records),
             'per_field_max_abs_difference':{field:max([r.get('max_abs_difference',0) or 0 for r in flat if r['field']==field],default=None)
                                             for field in ['frame_number','native_time_s','native_psm_l','native_psm_r','likelihood_l','likelihood_r']},
             'all_nan_masks_equal':all(r.get('nan_mask_equal',False) for r in flat),
             'test_video_count':len(tests),'test_all_inputs_exist':all(all(r[role+'_exists'] for role in ['gaze','stage','video_ts','mp4','dlc']) for r in tests)}
    (REPORT/'all84_native_dlc_audit.json').write_text(json.dumps({'summary':summary,'records':records,'test_inventory':tests},indent=2),encoding='utf-8')
    csv_out(REPORT/'all84_native_dlc_field_comparison.csv',flat)
    csv_out(REPORT/'all84_native_input_manifest.csv',input_inventory)
    csv_out(REPORT/'test10_source_inventory.csv',tests)
    print(json.dumps({k:v for k,v in summary.items() if k!='scanned_roots'},indent=2))
    print(json.dumps([{'root':r['root'],'files':r['file_count'],'csvs':r['csv_count'],'name_dlc_csvs':r['filename_dlc_csv_count'],
                       'header_dlc_csvs':r['standard_dlc_header_csv_count'],'read_errors':len(r['read_errors'])} for r in scans],indent=2))


if __name__=='__main__': main()
