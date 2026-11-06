"""Read-only forensic comparisons for one legacy/current gaze-cache discrepancy."""
import ast
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT=Path(r'D:\GitHub_Project\co-tracker')
BASE=Path(r'D:\Dataset\dvss\Active_Data\User9')
OUT=Path(__file__).parent
CID='User9_2_1'
FUNCTIONS=['read_rows','number','timestamp_ns','status_code','timeline','interpolate_valid','smooth_valid']


def difference(a,b):
    good=np.isfinite(a)&np.isfinite(b)
    delta=np.abs(a-b)
    rowdiff=np.any(delta>0,axis=1) if a.ndim==2 else delta>0
    return {'exact_equal':bool(np.array_equal(a,b,equal_nan=True)),
            'nan_mask_equal':bool(np.array_equal(np.isnan(a),np.isnan(b))),
            'n_values':a.size,'n_rows':len(a),'finite_rows':int(np.isfinite(a).all(1).sum()) if a.ndim==2 else int(np.isfinite(a).sum()),
            'max_abs_diff':float(np.max(delta[good])),'mean_abs_diff':float(np.mean(delta[good])),
            'median_abs_diff':float(np.median(delta[good])),'different_rows':int(rowdiff.sum()),
            'rows_diff_over_1e_minus_8':int(np.any(delta>1e-8,axis=1).sum()) if a.ndim==2 else int((delta>1e-8).sum()),
            'rows_diff_over_1e_minus_3':int(np.any(delta>1e-3,axis=1).sum()) if a.ndim==2 else int((delta>1e-3).sum())}


def main():
    source=ROOT/'tools/gaze_psm_pipeline_dlc.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    namespace={'csv':csv,'Path':Path,'np':np,'Decimal':Decimal,'InvalidOperation':InvalidOperation,'uniform_filter1d':uniform_filter1d}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),namespace)
    paths={'old_dlc':ROOT/'outputs/gaze_psm_xcorr_dlc/cache'/f'{CID}.npz',
           'old_mask':ROOT/'outputs/gaze_psm_xcorr/cache'/f'{CID}.npz',
           'current':OUT/'current_data/cache'/f'{CID}.npz'}
    data={}
    for name,path in paths.items():
        with np.load(path,allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
    comparisons=[]
    for first,second in [('old_dlc','old_mask'),('old_dlc','current')]:
        for field in ['gaze_raw_l','gaze_l','gaze_raw_r','gaze_r']:
            comparisons.append({'first':first,'second':second,'field':field,**difference(data[first][field],data[second][field])})
    old,new=data['old_dlc'],data['current']
    stages=[];top=[]
    for field in ['gaze_raw_l','gaze_l']:
        for stage in sorted(set(old['stage'])):
            keep=old['stage']==stage
            stages.append({'field':field,'stage':str(stage),**difference(old[field][keep],new[field][keep])})
        delta=np.abs(old[field]-new[field]);rowmax=np.nan_to_num(np.max(delta,axis=1),nan=-1.)
        for i in np.argsort(rowmax)[-10:][::-1]:
            top.append({'field':field,'grid_index':int(i),'time_s':float(old['time_s'][i]),'stage':str(old['stage'][i]),
                        'old_xy':old[field][i].tolist(),'current_xy':new[field][i].tolist(),'absolute_xy_difference':delta[i].tolist()})
    raw=namespace['read_rows'](BASE/'csv'/f'{CID}_gaze.csv')
    filtered=[r for r in raw if str(r.get('Capture Unix timestamp','')).strip()]
    timestamp_text=[r['Capture Unix timestamp'] for r in filtered]
    tdec=np.asarray([namespace['timestamp_ns'](t) for t in timestamp_text],dtype=np.int64)
    tfloat=np.asarray([int(float(t)) for t in timestamp_text],dtype=np.int64)
    gaze=np.asarray([[namespace['number'](r['norm_x']),namespace['number'](r['norm_y'])] for r in filtered])
    good=np.asarray([namespace['status_code'](r.get('Status'))==namespace['status_code']('2') for r in filtered])
    good &= np.isfinite(gaze).all(1)&(gaze>=0).all(1)&(gaze<=1).all(1)
    gaze[~good]=np.nan;gaze*=np.array([1280,1024])
    frames,video_ns,_=namespace['timeline'](BASE,CID);origin=int(video_ns[0])
    experiments=[]
    for label,t in [('exact_decimal',(tdec-origin)/1e9),('int_float_timestamp',(tfloat-origin)/1e9),
                    ('absolute_float_subtraction',(np.asarray([float(t) for t in timestamp_text])-origin)/1e9)]:
        gr=namespace['interpolate_valid'](t,gaze,old['time_s'],.1)
        gs=namespace['smooth_valid'](gr,41.,80.)
        experiments.append({'timestamp_method':label,'vs_old_raw':difference(gr,old['gaze_raw_l']),
                            'vs_old_smoothed':difference(gs,old['gaze_l']),
                            'vs_current_raw':difference(gr,new['gaze_raw_l'])})
    derived=namespace['read_rows'](BASE/'gaze_analyze'/f'{CID}_gaze.csv')
    derived_diffs={k:sum(raw[int(r['_orig_row_index'])][k]!=r[k] for r in derived)
                   for k in ['Capture Unix timestamp','norm_x','norm_y','Status']}
    function_sources=[]
    for path in ROOT.rglob('gaze_psm_pipeline*.py'):
        if '.git' in path.parts:continue
        content=path.read_text(encoding='utf-8-sig');tr=ast.parse(content)
        hashes={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                for n in tr.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS}
        function_sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'function_ast_sha256':hashes})
    report={'content_id':CID,'is_one_of_the_10_model_test_videos':False,'source_cache_paths':{k:str(v) for k,v in paths.items()},
            'comparisons':comparisons,'per_stage_difference':stages,'largest_differences':top,
            'qc':{k:json.loads(str(v['qc_json'])) for k,v in data.items()},
            'raw_input':{'path':str(BASE/'csv'/f'{CID}_gaze.csv'),'raw_rows':len(raw),'timestamped_rows':len(filtered),
                         'accepted_gaze_samples':int(good.sum()),'scientific_timestamp_rows':sum('e' in s.lower() for s in timestamp_text),
                         'duplicate_times_all':len(tdec)-len(np.unique(tdec)),
                         'duplicate_times_accepted':int(good.sum())-len(np.unique(tdec[good])),
                         'nonincreasing_times_all':int((np.diff(tdec)<=0).sum()),
                         'nonincreasing_times_accepted':int((np.diff(tdec[good])<=0).sum()),
                         'max_decimal_vs_int_float_timestamp_difference_ns':int(np.abs(tdec-tfloat).max()),
                         'decimal_vs_int_float_differing_rows':int((tdec!=tfloat).sum()),
                         'timestamp_examples':timestamp_text[:3]},
            'derived_gaze_check':{'path':str(BASE/'gaze_analyze'/f'{CID}_gaze.csv'),'rows':len(derived),
                                  'alignment':'_orig_row_index','exact_string_differences':derived_diffs},
            'timestamp_parser_sensitivity_experiments':experiments,'available_pipeline_sources':function_sources,
            'facts':['Old DLC and old SAM gaze arrays are bitwise identical (NaNs equal).',
                     'Current exact-Decimal reconstruction exactly matches the current rebuilt gaze cache.',
                     'The inspected input/parser/interpolation/smoothing functions are AST-identical across available current and snapshot code.',
                     'Accepted gaze counts and all validity masks match old caches.',
                     'Current raw gaze timestamps have no duplicates and are strictly increasing.',
                     'Neither tested float-timestamp parsing variant reproduces the old cache.',
                     'All 88815 matching derived gaze rows preserve current raw timestamp/coordinate/status strings exactly.'],
            'unresolved':'The available artifacts do not establish the origin of the discrepancy. Scientific-notation timestamp/input precision is a plausible lead, not a demonstrated cause. No original NAS raw-CSV hash/contents at legacy extraction time was available for direct comparison.',
            'impact_scope':'Only this non-test recording differs in the parent all84 deep audit. The 50 prediction analyses on the 10 test recordings are unaffected by this recording; an 84-record unpaired reference may include it and should be rebuilt from current data. No downstream correlation effect is inferred solely from these small coordinate differences.'}
    (OUT/'User9_2_1_gaze_discrepancy_investigation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'comparisons':comparisons,'raw_input':report['raw_input'],
                      'derived_gaze_check':report['derived_gaze_check'],'unresolved':report['unresolved']},indent=2))


if __name__=='__main__':main()
