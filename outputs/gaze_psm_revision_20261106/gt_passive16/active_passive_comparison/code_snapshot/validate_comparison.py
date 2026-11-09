"""Independent validation of the paired active/passive supplement.

Reads original CSV/JSONL files and reconstructs expected pairings/aggregates.
Does not import the analysis implementation.
"""
import argparse
import csv
import json
import hashlib
import pathlib
from collections import Counter, defaultdict

import numpy as np

BASE = pathlib.Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106')
FIELDS = ('variant', 'task', 'side', 'signal', 'stage', 'mode')

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def read_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)

def key(row):
    return tuple(row[field] for field in FIELDS)

def truth(value):
    return str(value).lower() in ('true', '1')

def numeric(value):
    return float(value) if value not in ('', None, 'None', 'null') else float('nan')

def close(a, b, atol=1e-12):
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=0, equal_nan=True))

def fisher(values):
    return np.tanh(np.mean(np.arctanh(np.clip(np.asarray(values, float), -1+1e-7, 1-1e-7)), axis=0))

def by_identity(rows, values, field):
    labels=sorted({r[field] for r in rows})
    return np.array([np.mean([v for r,v in zip(rows,values) if r[field]==person],axis=0) for person in labels])

def expected_mean(rows, values, weighting):
    field={'observer_equal':'observer_user','operator_equal':'source_user'}.get(weighting)
    arr=np.asarray(values,float)
    if field: arr=by_identity(rows,arr,field)
    return np.mean(arr,axis=0)

def independent_seed(value):
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:4], 'little')

def validate_outputs(out, mapping, expected, curves, errors, checks):
    metrics=('r_zero','peak_r','peak_lag_s')
    output_mapping=read_csv(out/'paired_record_mapping.csv')
    if len(output_mapping)!=16: errors.append('Output mapping count differs')
    for r in output_mapping:
        x=mapping[r['passive_id']]
        for of,sf in [('active_id','source_content_id'),('observer_user','observer_user'),('source_user','source_user'),('task','task')]:
            if r[of]!=x[sf]:errors.append('Output mapping field differs: '+str((r['passive_id'],of)))
    for domain in ['active','passive']:
        exported={}
        with open(out/(domain+'_matched_record_curves.jsonl'),encoding='utf-8') as handle:
            for line in handle:
                r=json.loads(line);exported[(r['content_id'],)+key(r)]=r
        if exported!=curves[domain]:errors.append(domain+' extracted curves differ from originals')
    checks['exact_extracted_curve_rows']={domain:len(curves[domain]) for domain in curves}
    rows=read_csv(out/'paired_metrics.csv'); grouped=defaultdict(list)
    if len(rows)!=len(expected): errors.append('Paired metrics row count mismatch')
    seen=set()
    for row in rows:
        rk=(row['passive_id'],)+key(row)
        if rk in seen:errors.append('Duplicate paired row '+str(rk))
        seen.add(rk)
        p,a,pc,ac,support=expected[rk]
        if truth(row['included'])!=support:errors.append('Pair support differs '+str(rk))
        for metric in metrics:
            pv,av=numeric(p[metric]),numeric(a[metric])
            for name,want in [('passive_'+metric,pv),('active_'+metric,av),('delta_'+metric,pv-av)]:
                if not close(numeric(row[name]),want):errors.append('Pair metric differs '+str((rk,name)))
        if support:grouped[key(row)].append(row)
    checks['paired_metrics_checked']=len(rows)

    # Add common-time sensitivity using the independently generated record outputs.
    cf=out/'common_time_record_curves.jsonl'
    if cf.exists():
        with open(cf,encoding='utf-8') as handle:
            for line in handle:
                r=json.loads(line);r['variant']='common_time_primary';domain=r['domain'];cid=r['passive_id'] if domain=='passive' else r['active_id']
                curves[domain][(cid,)+key(r)]=r
        cs=read_csv(out/'common_time_record_summary.csv')
        ci={(r['domain'],r['passive_id'])+key(r):r for r in cs}
        ct=read_csv(out/'common_time_paired_metrics.csv')
        for row in ct:
            p=ci[('passive',row['passive_id'])+key(row)];a=ci[('active',row['passive_id'])+key(row)]
            support=truth(p['full_lag_support']) and truth(a['full_lag_support'])
            if truth(row['included'])!=support:errors.append('Common-time inclusion differs')
            for metric in metrics:
                if not close(numeric(row['delta_'+metric]),numeric(p[metric])-numeric(a[metric])):errors.append('Common-time delta differs')
            if support:grouped[key(row)].append(row)
        checks['common_time_paired_rows_checked']=len(ct)

    # Spatial values must come exclusively from paired zero-lag source rows.
    sp_fields=('distance_median_px','proximity_25px','proximity_50px','proximity_100px')
    source_sp={d:{(r['content_id'],)+key(r):r for r in read_csv(BASE/f/'spatial_records.csv') if r['at']=='zero'} for d,f in [('active','gt84'),('passive','gt_passive16')]}
    spatial=read_csv(out/'paired_spatial_metrics.csv');spgroup=defaultdict(list)
    for row in spatial:
        if row['at']!='zero':errors.append('Nonzero-lag spatial row found')
        p=source_sp['passive'][(row['passive_id'],)+key(row)];a=source_sp['active'][(row['active_id'],)+key(row)]
        if not expected[(row['passive_id'],)+key(row)][4]:errors.append('Incomplete curve pair included in spatial')
        for metric in sp_fields:
            for name,want in [('passive_'+metric,numeric(p[metric])),('active_'+metric,numeric(a[metric])),('delta_'+metric,numeric(p[metric])-numeric(a[metric]))]:
                if not close(numeric(row[name]),want):errors.append('Spatial metric differs '+str((row['passive_id'],name)))
        spgroup[key(row)].append(row)
    if len(spatial)!=218:errors.append('Unexpected spatial row count')
    checks['zero_lag_spatial_pairs_checked']=len(spatial)
    common_spatial_path=out/'common_time_paired_spatial_metrics.csv'
    if common_spatial_path.exists():
        csr=read_csv(out/'common_time_spatial_records.csv')
        csi={(r['domain'],r['passive_id'])+key(r):r for r in csr}
        csp=read_csv(common_spatial_path)
        for row in csp:
            p=csi[('passive',row['passive_id'])+key(row)];a=csi[('active',row['passive_id'])+key(row)]
            if p['at']!='zero' or a['at']!='zero' or not truth(p['both_full_lag_support']) or not truth(a['both_full_lag_support']):
                errors.append('Common-time spatial row has inappropriate support')
            if p['pairs']!=a['pairs']:errors.append('Common-time spatial pair counts differ')
            for metric in sp_fields:
                for name,want in [('passive_'+metric,numeric(p[metric])),('active_'+metric,numeric(a[metric])),('delta_'+metric,numeric(p[metric])-numeric(a[metric]))]:
                    if not close(numeric(row[name]),want):errors.append('Common-time spatial metric differs '+str((row['passive_id'],name)))
            spgroup[key(row)].append(row)
        checks['common_time_zero_lag_spatial_pairs_checked']=len(csp)

    # All scalar summaries and all three weightings are reconstructed independently.
    scalar=read_csv(out/'sensitivity_summary.csv')
    for row in scalar:
        records=(spgroup if row['metric'] in sp_fields else grouped)[key(row)]
        vals={d:expected_mean(records,[numeric(r[d+'_'+row['metric']]) for r in records],row['weighting']) for d in ['active','passive']}
        for name,want in [('active_value',vals['active']),('passive_value',vals['passive']),('delta',vals['passive']-vals['active'])]:
            if not close(numeric(row[name]),want):errors.append('Scalar aggregation differs '+str((key(row),row['metric'],row['weighting'],name)))
    checks['scalar_weighting_rows_checked']=len(scalar)
    loo=read_csv(out/'loo_sensitivity.csv')
    for row in loo:
        records=[r for r in grouped[key(row)] if r['observer_user']!=row['excluded_identity'] and r['source_user']!=row['excluded_identity']]
        vals={d:expected_mean(records,[numeric(r[d+'_'+row['metric']]) for r in records],'observer_equal') for d in ['active','passive']}
        if not close(numeric(row['delta']),vals['passive']-vals['active']):errors.append('Leave-one-identity-out delta differs')
        if int(row['n_pairs'])!=len(records):errors.append('Leave-one-identity-out count differs')
    checks['leave_one_identity_out_rows_checked']=len(loo)

    # Group curves: reconstruct fixed-cohort observer-equal Fisher-z curves.
    longs=read_csv(out/'paired_group_curves.csv');longgroups=defaultdict(list)
    for row in longs:longgroups[key(row)].append(row)
    reconstructed={}
    max_curve_error=0.
    for k, records in grouped.items():
        observers={r['observer_user'] for r in records}
        if len(observers)<3:
            if k in longgroups:errors.append('Insufficient group has a reported group curve')
            continue
        if k not in longgroups:errors.append('Missing supported group curve '+str(k));continue
        pc=[curves['passive'][(r['passive_id'],)+k] for r in records]
        ac=[curves['active'][(r['active_id'],)+k] for r in records]
        lag=np.asarray(pc[0]['lag_s'],float)
        pz=np.arctanh(np.clip(np.asarray([c['r'] for c in pc],float),-1+1e-7,1-1e-7))
        az=np.arctanh(np.clip(np.asarray([c['r'] for c in ac],float),-1+1e-7,1-1e-7))
        up=by_identity(records,pz,'observer_user');ua=by_identity(records,az,'observer_user')
        pr=np.tanh(up.mean(axis=0));ar=np.tanh(ua.mean(axis=0))
        got=longgroups[k]
        expected_values={'lag_s':lag,'passive_r':pr,'active_r':ar,'delta_r':pr-ar,'delta_z':up.mean(axis=0)-ua.mean(axis=0)}
        for field,want in expected_values.items():
            actual=np.array([numeric(r[field]) for r in got]);max_curve_error=max(max_curve_error,float(np.max(np.abs(actual-want))))
            if not close(actual,want):errors.append('Group curve differs '+str((k,field)))
        reconstructed[k]=(lag,pr,ar,up,ua)
    checks['group_curves_checked']=len(reconstructed);checks['max_group_curve_absolute_error']=max_curve_error
    peaks=read_csv(out/'group_peak_comparison.csv')
    for row in peaks:
        lag,pr,ar,_,_=reconstructed[key(row)];pi,ai=np.argmax(pr),np.argmax(ar)
        vals={'passive_peak_r':pr[pi],'active_peak_r':ar[ai],'delta_peak_r':pr[pi]-ar[ai], 'passive_peak_lag_s':lag[pi],'active_peak_lag_s':lag[ai],'delta_group_peak_lag_s':lag[pi]-lag[ai]}
        for field,want in vals.items():
            if not close(numeric(row[field]),want):errors.append('Group peak differs '+str((key(row),field)))
    checks['group_peaks_checked']=len(peaks)

    # Bootstrap archives are checked against independent synchronized observer draws.
    archive=np.load(out/'bootstrap_distributions.npz');index=read_json(out/'bootstrap_index.json')
    metric_count=curve_count=0
    for item in index:
        k=key(item);draws=archive[item['array_key']]
        if item['metric']=='group_curve_peak_lags':
            # Validate all peak archives and every curve band's synchronized draw.
            lag,pr,ar,up,ua=reconstructed[k]
            rng=np.random.default_rng(independent_seed((k,'curves')))
            ix=rng.integers(0,len(up),(5000,len(up)))
            bp=np.tanh(up[ix].mean(axis=1));ba=np.tanh(ua[ix].mean(axis=1))
            want=np.stack([lag[np.argmax(bp,axis=1)],lag[np.argmax(ba,axis=1)]],axis=1)
            if not close(draws,want):errors.append('Peak bootstrap archive differs '+str(k))
            lo,hi=np.percentile(bp-ba,[2.5,97.5],axis=0)
            got=longgroups[k]
            if not close([numeric(r['delta_r_low']) for r in got],lo) or not close([numeric(r['delta_r_high']) for r in got],hi):errors.append('Curve bootstrap band differs '+str(k))
            curve_count+=1
        else:
            records=(spgroup if item['metric'] in sp_fields else grouped)[k]
            b=by_identity(records,[numeric(r['delta_'+item['metric']]) for r in records],'observer_user')
            rng=np.random.default_rng(independent_seed((k,item['metric'],'paired_metric_bootstrap')))
            want=b[rng.integers(0,len(b),(5000,len(b)))].mean(axis=1)
            if not close(draws,want):errors.append('Metric bootstrap differs '+str((k,item['metric'])))
            lo,hi=np.percentile(want,[2.5,97.5])
            sr=[r for r in scalar if key(r)==k and r['metric']==item['metric'] and r['weighting']=='observer_equal'][0]
            if not close(numeric(sr['low']),lo) or not close(numeric(sr['high']),hi):errors.append('Metric bootstrap range differs')
            metric_count+=1
    checks['bootstrap_metric_arrays_recomputed']=metric_count
    checks['bootstrap_curve_peak_arrays_and_bands_recomputed']=curve_count
    coverage=read_csv(out/'paired_condition_coverage.csv')
    if len(coverage)!=72 or sum(r['status']=='supported' for r in coverage)!=56:errors.append('Predefined coverage count differs')
    for r in coverage:
        rec=grouped.get(key(r),[])
        if int(r['n_pairs_complete'])!=len(rec):errors.append('Coverage pair count differs')
    checks['predefined_coverage_cells']=len(coverage)

def load_sources():
    passive_dir, active_dir = BASE/'gt_passive16', BASE/'gt84'
    inventory = read_csv(passive_dir/'analysis_record_inventory.csv')
    links = read_json(pathlib.Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json'))['clips']
    links = {k.rsplit('/', 1)[-1]: v for k, v in links.items()}
    mapping = {r['recording_id']: r for r in inventory}
    errors = []
    for pid, row in mapping.items():
        if links[pid]['meta']['content_id'] != row['source_content_id']:
            errors.append('Original JSON mapping mismatch: '+pid)
    passive = {(r['content_id'],)+key(r): r for r in read_csv(passive_dir/'record_summary.csv')}
    active_all = {(r['content_id'],)+key(r): r for r in read_csv(active_dir/'record_summary.csv')}
    active_ids = {r['source_content_id'] for r in inventory}
    active = {k:r for k,r in active_all.items() if r['content_id'] in active_ids}
    curves = {}
    for domain, directory, selected in [('passive', passive_dir, set(mapping)), ('active', active_dir, active_ids)]:
        index = {}
        with open(directory/'record_curves.jsonl',encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row['content_id'] in selected:
                    k=(row['content_id'],)+key(row)
                    if k in index: errors.append('Duplicate curve: '+str(k))
                    index[k] = row
        curves[domain] = index
    return mapping, passive, active, curves, errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path(__file__).parent/'output')
    args=parser.parse_args()
    mapping, passive, active, curves, errors=load_sources()
    checks={}
    counts=Counter()
    expected={}
    for pk, p in passive.items():
        pid=p['content_id']; aid=mapping[pid]['source_content_id']; ak=(aid,)+key(p)
        a=active.get(ak)
        counts['passive_conditions']+=1
        if a is None:
            errors.append('Missing active counterpart: '+str(pk)); continue
        counts['matched_conditions']+=1
        pc,ac=curves['passive'][pk],curves['active'][ak]
        if not close(pc['lag_s'], ac['lag_s']): errors.append('Different lag grids: '+str(pk))
        supported=truth(p['full_lag_support']) and truth(a['full_lag_support'])
        if supported != (np.isfinite(np.asarray(pc['r'],float)).all() and np.isfinite(np.asarray(ac['r'],float)).all()):
            errors.append('Support flags differ from finite curves: '+str(pk))
        if p['variant']=='primary':
            counts['primary_conditions']+=1
            counts['primary_complete_pairs']+=int(supported)
        expected[pk]=(p,a,pc,ac,supported)
    checks['source_counts']=dict(counts)
    checks['mapping_count']=len(mapping)
    checks['source_complete_flags_and_lag_grids']=not errors
    if counts['primary_conditions'] != 440 or counts['primary_complete_pairs'] !=436 or len(mapping)!=16:
        errors.append('Unexpected primary counts or mapping count')

    validate_outputs(args.output,mapping,expected,curves,errors,checks)
    result={'passed':not errors,'checks':checks,'errors':errors,'validation_scope':'original source extraction, pair metrics, all weighting summaries, fixed-cohort Fisher curves, group peaks, zero-lag spatial values, complete synchronized bootstrap archives and curve bands'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'independent_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__=='__main__':
    main()
