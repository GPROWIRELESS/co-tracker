"""Independent saved-output audit; no source data or numerical tables edited."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def records(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def key(c):
    return tuple(c[k] for k in ('content_id','source','side','signal','stage','mode'))


def read_primary(path):
    out={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            c=json.loads(line)
            if c['variant']=='primary':out[key(c)]=c
    return out


def main(root,old):
    root,old=Path(root),Path(old)
    validation=json.loads((root/'validation.json').read_text(encoding='utf8'))
    assert validation['legacy_regression']['passed']
    primary=read_primary(root/'record_curves.jsonl')
    historical=read_primary(old/'record_curves.jsonl')
    assert primary.keys()==historical.keys()
    differences=[]
    for k,c in primary.items():
        a,b=np.asarray(c['r'],float),np.asarray(historical[k]['r'],float)
        assert np.array_equal(np.isfinite(a),np.isfinite(b)),k
        differences.append(float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)]))) if np.isfinite(a).any() else 0.)
    assert max(differences)==0.
    # Reconstruct primary complete-record, equal-record/equal-user aggregation.
    grouped=defaultdict(lambda:defaultdict(list))
    for c in primary.values():
        a=np.asarray(c['r'],float)
        if np.isfinite(a).all():
            k=tuple(c[x] for x in ('task','side','signal','stage','mode','source'))
            grouped[k][c['user']].append(a)
    output_groups=defaultdict(list)
    for row in records(root/'group_curves.csv'):
        if row['variant']=='primary':
            k=tuple(row[x] for x in ('task','side','signal','stage','mode','source'))
            output_groups[k].append(row)
    group_errors=[]
    for k,rows in output_groups.items():
        users=grouped.get(k,{})
        zusers=[np.mean(np.arctanh(np.clip(a,-1+1e-7,1-1e-7)),axis=0) for a in users.values()]
        # Fisher nesting equals a user mean in z space after equal-record z mean.
        expected=np.tanh(np.mean(zusers,axis=0)) if len(users)>=3 else np.full(len(rows),np.nan)
        got=np.array([float(x['r']) if x['r'] else np.nan for x in rows])
        assert np.array_equal(np.isfinite(expected),np.isfinite(got)),k
        err=float(np.max(np.abs(expected[np.isfinite(expected)]-got[np.isfinite(got)]))) if np.isfinite(expected).any() else 0.
        assert err<1e-12,(k,err)
        group_errors.append(err)
    paired=records(root/'paired_metrics.csv')
    assert all(r['analysis_rule'] in ('fixed_left','fixed_right') and r['same_side']=='True' for r in paired)
    supplementary=records(root/'selected_rule_paired_metrics.csv')
    eligible=[r for r in supplementary if r['side']=='gt_selected_side' and r['common_input']=='True']
    assert all(r['same_side']=='True' and r['gt_side']==r['model_side'] for r in eligible)
    qc=records(root/'preprocessing_qc.csv')
    assert all(int(r['quality_rejected_count'])==0 for r in qc)
    diagnostics=[]
    for p in sorted((root/'native_qc_flags').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for side in ('l','r'):
                assert not z[side+'_quality_rejected'].any()
                assert len(z[side+'_native_time_s'])==len(z[side+'_native_nonfinite'])
                assert len(z[side+'_grid_time_s'])==len(z[side+'_grid_ordinary_resample'])
                assert len(z[side+'_velocity_time_s'])==len(z[side+'_velocity_psm_filter_edge'])
            diagnostics.append(p.name)
    manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf8'))
    changed_code=[p for p,h in manifest['code_sha256'].items() if sha(root/'code_snapshot'/p)!=h]
    assert not changed_code
    null=records(root/'unpaired_reference.csv')
    effective=[int(r['n_valid']) for r in null if int(r['n_target_users'])>=3]
    result=dict(passed=True,n_primary_curves=len(primary),primary_vs_previous_model_v2_max_abs_r=max(differences),
                independent_primary_group_curves=len(output_groups),max_abs_group_r_error=max(group_errors),
                n_fixed_pairs=len(paired),n_selected_pairs=len(supplementary),eligible_gt_selected_pairs_same_physical_side=len(eligible),
                quality_rejected_count=0,n_native_qc_files=len(diagnostics),code_snapshot_unchanged=True,
                reference_attempts=manifest['parameters']['null_replicates'],
                eligible_group_reference_effective_min=min(effective),eligible_group_reference_effective_max=max(effective),
                scope='Numerical and identity consistency; not biological validation or calibrated confidence coverage')
    (root/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--previous',required=True)
    a=p.parse_args();main(a.output,a.previous)
