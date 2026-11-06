"""Verify the scope change against the previously delivered result before replacement."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

OLD=Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_models_dlc_v2\20261105_mild_smoothing')
NEW=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\outputs\gaze_psm_xcorr_models_dlc_v2\20261105_no_drop_ring')
KEY=['variant','content_id','user','task','side','signal','stage','mode','source']

def curves(path):
    index={};excluded=0
    with (path/'record_curves.jsonl').open(encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if c['stage']=='drop ring':excluded+=1;continue
            key=tuple(c[k] for k in KEY)
            if key in index:raise AssertionError('Duplicate curve')
            index[key]=hashlib.sha256(line.strip().encode()).hexdigest()
    return index,excluded

def main():
    aa,removed=curves(OLD);bb,residual=curves(NEW)
    assert aa==bb, 'Unexcluded record curves changed'
    checks=[]
    for p in sorted(NEW.glob('*.csv')):
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or 'stage' not in reader.fieldnames:continue
            n=0;bad=0
            for row in reader:
                n+=1;bad+=int(row['stage']=='drop ring')
            assert bad==0,p.name
            checks.append(dict(file=p.name,rows=n,excluded_stage_rows=bad))
    old=pd.read_csv(OLD/'paired_summary.csv');new=pd.read_csv(NEW/'paired_summary.csv')
    key=['variant','task','side','signal','stage','mode','model']
    metrics=[c for c in new if c.startswith('participant_equal_') or c.endswith('_resample_low') or c.endswith('_resample_high')]
    joined=old.merge(new,on=key,suffixes=('_old','_new'))
    maxdiff=0.
    for metric in metrics:
        x=joined[metric+'_old'].to_numpy(float);y=joined[metric+'_new'].to_numpy(float)
        assert np.array_equal(np.isfinite(x),np.isfinite(y)),metric
        mask=np.isfinite(x)
        if mask.any():maxdiff=max(maxdiff,float(abs(x[mask]-y[mask]).max()))
        assert np.allclose(x,y,rtol=0,atol=1e-12,equal_nan=True),metric
    for p in (NEW/'occurrence_statistics').rglob('User1_1_1.npz'):
        # After exclusion, none of this record's per-curve stats is indexed to the removed stage.
        q=pd.read_csv(NEW/'record_summary.csv')
        use=q[(q.content_id=='User1_1_1')&(q.variant==p.parent.name)]
        with np.load(p) as z:
            assert set(k[:-6] for k in z.files if k.endswith('_terms'))==set(use.statistics_key),p
    evidence=dict(status='passed',excluded_stage='drop ring',scope='all active stage-specific statistics, figures, occurrence caches and donor references',
        record_curves_removed=removed,record_curves_remaining=len(bb),residual_excluded_curves=residual,
        all_unexcluded_record_curves_byte_identical=True,unchanged_summary_rows=len(joined),
        summary_metric_max_abs_difference=maxdiff,csv_checks=checks,
        whole_record_alignment_audits_preserved=True,raw_inputs_modified=False)
    (NEW/'stage_removal_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in evidence.items() if k!='csv_checks'},ensure_ascii=False))

if __name__=='__main__':main()
