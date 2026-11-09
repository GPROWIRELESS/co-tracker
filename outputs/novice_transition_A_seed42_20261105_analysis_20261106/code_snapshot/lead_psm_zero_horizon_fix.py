"""Compatibility entry: verify the corrected h=0 control without rewriting curves."""
import argparse,sys,json,time
from pathlib import Path
import numpy as np
DEFAULT_ROOT=Path(r'D:\GitHub_Project\co-tracker\outputs\novice_transition_A_seed42_20261105_analysis_20261106')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args();start=time.monotonic()
    cells={};fields=['lag_s','r','pairs','occurrences','gaze_rms','psm_rms']
    with (args.root/'gaze_psm_lead/record_curves.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['source'] not in ['gt','current_transition']:continue
            view,variant=row['variant'].split('__',1)
            if view not in ['anchor','target_current_transition']:continue
            key=(variant,)+tuple(row[k] for k in ['content_id','source','side','signal','stage','mode'])
            cells.setdefault(key,{})[view]={k:np.asarray(row[k],float) for k in fields}
    errors=[]
    for key,views in cells.items():
        assert set(views)=={'anchor','target_current_transition'},key
        for field in fields:
            a,b=views['anchor'][field],views['target_current_transition'][field]
            np.testing.assert_array_equal(a,b)
            if field=='r':errors.append(float(np.nanmax(abs(a-b))) if np.isfinite(a).any() else 0.)
    result={'complete':True,'n_record_curve_comparisons':len(cells),'array_comparisons':len(cells)*len(fields),
        'max_abs_anchor_vs_target0_correlation_difference':max(errors),'same_finite_masks':True,
        'gt_valid_column_used_for_selection':False,
        'definition':'h=0 uses native coordinate-only GT at shared sparse anchors; positive h uses target-time interpolation',
        'entry_mode':'validation only; zero-horizon correction is already in lead_psm_postprocess_driver.py',
        'elapsed_seconds':time.monotonic()-start}
    (args.root/'gaze_psm_lead/zero_horizon_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
