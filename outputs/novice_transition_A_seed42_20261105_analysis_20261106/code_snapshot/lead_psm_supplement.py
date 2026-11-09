"""Finalize GT sampling reference, matched sensitivities, and exact-support donor controls."""
import sys,json,csv,shutil,time,argparse
from pathlib import Path
import numpy as np
try:
    import lead_psm_postprocess_driver as a
except ModuleNotFoundError:
    import lead_psm_run as a

def readrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def proper_summary(pairs,fn):
    stages=fn([r for r in pairs if r['stage']!='whole'],bootstrap=5000,seed=42)
    whole=[r for r in fn([r for r in pairs if r['stage']=='whole'],bootstrap=5000,seed=42) if r['stage']=='whole']
    return stages+whole
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=a.DEFAULT_ROOT);args=parser.parse_args()
    start=time.time();root=args.root;out=root/'gaze_psm_lead';sys.path.insert(0,str(out/'code_snapshot'))
    from gaze_psm_statistics_v2 import group_curves,paired_group_differences,summarize_pairs
    meta=json.loads((root/'input_audit/metadata.json').read_text());models=meta['models'];records=meta['records']
    native={r['content_id']:a.load(root/'input_audit/native'/(r['content_id']+'.npz')) for r in records}
    predictions={r['content_id']:{m:a.load(root/'input_audit/predictions'/m/(r['content_id']+'.npz')) for m in models} for r in records}
    psm_qc=[]
    for rec in records:
        n=native[rec['content_id']]
        for j,(side,field) in enumerate([('l','left'),('r','right')]):
            xy=n[field];finite=np.isfinite(xy).all(1);lik=n['likelihood'][:,j]
            psm_qc.append({'content_id':rec['content_id'],'side':side,'native_count':len(xy),'finite_count':int(finite.sum()),
                'nonfinite_count':int((~finite).sum()),'low_likelihood_0p5_count':int(np.sum(np.isfinite(lik)&(lik<.5))),
                'jump_over_100px_count':int(np.sum(np.linalg.norm(np.diff(xy,axis=0),axis=1)>100)),
                'outside_image_count':int(np.sum(finite&((xy<0).any(1)|(xy>np.array([rec['width'],rec['height']])).any(1)))),
                'quality_rejected_count':0,'policy':'finite coordinates retained; confidence/jump/range diagnostics only'})
    a.csvsave(out/'psm_native_quality_diagnostics.csv',psm_qc)
    curves=[]
    with (out/'record_curves.jsonl').open(encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            for k in ['r','lag_s','pairs','occurrences','gaze_rms','psm_rms']:
                c[k]=np.array([float('nan') if x is None else x for x in c[k]])
            curves.append(c)
    common=a.shared_group_curves(curves)
    # Whole-task estimates are separate; they must not be counted as a fifth phase.
    pairs=readrows(out/'paired_metrics.csv');summary=proper_summary(pairs,summarize_pairs);a.csvsave(out/'paired_summary.csv',summary)
    ncurves=[];nrows=[];cache=out/'native_reference_cache';cache.mkdir(exist_ok=True)
    for r in records:
        n=native[r['content_id']];ix=np.arange(len(n['time']))
        p={'native_index':ix,'time':n['time'],'expectedxy':n['gaze'].copy(),'valid_maps':np.ones(len(ix),bool)}
        cc,rr,qq,aa,tt=a.analyze_job((r,n,{'native_gt_reference':p},{'native_gt_reference':0},'anchor','primary',cache))
        for c in cc:
            if c['source']=='gt':ncurves.append(c|{'variant':'native_reference__primary','view':'native_reference'})
        for x in rr:
            if x['source']=='gt':nrows.append(x|{'variant':'native_reference__primary','view':'native_reference'})
    a.csvsave(out/'native_gt_reference_record_summary.csv',nrows)
    ng,nl=group_curves(ncurves,bootstrap=5000,seed=42,min_users=3);a.annotate_groups(ng,nl)
    a.csvsave(out/'native_gt_reference_group_summary.csv',ng);a.csvsave(out/'native_gt_reference_group_curves.csv',nl)
    # Hold complete-record support fixed when quantifying sparse versus native GT differences.
    cc=[c|{'variant':'sampling_reference__primary','source':'sparse_gt'} for c in common if c['variant']=='anchor__primary' and c['source']=='gt']
    cc += [c|{'variant':'sampling_reference__primary','source':'gt'} for c in ncurves]
    cc=a.shared_group_curves(cc)
    sp=paired_group_differences(cc,bootstrap=5000,seed=42);a.csvsave(out/'gt_sampling_paired_group_summary.csv',sp)
    # Sensitivity table uses the same technical condition keys in primary and branch.
    sensitivity=[]
    key=lambda r:tuple(r[k] for k in ('content_id','user','task','side','signal','stage','mode','model'))
    for view in ['anchor']+['target_'+m for m in models]:
        base={key(r):r for r in pairs if r['variant']==view+'__primary'}
        for variant in a.VARIANTS:
            if variant in ['primary','window4']:continue
            branch={key(r):r for r in pairs if r['variant']==view+'__'+variant}
            shared=[k for k in base.keys()&branch.keys() if base[k]['common_input']=='True' and branch[k]['common_input']=='True']
            rows=[]
            for k in shared:
                rows.append(base[k]|{'variant':view+'__primary_matched_to_'+variant})
                rows.append(branch[k])
            if rows:sensitivity+=proper_summary(rows,summarize_pairs)
    a.csvsave(out/'sensitivity_matched_summary.csv',sensitivity)
    # Export ±4s narrowed to ±2s to compare identical lag domains as well as support.
    from gaze_psm_reliability_v2 import peak_features
    restricted=[];restricted_rows=[]
    for c in curves:
        if not c['variant'].endswith('__window4'):continue
        keep=np.abs(c['lag_s'])<=2+1e-12;new=dict(c,variant=c['variant'].replace('__window4','__window4_restricted2'))
        for field in ('lag_s','r','pairs','occurrences','gaze_rms','psm_rms'):new[field]=c[field][keep]
        restricted.append(new);restricted_rows.append({k:new[k] for k in a.KEY}|{'source':new['source']}|peak_features(new)|{'lag_identifiable':False,'lag_status':'restricted_domain_diagnostic_no_occurrence_resampling','direction_0p05':'undetermined','direction_0p10':'undetermined','direction_0p20':'undetermined','point_direction_0p05':'undefined','point_direction_0p10':'undefined','point_direction_0p20':'undefined'})
    rp=a.pair_curves(restricted,restricted_rows);a.csvsave(out/'window4_restricted2_paired_metrics.csv',rp)
    a.csvsave(out/'window4_restricted2_paired_summary.csv',proper_summary(rp,summarize_pairs))
    # The participant with a different training clock is also removed on target-time views.
    tc=[c for c in common if c['variant'].startswith('target_') and c['variant'].endswith('__primary') and c['user']!='User1']
    tg,tl=group_curves(tc,bootstrap=5000,seed=42,min_users=3);a.annotate_groups(tg,tl)
    a.csvsave(out/'target_time_leave_user1_out_group_summary.csv',[dict(r,excluded_user='User1') for r in tg])
    # Rebuild descriptive controls on exactly the observed complete-record set.
    groups=[]
    for r in readrows(out/'group_summary.csv'):
        for k in ('n_stable_users','peak_r'):
            r[k]=float(r[k]) if r[k] else np.nan
        groups.append(r)
    refs=a.references(native,predictions,models,records,groups,out,reps=5000,common_curves=common)
    # Display phases in action order rather than alphabetical order.
    plotted=[]
    for r in readrows(out/'group_curves.csv'):
        if r['variant']=='anchor__primary' or (r['variant'].startswith('target_') and r['variant'].endswith('__primary') and r['stage']=='whole'):
            for k in ('lag_s','r','r_resample_low','r_resample_high'):r[k]=float(r[k]) if r[k] else np.nan
            plotted.append(r)
    plotted_pairs=[]
    for r in readrows(out/'paired_group_summary.csv'):
        if r['variant']=='anchor__primary':
            for k in ('delta_group_peak_lag_s','delta_group_peak_lag_resample_low','delta_group_peak_lag_resample_high'):r[k]=float(r[k]) if r[k] else np.nan
            plotted_pairs.append(r)
    a.figures(groups,plotted,plotted_pairs,out,models)
    # Sampling-reference chart.
    import matplotlib.pyplot as plt
    stages=['whole','reach ring','grab ring','transfer ring','place ring'];fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,signal in zip(axes,['position','velocity']):
        for si,side in enumerate(['l','r']):
            for y,stage in enumerate(stages):
                rr=[r for r in sp if r['signal']==signal and r['mode']=='psm_stage' and r['side']==side and r['stage']==stage]
                if rr:
                    r=rr[0];v=r['delta_group_peak_lag_s'];lo=r['delta_group_peak_lag_resample_low'];hi=r['delta_group_peak_lag_resample_high'];yy=y+(si-.5)*.15
                    if np.isfinite(v):ax.plot([lo,hi],[yy,yy],color=a.COLORS[si+1]);ax.scatter([v],[yy],color=a.COLORS[si+1],label=('Left PSM' if si==0 else 'Right PSM') if y==0 else None)
        ax.axvline(0,color='#64748b',ls='--');ax.set_title(signal);ax.set_yticks(range(len(stages)),stages);ax.set_xlabel('Sparse GT − native GT peak lag (s)');ax.grid(axis='x',alpha=.2);ax.legend()
    fig.suptitle('GT sampling reference: matched complete recordings, participant equal');fig.tight_layout();plot=root/'figures/lead_gt_sampling_reference.png';fig.savefig(plot,dpi=180);plt.close(fig)
    # Final manifest explicitly records the postprocessing implementation.
    for source,name in [(Path(__file__),'lead_psm_supplement.py'),(Path(a.__file__),'lead_psm_postprocess_driver.py')]:
        destination=out/'code_snapshot'/name
        if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
    fallback_reproducer='''"""Run the frozen numerical implementation against an existing input_audit in a new output root."""
import argparse,sys
from pathlib import Path
import lead_psm_postprocess_driver as driver
import lead_psm_supplement as finish
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
driver.TOOLS=Path(__file__).resolve().parent
sys.argv=[__file__,'--root',str(args.root),'--workers',str(args.workers)]
driver.main()
sys.argv=[__file__,'--root',str(args.root)]
finish.main()
'''
    reproducer=Path(__file__).resolve().parent/'reproduce_lead.py'
    if reproducer.exists() and reproducer.resolve()!=(out/'code_snapshot/reproduce_lead.py').resolve():
        shutil.copy2(reproducer,out/'code_snapshot/reproduce_lead.py')
    elif not reproducer.exists():
        (out/'code_snapshot/reproduce_lead.py').write_text(fallback_reproducer,encoding='utf8')
    findings=json.loads((out/'findings.json').read_text());findings['gt_native_reference_groups']=a.clean(ng);findings['gt_sampling_paired_groups']=a.clean(sp);findings['figures'].append(str(plot))
    findings['reference_valid_draw_range']=[min((r['valid_draws'] for r in refs),default=0),max((r['valid_draws'] for r in refs),default=0)];a.jsave(out/'findings.json',findings)
    a.jsave(out/'supplement_validation.json',{'complete':True,'whole_excluded_from_phase_average':True,'native_gt_reference_curves':len(ncurves),
        'shared_complete_record_gt_sampling_comparison':True,'matched_sensitivity_rows':len(sensitivity),'exact_observed_record_support_in_donor_reference':True,
        'elapsed_seconds':time.time()-start,'execution_sequence':'lead_psm_run.py then lead_psm_supplement.py; reproduce with code_snapshot/reproduce_lead.py --root NEW_ROOT containing input_audit',
        'code_sha256':{p.name:a.sha(p) for p in (out/'code_snapshot').glob('*.py')}})
    print(json.dumps({'complete':True,'native_reference_curves':len(ncurves),'sampling_pairs':len(sp),'matched_sensitivity_rows':len(sensitivity),'elapsed_seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
