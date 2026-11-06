"""Build a reviewable overlay; source measurements and original inference stay intact."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).parent/'mpl')
import sys, json, shutil, argparse, importlib.util
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np
from scipy import stats
HERE=Path(__file__).parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=Path(r'D:\GitHub_Project\co-tracker'))
parser.add_argument('--output',type=Path,default=HERE/'overlay/outputs/score_corr_max_dlc')
parser.add_argument('--patched-tools',type=Path,default=HERE/'patched_tools')
OPTIONS=parser.parse_args()
ROOT=OPTIONS.repo.resolve()
sys.path.insert(0,str(ROOT))
from tools import analyze_score_corr_dlc as base
from tools.analyze_raw_score_corr_dlc import validate_results
from tools import plot_peak_correlation_vs_score as plots
from matplotlib.backends.backend_pdf import PdfPages

def typed(rows):
    result=[]
    for r in rows:
        d={}
        for k,v in r.items():
            if v=='' and k in ('association','p','q_bh','ci_low','ci_high','n_informative_users','bootstrap_valid','rank_slope','rank_slope_cluster_se','cluster_df'): d[k]=np.nan
            elif v in ('True','False'): d[k]=v=='True'
            else:
                try: d[k]=float(v)
                except (ValueError,TypeError): d[k]=v
        result.append(d)
    return result

def load_module(name):
    spec=importlib.util.spec_from_file_location(name,OPTIONS.patched_tools/f'{name}.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def main():
    source=ROOT/'outputs/score_corr_max_dlc'
    dest=OPTIONS.output.resolve()
    if dest==source.resolve() or source.resolve() in dest.parents or dest in source.resolve().parents:
        raise ValueError('Build into a separate review directory before installing')
    dest.mkdir(parents=True,exist_ok=True)
    corr=dest/'corrections'; corr.mkdir(exist_ok=True)
    source_hash={}
    inaccessible_links=[]
    for p in source.rglob('*'):
        try:
            if p.is_file(): source_hash[p.relative_to(source).as_posix()]=base.sha256(p)
        except OSError:
            inaccessible_links.append(p.relative_to(source).as_posix())
    json_dump=lambda path,x:path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    matched=typed(base.read_csv(source/'matched_scores.csv'))
    scores={}
    for task in ('A','C'):
        for r in base.read_csv(ROOT/'Index'/f'Task_{task}_Score.csv'):
            scores[f"User{r['UserID']}_{r['TaskNum']}_{r['Attempt']}"]=r
    fields=('Time','Eco','TimeRaw','EcoRaw','PenaltySum','PenaltyRawSum')
    duplicates={}
    for r in matched:
        vals=tuple(r[f] for f in fields)
        if all(np.isfinite(v) for v in vals): duplicates.setdefault(vals,[]).append(r['content_id'])
    for r in matched:
        ratio=r['TimeRaw']/r['duration_s'] if np.isfinite(r['TimeRaw']) and r['duration_s']>0 else np.nan
        r['time_raw_duration_ratio']=ratio
        r['duration_review_threshold']=.20
        r['duration_review_status']='unavailable' if not np.isfinite(ratio) else 'review' if abs(ratio-1)>.20 else 'within_tolerance'
        r['duration_flag_is_matching_error']=False
        r['severe_total_penalty_proxy']='unknown' if not np.isfinite(r['PenaltySum']) else str(r['PenaltySum']<=-100)
        raw=scores[r['content_id']]
        vals=[base.number(raw.get(k)) for k in ('Suture Break Count','Detached Towers')]
        r['suture_or_tower_minus100_flag']=('not_applicable' if r['task']=='A' else 'True' if any(v<=-100 for v in vals) else 'False' if all(np.isfinite(v) for v in vals) else 'unknown')
        r['duplicate_component_content_ids']=';'.join(x for x in duplicates.get(tuple(r[f] for f in fields),[]) if x!=r['content_id'])
    base.write_csv(corr/'matched_scores_audited.csv',matched)
    # Add audit columns to the requested matched table, preserving every original field.
    base.write_csv(dest/'matched_scores.csv',matched)
    for signal in ('position','velocity'):
        p=dest/'metrics'/signal/'Score'; p.mkdir(parents=True,exist_ok=True)
        base.write_csv(p/'matched_outcomes.csv',matched)
    diagnostics=[]; validations={}; all_results=[]; plot_jobs=[]
    outcomes=('Score','Time','Eco','TimeRaw','EcoRaw','PenaltySum','PenaltyRawSum')
    for outcome in outcomes:
        gathered=[]
        for signal in ('position','velocity'):
            d=source/'metrics'/signal/outcome
            rows=typed(base.read_csv(d/'associations.csv'))
            people=typed(base.read_csv(d/'participant_summaries.csv'))
            records=typed(base.read_csv(d/'record_phase_audit.csv'))
            phases=typed(base.read_csv(d/'phase_validity.csv'))
            for scenario in sorted({r['scenario'] for r in rows}):
                rr=[r for r in rows if r['scenario']==scenario]
                pp=[r for r in people if r['scenario']==scenario]
                validations[f'{outcome}/{signal}/{scenario}']=validate_results(rr,pp,base.scenario_rows(records,scenario),outcome)
            for r in rows:
                chosen=[p for p in people if p['scenario']==r['scenario'] and base.key(p)==base.key(r)]
                rec=[p for p in base.scenario_rows(records,r['scenario']) if base.key(p)==base.key(r)]
                variation_sample=chosen if r['method']=='between_participants' else rec
                reason=''
                if r['status']!='ok':
                    if r['status'].startswith('excluded'): reason='excluded'
                    elif variation_sample and np.ptp([p[outcome] for p in variation_sample])==0: reason='no_outcome_variation'
                    elif variation_sample and np.ptp([p[r['predictor']] for p in variation_sample])==0: reason='no_predictor_variation'
                    elif r['n_users']<6: reason='insufficient_users'
                    elif r['status']=='insufficient_users_with_within_variation': reason='insufficient_within_variation'
                    else: reason=r['status']
                r['unavailable_reason']=reason
                diagnostics.append({k:r.get(k,'') for k in ('outcome','scenario',*base.GROUP,'predictor','method','status','n_users','n_informative_users','unavailable_reason')})
            plot_jobs.append((outcome,signal,rows,phases,records,people))
            gathered+=rows
        all_results+=gathered
    base.write_csv(corr/'availability_diagnostics.csv',diagnostics)
    base.write_csv(corr/'significant_results.csv',[r for r in all_results if base.number(r.get('q_bh'))<.05])
    base.write_csv(corr/'key_order_comparison.csv',[r for r in all_results if r['task']=='A' and r['stage']=='transfer ring' and r['predictor']=='peak_r' and r['scenario'] in ('primary','velocity_secondary')])
    finite=[dict(r) for r in all_results if np.isfinite(base.number(r.get('p')))]
    for r,q in zip(finite,stats.false_discovery_control([r['p'] for r in finite])): r['q_global_all_999']=float(q)
    unique=[r for r in finite if r['outcome'] not in ('TimeRaw','EcoRaw')]
    for r,q in zip(unique,stats.false_discovery_control([r['p'] for r in unique])): r['q_global_without_raw_759']=float(q)
    base.write_csv(corr/'global_bh_sensitivity.csv',finite)
    # Optional post-hoc sensitivity. This is a known-total-penalty subset, NOT a critical-error diagnosis.
    lookup={r['content_id']:r for r in matched}
    sensitivity=[]; sensitivity_people=[]
    args=SimpleNamespace(bootstrap=10000,permutations=19999,seed=20261030,min_users=6)
    for outcome,signal,rows,phases,records,people in plot_jobs:
        if outcome!='Score': continue
        subset=[r for r in records if np.isfinite(lookup[r['content_id']]['PenaltySum']) and lookup[r['content_id']]['PenaltySum']>-100]
        scenario='primary' if signal=='position' else 'velocity_secondary'
        rr,pp=base.analyze(args,subset,phases,'Score',scenarios=(scenario,))
        validations[f'supplement/known_penalty_gt_minus100/{signal}']=validate_results(rr,pp,base.scenario_rows(subset,scenario),'Score')
        for r in rr+pp:
            r['subset']='known_penalty_gt_minus100'
            r['outcome']='Score'
            r['family']='posthoc_known_penalty_gt_minus100/'+r.get('family',scenario)
        sensitivity+=rr; sensitivity_people+=pp
    base.write_csv(corr/'penalty_subset_associations.csv',sensitivity)
    base.write_csv(corr/'penalty_subset_participants.csv',sensitivity_people)
    mod=load_module('analyze_score_corr_max_dlc'); layout=load_module('organize_score_corr_outputs')
    (dest/'plots').mkdir(exist_ok=True)
    with PdfPages(dest/'association_overviews.pdf') as legacy, PdfPages(dest/'plots'/'association_overviews.pdf') as book:
        for outcome,signal,rows,phases,records,people in plot_jobs:
            canonical=dest/'plots'/signal/outcome; canonical.mkdir(parents=True,exist_ok=True)
            # Layout accepts CSV strings; keep numerical fields parsed in its own code.
            phase_strings=[{k:str(v) for k,v in p.items()} for p in phases]
            fig=layout.make_overview({'associations':rows,'phase_validity':phase_strings},outcome,signal)
            fig.savefig(canonical/'overview.png',dpi=180,bbox_inches='tight'); fig.savefig(canonical/'overview.pdf',bbox_inches='tight'); book.savefig(fig,bbox_inches='tight'); base.plt.close(fig)
            if outcome in mod.OUTCOMES:
                fig=mod.overview(rows,phases,outcome,signal,'primary' if signal=='position' else 'velocity_secondary')
                folder=dest/'figures'; folder.mkdir(exist_ok=True)
                for ext in ('png','pdf'): fig.savefig(folder/f'association_overview_{outcome}_{signal}.{ext}',dpi=180,bbox_inches='tight')
                legacy.savefig(fig,bbox_inches='tight'); base.plt.close(fig)
    settings=json.loads((source/'provenance.json').read_text())['settings']
    coverage=json.loads((source/'validation.json').read_text())['coverage']
    mod.make_report(dest,coverage,all_results,settings,len(matched))
    (dest/'README.md').write_text('> **2026-11-05 修订：**请先读 [逐条复核与修正说明](corrections/review_zh.md) 和 [修订报告](report.md)。原有推断值保留；时长与重复值是待核查标志，非已确认匹配错误。新增筛选分析见 corrections/，不替代主分析。\n\n'+(source/'README.md').read_text(),encoding='utf-8')
    json_dump(corr/'validation.json',{'original_successful_tests_checked':sum(v.get('between_vs_scipy',0)+v.get('within_vs_full_dummy_regression',0) for k,v in validations.items() if not k.startswith('supplement')),'validation':validations,'source_files_unchanged':all(base.sha256(source/p)==sha for p,sha in source_hash.items()),'original_numeric_inference_preserved':True,'all_checks_passed':True})
    json_dump(corr/'original_file_sha256.json',source_hash)
    json_dump(corr/'unreadable_legacy_links.json',inaccessible_links)
    print(json.dumps({'overlay':str(dest),'original_tests':sum(r['status']=='ok' for r in all_results),'families':len(set((r['outcome'],r['family']) for r in all_results)),'supplement_tests':sum(r['status']=='ok' for r in sensitivity),'supplement_q_below_05':sum(r.get('q_bh',1)<.05 for r in sensitivity)},indent=2))

if __name__=='__main__':main()
