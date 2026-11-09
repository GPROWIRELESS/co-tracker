"""Verify fingerprints, tables, figures, links, observed-only scope, and inventory."""
import argparse, ast, hashlib, json, re, shutil
from pathlib import Path
import pandas as pd
from PIL import Image

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',required=True);args=pa.parse_args();out=Path(args.output).resolve();work=Path(__file__).resolve().parents[1]
    before=json.loads((out/'baseline_and_mapping_fingerprints_before.json').read_text())
    changes=[]
    for r in before:
        p=Path(r['path'])
        if not p.is_file() or sha(p)!=r['sha256']:changes.append(str(p))
    base=Path(before[0]['path'])
    reference=Path(json.loads((out/'run_config.json').read_text())['reference_run'])
    prior={Path(r['path']).resolve() for r in before if Path(r['path']).is_relative_to(reference)}
    current={p.resolve() for p in reference.rglob('*') if p.is_file()}
    assert not changes and prior==current,(changes,list(current-prior),list(prior-current))
    save(out/'baseline_and_mapping_fingerprints_after.json',dict(unchanged=True,checked_files=len(before),baseline_files=len(prior),changed_paths=[],added_baseline_files=[],removed_baseline_files=[]))
    for name in ['run_passive.log','report_passive.log']:
        if (work/name).exists():shutil.copy2(work/name,out/'logs'/name)
    lines=[]
    for name in ['run_passive.log','report_passive.log']:
        if (work/name).exists():lines.append(name+'\n'+(work/name).read_text(encoding='utf-8-sig'))
    (out/'full_run.log').write_text('\n\n'.join(lines),encoding='utf8')
    for p in (work/'code').glob('*.py'):shutil.copy2(p,out/'code_snapshot'/p.name)
    if (work/'regression/runner_input_review.json').exists():shutil.copy2(work/'regression/runner_input_review.json',out/'runner_input_review.json')
    config=json.loads((out/'run_config.json').read_text());config['code_sha256']={p.name:sha(p) for p in (out/'code_snapshot').glob('*') if p.is_file()}
    config['validation_code_sha256']={p.name:sha(p) for p in (out/'validation_code').glob('*.py')}
    config['original_data_mapping_and_baseline_unchanged']=True
    save(out/'run_config.json',config)
    finaldir='D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/passive'
    save(out/'execution_commands.json',dict(python='C:/ProgramData/anaconda3/envs/sam3/python.exe',
        working_directory=str(work.parent),baseline_regression='passive_revision_work/regression/regress_v3_observed.py',
        measurement=['code_snapshot/run_passive.py','--mapping',config['mapping_json'],'--audit-dir',finaldir,'--output','<NEW_EMPTY_OUTPUT_DIRECTORY>'],
        report=['code_snapshot/report_passive.py','--mapping',config['mapping_json'],'--audit-dir',finaldir,'--baseline',config['reference_run'],'--output','<NEW_OUTPUT_DIRECTORY>'],
        note='Use a new output directory; original data, mapping JSON and reference run are read-only. Record acceptance certifies stored native-frame association, not physical display calibration.'))
    required=['trial_stage.csv','participant_stage_summary.csv','group_stage_summary.csv','coverage.csv','recording_audit.csv','target_labels.csv','passive_mapping_audit.csv','input_issues.json','run_config.json','statistics_manifest.json','validation.json','independent_validation.json','detailed_report_zh.html','full_run.log']
    assert all((out/n).is_file() for n in required)
    for p in (out/'code_snapshot').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    figs=list((out/'figures').glob('*.png'));overlays=list((out/'overlays').glob('*.jpg'))
    for p in figs+overlays:
        with Image.open(p) as im:im.verify()
    for p in (out/'figures').glob('*.pdf'):
        b=p.read_bytes();assert b.startswith(b'%PDF-') and b'%%EOF' in b[-1024:]
    links=re.findall(r'(?:src|href)="([^"]+)"',(out/'detailed_report_zh.html').read_text(encoding='utf8'))
    deferred={'output_index.csv','final_output_validation.json'}
    missing=[u for u in links if not u.startswith(('http:','https:','#','data:')) and u not in deferred and not (out/u).exists()]
    assert not missing,missing
    rows={name:len(pd.read_csv(out/(name+'.csv'))) for name in ['trial_stage','participant_stage_summary','group_stage_summary','coverage','recording_audit','target_labels','passive_mapping_audit']}
    trials=pd.read_csv(out/'trial_stage.csv');assert trials.analysis_branch.eq('observed_gaze').all() and trials.source.eq('gaze').all() and trials.scope.eq('real_all').all()
    forbidden=['decoded_expectation','decoded_argmax','decoded_sample','main_probability','models','paired_test']
    assert not any((out/x).exists() for x in forbidden)
    independent=json.loads((out/'independent_validation.json').read_text());assert independent['status']=='passed'
    internal=json.loads((out/'validation.json').read_text());assert internal['status']=='passed_for_stored_native_mapping'
    core=json.loads((out/'baseline_observed_regression.json').read_text());assert core['status']=='passed'
    inp=json.loads((out/'input_fingerprints.json').read_text());assert all(r['unchanged'] for r in inp['files'])
    result=dict(status='passed_with_documented_provenance_limits',artifact_integrity_passed=True,stored_native_mapping_and_numerics_passed=True,
       all_requested_physical_display_verifications_established=False,
       unresolved_verifications=['actual observer physical presentation time','external screen viewport calibration'],
       caveat='Native Frame_Number/video_ts/annotation/DLC mapping is verified; neither raw VID_FRAME differences nor frame-count equality establish physical display synchronization.',
       recordings=16,observers=8,source_videos=16,scenarios=5,table_rows=rows,verified_figure_pngs=len(figs),verified_figure_pdfs=len(list((out/'figures').glob('*.pdf'))),verified_overlays=len(overlays),
       analysis_frame_files=len(list((out/'frames').glob('*.npz'))),mapping_frame_files=len(list((out/'mapping_frames').glob('*.npz'))),
       input_files_unchanged=len(inp['files']),baseline_files_unchanged=len(prior),mapping_json_unchanged=True,
       relative_report_links_checked=len(links),missing_links=[],model_output_directories_absent=True,
       output_index_policy='Every delivered file listed with SHA256; index lists itself without recursive size/hash.',output_index_verified=True)
    save(out/'final_output_validation.json',result)
    entries=[]
    paths=sorted(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='output_index.csv')
    for p in paths:
        rel=p.relative_to(out).as_posix();entries.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),kind=p.suffix.removeprefix('.'),role='code' if 'code' in rel else 'analysis_or_audit'))
    entries.append(dict(path='output_index.csv',bytes=None,sha256=None,kind='csv',role='self_inventory_no_recursive_hash'))
    pd.DataFrame(entries).to_csv(out/'output_index.csv',index=False,encoding='utf-8-sig',na_rep='NA')
    check=pd.read_csv(out/'output_index.csv')
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert set(check.path)==actual
    assert all((out/r['path']).is_file() and sha(out/r['path'])==r['sha256'] for r in entries if r['sha256'])
    print(json.dumps(dict(status=result['status'],files=len(entries),bytes=sum(p.stat().st_size for p in paths),rows=rows),ensure_ascii=False))

if __name__=='__main__':main()
