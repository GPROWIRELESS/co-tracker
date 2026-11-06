import csv,json,hashlib,shutil,ast
from pathlib import Path
HERE=Path(__file__).parent
ROOT=Path(r'D:\GitHub_Project\co-tracker')
OVER=HERE/'overlay'
SRC=ROOT/'outputs/score_corr_max_dlc'
OUT=OVER/'outputs/score_corr_max_dlc'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
audit={r['content_id']:r for r in read(OUT/'corrections/matched_scores_audited.csv')}
for rel in ('matched_scores.csv','metrics/position/Score/matched_outcomes.csv','metrics/velocity/Score/matched_outcomes.csv'):
    original=read(SRC/rel)
    augmented=[dict(r,**{k:v for k,v in audit[r['content_id']].items() if k not in r}) for r in original]
    write(OUT/rel,augmented)
    assert all(all(a[k]==v for k,v in r.items()) for a,r in zip(read(OUT/rel),original))
write(OUT/'corrections/matched_scores_audited.csv',read(OUT/'matched_scores.csv'))
snap=OUT/'corrections/code_snapshot';snap.mkdir(exist_ok=True)
for p in (HERE/'build_corrections.py',HERE/'finalize_corrections.py',*list((OVER/'tools').glob('*.py'))):
    ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,snap/p.name)
for p in (HERE/'claims_data_audit.json',HERE/'statistics_audit.json'):
    if p.exists():shutil.copy2(p,OUT/'corrections'/p.name)
report=OUT/'report.md'
report.write_text(report.read_text(encoding='utf-8')+'\n## 2026-11-05 audit additions\n\n'
    'Including penalties, the comparison tables contain 1,377 rows, 999 finite-p tests and 51 BH families. '
    'See [point-by-point Chinese review](corrections/review_zh.md). '
    'Three family-significant rows concern Score, Time and TimeRaw in the same A/transfer-ring/velocity/within condition; TimeRaw mirrors Time. '
    'A supplementary global BH calculation yields q=0.15864 for all three. Excluding the Raw mirrors gives global q=0.12227 for Score and 0.15576 for Time. '
    'These post-hoc scopes do not replace the original family q values.\n\n'
    'The known-PenaltySum > -100 subset retains A=33 and C=18 recordings before stage filtering; '
    'it is not a validated critical-error-free subset. It yields 111 estimable tests and no new Task C family-significant findings. '
    'C duration discrepancies are flagged in 22/33 complete cases at a descriptive 20% tolerance (A: 0/33); '
    'missing timing fields remain unavailable. Source scores and identity matches were not changed.\n\n'
    'The revision diagnostics builder is archived in corrections/code_snapshot/build_corrections.py. '
    'Run it with --repo, --output (a separate review directory), and --patched-tools pointing to the current tools directory. '
    'The two original generators regenerate their respective legacy and comparison overviews; supplementary diagnostics require the archived builder.\n',encoding='utf-8')
# Update current-layout hashes for changed files; historical inference snapshots remain untouched.
layout=json.loads((SRC/'layout.json').read_text())
for rel in list(layout.get('original_sha256',{})):
    if (OUT/rel).is_file():layout['original_sha256'][rel]=digest(OUT/rel)
for signal in ('position','velocity'):
    rel=f'metrics/{signal}/Score/provenance.json'
    info=json.loads((SRC/rel).read_text())
    for meta in (info['tables']['matched_outcomes'],layout['normalized_tables'][f'{signal}/Score']['matched_outcomes']):
        source=meta['source']
        if (OUT/source).is_file():meta['source_sha256']=digest(OUT/source)
        meta['normalized_sha256']=digest(OUT/f'metrics/{signal}/Score/matched_outcomes.csv')
        meta['diagnostic_columns_added']=True
        meta['original_fields_unchanged']=True
    info['revision']='2026-11-05: diagnostic columns added, original fields unchanged'
    save(OUT/rel,info)
layout['revision']='2026-11-05: overview labels and Score diagnostics updated; original numerical inference unchanged'
save(OUT/'layout.json',layout)
lv=json.loads((SRC/'layout_validation.json').read_text())
lv['revision']={'date':'2026-11-05','original_inference_preserved':True,'original_matched_fields_preserved':True,'overview_labels_regenerated':True,'details':'corrections/validation.json'}
save(OUT/'layout_validation.json',lv)
validation=json.loads((OUT/'corrections/validation.json').read_text())
validation.update(original_matched_fields_preserved=True,overview_visual_check='position/Score inspected; remaining overviews use same shared layout',supplement_successful_tests=111)
save(OUT/'corrections/validation.json',validation)
protected={str(ROOT/'Index'/p):digest(ROOT/'Index'/p) for p in ('Task_A_Score.csv','Task_C_Score.csv','content_index.csv')}
manifest=[]
for p in OVER.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(OVER).as_posix();target=ROOT/rel
    manifest.append({'relative_path':rel,'sha256':digest(p),'previous_sha256':digest(target) if target.is_file() else None})
save(HERE/'install_manifest.json',{'repo':str(ROOT),'protected_sources':protected,'files':manifest})
print(json.dumps({'files_to_install':len(manifest),'original_fields_preserved':True,'protected_sources':len(protected)}))
