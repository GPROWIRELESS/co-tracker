"""Validate and publish the reviewed comparison to the requested directory."""
from pathlib import Path
import argparse,json,hashlib,shutil,re,sys,importlib.metadata,csv
from html.parser import HTMLParser
from urllib.parse import unquote

WORK=Path(__file__).resolve().parent
OUT=WORK/'output'
DEST=Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106\gt_passive16')
TARGET=DEST/'active_passive_comparison'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def jsave(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['src','href'] and v:self.links.append(v)

parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
assert OUT.is_dir()
for name in ['independent_validation.json','support_audit.json']:
 d=json.loads((OUT/name).read_text(encoding='utf-8'));assert d['passed'],name
assert (OUT/'report.html').is_file()
with open(OUT/'row_correspondence_audit.csv',encoding='utf-8-sig') as f:row_audit=list(csv.DictReader(f))
assert len(row_audit)==16 and all(r['row_counts_equal']=='True' and r['passive_rows']==r['active_rows'] for r in row_audit)
for r in row_audit:
 assert sha(Path(r['passive_npy_path']))==r['passive_sha256']
 assert sha(Path(r['active_npy_path']))==r['active_sha256']
code=OUT/'code_snapshot';code.mkdir(exist_ok=True)
for name in ['build_comparison.py','support_audit.py','build_report.py','validate_comparison.py','finalize_delivery.py']:
 shutil.copy2(WORK/name,code/name)
kernels=code/'frozen_kernels';kernels.mkdir(exist_ok=True)
for f in (DEST/'code_snapshot').glob('*.py'):shutil.copy2(f,kernels/f.name)
repro='''# Reproduction\n\nRun in an environment with Python, NumPy, pandas, SciPy and Matplotlib. Original source directories and input caches are recorded with SHA256 in comparison_manifest.json and support_audit.json. Use a NEW output directory.\n\n```powershell\npython code_snapshot/support_audit.py --output NEW_OUTPUT\npython code_snapshot/build_comparison.py --output NEW_OUTPUT\npython code_snapshot/validate_comparison.py --output NEW_OUTPUT\npython code_snapshot/build_report.py --output NEW_OUTPUT\n```\n\nThe supplement reuses frozen original per-record results; common-time sensitivity additionally needs the referenced original caches. These paths are explicit in support_audit.json. All frozen numerical Python modules used by the audit are included in code_snapshot/frozen_kernels. Existing source data and baseline result files are not regenerated.\n\nThe bootstrap archive stores scalar paired-metric difference draws and paired group-curve peak-lag draws; bootstrap_index.json maps arrays to conditions. Pointwise difference-curve ranges are exported in paired_group_curves.csv and reproducible from the documented deterministic seeds. These ranges are exploratory, not simultaneous or calibrated confidence bands.\n'''
(OUT/'REPRODUCE.md').write_text(repro,encoding='utf-8')
jsave(OUT/'runtime_packages.json',dict(python=sys.version,packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','matplotlib']}))
jsave(OUT/'execution_commands.json',dict(working_directory=str(WORK),commands=['python support_audit.py','python build_comparison.py','python validate_comparison.py --output output','python build_report.py','python finalize_delivery.py','python finalize_delivery.py --publish'],reproduction='See REPRODUCE.md; specify a new output directory.'))
h=Links();h.feed((OUT/'report.html').read_text(encoding='utf-8'))
local=[unquote(v.split('#')[0]) for v in h.links if not v.startswith(('http:','https:','data:','mailto:','#'))]
missing=[v for v in local if not (OUT/v).exists()];assert not missing,missing
png=list((OUT/'figures').glob('*.png'));pdf=list((OUT/'figures').glob('*.pdf'));assert len(png)>0 and {p.stem for p in png}=={p.stem for p in pdf}
validation=dict(passed=True,row_correspondence_verified_pairs=16,pairing_rule='source mapping + equal active/passive NPY rows, row i to row i; no timestamp criterion',numerical_validation='independent_validation.json',optional_coordinate_availability_validation='support_audit.json',local_links_checked=len(local),missing_links=missing,png_count=len(png),pdf_count=len(pdf),numerical_source_files_changed=False)
jsave(OUT/'validation.json',validation)
manifest=dict(files=[dict(path=f.relative_to(OUT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='file_manifest.json'],excludes_self=True)
manifest['files_listed']=len(manifest['files']);jsave(OUT/'file_manifest.json',manifest)
print(json.dumps(validation),flush=True)
if not args.publish:sys.exit(0)
assert TARGET.parent.resolve()==DEST.resolve()
if TARGET.exists():
 assert args.resume and not (TARGET/'delivery_validation.json').exists(),f'Refusing to overwrite completed comparison: {TARGET}'
 for name in ['report.html','README.md','file_manifest.json']:assert sha(DEST/name)==sha(TARGET/'provenance'/(name+'.before_comparison'))
# Original source numerical inputs must still match pre-comparison hashes.
cm=json.loads((OUT/'comparison_manifest.json').read_text(encoding='utf-8'))
for row in cm['inputs']:assert sha(Path(row['path']))==row['sha256'],row['path']
shutil.copytree(OUT,TARGET,dirs_exist_ok=args.resume)
backup=TARGET/'provenance';backup.mkdir(exist_ok=args.resume)
for name in ['report.html','README.md','file_manifest.json']:
 if not (backup/(name+'.before_comparison')).exists():shutil.copy2(DEST/name,backup/(name+'.before_comparison'))
report=DEST/'report.html';text=report.read_text(encoding='utf-8')
link='<section style="margin:24px;padding:18px;border:2px solid #196b76;background:#f1faf9"><h2>新增：Passive 与 Active 配对比较</h2><p>16 对记录按来源映射与 NPY 行数核对后逐行对应，含比较图表、验证及坐标可用性敏感性附录。<a href="active_passive_comparison/report.html">打开补充比较报告</a></p></section>'
match=re.search(r'<main\b[^>]*>',text,re.I) or re.search(r'<body\b[^>]*>',text,re.I)
assert match,'No main or body in existing report'
report.write_text(text[:match.end()]+link+text[match.end():],encoding='utf-8')
readme=DEST/'README.md';readme.write_text(readme.read_text(encoding='utf-8')+'\n\n## Passive 与 Active 配对比较补充\n\n已按来源视频匹配当前16条passive与gt84中的active记录。包含共同支持的相关曲线、配对指标、空间比较、共同有效时间点敏感性及独立验证。\n\n[打开补充比较报告](active_passive_comparison/report.html) · [配对清单](active_passive_comparison/paired_record_mapping.csv) · [验证记录](active_passive_comparison/validation.json)\n',encoding='utf-8')
for row in cm['inputs']:assert sha(Path(row['path']))==row['sha256'],row['path']
delivery=dict(passed=True,destination=str(TARGET),original_numeric_inputs_unchanged=True,original_entry_files_backed_up=True,parent_updates=['report.html','README.md','file_manifest.json'])
jsave(TARGET/'delivery_validation.json',delivery)
new_files=[dict(path=f.relative_to(TARGET).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(TARGET.rglob('*')) if f.is_file() and f.relative_to(TARGET).as_posix()!='file_manifest.json']
jsave(TARGET/'file_manifest.json',dict(files=new_files,files_listed=len(new_files),excludes_self=True))
pm=json.loads((DEST/'file_manifest.json').read_text(encoding='utf-8'))
entries={r['path']:r for r in pm['files']}
for f in [DEST/'report.html',DEST/'README.md']+[f for f in TARGET.rglob('*') if f.is_file()]:
 rel=f.relative_to(DEST).as_posix();entries[rel]=dict(path=rel,bytes=f.stat().st_size,sha256=sha(f))
pm['files']=[entries[k] for k in sorted(entries)];pm['files_listed']=len(entries);jsave(DEST/'file_manifest.json',pm)
# Verify every delivered file after writing and the parent navigation link.
for row in new_files:assert sha(TARGET/row['path'])==row['sha256']
assert 'active_passive_comparison/report.html' in report.read_text(encoding='utf-8')
print(json.dumps(delivery),flush=True)
