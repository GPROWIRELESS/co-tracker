"""Read-only comparison of every current same-stem gaze NPY/CSV pair."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import argparse,csv,json,hashlib,html,shutil,time
from collections import Counter,defaultdict
import numpy as np
CORE=('Frame_Number','Timestamp(ns)','gaze_x','gaze_y','valid')
MISSING={'','nan','NaN','NAN'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return str(x)
    return x
def dump(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')
def save(p,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def integer(s):
    d=Decimal(s.strip())
    if not d.is_finite() or d != d.to_integral_value():raise ValueError('not an exact integer')
    return int(d)
def compare_col(arr,texts,name,ident,frames):
    n=len(arr);kind=arr.dtype.kind;errors=[];numeric_diff=[]
    same=np.zeros(n,bool);parse_fail=[]
    for i,(a,s) in enumerate(zip(arr,texts)):
        try:
            if kind in 'iu':
                v=integer(s);same[i]=int(a)==v;diff=abs(int(a)-v)
                if diff:numeric_diff.append(diff)
            elif kind=='f':
                v=float('nan') if s.strip() in MISSING else float(s)
                cast=np.asarray(v,dtype=arr.dtype).item()
                same[i]=(np.isnan(a) and np.isnan(cast)) or a==cast
                if np.isfinite(a) and np.isfinite(v):numeric_diff.append(abs(float(a)-v))
            elif kind in 'US':
                v=s;s0=a.decode('utf-8') if kind=='S' else str(a);same[i]=s0==s
            elif kind=='b':
                if s.strip().lower() in ('true','false'):v=s.strip().lower()=='true'
                else:v=bool(integer(s))
                same[i]=bool(a)==v
            else:raise ValueError('unsupported dtype '+str(arr.dtype))
            if not same[i] and len(errors)<5:errors.append(dict(row_index=i,csv_line=i+2,frame_number=int(frames[i]) if frames is not None else None,npy=clean(a),csv=s))
        except (ValueError,InvalidOperation,OverflowError) as exc:
            parse_fail.append(i)
            if len(errors)<5:errors.append(dict(row_index=i,csv_line=i+2,npy=clean(a),csv=s,error=str(exc)))
    return dict(**ident,field=name,dtype=str(arr.dtype),compared_rows=n,mismatch_n=int((~same).sum()),
                parse_error_n=len(parse_fail),equal=bool(same.all()),max_raw_numeric_difference=max(numeric_diff,default=0),
                comparison='CSV cast to NPY floating dtype, NaNs equivalent' if kind=='f' else 'exact Decimal integer' if kind in 'iu' else 'exact string/boolean',examples=errors)
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--data-root',type=Path,default=Path(r'D:\Dataset\dvss'))
    pa.add_argument('--output',type=Path,default=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\outputs\gaze_npy_csv_audit_20261106'))
    args=pa.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.time()
    inventory=[];files=[];field_rows=[];manifest=[];missing=[];ignored=[];load_errors=[];users=[]
    for domain in ['Active_Data','Passive_Data']:
        root=args.data_root/domain
        for user in sorted(root.glob('User*'),key=lambda p:int(p.name[4:])):
            folder=user/'npy'
            if not folder.is_dir():users.append(dict(domain=domain,user=user.name,npy_folder_exists=False));continue
            npys={p.stem:p for p in folder.glob('*.npy')};csvs={p.stem:p for p in folder.glob('*_gaze.csv')}
            users.append(dict(domain=domain,user=user.name,npy_folder_exists=True,npy_files=len(npys),current_csv_files=len(csvs)))
            for p in folder.iterdir():
                if p.is_file() and p not in set(npys.values())|set(csvs.values()):ignored.append(dict(domain=domain,user=user.name,path=str(p),reason='not a current same-stem gaze NPY/CSV; backup or other file'))
            for stem in sorted(set(npys)|set(csvs)):
                npy=npys.get(stem);cp=csvs.get(stem);ident=dict(domain=domain,user=user.name,record=stem)
                if npy is None or cp is None:missing.append(dict(**ident,npy=str(npy) if npy else None,csv=str(cp) if cp else None));continue
                before={str(p):(p.stat().st_size,p.stat().st_mtime_ns,sha(p)) for p in [npy,cp]}
                try:
                    a=np.load(npy,allow_pickle=False)
                    if a.ndim!=1 or a.dtype.names is None:raise ValueError('Expected structured 1D NPY')
                    with cp.open(encoding='utf-8-sig',newline='') as f:
                        reader=csv.reader(f);header=next(reader);rows=list(reader)
                    if len(set(header))!=len(header):raise ValueError('Duplicate CSV columns')
                    malformed=[i+2 for i,r in enumerate(rows) if len(r)!=len(header)]
                    if malformed:raise ValueError('CSV row widths mismatch: '+str(malformed[:5]))
                    common=[c for c in a.dtype.names if c in header];only_npy=[c for c in a.dtype.names if c not in header];only_csv=[c for c in header if c not in a.dtype.names]
                    n=min(len(a),len(rows));indices={c:header.index(c) for c in common};local=[]
                    frames=a['Frame_Number'][:n] if 'Frame_Number' in a.dtype.names else None
                    for c in common:local.append(compare_col(a[c][:n],[r[indices[c]] for r in rows[:n]],c,ident,frames))
                    field_rows.extend(local);byfield={r['field']:r for r in local}
                    core_present=all(k in byfield for k in CORE)
                    counts_equal=len(a)==len(rows)
                    core_equal=counts_equal and core_present and all(byfield[k]['equal'] for k in CORE)
                    common_equal=counts_equal and all(r['equal'] for r in local)
                    row_order_equal=counts_equal and 'Frame_Number' in byfield and byfield['Frame_Number']['equal']
                    aligned_core_equal=None
                    if counts_equal and core_present and not row_order_equal:
                        nf=a['Frame_Number'].tolist();cf=[integer(r[indices['Frame_Number']]) for r in rows]
                        if len(set(nf))==len(nf) and len(set(cf))==len(cf) and set(nf)==set(cf):
                            mapping={f:i for i,f in enumerate(cf)};rr=[rows[mapping[int(f)]] for f in nf]
                            aligned_core_equal=all(compare_col(a[c],[r[indices[c]] for r in rr],c,ident,a['Frame_Number'])['equal'] for c in CORE)
                    timestamps=a['Timestamp(ns)'] if 'Timestamp(ns)' in a.dtype.names else np.array([],dtype=np.int64)
                    record=dict(**ident,npy_path=str(npy),csv_path=str(cp),npy_rows=len(a),csv_rows=len(rows),row_counts_equal=counts_equal,
                                core_fields_present=core_present,core_gaze_equal=core_equal,frame_order_equal=row_order_equal,
                                core_equal_after_frame_alignment=aligned_core_equal,common_fields_equal=common_equal,
                                schema_equal=set(a.dtype.names)==set(header),column_order_equal=list(a.dtype.names)==header,
                                all_fields_equal=common_equal and set(a.dtype.names)==set(header),npy_only_fields=only_npy,csv_only_fields=only_csv,
                                unequal_common_fields=[r['field'] for r in local if not r['equal']],
                                nonincreasing_timestamp_intervals=int((np.diff(timestamps)<=0).sum()),
                                duplicate_frame_ids=int(len(frames)-len(np.unique(frames))) if frames is not None else None)
                    files.append(record)
                except Exception as exc:load_errors.append(dict(**ident,npy_path=str(npy),csv_path=str(cp),error=repr(exc)))
                for p in [npy,cp]:
                    size,mtime,h=before[str(p)];unchanged=(p.stat().st_size,p.stat().st_mtime_ns)==(size,mtime) and sha(p)==h
                    manifest.append(dict(path=str(p),bytes=size,sha256=h,unchanged_during_audit=unchanged))
                if len(files)%20==0:print(f'Checked {len(files)} pairs',flush=True)
    save(out/'pair_summary.csv',files);save(out/'field_comparison.csv',field_rows);dump(out/'field_comparison.json',field_rows)
    dump(out/'source_manifest.json',manifest);dump(out/'missing_pairs.json',missing);dump(out/'ignored_files.json',ignored);dump(out/'load_errors.json',load_errors)
    peruser=[]
    for u in users:
        rr=[r for r in files if r['domain']==u['domain'] and r['user']==u['user']]
        peruser.append(dict(**u,pairs_checked=len(rr),npy_rows=sum(r['npy_rows'] for r in rr),
                            core_equal_pairs=sum(r['core_gaze_equal'] for r in rr),common_equal_pairs=sum(r['common_fields_equal'] for r in rr),
                            schema_equal_pairs=sum(r['schema_equal'] for r in rr),whole_equal_pairs=sum(r['all_fields_equal'] for r in rr),
                            files_with_nonincreasing_timestamps=sum(r['nonincreasing_timestamp_intervals']>0 for r in rr)))
    save(out/'per_user_summary.csv',peruser)
    summary=dict(data_root=str(args.data_root),pairs_checked=len(files),rows_checked=sum(r['npy_rows'] for r in files),
                 domains={d:dict(pairs=sum(r['domain']==d for r in files),users=sum(u['domain']==d for u in users),
                                  core_equal=sum(r['core_gaze_equal'] for r in files if r['domain']==d),
                                  common_equal=sum(r['common_fields_equal'] for r in files if r['domain']==d),
                                  schema_equal=sum(r['schema_equal'] for r in files if r['domain']==d)) for d in ['Active_Data','Passive_Data']},
                 core_equal_pairs=sum(r['core_gaze_equal'] for r in files),common_equal_pairs=sum(r['common_fields_equal'] for r in files),
                 schema_equal_pairs=sum(r['schema_equal'] for r in files),whole_equal_pairs=sum(r['all_fields_equal'] for r in files),
                 unequal_pairs=[r for r in files if not r['core_gaze_equal'] or not r['common_fields_equal']],
                 missing_pairs=len(missing),load_errors=load_errors,ignored_file_count=len(ignored),
                 source_files_unchanged=all(r['unchanged_during_audit'] for r in manifest),
                 csv_only_field_counts=dict(Counter(c for r in files for c in r['csv_only_fields'])),
                 npy_only_field_counts=dict(Counter(c for r in files for c in r['npy_only_fields'])),
                 nonincreasing_timestamp_files=[dict(domain=r['domain'],record=r['record'],count=r['nonincreasing_timestamp_intervals']) for r in files if r['nonincreasing_timestamp_intervals']],
                 rules=dict(integer='exact Decimal parsing, including scientific timestamp text; no float conversion',
                            floating='round-trip to stored NPY float dtype; missing blanks/NaN equal; raw decimal differences separately recorded',
                            text='exact string comparison; no stripping or aliasing of stage labels',
                            pairing='current same-stem .npy/.csv only; backups excluded',core_fields=list(CORE)),elapsed_seconds=time.time()-start)
    dump(out/'summary.json',summary)
    lines=['NPY / CSV gaze一致性核验（2026-11-06）',f'核验根目录：{args.data_root}',
           f'同名文件对：{len(files)}；NPY总行数：{summary["rows_checked"]:,}',
           f'核心gaze字段一致：{summary["core_equal_pairs"]}/{len(files)}',
           f'全部共有字段一致：{summary["common_equal_pairs"]}/{len(files)}',
           f'字段集合一致：{summary["schema_equal_pairs"]}/{len(files)}',
           f'完整字段和值一致：{summary["whole_equal_pairs"]}/{len(files)}',
           f'缺失配对：{len(missing)}；读取错误：{len(load_errors)}；原始文件未变化：{summary["source_files_unchanged"]}',
           '','核心字段：Frame_Number、Timestamp(ns)、gaze_x、gaze_y、valid。',
           '整数/时间戳按Decimal精确比较；浮点按NPY存储精度比较，避免将float32文本序列化误差当作数据不一致。',
           'NPY与CSV一致不等于时间戳本身有效；非递增时间戳另行列出，不在本次修改。','',
           'CSV独有字段文件数：'+str(summary['csv_only_field_counts']),'NPY独有字段文件数：'+str(summary['npy_only_field_counts']),
           '','逐用户结果：']
    for r in peruser:lines.append(f'{r["domain"]}/{r["user"]}: {r["pairs_checked"]}对，核心一致{r["core_equal_pairs"]}，共有字段一致{r["common_equal_pairs"]}，字段集合一致{r["schema_equal_pairs"]}')
    lines+=['','非一致文件详情：',json.dumps(clean(summary['unequal_pairs']),ensure_ascii=False,indent=2),'','时间戳非递增文件：',json.dumps(summary['nonincreasing_timestamp_files'],ensure_ascii=False,indent=2)]
    (out/'report_zh.txt').write_text('\n'.join(lines),encoding='utf-8')
    body='<br>'.join(html.escape(x) for x in lines)
    links=' '.join(f'<a href="{f}">{f}</a>' for f in ['pair_summary.csv','per_user_summary.csv','field_comparison.csv','summary.json','source_manifest.json'])
    (out/'report_zh.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NPY/CSV gaze一致性核验</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#213747}a{color:#087b8c;margin-right:15px}pre{white-space:pre-wrap}</style><h1>NPY / CSV gaze一致性核验</h1><p>'+links+'</p><pre>'+body+'</pre></html>',encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps(clean({k:v for k,v in summary.items() if k not in ['unequal_pairs','nonincreasing_timestamp_files','rules','load_errors']}),ensure_ascii=True,indent=2),flush=True)
if __name__=='__main__':main()
