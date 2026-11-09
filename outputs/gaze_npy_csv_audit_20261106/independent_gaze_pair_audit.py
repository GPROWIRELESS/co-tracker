"""Independent read-only same-stem CSV/NPY comparison, Python csv + Decimal."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import csv,json,collections
import numpy as np
ROOT=Path(r'D:\Dataset\dvss')
OUT=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\gaze_pair_audit_work')
CRITICAL=['Frame_Number','Timestamp(ns)','gaze_x','gaze_y','valid']
results=[]
for p in sorted(ROOT.glob('*/User*/npy/*.npy')):
    q=p.with_suffix('.csv');r={'domain':p.parents[2].name,'user':p.parents[1].name,'stem':p.stem,'npy':str(p),'csv':str(q)}
    try:
        a=np.load(p,allow_pickle=False)
        with q.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); columns=reader.fieldnames; rows=list(reader)
        r.update(npy_n=len(a),csv_n=len(rows),npy_fields=list(a.dtype.names or []),csv_fields=columns)
        r['duplicate_csv_column_names']=[k for k,v in collections.Counter(columns).items() if v>1]
        r['only_npy_fields']=sorted(set(a.dtype.names or [])-set(columns));r['only_csv_fields']=sorted(set(columns)-set(a.dtype.names or []))
        if len(a)!=len(rows):r['error']='row-count mismatch';results.append(r);continue
        r['fields']={}
        for field in a.dtype.names or []:
            if field not in columns:continue
            values=[row[field] for row in rows];dtype=a.dtype[field];orig=a[field];bad=[];parsebad=[];maxerr=0.
            if dtype.kind in 'iu':
                # Decimal avoids both timestamp double-rounding and silent byte overflow.
                for i,s in enumerate(values):
                    try:
                        dec=Decimal(s.strip())
                        if not dec.is_finite() or dec!=dec.to_integral_value():raise ValueError('noninteger')
                        if int(dec)!=int(orig[i]):bad.append(i)
                    except (InvalidOperation,ValueError):parsebad.append(i)
            elif dtype.kind=='f':
                normalized=[('nan' if s.strip().lower() in ('','nan','na','null','none') else s) for s in values]
                try:
                    cast=np.array(normalized,dtype=dtype)
                    match=(cast==orig)|(np.isnan(cast)&np.isnan(orig));bad=np.flatnonzero(~match).tolist()
                    finite=np.isfinite(orig)&np.isfinite(cast)
                    maxerr=float(np.max(np.abs(orig[finite].astype(np.float64)-cast[finite].astype(np.float64)))) if finite.any() else 0.
                except (ValueError,OverflowError):
                    for i,s in enumerate(normalized):
                        try:
                            v=np.asarray(s,dtype=dtype).item();o=orig[i]
                            if not (v==o or (np.isnan(v) and np.isnan(o))):bad.append(i)
                        except (ValueError,OverflowError):parsebad.append(i)
            elif dtype.kind in 'US':
                # Compare untruncated strings: casting CSV to NPY U-width could hide loss.
                bad=[i for i,s in enumerate(values) if str(orig[i])!=s]
            else:
                r['fields'][field]={'unsupported_dtype':str(dtype)};continue
            sample_indices=(bad+parsebad)[:3]
            r['fields'][field]={'dtype':str(dtype),'mismatches':len(bad),'parse_errors':len(parsebad),
                'max_abs_dtype_cast_difference':maxerr,'examples':[{'row0':i,'npy':str(orig[i]),'csv':values[i]} for i in sample_indices]}
        r['critical_match']=all(k in r['fields'] and r['fields'][k].get('mismatches',1)==0 and r['fields'][k].get('parse_errors',1)==0 for k in CRITICAL)
        r['shared_fields_match']=all(v.get('mismatches',1)==0 and v.get('parse_errors',1)==0 for v in r['fields'].values())
    except Exception as e:r['error']=type(e).__name__+': '+str(e)
    results.append(r)
    if len(results)%25==0:print('Checked',len(results),flush=True)
summary={'pairs':len(results),'rows':sum(r.get('npy_n',0) for r in results),
         'domain_counts':dict(collections.Counter(r['domain'] for r in results)),
         'critical_match_pairs':sum(r.get('critical_match',False) for r in results),
         'shared_field_match_pairs':sum(r.get('shared_fields_match',False) for r in results),
         'schema_equal_pairs':sum(not r.get('only_npy_fields') and not r.get('only_csv_fields') for r in results),
         'whole_schema_and_values_equal_pairs':sum(r.get('shared_fields_match',False) and not r.get('only_npy_fields') and not r.get('only_csv_fields') for r in results),
         'errors':[{'stem':r['stem'],'error':r['error']} for r in results if 'error' in r],
         'source_files_modified':False,'timestamp_comparison':'CSV Decimal integer versus NPY integer exactly',
         'float_comparison':'CSV numeric cast to NPY field dtype, NaN equal; no absolute tolerance',
         'string_comparison':'CSV string exactly vs untruncated NPY string'}
(OUT/'independent_gaze_pair_results.json').write_text(json.dumps({'summary':summary,'records':results},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
