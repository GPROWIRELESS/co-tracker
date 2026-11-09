"""Matched source-video comparison; original result directories are read only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
import csv,json,hashlib,itertools,platform,shutil,argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'output')
OUT=parser.parse_args().output.resolve()
BASE=Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106')
P=BASE/'gt_passive16'; A=BASE/'gt84'
LINKS=Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json')
KEYS=['variant','task','side','signal','stage','mode']
STAGES={'A':['reach ring','grab ring','transfer ring','place ring','drop ring'],
        'C':['reach string','wrap around psm','threading string through tower','pull through and cinch']}
METRICS=['r_zero','peak_r','peak_lag_s']
SPATIAL=['distance_median_px','proximity_25px','proximity_50px','proximity_100px']
NBOOT=5000

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def jsave(path,x):path.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8')
def save(name,rows):
 df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def read(p):return pd.read_csv(p,keep_default_na=True)
def ck(r):return (r['content_id'],)+tuple(r[k] for k in KEYS)
def seed(key):return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4],'little')
def groups(x,field):
 labels=sorted(set(x[field]))
 return labels,[np.flatnonzero(x[field].to_numpy()==s) for s in labels]
def mean_matrix(mat,df,weighting):
 if weighting=='pair_equal':return mat
 field='observer_user' if weighting=='observer_equal' else 'source_user'
 return np.array([mat[ix].mean(axis=0) for ix in groups(df,field)[1]])
def summary_values(df,metric,weighting):
 # Per-record metrics are arithmetic means; full curves use Fisher z separately.
 p=mean_matrix(df['passive_'+metric].to_numpy(float),df,weighting).mean(axis=0)
 a=mean_matrix(df['active_'+metric].to_numpy(float),df,weighting).mean(axis=0)
 return float(p),float(a),float(p-a)
def ninfo(df):return dict(n_pairs=len(df),n_observers=df.observer_user.nunique(),n_operators=df.source_user.nunique())
def srow(df,k,metric,weighting,boot=False):
 info=ninfo(df); status='supported' if info['n_observers']>=3 else 'insufficient_observers'
 p,a,d=summary_values(df,metric,weighting)
 lo=hi=np.nan
 if boot and status=='supported':
  b=mean_matrix(df['delta_'+metric].to_numpy(float),df,'observer_equal')
  rng=np.random.default_rng(seed((tuple(k.values()),metric,'paired_metric_bootstrap')))
  draws=b[rng.integers(0,len(b),(NBOOT,len(b)))].mean(axis=1)
  lo,hi=np.percentile(draws,[2.5,97.5]);BOOT['metric_'+str(len(BOOT))]=draws
  BINDEX.append(dict(array_key='metric_'+str(len(BOOT)-1),**k,metric=metric))
 return dict(**k,metric=metric,weighting=weighting,passive_value=p,active_value=a,delta=d,low=lo,high=hi,**info,status=status,inference='exploratory; observer clusters only; crossed identity dependence not fully accounted')

OUT.mkdir(parents=True,exist_ok=True)
inv=read(P/'analysis_record_inventory.csv')
links=json.loads(LINKS.read_text(encoding='utf-8'))['clips']
links={k.split('/')[-1]:v for k,v in links.items()}
mapping=[]
row_audit=[]
for r in inv.to_dict('records'):
 link=links[r['recording_id']]
 assert link['meta']['content_id']==r['source_content_id']
 assert 'User'+str(link['meta']['observer'])==r['observer_user']
 assert Path(link['source_video_path']).stem==r['source_content_id']
 pnpy=Path(r['gaze_npy_path']);anpy=Path(r['stage_event_npy_path'])
 plen=len(np.load(pnpy,allow_pickle=False));alen=len(np.load(anpy,allow_pickle=False))
 assert plen==alen,('gaze NPY row mismatch',r['recording_id'],plen,alen)
 row_audit.append(dict(passive_id=r['recording_id'],active_id=r['source_content_id'],passive_npy_path=str(pnpy),active_npy_path=str(anpy),passive_rows=plen,active_rows=alen,row_counts_equal=plen==alen,passive_sha256=sha(pnpy),active_sha256=sha(anpy),pairing_rule='source mapping + equal gaze NPY rows; row i corresponds to row i; timestamps are not a pairing criterion'))
 mapping.append(dict(passive_id=r['recording_id'],active_id=r['source_content_id'],observer_user=r['observer_user'],source_user=r['source_user'],task=r['task'],mapping_verified=True,source_video_path=r['source_video_path']))
mapping=pd.DataFrame(mapping);save('paired_record_mapping.csv',mapping)
save('row_correspondence_audit.csv',row_audit)
assert len(mapping)==16 and mapping.passive_id.is_unique and mapping.active_id.is_unique
mp=mapping.set_index('passive_id').to_dict('index')
ps=read(P/'record_summary.csv');aa=read(A/'record_summary.csv'); aa=aa[aa.content_id.isin(mapping.active_id)]
save('active_matched_record_summary.csv',aa)
save('passive_matched_record_summary.csv',ps)
ad={ck(r):r for r in aa.to_dict('records')}
paired=[]
for r in ps.to_dict('records'):
 m=mp[r['content_id']]; x=ad.get((m['active_id'],)+tuple(r[k] for k in KEYS))
 z={k:r[k] for k in KEYS}; z.update(passive_id=r['content_id'],**{k:m[k] for k in ['active_id','observer_user','source_user']})
 z['included']=bool(x and r['full_lag_support'] and x['full_lag_support'])
 z['exclusion_reason']='' if z['included'] else ('missing_active_condition' if x is None else 'active_or_passive_incomplete_lag_support')
 for metric in METRICS:
  z['passive_'+metric]=r[metric];z['active_'+metric]=x[metric] if x else np.nan
  z['delta_'+metric]=z['passive_'+metric]-z['active_'+metric]
 for flag in ['full_lag_support','lag_identifiable','peak_at_boundary','peak_at_support_edge','reliability_reasons']:
  z['passive_'+flag]=r[flag];z['active_'+flag]=x[flag] if x else None
 paired.append(z)
paired=pd.DataFrame(paired); save('paired_metrics.csv',paired)
assert len(paired)==2420 and len(paired.query("variant=='primary' and included"))==436
print('Matched metrics ready',flush=True)

def load_curves(path,ids,export):
 data={}
 with open(path,encoding='utf-8') as f,open(OUT/export,'w',encoding='utf-8') as w:
  for line in f:
   r=json.loads(line)
   if r['content_id'] in ids:data[ck(r)]=r;w.write(line)
 return data
pc=load_curves(P/'record_curves.jsonl',set(mapping.passive_id),'passive_matched_record_curves.jsonl')
ac=load_curves(A/'record_curves.jsonl',set(mapping.active_id),'active_matched_record_curves.jsonl')
BOOT={};BINDEX=[];curverows=[];summaries=[];sens=[];peakrows=[];loorows=[]

def process(df,pc,ac,source_label='original_support'):
 for key,g in df[df.included].groupby(KEYS,sort=True):
  k=dict(zip(KEYS,key));g=g.reset_index(drop=True);info=ninfo(g);supported=info['n_observers']>=3
  mats=[[],[]];lag=None
  for r in g.to_dict('records'):
   p=pc[(r['passive_id'],)+key];a=ac[(r['active_id'],)+key]
   lp=np.array(p['lag_s']);la=np.array(a['lag_s']);assert np.array_equal(lp,la)
   if lag is not None:assert np.array_equal(lp,lag)
   lag=lp
   for dst,v in zip(mats,[p,a]):
    arr=np.array(v['r'],dtype=float);assert np.isfinite(arr).all();dst.append(np.arctanh(np.clip(arr,-1+1e-7,1-1e-7)))
  pz,az=np.array(mats)
  if supported:
   up=mean_matrix(pz,g,'observer_equal');ua=mean_matrix(az,g,'observer_equal')
   pp=np.tanh(up.mean(axis=0));ap=np.tanh(ua.mean(axis=0));dz=up.mean(axis=0)-ua.mean(axis=0)
   low=np.full(len(lag),np.nan);high=low.copy()
   if k['variant'] in ['primary','common_time_primary']:
    rng=np.random.default_rng(seed((key,'curves')));draw=rng.integers(0,len(up),(NBOOT,len(up)))
    bp=np.tanh(up[draw].mean(axis=1));ba=np.tanh(ua[draw].mean(axis=1))
    low,high=np.percentile(bp-ba,[2.5,97.5],axis=0)
    bkey='curve_'+str(len(BOOT));BOOT[bkey]=np.stack([lag[np.argmax(bp,axis=1)],lag[np.argmax(ba,axis=1)]],axis=1)
    BINDEX.append(dict(array_key=bkey,**k,metric='group_curve_peak_lags',columns=['passive','active']))
   for j,t in enumerate(lag):curverows.append(dict(**k,lag_s=t,passive_r=pp[j],active_r=ap[j],delta_r=pp[j]-ap[j],delta_z=dz[j],delta_r_low=low[j],delta_r_high=high[j],**info,support=source_label))
   pi,ai=np.argmax(pp),np.argmax(ap)
   peakrows.append(dict(**k,**info,passive_peak_r=pp[pi],active_peak_r=ap[ai],delta_peak_r=pp[pi]-ap[ai],passive_peak_lag_s=lag[pi],active_peak_lag_s=lag[ai],delta_group_peak_lag_s=lag[pi]-lag[ai],passive_boundary=pi in [0,len(lag)-1],active_boundary=ai in [0,len(lag)-1],estimand='difference between peaks of matched observer-weighted GROUP curves'))
  for metric in METRICS:
   for weighting in ['observer_equal','pair_equal','operator_equal']:
    row=srow(g,k,metric,weighting,boot=(weighting=='observer_equal' and k['variant'] in ['primary','common_time_primary']))
    row['support']=source_label;sens.append(row)
    if weighting=='observer_equal' and k['variant'] in ['primary','common_time_primary']:summaries.append(row)
   if k['variant']=='primary' and supported:
    for person in sorted(set(g.observer_user)|set(g.source_user)):
     rem=g[(g.observer_user!=person)&(g.source_user!=person)]
     if len(rem):
      p,a,d=summary_values(rem,metric,'observer_equal')
      loorows.append(dict(**k,metric=metric,excluded_identity=person,passive_value=p,active_value=a,delta=d,**ninfo(rem),status='supported' if rem.observer_user.nunique()>=3 else 'insufficient_observers'))

process(paired,pc,ac)
print('Curve aggregation and bootstrap ready',flush=True)
coverage=[]
for task,stages in STAGES.items():
 for stage,side,signal,mode in itertools.product(stages,['l','r'],['position','velocity'],['PSM in stage','Both in stage']):
  k=dict(variant='primary',task=task,side=side,signal=signal,stage=stage,mode=mode)
  g=paired
  for col,v in k.items():g=g[g[col]==v]
  valid=g[g.included];ni=ninfo(valid);status='supported' if ni['n_observers']>=3 else ('stage_unannotated' if not len(g) else 'insufficient_observers')
  coverage.append(dict(**k,n_pairs_total=len(g),n_pairs_complete=len(valid),n_observers=ni['n_observers'],n_operators=ni['n_operators'],status=status))
  if not len(valid):
   for metric in METRICS:summaries.append(dict(**k,metric=metric,weighting='observer_equal',passive_value=np.nan,active_value=np.nan,delta=np.nan,low=np.nan,high=np.nan,**ni,status=status,support='original_support'))
save('paired_condition_coverage.csv',coverage)

# Only original zero-lag spatial summaries; never reuse original group-peak rows.
pas=read(P/'spatial_records.csv');aas=read(A/'spatial_records.csv')
aas=aas[aas['at']=='zero'];asd={ck(r):r for r in aas.to_dict('records')}
included={(r['passive_id'],)+tuple(r[k] for k in KEYS) for r in paired[paired.included].to_dict('records')}
sp=[]
for r in pas[pas['at']=='zero'].to_dict('records'):
 if ck(r) not in included:continue
 m=mp[r['content_id']];x=asd.get((m['active_id'],)+tuple(r[k] for k in KEYS))
 if not x:continue
 z={k:r[k] for k in KEYS};z.update(passive_id=r['content_id'],**{k:m[k] for k in ['active_id','observer_user','source_user']},at='zero')
 for metric in SPATIAL:z['passive_'+metric]=r[metric];z['active_'+metric]=x[metric];z['delta_'+metric]=r[metric]-x[metric]
 sp.append(z)
sp=pd.DataFrame(sp);save('paired_spatial_metrics.csv',sp)
for key,g in sp.groupby(KEYS):
 k=dict(zip(KEYS,key))
 for metric in SPATIAL:
  for weighting in ['observer_equal','pair_equal','operator_equal']:
   row=srow(g,k,metric,weighting,boot=weighting=='observer_equal');row['support']='original_support';sens.append(row)
   if weighting=='observer_equal':summaries.append(row)

# Optional independently generated common-time sensitivity.
cf=OUT/'common_time_record_curves.jsonl';sf=OUT/'common_time_record_summary.csv'
if cf.exists() and sf.exists():
 cs=read(sf); print('Common-time schema',list(cs.columns),flush=True)
 records=[json.loads(line) for line in cf.read_text(encoding='utf-8').splitlines()]
 cpc={};cac={}
 for r in records:
  r['variant']='common_time_primary';r['content_id']=r['passive_id'] if r['domain']=='passive' else r['active_id']
  (cpc if r['domain']=='passive' else cac)[ck(r)]=r
 ca={(r['active_id'],)+tuple(r[k] for k in KEYS):r for r in cs[cs.domain=='active'].to_dict('records')}
 cr=[]
 for r in cs[cs.domain=='passive'].to_dict('records'):
  r['variant']='common_time_primary';key=tuple(r[k] for k in KEYS)
  x=ca[(r['active_id'],)+key];z={k:r[k] for k in KEYS+['passive_id','active_id','observer_user','source_user']}
  z['included']=bool(r['full_lag_support'] and x['full_lag_support'])
  for metric in METRICS:z['passive_'+metric]=r[metric];z['active_'+metric]=x[metric];z['delta_'+metric]=r[metric]-x[metric]
  cr.append(z)
 cr=pd.DataFrame(cr);save('common_time_paired_metrics.csv',cr);process(cr,cpc,cac,'common_postfilter_gaze_mask')

csf=OUT/'common_time_spatial_records.csv'
if csf.exists():
 csp=read(csf)
 csa={(r['active_id'],)+tuple(r[k] for k in KEYS):r for r in csp[csp.domain=='active'].to_dict('records')}
 spatial_pairs=[]
 for r in csp[csp.domain=='passive'].to_dict('records'):
  x=csa[(r['active_id'],)+tuple(r[k] for k in KEYS)]
  z={k:r[k] for k in KEYS+['passive_id','active_id','observer_user','source_user']};z['at']='zero'
  for metric in SPATIAL:z['passive_'+metric]=r[metric];z['active_'+metric]=x[metric];z['delta_'+metric]=r[metric]-x[metric]
  spatial_pairs.append(z)
 csp=pd.DataFrame(spatial_pairs);save('common_time_paired_spatial_metrics.csv',csp)
 for key,g in csp.groupby(KEYS):
  k=dict(zip(KEYS,key))
  for metric in SPATIAL:
   for weighting in ['observer_equal','pair_equal','operator_equal']:
    row=srow(g,k,metric,weighting,boot=weighting=='observer_equal');row['support']='common_postfilter_gaze_mask';sens.append(row)
    if weighting=='observer_equal':summaries.append(row)

save('paired_group_curves.csv',curverows);save('paired_group_summary.csv',summaries);save('sensitivity_summary.csv',sens)
save('group_peak_comparison.csv',peakrows);save('loo_sensitivity.csv',loorows)
np.savez_compressed(OUT/'bootstrap_distributions.npz',**BOOT);jsave(OUT/'bootstrap_index.json',BINDEX)
inputs=[LINKS]+[folder/name for folder in [P,A] for name in ['record_summary.csv','record_curves.jsonl','spatial_records.csv','run_manifest.json']]+[P/'analysis_record_inventory.csv']
jsave(OUT/'comparison_manifest.json',dict(version='paired-source-video-v1',source_passive=str(P),source_active=str(A),output_requested=str(P/'active_passive_comparison'),inputs=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in inputs],primary_hz=41,max_lag_s=2,bootstrap=NBOOT,seed='SHA256 deterministic condition-specific seeds',pairing='source video, NOT within-person active/passive treatment',difference='passive minus active',curve_aggregation='equal records within passive observer; equal observers; Fisher z; same weights for active',scalar_aggregation='arithmetic mean per-record metrics within passive observer then equal observers; distinct from peaks of group curves',bootstrap_interpretation='exploratory observer-cluster percentile ranges; no calibrated coverage or formal crossed-dependence inference',lag_inference='not calibrated physiological delay; inspect boundary and identifiability',sensitivity_weightings=['observer_equal','pair_equal','operator_equal'],leave_one_out='delete identity from both observer and source roles',common_time_available=cf.exists(),versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)))
manifest=json.loads((OUT/'comparison_manifest.json').read_text(encoding='utf-8'))
manifest.update(pairing_rule='User-specified: source mapping plus equal active/passive gaze NPY row counts; row i to row i. No timestamp matching criterion.',row_audit='row_correspondence_audit.csv',coordinate_availability_sensitivity='Optional appendix only; does not determine record correspondence. Existing 41 Hz source results reused without new temporal realignment.')
jsave(OUT/'comparison_manifest.json',manifest)
(OUT/'code_snapshot').mkdir(exist_ok=True);shutil.copy2(__file__,OUT/'code_snapshot'/Path(__file__).name)
print(json.dumps(dict(pairs=len(mapping),primary_conditions=len(paired[paired.variant=='primary']),primary_complete=len(paired.query("variant=='primary' and included")),supported_primary_groups=sum(r['status']=='supported' for r in coverage),group_curve_rows=len(curverows)),indent=2),flush=True)
