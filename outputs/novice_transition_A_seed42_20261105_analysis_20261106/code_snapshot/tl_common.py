"""Shared reporting and point metrics; probability logic lives in the main entry."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tl_geometry as geo
STAGES={"A":("reach ring","grab ring","transfer ring","place ring"),"C":("reach string","wrap around psm","pull through and cinch","threading string through tower")}
def dump_json(path, obj):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(str(type(value)))
    path.write_text(json.dumps(obj, default=convert, ensure_ascii=False, indent=2),
                    encoding='utf-8')

def save_rows(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig', na_rep='NA')

def in_image(xy, width, height):
    return np.isfinite(xy).all(axis=1) & (xy>=0).all(axis=1) & (xy<[width,height]).all(axis=1)

def stage_masks(d):
    yield 'whole', d['included']
    for name in STAGES[d['task']]:
        yield name, d['stage']==name

def ratio(num,den):
    return num/den if den>0 else np.nan

def summarize(labels, ok, area, d):
    # Code -1 is never part of the occupancy denominator.
    idx=ok & (labels>=0)
    w=d['dt'][idx]
    seconds=w.sum()
    h=labels[idx]
    share=(np.bincount(h,weights=w,minlength=4)/seconds if seconds>0 else np.full(4,np.nan))
    baseline=((area[idx]*w[:,None]).sum(axis=0)/seconds if seconds>0 else np.full(4,np.nan))
    n,t,p,o=share
    an,at,ap,ao=baseline
    c=ratio(t,t+p)
    cu=ratio(at,at+ap)
    out=dict(valid_n=int(idx.sum()),valid_s=seconds,L_N=n,L_T=t,L_P=p,L_O=o,
             L_target=t+o,L_gripper=p+o,D=t+p,C=c,B=t-p,area_N=an,area_T=at,
             area_P=ap,area_O=ao,C_U=cu,C_excess=c-cu,E_target=t+o-at-ao,
             E_gripper=p+o-ap-ao,B_excess=t-p-at+ap,
             C_reason='' if np.isfinite(c) else ('zero_exclusive_occupancy' if seconds else 'no_valid_frames'))
    out.update(geo.temporal(np.where(idx,labels,-1),d['dt'],d['occurrence'],d['breaks']))
    return out

