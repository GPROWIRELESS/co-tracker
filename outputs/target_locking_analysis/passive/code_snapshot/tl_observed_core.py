"""Target Locking v3 real-observation measurement entry points."""
import numpy as np
import tl_observed_geometry as geo

STAGES = {"A": ("reach ring", "grab ring", "transfer ring", "place ring"),
          "C": ("reach string", "wrap around psm", "pull through and cinch", "threading string through tower")}
SCENARIOS = {'main': (50,50,None,False), 'r25': (25,25,None,False),
             'r100': (100,100,None,False), 'gripper100': (50,100,None,False),
             'smooth': (50,50,None,True)}

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


def prepare_arrays(d):
    d['z'],d['count'],d['slotmask']=geo.target_center(d['slots'],d['width'],d['height'])
    d['included']=np.isin(d['stage'],STAGES[d['task']])
    d['gripper_ok']=in_image(d['left'],d['width'],d['height']) & in_image(d['right'],d['width'],d['height'])
    d['dt'],d['breaks']=geo.times(d['timestamp_ns'],d['frames'])
    d['breaks'] |= d['identity_change'] | ~d['included']
    d['breaks'][1:] |= (d['occurrence'][1:]!=d['occurrence'][:-1]) | ~d['included'][:-1]
    d['smooth_xy'],d['interpolated']=geo.smooth_gaze(d['timestamp_ns'],d['gaze'],d['occurrence'],d['breaks'])
    return d


def support(d,scenario):
    _,_,codes,smooth=SCENARIOS[scenario]
    xy=d['smooth_xy'] if smooth else d['gaze_unmasked']
    gaze_ok=np.isfinite(xy).all(axis=1)
    gaze_ok &= (xy >= 0).all(axis=1) & (xy <= [d['width'],d['height']]).all(axis=1)
    return xy,d['included'] & (d['count']>0) & d['gripper_ok'] & gaze_ok

