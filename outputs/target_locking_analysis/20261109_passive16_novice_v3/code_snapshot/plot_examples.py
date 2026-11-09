"""Deterministic paired source-frame overlays and native-time state traces."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

ACTIVE='#2563a6'
PASSIVE='#cf6239'
STATE_COLORS=['#c9ced0','#238975','#c6733c','#826cad']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'output')
    out=ap.parse_args().output
    folder=out/'figures/examples';folder.mkdir(parents=True,exist_ok=True)
    datafolder=out/'figure_data/examples';datafolder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in pd.read_csv(out/'pair_manifest.csv').itertuples():
        with np.load(out/'frames'/f'{r.pair_id}.npz',allow_pickle=False) as z:
            f={k:z[k] for k in z.files}
        ids=np.flatnonzero(f['valid_main'])
        if not len(ids):continue
        i=int(ids[len(ids)//2])
        # The source MP4 and DLC are both indexed by retained video position,
        # exactly as the original v3 point_overlay(), not by original frame ID.
        cap=cv2.VideoCapture(r.source_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ok,img=cap.read();cap.release()
        if not ok:raise RuntimeError(f'Cannot decode {r.pair_id} frame {i}')
        fig,ax=plt.subplots(figsize=(9.5,8))
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        for key,color in [('target_xy','#FFD84A'),('left_xy','#34DE8C'),('right_xy','#34DE8C')]:
            pt=f[key][i];ax.add_patch(Circle(pt,50,fill=False,color=color,lw=1.7))
            ax.scatter(*pt,c=color,marker='+',s=80,lw=1.3)
        ax.scatter(*f['active_xy'][i],s=130,c=ACTIVE,edgecolors='white',lw=1.2,marker='o',label='Active gaze',zorder=5)
        ax.scatter(*f['passive_xy'][i],s=150,c=PASSIVE,edgecolors='white',lw=1.2,marker='^',label='Passive gaze',zorder=6)
        handles,labels=ax.get_legend_handles_labels()
        handles += [Line2D([0],[0],color='#E0B923',label='Target (50 px)'),Line2D([0],[0],color='#25AE68',label='DLC wrist regions (50 px)')]
        ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=2,frameon=False)
        ax.set_title(f'Task {r.task} | {r.content_id} | observer {r.observer_id}\n{f["stage"][i]} | original frame {f["frame_number"][i]} | t={f["time_s"][i]:.2f} s',fontsize=11)
        ax.set_axis_off();fig.tight_layout()
        opath=folder/f'{r.pair_id}_overlay.png';fig.savefig(opath,dpi=160,bbox_inches='tight');plt.close(fig)
        t=f['time_s'];fig,axes=plt.subplots(2,1,figsize=(13,5.1),sharex=True)
        starts=np.r_[0,np.flatnonzero(f['occurrence'][1:]!=f['occurrence'][:-1])+1]
        stops=np.r_[starts[1:],len(t)]
        for ax,domain,color in zip(axes,['active','passive'],[ACTIVE,PASSIVE]):
            state=f[f'{domain}_state_main']
            for k in range(4):
                mask=state==k;ax.scatter(t[mask],state[mask],s=2,c=STATE_COLORS[k],rasterized=True)
            invalid=state<0;ax.scatter(t[invalid],np.full(invalid.sum(),-1),s=.6,color='#d9d9d9')
            for a,b in zip(starts,stops):
                ax.axvline(t[a],color='#b6bbc0',lw=.4)
            ax.set_yticks([-1,0,1,2,3],['Excluded','Other','Target-only','Wrist-only','Overlap'])
            ax.set_ylabel(domain.title(),color=color,fontweight='bold');ax.set_ylim(-1.4,3.5)
            ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        axes[0].set_title(f'Task {r.task} | {r.content_id} | observer {r.observer_id}\nPaired common-frame states; native source-video time; main radius 50/50 px',fontsize=11)
        axes[-1].set_xlabel('Source video time (s); vertical lines mark stage-occurrence boundaries')
        fig.tight_layout();tpath=folder/f'{r.pair_id}_timeline.png';fig.savefig(tpath,dpi=160,bbox_inches='tight');plt.close(fig)
        sample=dict(pair_id=r.pair_id,content_id=r.content_id,observer_id=r.observer_id,task=r.task,
            stage=str(f['stage'][i]),retained_video_index=i,original_frame_number=int(f['frame_number'][i]),
            time_s=float(t[i]),selection='median-index of main common-valid frames (no outcome-based selection)',
            active_x=float(f['active_xy'][i,0]),active_y=float(f['active_xy'][i,1]),
            passive_x=float(f['passive_xy'][i,0]),passive_y=float(f['passive_xy'][i,1]),
            overlay=opath.relative_to(out).as_posix(),timeline=tpath.relative_to(out).as_posix())
        rows.append(sample)
        pd.DataFrame(dict(time_s=t,stage=f['stage'],active_state=f['active_state_main'],
            passive_state=f['passive_state_main'],common_valid=f['valid_main'])).to_csv(datafolder/f'{r.pair_id}_timeline.csv.gz',index=False,compression='gzip')
    pd.DataFrame(rows).to_csv(out/'example_index.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(pairs=len(rows),figures=2*len(rows))))


if __name__=='__main__':main()
