from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path(__file__).resolve().parents[1]
WORK = SOURCE / 'paper_summary_data'
FIG = SOURCE / 'paper_summary_figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.titlesize':11, 'axes.labelsize':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42,
    'savefig.dpi':300})
NAVY, ORANGE, TEAL = '#22577A', '#C66A25', '#087F8C'
STAGES = [('A','reach ring'), ('A','transfer ring'), ('A','place ring'),
          ('C','reach string'), ('C','wrap around psm')]
KEYS = [(t,s,side) for t,s in STAGES for side in ['l','r']]
LABELS = [f'{t} | {s} | {side.upper()}' for t,s,side in KEYS]

def save(fig, name):
    fig.savefig(FIG / (name+'.png'), facecolor='white')
    fig.savefig(FIG / (name+'.pdf'), facecolor='white')
    plt.close(fig)

def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x',color='#E4E8EC',lw=.6)
    ax.spines['left'].set_color('#A8AFB6')
    ax.spines['bottom'].set_color('#A8AFB6')

g=pd.read_csv(SOURCE/'gt84/group_summary.csv')
g=g[g.variant.eq('primary')]
def group(task,stage,side,signal='position',mode='PSM in stage'):
    q=g[g.task.eq(task)&g.stage.eq(stage)&g.side.eq(side)&g.signal.eq(signal)&g['mode'].eq(mode)]
    assert len(q)==1
    return q.iloc[0]

fig,ax=plt.subplots(figsize=(9,6.5))
fig.subplots_adjust(left=.31,right=.97,top=.85,bottom=.16)
fig.suptitle('Peak lag depends on task stage and pairing rule',x=.04,y=.965,ha='left',fontsize=15,fontweight='bold')
fig.text(.04,.917,'GT84 | Position | Five stages specified for correlation in the paper outline',fontsize=10,color='#475569')
gtrows=[]
for j,(mode,label,color,offset,marker) in enumerate([
    ('PSM in stage','PSM in stage',NAVY,-.15,'o'),
    ('Both in stage','Both in stage',ORANGE,.15,'s')]):
    for i,(task,stage,side) in enumerate(KEYS):
        r=group(task,stage,side,mode=mode)
        y=i+offset
        ax.plot([r.peak_lag_resample_low,r.peak_lag_resample_high],[y,y],color=color,lw=1.5,alpha=.8)
        ax.scatter(r.peak_lag_s,y,s=29,color=color,marker=marker,zorder=3,label=label if i==0 else None)
        gtrows.append(dict(task=task,stage=stage,side=side,mode=mode,
            peak_lag_s=r.peak_lag_s,peak_r=r.peak_r,r_zero=r.r_zero,
            range_low=r.peak_lag_resample_low,range_high=r.peak_lag_resample_high,
            n_users=int(r.n_stable_users),n_records=int(r.n_complete_records)))
ax.set_yticks(range(10),LABELS)
ax.set_ylim(9.6,-.6)
ax.axvline(0,color='#667085',lw=.8,ls='--')
ax.set_xlim(-2.1,2.1)
ax.set_xlabel('Group peak lag (s); positive = gaze occurs earlier')
style(ax)
ax.legend(loc='upper left',bbox_to_anchor=(-.01,1.10),ncol=2,frameon=False,fontsize=9)
fig.text(.04,.065,'Points: peaks of participant-equal group curves; support counts are in the accompanying CSV.',fontsize=9)
fig.text(.04,.035,'Lines: 2.5–97.5% participant resampling ranges; exploratory, not calibrated confidence intervals.',fontsize=8.6)
save(fig,'G1_stage_lags')

sp=pd.read_csv(SOURCE/'gt84/spatial_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(10,6.4),sharey=True,gridspec_kw={'width_ratios':[1.1,1]})
fig.subplots_adjust(left=.28,right=.96,wspace=.19,top=.81,bottom=.18)
fig.suptitle('Trajectory coupling and spatial proximity provide different evidence',x=.03,y=.965,ha='left',fontsize=13.5,fontweight='bold')
fig.text(.03,.92,'GT84 | PSM in stage analysis | NPY gaze; finite normalized coordinates',fontsize=10,color='#475569')
sp_rows=[]
for i,(task,stage,side) in enumerate(KEYS):
    rp=group(task,stage,side)
    rv=group(task,stage,side,signal='velocity')
    axes[0].scatter(rp.peak_r,i-.12,color=NAVY,s=30,label='Position' if i==0 else None)
    axes[0].scatter(rv.peak_r,i+.12,color=ORANGE,s=28,marker='s',label='Velocity' if i==0 else None)
    q=sp[sp.task.eq(task)&sp.stage.eq(stage)&sp.side.eq(side)&sp['mode'].eq('PSM in stage')&sp['at'].eq('zero')]
    assert len(q)==1
    r=q.iloc[0]
    axes[1].hlines(i,0,r.proximity_50px*100,color='#BAD7DC',lw=2)
    axes[1].scatter(r.proximity_50px*100,i,color=TEAL,s=31)
    axes[1].text(r.proximity_50px*100+.5,i,f'{r.proximity_50px*100:.1f}%',va='center',fontsize=9)
    sp_rows.append(dict(task=task,stage=stage,side=side,position_peak_r=rp.peak_r,
        velocity_peak_r=rv.peak_r,zero_proximity_50px=r.proximity_50px,
        zero_distance_median_px=r.distance_median_px))
axes[0].set_yticks(range(10),LABELS)
axes[0].set_ylim(9.6,-.6)
axes[0].set_xlim(min(0,min(min(q['position_peak_r'],q['velocity_peak_r']) for q in sp_rows))-.015,max(.1,max(max(q['position_peak_r'],q['velocity_peak_r']) for q in sp_rows))*1.15)
axes[0].set_title('(a) Peak trajectory correlation',loc='left')
axes[0].set_xlabel('Peak r (dimensionless)')
axes[0].legend(frameon=False,ncol=2,fontsize=9,loc='upper left',bbox_to_anchor=(0,1.13))
axes[1].set_title('(b) Spatial proximity at zero lag',loc='left')
axes[1].set_xlabel('Within 50 px of instrument (%)')
axes[1].set_xlim(0,max(10,max(q['zero_proximity_50px'] for q in sp_rows)*100)*1.28)
for ax in axes:style(ax)
fig.text(.03,.09,'Correlation describes centered trajectory covariation; proximity describes absolute spatial closeness.',fontsize=9)
fig.text(.03,.057,'Proximity: record-level proportions averaged within participants, then across participants.',fontsize=9)
fig.text(.03,.025,'The 50 px neighborhood is a descriptive threshold, not a validated fixation or skill label.',fontsize=9)
save(fig,'G2_coupling_vs_proximity')

m=pd.read_csv(WORK/'models_selected_summary.csv')
models=['all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot']
colors=['#22577A','#008571','#D08B22','#8061A8','#BF4D53']
for metric,filename,title in [
    ('curve_mae','M1_curve_fidelity','Correlation-curve fidelity by task and instrument side'),
    ('abs_lag_difference_s','M2_timing_fidelity','Peak-lag differences and identifiable-peak coverage')]:
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.1),sharex=True)
    fig.subplots_adjust(left=.19,right=.98,bottom=.18,top=.82,wspace=.65,hspace=.43)
    fig.suptitle(title,x=.025,y=.965,ha='left',fontsize=14,fontweight='bold')
    subtitle='Selected correlation stages | Participant-equal estimates | Same-record GT and model pairs'
    fig.text(.025,.922,subtitle,fontsize=10,color='#475569')
    max_upper=float(m[metric+'_resample_high'].max())
    for row,task in enumerate(['A','C']):
        for col,side in enumerate(['l','r']):
            ax=axes[row,col]
            q=m[m.task.eq(task)&m.side.eq(side)].set_index('model').loc[models]
            names=[]
            for i,model in enumerate(models):
                r=q.loc[model]
                y=i
                ax.plot([r[metric+'_resample_low'],r[metric+'_resample_high']],[y,y],color=colors[i],lw=1.5)
                ax.scatter(r['participant_equal_'+metric],y,color=colors[i],s=31,zorder=3)
                names.append(model if metric=='curve_mae' else f'{model}  [{int(r.n_both_localizable)}/{int(r.n_conditions)}]')
            ax.set_yticks(range(5),names,fontsize=9)
            ax.set_ylim(4.6,-.6)
            ax.set_xlim(0,max_upper*1.1)
            ax.set_title(f'({chr(97+row*2+col)}) Task {task} | Fixed {"left" if side=="l" else "right"}',loc='left',pad=9)
            style(ax)
            ax.tick_params(axis='x',labelbottom=True)
            ax.set_xlabel('Curve MAE (correlation units)' if metric=='curve_mae' else 'Mean absolute peak-lag difference (s)')
    fig.text(.025,.10,'Points: participant-equal means across selected stages. Lines: 2.5–97.5% ranges from 5,000 resamples.',fontsize=9)
    fig.text(.025,.067,'Resampling ranges are exploratory and have not been calibrated as confidence intervals.',fontsize=9)
    if metric=='abs_lag_difference_s':
        fig.text(.025,.032,'Brackets: identifiable peaks in BOTH GT and model / common complete conditions.',fontsize=9)
    else:
        fig.text(.025,.032,'A: reach / transfer / place ring (15 conditions per side). C: reach string / wrap around PSM (10 per side).',fontsize=9)
    save(fig,filename)

print(json.dumps({'figures':[p.name for p in FIG.iterdir()],'status':'generated'},indent=2))
