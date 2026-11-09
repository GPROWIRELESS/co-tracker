import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path('D:\\GitHub_Project\\co-tracker\\outputs\\novice_transition_A_seed42_20261105_analysis_20261106')
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
df=pd.read_csv(ROOT/'gaze_psm_lead/group_curves.csv')
names={'gt':'GT','current_transition':'Current','future500_transition':'Future 500','future1000_transition':'Future 1000','future1500_transition':'Future 1500'}
colors=['#152c3b','#355c93','#0b9a9c','#cf8a25','#b74972']
df=df[(df.variant=='anchor__primary')&(df['mode']=='psm_stage')&(df.stage=='whole')]
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey='row',layout='constrained')
for row,sig in enumerate(['position','velocity']):
    for col,side in enumerate(['l','r']):
        ax=axes[row,col]
        ax.axvspan(-.1,.1,color='#dbe4e5',alpha=.5,label='_nolegend_')
        for (source,label),color in zip(names.items(),colors):
            d=df[(df.signal==sig)&(df.side==side)&(df.source==source)].sort_values('lag_s')
            ax.plot(d.lag_s,d.r,color=color,lw=2.3 if source=='gt' else 1.6,label=label)
        ax.axvline(0,color='#89969c',lw=.8,ls='--');ax.axhline(0,color='#9ba6ab',lw=.6)
        ax.set_title(('位置' if sig=='position' else '速度')+' · '+('左' if side=='l' else '右')+' PSM')
        ax.set_ylabel('相关系数 r');ax.set_xlabel('lag（秒）；正值 = gaze领先PSM')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
axes[0,0].legend(loc='best',fontsize=9,frameon=True)
fig.suptitle('Task A｜同一预测锚点支持下的 gaze–PSM 互相关\n四阶段合并，参与者等权；灰带为 ±100 ms 同步范围',fontsize=15)
fig.savefig(ROOT/'figures/lead_whole_curves_zh.png',dpi=180)
fig.savefig(ROOT/'figures/lead_whole_curves_zh.pdf')
