"""Render existing target-locking estimates only; no analysis is recomputed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

ROOT=Path('D:/GitHub_Project/co-tracker/outputs/novice_transition_A_seed42_20261105_analysis_20261106')
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
FONT=Path('C:/Windows/Fonts/msyh.ttc')
fontManager.addfont(str(FONT))
family=FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':family,'font.size':11,'axes.unicode_minus':False,
                    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

table=pd.read_csv(ROOT/'target_locking'/'group_metrics.csv')
select=table[(table.axis=='anchor')&(table.scenario=='main')&(table.stage=='whole')&(table.metric=='L_target')]
models=['current_transition','future500_transition','future1000_transition','future1500_transition']
branches=['probability_map','decoded_expectation','decoded_argmax']
legend=['模型热图：目标区域概率质量','模型期望点：目标区域命中率','模型 argmax 点：目标区域命中率']
colors=['#19769b','#c37e28','#825a9b']
markers=['o','s','D']
gt=select[(select.model==models[0])&(select.source=='GT')&(select.branch=='probability_map')].iloc[0]
fig=plt.figure(figsize=(12.2,7.4),facecolor='white')
ax=fig.add_axes([.075,.31,.89,.51])
ax.axhspan(100*gt.ci_low,100*gt.ci_high,color='#69727d',alpha=.10,zorder=0)
ax.axhline(100*gt.estimate,color='#37424c',ls=(0,(5,3)),lw=1.5,label=f'配对 GT 点命中率：{100*gt.estimate:.2f}%')
records=[]
for j,(branch,label,color,marker) in enumerate(zip(branches,legend,colors,markers)):
    rows=select[(select.source=='model')&(select.branch==branch)].set_index('model').loc[models]
    x=np.arange(4)+(j-1)*.20
    y=100*rows.estimate.to_numpy()
    low=100*rows.ci_low.to_numpy();high=100*rows.ci_high.to_numpy()
    ax.errorbar(x,y,yerr=[y-low,high-y],fmt=marker,markersize=7,capsize=5,
                color=color,lw=1.8,label=label,zorder=3)
    for i,model in enumerate(models):
        ax.annotate(f'{y[i]:.2f}%',xy=(x[i],high[i]),xytext=(0,7),textcoords='offset points',
                    ha='center',va='bottom',fontsize=10,color=color)
        records.append(dict(model=model,branch=branch,estimate_percent=y[i],
                            ci_low_percent=low[i],ci_high_percent=high[i]))

ax.set_xlim(-.55,3.55);ax.set_ylim(0,max(35,100*select.ci_high.max()+5))
ax.set_xticks(range(4),['Current\n预测当前注视','Future 500\n预测 500 ms 后注视','Future 1000\n预测 1000 ms 后注视','Future 1500\n预测 1500 ms 后注视'])
ax.tick_params(axis='x',pad=10,length=0)
ax.set_ylabel('目标区域占用（%）',labelpad=12)
ax.grid(axis='y',color='#e7ebef',lw=.8)
ax.set_axisbelow(True)
ax.text(3.54,100*gt.estimate+.55,f'GT {100*gt.estimate:.2f}%',ha='right',va='bottom',color='#37424c',fontsize=10)

fig.text(.075,.955,'模型“看向目标”多少，取决于如何表示输出',fontsize=20,weight='bold',ha='left')
fig.text(.075,.901,'同一热图的概率质量、期望点命中率与 argmax 点命中率，统计含义不同',fontsize=12,color='#4d5964')
handles,labels=ax.get_legend_handles_labels()
# Put the three model representations together, then the paired observation.
order=[1,2,3,0]
fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower left',
           bbox_to_anchor=(.066,.145),frameon=False,ncol=2,columnspacing=2.0,handlelength=2.4,
           fontsize=10.5,labelspacing=.9)
fig.text(.075,.105,'点与标注值：参与者等权均值；误差线及灰色 GT 区间：10,000 次参与者 bootstrap 的 95% 探索性范围。',
         fontsize=10,color='#4d5964')
fig.text(.075,.071,'5 段验证录像、4 名参与者；GT 仅按有限且归一化 [0,1] 筛选，不依据 valid。',fontsize=10,color='#4d5964')
fig.text(.075,.035,'argmax 点命中率接近 GT，不代表完整注视分布已与 GT 一致；该验证集参与了模型选择。',
         fontsize=10,color='#4d5964')
for ext in ('png','pdf'):
    fig.savefig(OUT/f'tl_decoding_occupancy_zh.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
plt.close(fig)
print(OUT/'tl_decoding_occupancy_zh.png')
