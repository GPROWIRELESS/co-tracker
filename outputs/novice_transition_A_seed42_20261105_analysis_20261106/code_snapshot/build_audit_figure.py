import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(r'D:\GitHub_Project\co-tracker\outputs\novice_transition_A_seed42_20261105_analysis_20261106')
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
meta=json.loads((ROOT/'input_audit/metadata.json').read_text(encoding='utf-8'))
fig,axs=plt.subplots(2,1,figsize=(11,7),gridspec_kw={'height_ratios':[1,1.4]},constrained_layout=True)
rows=[]
for i,r in enumerate(meta['records']):
    cid=r['content_id']; n=np.load(ROOT/'input_audit/native'/f'{cid}.npz')
    p=np.load(ROOT/'input_audit/predictions/current_transition'/f'{cid}.npz')
    t=n['time'];g=n['gaze_valid'];sel=np.isin(n['stage'],meta['stages']);a=p['native_index']
    axs[0].plot([t[0],t[-1]],[i,i],lw=10,color='#d9e3ec',solid_capstyle='butt')
    axs[0].scatter(t[a][g[a]&sel[a]],np.full(np.sum(g[a]&sel[a]),i),s=5,color='#147d92',zorder=3)
    delta=(n['training_time']-t)*1000
    rows.append(dict(content_id=cid,native_n=len(t),prediction_n=len(a),duration_s=float(t[-1]),
                     common_stage_gaze_anchor_n=int(np.sum(g[a]&sel[a])),clock_delta_median_ms=float(np.median(delta)),
                     clock_delta_p95_abs_ms=float(np.percentile(abs(delta),95)),clock_delta_max_abs_ms=float(max(abs(delta)))))
    if cid=='User1_6_1':axs[1].plot(t,delta,lw=.8,color='#bb5f36')
axs[0].set_yticks(range(5),[r['content_id'] for r in meta['records']]);axs[0].invert_yaxis()
axs[0].set(xlabel='录制时间（秒）',title='5段验证录像：灰色为完整时间线，蓝色为四阶段内有效 GT 的预测锚点')
axs[1].axhline(0,color='#777',lw=.8);axs[1].set(xlabel='User1_6_1 录制时间（秒）',ylabel='训练 PTS − 录制时间（毫秒）',title='User1 的时钟差异单独审计；其余4段差异为0')
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/input_time_audit.png',dpi=180);fig.savefig(ROOT/'figures/input_time_audit.svg')
(ROOT/'input_audit/timeline_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
