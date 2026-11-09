"""Render standalone Chinese report and export scientific PNG/PDF figures."""
from pathlib import Path
import json, html, math, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
P=ROOT/'output'
FIG=P/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'savefig.dpi':170})
COL={'active':'#166583','passive':'#dc7543','delta':'#6c4b9c'}
STAGES={'A':['grab ring','place ring','reach ring','drop ring','move ring'],
 'C':['grab string','threading string through tower','pull string','release string']}
FIGURES=[]

def read(name):
    f=P/name
    return pd.read_csv(f) if f.exists() and f.stat().st_size>4 else pd.DataFrame()

def filt(df,**kw):
    out=df
    for k,v in kw.items():
        if k in out:out=out[out[k].eq(v)]
    return out

def true(s):
    return s.astype(str).str.lower().eq('true')

def stages(df,task):
    present=list(dict.fromkeys(filt(df,task=task)['stage'].dropna()))
    # Stage labels come from the actual exported frozen definitions.
    return present

def save(fig,stem,title,caption):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(stem+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(stem+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    FIGURES.append({'stem':stem,'title':title,'caption':caption})

def curve_figs(curves,coverage):
    for signal in ['position','velocity']:
      for task in ['A','C']:
        d=filt(curves,variant='primary',task=task,signal=signal)
        ss=stages(coverage,task) if not coverage.empty else stages(d,task)
        cols=[('l','PSM in stage'),('r','PSM in stage'),('l','Both in stage'),('r','Both in stage')]
        for diff in [False,True]:
            fig,axes=plt.subplots(len(ss),4,figsize=(14,2.25*len(ss)),squeeze=False,sharex=True)
            for i,stage in enumerate(ss):
              for j,(side,mode) in enumerate(cols):
                ax=axes[i,j];q=filt(d,stage=stage,side=side,mode=mode).sort_values('lag_s')
                ax.axhline(0,color='#c5ccd1',lw=.7);ax.axvline(0,color='#c5ccd1',lw=.7)
                n=int(q.n_observers.iloc[0]) if len(q) else 0
                if len(q) and n>=3:
                    x=q.lag_s.to_numpy(float)
                    if diff:
                        ax.plot(x,q.delta_r,color=COL['delta'],lw=1.3)
                        if {'delta_r_low','delta_r_high'}.issubset(q):
                            ax.fill_between(x,q.delta_r_low.to_numpy(float),q.delta_r_high.to_numpy(float),color=COL['delta'],alpha=.16,lw=0)
                    else:
                        ax.plot(x,q.active_r,color=COL['active'],lw=1.3,label='Active')
                        ax.plot(x,q.passive_r,color=COL['passive'],lw=1.3,label='Passive')
                    ax.text(.02,.96,f"Pairs {int(q.n_pairs.iloc[0])} | observers {n}",transform=ax.transAxes,va='top',fontsize=7,color='#455460')
                else:
                    ax.text(.5,.5,'Insufficient support',transform=ax.transAxes,ha='center',color='#7a8792')
                if i==0:ax.set_title(('Left' if side=='l' else 'Right')+' · '+mode,fontsize=9)
                if j==0:ax.set_ylabel(stage+'\n'+('Passive − Active r' if diff else 'Correlation r'),fontsize=8)
                if i==len(ss)-1:ax.set_xlabel('Lag (s); positive = gaze earlier')
                ax.set_xlim(-2,2);ax.grid(alpha=.13)
            fig.suptitle(f"Task {task} · {signal.title()} · "+('Paired group curve difference' if diff else 'Matched-source group curves'),fontsize=16,y=1.01)
            if not diff:fig.legend(handles=[Line2D([0],[0],color=COL['active'],label='Active'),Line2D([0],[0],color=COL['passive'],label='Passive')],loc='upper center',bbox_to_anchor=(.5,.992),ncol=2,frameon=False)
            fig.tight_layout(rect=(0,0,1,.975))
            zh='位置' if signal=='position' else '速度'
            save(fig,f"{'delta' if diff else 'curves'}_{signal}_{task}",f"Task {task}：{zh}{'相关差值曲线' if diff else '相关曲线对照'}",
                '每个条件固定共同完整支持的记录对，按观看者等权汇总。阴影为按观看者同步重采样的逐 lag 探索性百分位范围，不是同时置信带。' if diff else '蓝色为来源操作者 active，橙色为对应观看者 passive。两条曲线采用同一配对集合和同一权重；少于 3 名观看者时不绘制组曲线。')

def paired_figs(metrics):
    d=filt(metrics,variant='primary')
    if 'included' in d:d=d[true(d.included)]
    for metric,zh in [('r_zero','零 lag 相关'),('peak_r','峰值相关')]:
        fig,axes=plt.subplots(2,2,figsize=(10,9))
        for i,task in enumerate(['A','C']):
          for j,signal in enumerate(['position','velocity']):
            ax=axes[i,j];q=filt(d,task=task,signal=signal)
            x=q['active_'+metric].to_numpy(float);y=q['passive_'+metric].to_numpy(float)
            valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
            if len(x):
                low=min(x.min(),y.min())-.03;high=max(x.max(),y.max())+.03
                ax.plot([low,high],[low,high],color='#8f9aa5',ls='--',lw=1)
                ax.scatter(x,y,c=COL['passive'],s=20,alpha=.56,edgecolor='none')
                ax.set_xlim(low,high);ax.set_ylim(low,high)
                ax.text(.03,.97,f'{len(x)} record-condition pairs',transform=ax.transAxes,va='top',fontsize=9)
            ax.set_title(f'Task {task} · {signal.title()}');ax.set_xlabel('Active '+metric);ax.set_ylabel('Passive '+metric);ax.grid(alpha=.15)
        fig.suptitle(f'Paired record-condition comparison · {metric}',fontsize=16)
        fig.tight_layout(rect=(0,0,1,.97))
        save(fig,'paired_'+metric,zh+'的逐记录配对比较','每个点代表同一来源视频、同阶段、同侧、同信号和同 mode 的一对结果。虚线为两者相等；点不代表彼此独立的受试者。')

def spatial_fig(spatial):
    if spatial.empty:return
    d=filt(spatial,variant='primary')
    if 'included' in d:d=d[true(d.included)]
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for i,task in enumerate(['A','C']):
      for j,(metric,label,scale) in enumerate([('distance_median_px','Median distance (px)',1),('proximity_50px','Within 50 px (%)',100)]):
        ax=axes[i,j];q=filt(d,task=task)
        a='active_'+metric;p='passive_'+metric
        if a not in q:continue
        x=q[a].to_numpy(float)*scale;y=q[p].to_numpy(float)*scale;v=np.isfinite(x)&np.isfinite(y);x=x[v];y=y[v]
        if len(x):
            m=max(x.max(),y.max())*1.05;ax.plot([0,m],[0,m],ls='--',c='#8f9aa5',lw=1)
            ax.scatter(x,y,s=22,c=COL['passive'],alpha=.6,edgecolor='none');ax.set_xlim(0,m);ax.set_ylim(0,m)
        ax.set_title(f'Task {task} · zero lag');ax.set_xlabel('Active '+label);ax.set_ylabel('Passive '+label);ax.grid(alpha=.15)
    fig.suptitle('Paired spatial proximity at zero lag',fontsize=16);fig.tight_layout(rect=(0,0,1,.97))
    save(fig,'spatial_zero','零 lag 空间接近度','距离越小、接近比例越高，表示 gaze 越接近器械。这里只比较共同零 lag 条件，未混用原来两组各自的组峰位。')

def lag_fig(metrics):
    d=filt(metrics,variant='primary')
    if 'included' in d:d=d[true(d.included)]
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for i,task in enumerate(['A','C']):
      for j,signal in enumerate(['position','velocity']):
        ax=axes[i,j];q=filt(d,task=task,signal=signal)
        a=q.active_peak_lag_s.to_numpy(float);p=q.passive_peak_lag_s.to_numpy(float)
        ident=np.zeros(len(q),dtype=bool)
        if {'active_lag_identifiable','passive_lag_identifiable'}.issubset(q):ident=(true(q.active_lag_identifiable)&true(q.passive_lag_identifiable)).to_numpy()
        boundary=np.isclose(np.abs(a),2)|np.isclose(np.abs(p),2)
        ax.plot([-2,2],[-2,2],c='#909aa4',ls='--',lw=1)
        ax.scatter(a[~boundary],p[~boundary],s=18,c='#a6b0b9',alpha=.5,label='Other peaks')
        ax.scatter(a[boundary],p[boundary],s=30,c='#d3794e',marker='x',alpha=.7,label='Boundary in either domain')
        if ident.any():ax.scatter(a[ident],p[ident],s=40,c='#166583',label='Both identifiable')
        ax.set_xlim(-2.15,2.15);ax.set_ylim(-2.15,2.15);ax.set_title(f'Task {task} · {signal.title()}')
        ax.set_xlabel('Active peak lag (s)');ax.set_ylabel('Passive peak lag (s)');ax.grid(alpha=.15)
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#a6b0b9',label='Other peaks'),Line2D([0],[0],marker='x',ls='',color='#d3794e',label='Boundary in either domain'),Line2D([0],[0],marker='o',ls='',color='#166583',label='Both identifiable')],loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Peak lag and reliability · descriptive only',fontsize=16);fig.tight_layout(rect=(0,.04,1,.97))
    save(fig,'lag_reliability','峰值 lag 与定位可靠性','峰位图仅作描述。边界峰及未通过可靠性规则的峰值不能解释为精确的生理领先／滞后时间。重合点可能覆盖。')

def sensitivity_fig(sens):
    if sens.empty:return
    sens=sens[sens.variant.ne('common_time_primary')]
    if 'n_observers' in sens:sens=sens[sens.n_observers.ge(3)]
    d=filt(sens,metric='r_zero',weighting='observer_equal')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for i,task in enumerate(['A','C']):
      for j,signal in enumerate(['position','velocity']):
        ax=axes[i,j];q=filt(d,task=task,signal=signal).copy()
        q['condition']=q.stage.astype(str)+' / '+q.side.astype(str)+' / '+q['mode'].map({'PSM in stage':'P','Both in stage':'B'}).fillna(q['mode'])
        if not q.empty:
            pivot=q.pivot_table(index='variant',columns='condition',values='delta',aggfunc='first',dropna=False)
            if 'primary' in pivot.index:pivot=pivot.reindex(['primary']+[x for x in pivot.index if x!='primary'])
            v=pivot.to_numpy(float);lim=max(.01,float(np.nanmax(np.abs(v))))
            im=ax.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.set_yticks(range(len(pivot)),pivot.index,fontsize=7);ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=65,ha='right',fontsize=6)
            fig.colorbar(im,ax=ax,shrink=.7,label='Passive − Active r_zero')
        ax.set_title(f'Task {task} · {signal.title()}')
    fig.suptitle('Sensitivity to preprocessing variant',fontsize=16);fig.tight_layout(rect=(0,0,1,.96))
    save(fig,'sensitivity_variants','预处理配置敏感性','颜色表示零 lag 相关的配对组差异。各配置按自身双方完整支持的配对集合汇总；覆盖差异应结合 sensitivity_summary.csv 查看。P 为 PSM in stage，B 为 Both in stage。')
    d=filt(sens,variant='primary',metric='r_zero')
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for i,task in enumerate(['A','C']):
      for j,signal in enumerate(['position','velocity']):
        ax=axes[i,j];q=filt(d,task=task,signal=signal)
        keys=['stage','side','mode'];pivot=q.pivot_table(index=keys,columns='weighting',values='delta',aggfunc='first')
        if 'observer_equal' in pivot:
            for name,c in [('pair_equal','#166583'),('operator_equal','#dc7543')]:
                if name in pivot:ax.scatter(pivot.observer_equal,pivot[name],s=27,alpha=.7,c=c,label=name)
            v=pivot.to_numpy(float);lo=np.nanmin(v)-.01;hi=np.nanmax(v)+.01;ax.plot([lo,hi],[lo,hi],ls='--',c='#8f9aa5',lw=1)
            ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.axhline(0,lw=.7,c='#c5ccd1');ax.axvline(0,lw=.7,c='#c5ccd1');ax.set_title(f'Task {task} · {signal.title()}');ax.set_xlabel('Observer-equal difference');ax.set_ylabel('Alternative-weight difference');ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8);fig.suptitle('Sensitivity to paired aggregation weights',fontsize=16);fig.tight_layout(rect=(0,0,1,.97))
    save(fig,'sensitivity_weights','汇总权重敏感性','观看者等权为主分析，另外比较记录对等权和来源操作者等权。每种权重内部，active/passive 使用相同配对和权重。')

def common_time_fig(summary):
    d=filt(summary,weighting='observer_equal')
    if 'n_observers' in d:d=d[d.n_observers.ge(3)]
    base=filt(d,variant='primary');common=filt(d,variant='common_time_primary')
    keys=['task','side','signal','stage','mode','metric']
    q=base.merge(common,on=keys,suffixes=('_original','_common'))
    if q.empty:return
    for spatial in [False,True]:
      metrics=['distance_median_px','proximity_50px'] if spatial else ['r_zero','peak_r']
      if not q.metric.isin(metrics).any():continue
      fig,axes=plt.subplots(2,2,figsize=(11,9))
      for i,kind in enumerate(['A','C'] if spatial else ['position','velocity']):
       for j,metric in enumerate(metrics):
        ax=axes[i,j];z=filt(q,metric=metric,**({'task':kind} if spatial else {'signal':kind}))
        for task,color in [('A','#166583'),('C','#dc7543')]:
            zz=filt(z,task=task);ax.scatter(zz.delta_original,zz.delta_common,s=30,alpha=.72,c=color,label='Task '+task)
        vals=z[['delta_original','delta_common']].to_numpy(float)
        if vals.size and np.isfinite(vals).any():
            lo=np.nanmin(vals);hi=np.nanmax(vals);pad=max(.01,(hi-lo)*.07);lo-=pad;hi+=pad
            ax.plot([lo,hi],[lo,hi],ls='--',lw=1,c='#8f9aa5');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.axhline(0,color='#bac6ce',lw=.7);ax.axvline(0,color='#bac6ce',lw=.7)
        ax.set_title(('Task '+kind if spatial else kind.title())+' · '+metric);ax.set_xlabel('Original-result difference');ax.set_ylabel('Common-coordinate difference');ax.grid(alpha=.13)
      axes[0,0].legend(frameon=False,fontsize=8)
      fig.suptitle('Common available coordinate samples · '+('spatial' if spatial else 'correlation')+' sensitivity',fontsize=15)
      fig.tight_layout(rect=(0,0,1,.97))
      save(fig,'common_time_'+('spatial' if spatial else 'correlation'),'附录：共同可用坐标样本的'+('空间指标' if spatial else '相关指标')+'敏感性','每点为同一组条件的 Passive − Active：横轴复用原结果，纵轴将两域过滤后的 gaze 有限有效坐标掩码取交集后重算。虚线表示差异不变。此项仅为可选的坐标可用性敏感性分析，不改变已按来源映射和 NPY 行号确定的配对，不新增时间戳对齐条件。')

def difference_fig(summary):
    d=filt(summary,variant='primary',weighting='observer_equal')
    d=d[d.n_observers.ge(3)]
    short={'threading string through tower':'thread tower','pull through and cinch':'pull & cinch','wrap around psm':'wrap PSM'}
    for signal in ['position','velocity']:
        fig,axes=plt.subplots(2,2,figsize=(14,12))
        for i,task in enumerate(['A','C']):
          for j,metric in enumerate(['r_zero','peak_r']):
            ax=axes[i,j];q=filt(d,task=task,signal=signal,metric=metric).sort_values(['stage','side','mode'])
            yy=np.arange(len(q));labels=[short.get(r.stage,r.stage)+' / '+r.side+' / '+('P' if r.mode=='PSM in stage' else 'B') for r in q.itertuples()]
            ax.hlines(yy,q.low,q.high,color=COL['delta'],lw=1.3,alpha=.7);ax.scatter(q.delta,yy,c=COL['delta'],s=25,zorder=3)
            ax.axvline(0,color='#8496a3',lw=1,ls='--');ax.set_yticks(yy,labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel('Passive − Active');ax.set_title(f'Task {task} · {metric}');ax.grid(axis='x',alpha=.14)
        fig.suptitle(signal.title()+' · Paired scalar differences with exploratory ranges',fontsize=16);fig.tight_layout(rect=(0,0,1,.97))
        save(fig,'group_differences_'+signal,('位置' if signal=='position' else '速度')+'指标的配对差异汇总','点为观看者分层算术平均的配对差异，横线为 5000 次观看者同步重采样的探索性 2.5–97.5 百分位范围，不作正式显著性判定。P 为 PSM in stage，B 为 Both in stage；l/r 为固定左/右器械。')

def table(df,cols=None):
    if df.empty:return '<p class="muted">暂无可用结果。</p>'
    if cols is not None:df=df[[x for x in cols if x in df]]
    return '<div class="table-wrap">'+df.to_html(index=False,escape=True,classes='data',float_format=lambda x:f'{x:.4g}',na_rep='NA')+'</div>'

def format_range(s,digits=3):
    s=pd.to_numeric(s,errors='coerce').dropna()
    return f'{s.min():.{digits}f} 至 {s.max():.{digits}f}' if len(s) else 'NA'

def build_html(mapping,metrics,summary,coverage,spatial):
    main=filt(metrics,variant='primary');complete=main[true(main.included)] if 'included' in main else main
    s=filt(summary,variant='primary',weighting='observer_equal')
    if 'n_observers' in s:s=s[s.n_observers.ge(3)]
    lines=[]
    for signal,zh in [('position','位置'),('velocity','速度')]:
      q=filt(s,signal=signal,metric='r_zero')
      if len(q):lines.append(f'<li>{zh}零 lag 相关：{len(q)} 个支持充分条件，Passive − Active 范围为 <b>{format_range(q.delta)}</b>；其中 {int((q.delta<0).sum())} 个为负、{int((q.delta>0).sum())} 个为正。各条件彼此相关，此计数不构成显著性检验。</li>')
    dq=filt(s,metric='distance_median_px')
    if len(dq):lines.append(f'<li>零 lag 空间距离：{len(dq)} 个支持充分条件的 Passive − Active 为 <b>{format_range(dq.delta,1)} px</b>。正值表示 passive 距离更远。</li>')
    manifest={}
    if (P/'comparison_manifest.json').exists():manifest=json.loads((P/'comparison_manifest.json').read_text(encoding='utf-8'))
    audit={}
    if (P/'support_audit.json').exists():audit=json.loads((P/'support_audit.json').read_text(encoding='utf-8'))
    common=filt(summary,variant='common_time_primary',weighting='observer_equal')
    if 'n_observers' in common:common=common[common.n_observers.ge(3)]
    common_rows=[]
    for signal in ['position','velocity']:
      for metric in ['r_zero','peak_r']:
        a=filt(s,signal=signal,metric=metric);b=filt(common,signal=signal,metric=metric)
        keys=['task','side','signal','stage','mode','metric']
        j=a.merge(b,on=keys,suffixes=('_original','_common'))
        common_rows.append({'signal':signal,'metric':metric,'primary_supported':len(a),'primary_negative':int(a.delta.lt(0).sum()),'common_supported':len(b),'common_negative':int(b.delta.lt(0).sum()),'matched_conditions':len(j),'sign_changes':int((j.delta_original*j.delta_common<0).sum())})
    audit_text=f"NPY 的逐行配对与坐标是否有限有效是两件事。已配对的对应行中，某一域可能没有可用坐标；64 个信号×侧别审计单元的过滤后 gaze 有效掩码不一致比例为 {100*audit.get('mask_disagreement_fraction_min',float('nan')):.2f}%–{100*audit.get('mask_disagreement_fraction_max',float('nan')):.2f}%。这不表示行号错配，也不是否定原结果配对提取的依据。"
    downloads=[('paired_record_mapping.csv','16 对记录映射'),('row_correspondence_audit.csv','原始 NPY 行数与逐行对应核查'),('paired_condition_coverage.csv','预定条件完整覆盖'),('paired_metrics.csv','逐记录配对指标'),('paired_group_summary.csv','主分析及坐标可用性补充汇总'),('paired_group_curves.csv','组曲线与差值'),('group_peak_comparison.csv','组曲线峰位比较'),('paired_spatial_metrics.csv','零 lag 空间配对'),('sensitivity_summary.csv','配置和权重敏感性'),('loo_sensitivity.csv','逐身份剔除敏感性'),('common_time_record_summary.csv','附录：共同可用坐标补充指标'),('common_time_paired_metrics.csv','附录：共同可用坐标配对指标'),('common_time_paired_spatial_metrics.csv','附录：共同可用坐标空间配对'),('support_audit.csv','原网格与坐标有效性审计'),('support_audit.json','坐标有效性审计说明'),('bootstrap_distributions.npz','重采样分布'),('comparison_manifest.json','运行与输入清单'),('validation.json','验证结果'),('independent_validation.json','独立验证')]
    downloads.extend([('REPRODUCE.md','复现说明'),('bootstrap_index.json','重采样数组索引'),('active_matched_record_summary.csv','提取的 Active 单记录结果'),('active_matched_record_curves.jsonl','提取的 Active 完整曲线')])
    dl=''.join(f'<a class="download" href="{f}">{html.escape(label)}</a>' for f,label in downloads if (P/f).exists())
    def gallery_for(appendix):
        return ''.join(f'<article class="figure"><h3>{html.escape(x["title"])}</h3><p>{html.escape(x["caption"])}</p><a href="figures/{x["stem"]}.png"><img loading="lazy" src="figures/{x["stem"]}.png" alt="{html.escape(x["title"])}"></a><div class="links"><a href="figures/{x["stem"]}.png">PNG</a><a href="figures/{x["stem"]}.pdf">PDF</a></div></article>' for x in FIGURES if x['stem'].startswith('common_time_')==appendix)
    gallery=gallery_for(False);appendix_gallery=gallery_for(True)
    covcols=['task','side','signal','stage','mode','n_pairs_total','n_pairs_complete','n_observers','n_operators','status']
    sumcols=['variant','task','side','signal','stage','mode','metric','passive_value','active_value','delta','low','high','n_pairs','n_observers','n_operators','status']
    summary_display=summary.copy()
    if 'status' in summary_display:
        summary_display.loc[summary_display.status.ne('supported'),['passive_value','active_value','delta','low','high']]=np.nan
    unsupported=coverage[coverage.n_observers.lt(3)] if 'n_observers' in coverage else coverage.iloc[0:0]
    htmltext='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Passive 与 Active gaze–PSM 配对比较</title><style>
    :root{--ink:#183348;--muted:#566d7a;--line:#dce5ea;--blue:#166583;--orange:#dc7543}*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:var(--ink);font:15px/1.75 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1240px;margin:auto;background:white;padding:44px 48px}h1{font-size:31px;line-height:1.35;margin:8px 0 18px}h2{font-size:23px;margin:36px 0 14px;border-top:1px solid var(--line);padding-top:24px}h3{font-size:18px;margin:12px 0}p{margin:10px 0}.eyebrow{color:var(--blue);letter-spacing:.12em;font-size:12px;font-weight:700}.lead{font-size:18px;max-width:960px}.muted{color:var(--muted)}.badge{display:inline-block;padding:4px 12px;background:#e7f1f4;border-radius:20px;margin:0 6px 5px 0;font-size:13px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}.card{background:#f4f8fa;padding:18px;border-radius:8px;border-left:3px solid var(--blue)}.card b{display:block;font-size:29px}.note{background:#fff5eb;border-left:4px solid var(--orange);padding:15px 20px;margin:22px 0}.table-wrap{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:5px;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{text-align:left;background:#edf4f7;font-weight:600}th,td{padding:8px 10px;border-bottom:1px solid var(--line)}tr:nth-child(even){background:#fafcfd}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.download{display:inline-block;border:1px solid var(--line);padding:7px 12px;border-radius:5px;margin:4px}.figure{margin:22px 0 34px;padding:20px;border:1px solid var(--line);border-radius:9px}.figure img{width:100%;height:auto;background:white}.figure p{color:var(--muted);font-size:13px}.links{display:flex;gap:20px;font-size:13px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}footer{border-top:1px solid var(--line);margin-top:36px;padding-top:20px;font-size:12px;color:var(--muted)}@media(max-width:700px){main{padding:25px 16px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:26px}.figure{padding:10px}}@media print{body{background:white}main{padding:0;max-width:none}.figure{break-inside:avoid}.figure img{max-height:850px;object-fit:contain}.download{display:none}}
    </style></head><body><main>'''
    htmltext+=f'''<div class="eyebrow">MATCHED SOURCE VIDEO · SUPPLEMENTARY ANALYSIS</div><h1>Passive 与 Active gaze–PSM 配对比较</h1><p class="lead">同一段器械运动中，观看者的眼动与来源操作者的眼动，分别与器械轨迹呈现怎样的时间与空间关系？</p><span class="badge">41 Hz</span><span class="badge">±2 s 主窗口</span><span class="badge">固定左右器械</span><span class="badge">差值 = Passive − Active</span><div class="cards"><div class="card"><b>{len(mapping)}</b>来源视频配对</div><div class="card"><b>{mapping.observer_user.nunique()}</b>观看者</div><div class="card"><b>{mapping.source_user.nunique()}</b>来源操作者</div><div class="card"><b>{len(complete)} / {len(main)}</b>主分析完整条件对</div></div>
    <div class="note"><b>比较单位是来源视频，不是同一人的两种实验状态。</b>观看者与操作者身份可能重复、跨角色出现。所有区间为探索性重采样范围，未处理全部交叉依赖；本报告不提供正式 p 值、FDR 结论或精确生理时延结论。</div>
    <h2>主要观察</h2><ul>{''.join(lines)}</ul><p>以上为复用原单记录结果、对标量指标作观看者分层算术平均后的比较。峰值 lag 的可靠性标记保留在逐记录结果中。峰值相关、零 lag 相关与空间距离分别描述，相关更高不自动等于空间距离更近。</p>
    <h2>配对与分析方法</h2><p><b>按用户指定规则，记录配对只依赖来源映射和原始 active/passive gaze NPY 行数相等；第 i 行对应第 i 行。时间戳不作为配对条件，本次不新增时间戳重对齐。</b>以当前 16 条 passive 记录为纳入集合，使用 passive_gaze_links.json 的来源 content_id 与 gt84 单记录结果连接，并在 row_correspondence_audit.csv 核查原始 NPY 行数。41 Hz 仅沿用原分析的数值定义，不用于重新确定记录或行的对应关系。</p><p>主比较直接复用原有逐行对应分析结果，再按 task、side、signal、stage、mode 和 variant 匹配结果条件。原 gt84 的 84 条记录总体汇总不用于本次对照。记录已配对之后，每个结果条件仅汇总双方均具有完整 lag 支持的记录对，整条曲线固定同一集合；这是结果可计算性筛选，不改变输入行号对应规则。</p><p>完整组曲线先在观看者内对相关作 Fisher-z 等权平均，再在观看者间等权平均并反变换；active 与 passive 使用同一配对权重。单记录 r_zero、peak_r、peak_lag_s 等标量指标采用观看者内、观看者间两级算术平均，因此标量汇总的 r_zero 不必等于 Fisher-z 组曲线的零点。空间指标在共同零 lag 条件下按同一算术层级汇总。组结果至少需要 3 名观看者。</p><p>以观看者为单位进行 5000 次同步有放回重采样，保留 active/passive 配对。报告逐 lag 和各指标的 2.5–97.5 百分位范围；这些范围并非已校准的置信区间或多重比较同时区间。另给出记录对等权、来源操作者等权、预处理变体和按身份从任一角色剔除的敏感性结果。</p><p>正 lag 表示 gaze 对应较早的时刻。单记录峰位差汇总与重新汇总后的组曲线峰位差是不同统计量，分别保存在 paired_group_summary.csv 和 group_peak_comparison.csv。坐标有限有效性与输入行号配对分开处理，可选敏感性分析见附录。</p>
    <h2>16 对记录映射</h2>{table(mapping,['task','passive_id','active_id','observer_user','source_user'])}
    <h2>覆盖与缺失</h2><p>预定阶段即使没有数据也保留。少于 3 名观看者的条件不呈现可推断的组比较；缺失不置零。</p>{table(unsupported,covcols)}<details><summary>查看完整条件覆盖表</summary>{table(coverage,covcols)}</details>
    <h2>图形比较</h2>{gallery}
    <h2>数值汇总</h2><p>主分析标记为 primary，直接复用原有逐行对应结果。未达到 3 名观看者的条件在表中显示 NA；CSV 中保留的对应算术汇总仅供审计。</p><details><summary>展开主分析组指标与探索性范围</summary>{table(filt(summary_display,variant='primary'),sumcols)}</details>
    <h2>附录：共同可用坐标样本敏感性</h2><p>{audit_text}</p><p>以下为可选的描述性补充：两域分别按冻结规则预处理，再取过滤后有限有效 gaze 坐标掩码的交集，保留原来的 gaze 数值、PSM、阶段和数值核。此项不重新配对行，也不要求时间戳一致。文件中保留技术名称 common_time_primary，以便追溯既有输出。</p><p>下表比较原结果与共同可用坐标样本下的相关差异方向；negative 表示 passive 低于 active，sign_changes 为两套均可汇总条件中的方向改变数。条件之间并非独立观察，坐标可用性控制也不能消除上游历史插补或测量质量差异。</p>{table(pd.DataFrame(common_rows))}{appendix_gallery}<details><summary>展开可选坐标可用性敏感性数值</summary>{table(filt(summary_display,variant='common_time_primary'),sumcols)}</details>
    <h2>结果下载与复现</h2><p>所有链接均为本地相对路径，离线可用。源码与运行清单保留输入指纹、参数和随机种子信息。</p>{dl}
    <footer>补充报告 · gt_passive16 / active_passive_comparison。原有 passive 与 gt84 结果作为来源保留。图表中的每个记录条件并非独立受试者；任务、阶段、侧别与模式均按冻结规则定义。</footer></main></body></html>'''
    (P/'report.html').write_text(htmltext,encoding='utf-8')
    readme='''# Passive–Active 来源视频配对比较\n\n打开 [report.html](report.html) 查看中文离线报告。所有图形在 figures/ 中，同时提供 PNG 与 PDF。\n\n按用户指定规则，以 passive_gaze_links.json 的来源映射及原始 active/passive gaze NPY 行数相等确定配对，第 i 行对应第 i 行。时间戳不作为配对条件，本次不新增时间戳重对齐。原始行数见 row_correspondence_audit.csv。\n\n以 gt_passive16 的 16 条记录为起点，从 gt84 提取对应 active 单记录结果。配对单位是来源视频，不是同一人的两种实验状态。主分析直接复用原有逐行对应结果，观看者等权，active/passive 使用同一完整支持配对集合。差值统一为 Passive − Active。\n\n41 Hz 仅沿用原分析数值定义；主窗口 ±2 秒、固定左右器械、两种信号及两种阶段模式保持不变。5000 次观看者同步重采样的百分位范围仅为探索性描述，未处理全部参与者交叉依赖；不报告正式 p/FDR 结论。\n\n附录共同可用坐标样本敏感性仅为可选描述，文件保留 common_time_primary 技术名称。它处理坐标有限有效性，不改变来源映射和逐行对应关系。\n\n'''
    readme+='\n'.join(f'- [{label}]({f})' for f,label in downloads if (P/f).exists())+'\n'
    (P/'README.md').write_text(readme,encoding='utf-8')
    (P/'figure_manifest.json').write_text(json.dumps(FIGURES,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    global P,FIG,FIGURES
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=P);parser.add_argument('--html-only',action='store_true');args=parser.parse_args();P=args.output;FIG=P/'figures'
    mapping=read('paired_record_mapping.csv');metrics=read('paired_metrics.csv');summary=read('paired_group_summary.csv');coverage=read('paired_condition_coverage.csv');spatial=read('paired_spatial_metrics.csv')
    if args.html_only:
        FIGURES=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
    else:
        curve_figs(read('paired_group_curves.csv'),coverage);paired_figs(metrics);difference_fig(summary);spatial_fig(spatial);lag_fig(metrics);sensitivity_fig(read('sensitivity_summary.csv'));common_time_fig(summary)
    build_html(mapping,metrics,summary,coverage,spatial)
    print(json.dumps({'report':str(P/'report.html'),'figures':len(FIGURES)},ensure_ascii=False))

if __name__=='__main__':main()
