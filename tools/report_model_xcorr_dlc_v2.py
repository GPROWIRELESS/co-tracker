"""Standalone local HTML report for fixed-instrument exploratory xcorr v2.

Run: python report_model_xcorr_dlc_v2.py --output <run-directory>
All plotted PNGs are embedded in the HTML; no remote scripts or styles are used.
"""
from __future__ import annotations
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot']
COLORS = {'gt':'#111827', 'all_ABCD':'#2563eb', 'single_task':'#16a34a',
          'leave_out':'#ea580c', 'spd_finetune':'#9333ea', 'spd_zero_shot':'#dc2626'}
OBSERVED = '__observed_stage_average__'


def _read(root, name):
    path = root/name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _select(df, **kw):
    if df.empty:
        return df
    q = df
    for col, val in kw.items():
        if col not in q:
            return q.iloc[:0]
        q = q[q[col] == val]
    return q


def _table(df, cols, names=None, max_rows=None):
    cols = [c for c in cols if c in df]
    if not cols or df.empty:
        return '<p class="muted">没有可用结果；请查看对应CSV中的支持状态。</p>'
    q = df[cols].copy()
    if max_rows:
        q = q.head(max_rows)
    if names:
        q = q.rename(columns=names)
    return '<div class="table-scroll">'+q.to_html(index=False, border=0, na_rep='—',float_format=lambda x:f'{x:.3f}',escape=True)+'</div>'


def _image(path, caption):
    b64=base64.b64encode(path.read_bytes()).decode('ascii')
    return '<figure><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+b64+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def _finish(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_summary(df, root):
    q = _select(df,variant='primary',signal='position',mode='psm_stage',stage=OBSERVED)
    if q.empty:
        return None
    keys=sorted(q[['task','side']].drop_duplicates().itertuples(index=False,name=None))
    fig, axes=plt.subplots(len(keys),2,figsize=(11,max(3.2,2.8*len(keys))),squeeze=False)
    for k,(task,side) in enumerate(keys):
        qq=q[(q.task==task)&(q.side==side)].set_index('model').reindex(MODELS)
        for j,(metric,title) in enumerate([('curve_mae','Curve MAE (correlation units)'),('delta_lag_s','Signed model - GT peak lag (s)')]):
            ax=axes[k,j]
            y=np.arange(len(MODELS))
            values=qq['participant_equal_'+metric].to_numpy(float)
            lo=qq[metric+'_resample_low'].to_numpy(float)
            hi=qq[metric+'_resample_high'].to_numpy(float)
            for n,m in enumerate(MODELS):
                if np.isfinite(values[n]):
                    ax.plot(values[n],n,'o',color=COLORS[m],markersize=5)
                    if np.isfinite(lo[n]) and np.isfinite(hi[n]):
                        ax.plot([lo[n],hi[n]],[n,n],color=COLORS[m],lw=2)
            ax.set_yticks(y,MODELS)
            ax.invert_yaxis()
            ax.grid(axis='x',alpha=.2)
            if metric=='delta_lag_s':ax.axvline(0,color='#94a3b8',lw=1,ls='--')
            ax.set_title(f'Task {task} | {side.upper()} | {title}',fontsize=10)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed instruments | participant-equal observed-stage average\nRanges: participant resampling, exploratory and uncalibrated',fontsize=12,y=1.005)
    fig.tight_layout()
    path=root/'figures'/'primary_summary.png';_finish(fig,path)
    return path


def _plot_curves(df, root):
    q=_select(df,variant='primary',signal='position',mode='psm_stage')
    paths=[]
    if q.empty:return paths
    for (task,side),data in q.groupby(['task','side'],sort=True):
        stages=sorted(data.stage.unique())
        nrow=(len(stages)+1)//2
        fig, axes=plt.subplots(nrow,2,figsize=(12,3.1*nrow),squeeze=False)
        for ax,stage in zip(axes.flat,stages):
            part=data[data.stage==stage]
            has_data=False
            for src in ['gt',*MODELS]:
                line=part[part.source==src].sort_values('lag_s')
                if line.empty:continue
                x=line.lag_s.to_numpy(float);y=line.r.to_numpy(float)
                has_data = has_data or np.isfinite(y).any()
                ax.plot(x,y,label=src,color=COLORS[src],lw=2.1 if src=='gt' else 1.2,alpha=1 if src=='gt' else .85)
                if src=='gt' and 'r_resample_low' in line:
                    ax.fill_between(x,line.r_resample_low.to_numpy(float),line.r_resample_high.to_numpy(float),color='#64748b',alpha=.16)
            ax.axvline(0,color='#94a3b8',ls='--',lw=.8)
            ax.axhline(0,color='#94a3b8',lw=.5)
            if len(part): ax.set_xlim(float(part.lag_s.min()),float(part.lag_s.max()))
            if not has_data:
                ax.text(.5,.65,'Insufficient support; no estimated curve',transform=ax.transAxes,
                        ha='center',va='center',fontsize=9,color='#64748b')
            ax.set_title(stage,fontsize=10)
            ax.set_xlabel('Lag (s); positive = gaze leads PSM')
            ax.set_ylabel('Pooled correlation')
            ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
        for ax in list(axes.flat)[len(stages):]:ax.axis('off')
        handles, labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.025),fontsize=9)
        fig.suptitle(f'Task {task} | fixed {side.upper()} instrument | position / PSM-stage',fontsize=13)
        fig.tight_layout(rect=(0,.035,1,.97))
        path=root/'figures'/f'curves_{task}_{side}.png';_finish(fig,path);paths.append(path)
    return paths


def _plot_peak_distributions(records,root):
    q=_select(records,variant='primary',source='gt',signal='position',mode='psm_stage')
    if q.empty or 'bootstrap_peak_distribution' not in q:return None
    examples=[('User1_1_1','transfer ring','r'),('User7_1_1','transfer ring','l'),
              ('User6_1_2','pull through and cinch','r'),('User1_5_2','pull through and cinch','l')]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,(cid,stage,side) in zip(axes.flat,examples):
        cell=q[(q.content_id==cid)&(q.stage==stage)&(q.side==side)]
        ax.set_xlim(-2,2);ax.axvline(0,color='#94a3b8',lw=1,ls='--')
        ax.set_title(f'{cid} | {side.upper()} | {stage}',fontsize=10)
        if not cell.empty:
            row=cell.iloc[0]
            try:values=np.array(json.loads(row.bootstrap_peak_distribution),float)
            except (ValueError,TypeError):values=np.empty((0,2))
            if values.size:
                ax.bar(values[:,0],values[:,1]/values[:,1].sum(),width=1/41,color='#2563eb',alpha=.8)
                ax.set_ylabel('Resampling frequency')
            else:
                ax.text(.5,.5,'Insufficient repeats for occurrence bootstrap',ha='center',va='center',transform=ax.transAxes,color='#64748b',fontsize=9)
            ax.text(.02,.97,str(row.lag_status).replace('_',' '),va='top',transform=ax.transAxes,fontsize=8)
        ax.set_xlabel('Peak lag (s)');ax.grid(axis='y',alpha=.15)
    fig.suptitle('GT occurrence-block peak distributions | uncalibrated exploratory diagnostics',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    path=root/'figures'/'peak_distributions.png';_finish(fig,path);return path


def generate_report(outdir):
    root=Path(outdir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    summary=_read(root,'paired_summary.csv')
    groups=_read(root,'group_summary.csv')
    curves=_read(root,'group_curves.csv')
    records=_read(root,'record_summary.csv')
    pairs=_read(root,'paired_metrics.csv')
    direct=_read(root,'direct_model_delay.csv')
    qc=_read(root,'preprocessing_qc.csv')
    exact=_read(root,'model_comparisons.csv')
    identity=_read(root,'identity_reference.csv')
    paired_group=_read(root,'paired_group_summary.csv')
    unpaired=_read(root,'unpaired_reference.csv')
    matched_sensitivity=_read(root,'sensitivity_summary.csv')
    # Accept alternate names used by callers without losing any table.
    if exact.empty:exact=_read(root,'exact_model_comparisons.csv')
    if identity.empty:identity=_read(root,'identity_null.csv')
    primary=_select(summary,variant='primary',signal='position',mode='psm_stage',stage=OBSERVED)
    overview_cols=['task','side','model','n_total_users','n_users','n_total_conditions','n_conditions','participant_equal_curve_mae','participant_equal_delta_lag_s','participant_equal_abs_lag_difference_s','gt_localizable_rate','model_localizable_rate','model_boundary_rate']
    names={'task':'任务','side':'器械侧','model':'模型','n_total_users':'总人数','n_users':'可用人数','n_total_conditions':'总条件','n_conditions':'输入可用条件','participant_equal_curve_mae':'曲线MAE','participant_equal_delta_lag_s':'有符号Δlag(s)','participant_equal_abs_lag_difference_s':'绝对Δlag(s)','gt_localizable_rate':'GT可定位率','model_localizable_rate':'模型可定位率','model_boundary_rate':'模型边界率'}
    plots=[]
    mainplot=_plot_summary(summary,root)
    if mainplot:plots.append(mainplot)
    curveplots=_plot_curves(curves,root);plots.extend(curveplots)
    peakplot=_plot_peak_distributions(records,root)
    if peakplot:plots.append(peakplot)
    body=['<h1>Gaze–PSM 时序评价 v2</h1>',
      '<p class="subtitle">固定左右器械 · 保留低 likelihood DLC 点 · 轻度平滑 · 探索性重分析</p>',
      '<div class="notice"><strong>结果用途：</strong>评价相关曲线与峰位置的复现，不证明因果领先。每任务最多5名参与者；重采样范围未验证为具有95%覆盖率的置信区间。模型排列为固定名称顺序，不构成排名。</div>',
      '<h2>主分析定义</h2><p>主分析为 position / psm_stage。GT与全部模型比较同一左或右器械，并使用共同技术支持。DLC likelihood、坐标跳变只记录作诊断，不触发删除。位置主配置为3点居中轻度平滑；模型原始时序保留，不用测试集最佳平移消除偏差。边界峰、多峰、非正峰和不可定位状态保留。</p>',
      '<p>曲线MAE是固定lag网格上相关系数差的平均绝对值，单位不是秒；Δlag定义为模型峰位置减GT峰位置。跨阶段汇总先在参与者内对实际可用阶段等权，再对参与者等权，故称“可观测阶段平均”，不能外推到缺失阶段。</p>',
      '<h2>固定侧主结果</h2>',_table(primary,overview_cols,names),
      '<p class="muted">表内lag点值包括不可定位条件的描述性估计。这里的可定位率来自 lag_identifiable 操作性规则，尚未验证为确证判定。解释时间误差时必须同时查看该比例、边界率和逐条件状态；不可定位不能默认为零误差。曲线误差与各项范围的详细字段见CSV。</p>']
    if mainplot:body.append(_image(mainplot,'点为参与者等权估计，线为未校准的参与者重采样范围；不是正式模型排名。'))
    body.append('<h2>各阶段的相关曲线</h2><p>黑线为GT，彩线为模型；灰带仅表示GT逐lag重采样范围。各来源完整支持参与者可能不同，准确人数见组汇总表。正lag表示gaze领先PSM。</p>')
    for p in curveplots:body.append(_image(p,p.stem))
    if peakplot:
        body.append(_image(peakplot,'四个GT条件示例：展示峰的重采样分布及重复不足的情况。窄分布不自动证明生理领先。'))
    pg=_select(groups,variant='primary',signal='position',mode='psm_stage',source='gt')
    body.extend(['<details><summary>GT组峰与支持情况</summary>',_table(pg,['task','side','stage','n_stable_users','peak_lag_s','peak_r','peak_at_boundary','peak_lag_resample_low','peak_lag_resample_high','bootstrap_boundary_frequency','status']),'</details>'])
    body.append('<h2>直接GT–模型相对时序</h2><p>直接相关峰描述信号相对偏移，不自动等于时间戳错误或计算延迟。只有独立验证集校准才能作为预先冻结的校正；本分析保留原始模型时序。</p>')
    if not direct.empty:
        dd=direct
        if 'variant' in dd:dd=dd[dd.variant=='primary']
        if 'signal' in dd:dd=dd[dd.signal=='position']
        whole=dd[dd.segment=='whole'] if 'segment' in dd else dd
        halves=dd[dd.segment!='whole'] if 'segment' in dd else dd.iloc[:0]
        if 'model' in whole and 'peak_lag_s' in whole:
            model_rows=[]
            for model in MODELS:
                z=whole[whole.model==model]
                x=pd.to_numeric(z.peak_lag_s,errors='coerce').dropna()
                model_rows.append(dict(model=model,n_videos=len(z),n_finite=len(x),
                    median_relative_delay_s=x.median() if len(x) else np.nan,
                    n_positive=int((x>0).sum()),n_negative=int((x<0).sum()),n_zero=int((x==0).sum())))
            body.append(_table(pd.DataFrame(model_rows),['model','n_videos','n_finite','median_relative_delay_s','n_positive','n_negative','n_zero']))
        body.append('<p class="muted">本节正lag表示模型gaze落后于GT gaze；中位数为视频等权的直接相关峰偏移，不能与对PSM的组峰差互换。</p>')
        body.extend(['<details><summary>全视频直接时序结果（全部视频×模型）</summary>',
            _table(whole,['content_id','task','model','segment','peak_lag_s','peak_r','peak_at_boundary']),'</details>',
            '<details><summary>前半段与后半段稳定性（保留segment标识）</summary>',
            _table(halves,['content_id','task','model','segment','peak_lag_s','peak_r','peak_at_boundary']),'</details>'])
    else:body.append('<p>直接时序审计结果见运行目录中的对应CSV。</p>')
    body.append('<h2>敏感性与不可估计情况</h2><p>敏感性分支检验不平滑、5点轻度平滑、边缘处理、±4秒窗口及速度滤波选择。不得根据结果选择最有利分支。分支间输入覆盖可能变化，应优先查阅固定共同条件的敏感性结果；以下各自支持汇总不能单独归因于参数变化。</p>')
    ms=_select(matched_sensitivity,signal='position',mode='psm_stage',stage=OBSERVED)
    body.extend(['<details><summary>共同条件上的敏感性比较（优先解读）</summary>',
        '<p>每个分支与其 primary_matched_to_* 配对行使用完全相同的条件keys。边缘处理可改变条件内部的时间支持；window4与±2秒主分析的lag域不同，应优先比较window4_restricted2分支。</p>',
        _table(ms,['sensitivity_branch','variant',*overview_cols],names),'</details>'])
    sens=_select(summary,signal='position',mode='psm_stage',stage=OBSERVED)
    if not sens.empty:sens=sens[sens.variant!='primary']
    body.extend(['<details><summary>敏感性分支各自支持汇总</summary>',_table(sens,['variant',*overview_cols],names),'</details>'])
    if not records.empty and 'lag_status' in records:
        states=_select(records,variant='primary',signal='position',mode='psm_stage').groupby(['source','lag_status'],dropna=False).size().reset_index(name='conditions')
        body.extend(['<details><summary>逐来源lag状态分布</summary>',_table(states,['source','lag_status','conditions']),'</details>'])
    body.append('<h2>参照与配对比较</h2><p>他人gaze参照、参与者身份重配参照和模型间配对比较回答不同问题。身份参照对曲线误差和绝对峰lag差使用相同的参与者整体映射，保留stage×side结构。峰lag参照未按可靠性筛选，因此只评价统计峰匹配，不能称为真实生理延迟的null。该参照依赖零假设下的交换性假设；不能把经验秩解释为普遍独立性或因果领先检验。每任务5人时，模型标签整体交换的双侧精确检验最小p值为0.0625。</p>')
    pgd=_select(paired_group,variant='primary',signal='position',mode='psm_stage')
    body.extend(['<details><summary>共同参与者上的配对组峰差</summary>',
        '<p>这是两条参与者组曲线的峰位置之差，不是逐记录绝对lag误差的平均。各模型对可使用不同的共同完整参与者；配对重采样使用相同用户索引，范围仍属未校准的探索性范围。</p>',
        _table(pgd,['task','side','stage','model','n_shared_users','gt_peak_lag_s','model_peak_lag_s','delta_group_peak_lag_s','delta_group_peak_lag_resample_low','delta_group_peak_lag_resample_high','gt_boundary','model_boundary','status']),'</details>'])
    body.extend(['<details><summary>他人gaze配对参照及有效次数</summary>',
        '<p>以同任务其他参与者的gaze按elapsed time配对PSM。每次保留观测组完整支持的所有目标参与者，缺少支持的抽样不补齐。超越比例只描述此参照，不能解释为正式的领先p值。</p>',
        _table(unpaired,['task','side','stage','source','n_target_users','n_requested','n_valid','observed_peak_r','reference_peak_r_95','exceedance_fraction_corrected','status']),'</details>'])
    body.extend(['<details><summary>参与者整体身份参照</summary>',_table(identity,['task','side','model','n_users','n_common_stages','n_permutations','observed_curve_mae','null_mean_curve_mae','empirical_lower_tail_rank','observed_abs_lag_difference_s','null_mean_abs_lag_difference_s','lag_empirical_lower_tail_rank','status']),'</details>',
      '<details><summary>模型间参与者整块比较</summary>',_table(exact,['task','side','stage','model_a','model_b','n_users','mean_curve_mae_b_minus_a','p_exact','p_holm','n_permutations','status']),'</details>'])
    body.append('<h2>审计、验证与尚未验证的限制</h2><ul><li>当前10个视频已用于方法审查，属于探索性重分析；不能追溯称为预注册验证。</li><li>未按likelihood或跳变剔除DLC点，遵循用户目视检查结论。剩余轻度滤波影响通过敏感性分析展示。</li><li>低重复数量下的记录内不确定性不可可靠估计；单occurrence不产生伪bootstrap区间。</li><li>现有样本不足以验证人群覆盖率、统一生理同步阈值或稳定模型排名。增加重采样次数不增加参与者。</li><li>原始高频gaze重新预处理、独立操作器械标注、独立时间校准和新参与者验证需另行数据支持。</li></ul>')
    for filename,title in [('validation.json','计算验证'),('simulation_validation.json','仿真验证'),('run_manifest.json','运行清单')]:
        path=root/filename
        if path.exists():
            try:obj=json.loads(path.read_text(encoding='utf-8-sig'));content=json.dumps(obj,ensure_ascii=False,indent=2)
            except Exception:content=path.read_text(encoding='utf-8',errors='replace')
            body.append('<details><summary>'+title+'</summary><pre>'+html.escape(content)+'</pre></details>')
    tables=sorted(p.name for p in root.glob('*.csv'))
    body.append('<h2>完整数据表</h2><ul>'+''.join('<li><a href="'+html.escape(n,quote=True)+'">'+html.escape(n)+'</a></li>' for n in tables)+'</ul>')
    css='''body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;color:#172033;background:#fff;line-height:1.65}h1{font-size:32px;margin-bottom:4px}h2{font-size:22px;margin-top:36px;border-top:1px solid #e2e8f0;padding-top:20px}.subtitle,.muted,figcaption{color:#64748b}.notice{background:#eff6ff;border-left:4px solid #2563eb;padding:14px 18px;margin:22px 0}.table-scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:right}th{background:#f1f5f9}td:first-child,th:first-child{text-align:left}img{width:100%;height:auto}figure{margin:26px 0}figcaption{font-size:13px}details{margin:16px 0;padding:12px;background:#f8fafc;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font-size:12px;max-height:450px;overflow:auto}a{color:#1d4ed8}'''
    document='<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Gaze–PSM v2</title><style>'+css+'</style></head><body>'+''.join(body)+'</body></html>'
    (root/'report.html').write_text(document,encoding='utf-8')
    readme='''# Gaze–PSM fixed-instrument v2

打开 `report.html` 查看中文完整报告。图像内嵌，离线可用。

- 主分析：position / psm_stage，左右器械分别固定。
- DLC likelihood 和跳变只用于诊断；保留低 likelihood 点，采用轻度平滑。
- 主比较集合仅按技术输入支持确定，不按峰正负、边界或模型可定位性删除条件。
- 曲线 MAE 评价相关曲线复现；signed/absolute lag 差同时报告可定位覆盖及边界比例。
- 组汇总以参与者为单位；跨阶段值明确称 observed-stage average。
- lag_identifiable是操作性规则，尚未验证为确证判定。
- 重采样范围是小样本探索性范围，未验证为95%置信区间。
- 模型原始时序保留，不用测试集最佳平移校准。
- 运行清单、输入/代码哈希、验证与敏感性结果一并保存。

## 结果入口

- `paired_summary.csv`：参与者等权配对汇总与重采样范围。
- `paired_metrics.csv`：全部逐条件同侧比较和状态。
- `record_summary.csv`：每条相关曲线的峰、支持和可靠性。
- `group_summary.csv` / `group_curves.csv`：组曲线及参与者支持。
- `paired_group_summary.csv`：共同完整参与者上的组曲线峰差，区别于逐记录MAE。
- `sensitivity_summary.csv`：每一分支与其matched primary的共同条件比较。
- `unpaired_reference.csv`：他人gaze参照及实际有效抽样数。
- `direct_model_delay.csv`：直接GT–模型相对时序审计。
- `identity_reference.csv`：参与者整体身份重配参照。
- `model_comparisons.csv`：参与者整块模型标签交换及Holm校正。
- `run_manifest.json` / `validation.json`：版本和验收记录。

## 推断边界

当前每任务最多5名参与者，A/C合计7名独立参与者。不得把重复阶段当作独立样本。
完整敏感性方案须一起报告，不能挑选最有利分支。新参与者验证、独立校准和独立器械标注尚需额外数据。
'''
    (root/'README.md').write_text(readme,encoding='utf-8')
    return {'report':str(root/'report.html'),'figures':[str(p) for p in plots], 'tables':tables}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(generate_report(args.output),ensure_ascii=False))
