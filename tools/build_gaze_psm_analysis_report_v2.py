"""Build an evidence-linked Chinese interpretation of the fixed-side v2 results."""
from pathlib import Path
import argparse
import base64
import html
import json
import numpy as np
import pandas as pd

MODELS=['all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot']
OBS='__observed_stage_average__'

def select(df,**kw):
    out=df
    for k,v in kw.items():out=out[out[k]==v]
    return out

def table(df,columns,labels=None,digits=3):
    q=df[columns].copy()
    if labels:q=q.rename(columns=labels)
    return '<div class="scroll">'+q.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.{digits}f}',escape=True)+'</div>'

def link(file,label=None):return f'<a href="{html.escape(file,quote=True)}">{html.escape(label or file)}</a>'

def fig(root,name,caption):
    p=root/'figures'/name
    if not p.exists():return ''
    uri='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
    return '<figure><img src="'+uri+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

def main(outdir):
    root=Path(outdir).resolve()
    read=lambda name:pd.read_csv(root/name)
    groups=read('group_summary.csv');pairs=read('paired_metrics.csv');summaries=read('paired_summary.csv')
    records=read('record_summary.csv');direct=read('direct_model_delay.csv');identity=read('identity_reference.csv')
    comparisons=read('model_comparisons.csv');sens=read('sensitivity_summary.csv');pg=read('paired_group_summary.csv')
    gt=select(groups,variant='primary',source='gt',signal='position',mode='psm_stage')
    primary=select(summaries,variant='primary',signal='position',mode='psm_stage',stage=OBS)
    p=select(pairs,variant='primary',signal='position',mode='psm_stage')
    valid=p[p.common_input.astype(bool)]
    for frame in [groups,pairs,summaries,records,pg]:
        assert not frame.stage.eq('drop ring').any(), 'Excluded stage remains in current analysis'
    # This interpretation is authored for the frozen ten-video study, not a generic
    # narrative generator. Reject materially changed inputs rather than reuse claims.
    expected={'all_ABCD':.280487804878,'single_task':.268292682927,'leave_out':.365853658537,'spd_finetune':.304878048780,'spd_zero_shot':.5}
    full=select(direct,segment='whole')
    assert len(full)==50 and full.content_id.nunique()==10, 'Narrative requires the audited ten-video cohort'
    for model,value in expected.items():
        assert abs(full[full.model==model].peak_lag_s.median()-value)<1e-9, 'Delay evidence changed; revise interpretation'
    tr=gt[(gt.task=='A')&(gt.side=='r')&(gt.stage=='transfer ring')]
    assert len(tr)==1 and abs(float(tr.iloc[0].peak_lag_s)-21/41)<1e-10, 'GT evidence changed; revise interpretation'
    assert comparisons.p_holm.dropna().eq(1).all(), 'Model comparison evidence changed; revise interpretation'
    delay=full.groupby('model',sort=False).agg(median_delay_s=('peak_lag_s','median'),min_delay_s=('peak_lag_s','min'),max_delay_s=('peak_lag_s','max'),n_videos=('content_id','size'),positive_videos=('peak_lag_s',lambda x:int((x>0).sum()))).reindex(MODELS).reset_index()
    def metric_matrix(metric):
        x=primary.pivot(index='model',columns=['task','side'],values=metric).reindex(MODELS)
        x.columns=[f'Task {t} {"左" if s=="l" else "右"}' for t,s in x.columns]
        return x.reset_index().rename(columns={'model':'模型'})
    n_conditions=valid[['content_id','stage','side']].drop_duplicates().shape[0]
    both=int((valid.gt_localizable.astype(bool)&valid.model_localizable.astype(bool)).sum())
    gt_recs=select(records,variant='primary',source='gt',signal='position',mode='psm_stage')
    gt_complete=gt_recs[gt_recs.full_support.astype(bool)]
    gt_local=int(gt_complete.lag_identifiable.astype(bool).sum())
    scope=[]
    for (task,side),q in valid.groupby(['task','side']):
        uq=q[['content_id','user','stage']].drop_duplicates()
        scope.append(dict(task=task,side=side,n_people=uq.user.nunique(),n_conditions=len(uq),stages='；'.join(sorted(uq.stage.unique()))))
    scope=pd.DataFrame(scope)
    transfer=select(pg,variant='primary',signal='position',mode='psm_stage',stage='transfer ring')
    change=[]
    for signal in ['position','velocity']:
        q=select(sens,signal=signal,mode='psm_stage',stage=OBS)
        for branch in q.sensitivity_branch.unique():
            a=q[q.variant==branch];b=q[q.variant=='primary_matched_to_'+branch]
            if a.empty or b.empty:continue
            joined=a.merge(b,on=['task','side','model'],suffixes=('_v','_p'))
            de=(joined.participant_equal_curve_mae_v-joined.participant_equal_curve_mae_p).abs()
            dt=(joined.participant_equal_abs_lag_difference_s_v-joined.participant_equal_abs_lag_difference_s_p).abs()
            change.append(dict(signal=signal,branch=branch,n_summaries=len(joined),median_curve_change=float(de.median()),max_curve_change=float(de.max()),median_lag_mae_change_s=float(dt.median()),max_lag_mae_change_s=float(dt.max())))
    changes=pd.DataFrame(change)
    changes.to_csv(root/'analysis_sensitivity_effects.csv',index=False,encoding='utf-8-sig')
    body=['<h1>Gaze 与 PSM 时序关系及模型复现分析</h1>',
      '<p class="subtitle">固定左右器械的 v2 结果分析 · 2026年11月5日</p>',
      '<div class="lead"><p><strong>主要结论：</strong>本批数据支持某些阶段存在可复现的位置相关结构，最清楚的正向组峰出现在 Task A 的右侧 <em>transfer ring</em>。模型通常能复现部分曲线形状，但存在约数百毫秒的 GT–模型相对时序偏移；曲线相似不代表精确领先时间已经复现。当前样本不能支持五模型的显著优劣排名。</p></div>',
      '<p>这一判断来自固定物理器械、共同输入支持上的比较。DLC 低 likelihood 和跳变点均保留，仅采用轻度平滑；未用测试视频的最优时间平移校正模型。所有边界峰、非正峰和不可定位结果均保留在主分析中。</p>',
      '<h2>分析范围与统计单位</h2>',
      f'<p>共10个视频，A/C各5个视频，每任务5名参与者，跨任务合计7名独立参与者。排除指定阶段后，主 position / psm_stage 分析共有 <strong>{n_conditions} 个 recording×stage×side 条件</strong>、{len(valid)}条五模型配对，均具有共同完整 lag 支持。条件数不能当作独立参与者数。</p>',
      table(scope,['task','side','n_people','n_conditions','stages'],{'task':'任务','side':'器械侧','n_people':'参与者','n_conditions':'主条件数','stages':'纳入阶段'}),
      '<p>位置采用3点居中平滑，采样率41 Hz，主搜索窗口±2秒。每个 occurrence 分别中心化后池化；参与者内录制等权、参与者之间等权的层级规则保持明确。跨阶段汇总为“每人实际可用阶段平均”，不代表各阶段具有相同的人群覆盖。数值和覆盖来源：'+link('paired_summary.csv')+'。</p>',
      '<p class="note">下文方括号内范围是参与者或 occurrence 重采样的2.5%–97.5%分位范围，未经覆盖率校准，不称为正式95%置信区间。正 lag 表示 gaze 领先 PSM；signed Δlag＝模型峰减GT峰。曲线 MAE 的单位是相关系数，不能换算成秒。</p>',
      '<h2>GT 的领先结构具有阶段和器械侧别特异性</h2>',
      '<p><strong>Task A 右侧 transfer ring 是主要正向时序线索。</strong>GT组峰约为 +0.512秒，重采样范围约[+0.268,+0.659]秒，峰值相关约0.450。左侧同一阶段的组峰只有+0.195秒，范围约[−0.049,+0.415]秒。因此，不能把右侧的证据直接推广为“整个 transfer ring 都领先约0.5秒”。</p>',
      '<p>该阶段两侧峰值相关都高于绝大多数他人 gaze 参照，1000次有效参照的校正超越比例均约0.001。这支持该配对具有相关结构；它检验的是峰值相关，而不是生理领先时间的准确性。来源：'+link('unpaired_reference.csv')+'。</p>',
      '<p>其他 Task A 阶段多存在跨零或较宽的峰位置范围。Task C 的右侧 wrap around psm 相关较高（约0.473），但峰在−0.146秒，范围约[−0.195,+0.244]秒，说明“高相关”不能直接作为领先证据。右侧 threading string through tower 的范围偏负，但只有3名参与者，且小样本峰范围未校准，应仅保留为滞后倾向的探索性线索。</p>',
      table(gt,['task','side','stage','n_stable_users','peak_r','peak_lag_s','peak_lag_resample_low','peak_lag_resample_high','peak_at_boundary'],{'task':'任务','side':'侧','stage':'阶段','n_stable_users':'人数','peak_r':'峰值r','peak_lag_s':'峰lag(s)','peak_lag_resample_low':'范围下限(s)','peak_lag_resample_high':'范围上限(s)','peak_at_boundary':'边界峰'}),
      '<p class="source">来源：'+link('group_summary.csv')+'；图中灰带为GT逐lag重采样范围，不是同时置信带。</p>']
    for name in ['curves_A_l.png','curves_A_r.png','curves_C_l.png','curves_C_r.png']:
        body.append(fig(root,name,'固定器械的主相关曲线；黑色为GT，彩色为模型。'))
    body.extend(['<h2>模型存在相对时序偏移 但不是已证实的帧索引错误</h2>',
      '<p>直接对整段GT gaze与模型 gaze做相关，五模型各有9/10个视频的最佳相关峰显示模型落后于GT。视频间中位偏移约0.27–0.50秒，其中zero-shot最大。每个视频并不共享同一偏移，且存在反向视频，所以不能简单把全部结果减去一个“统一延迟”。</p>',
      table(delay,['model','n_videos','positive_videos','median_delay_s','min_delay_s','max_delay_s'],{'model':'模型','n_videos':'视频数','positive_videos':'正向滞后视频数','median_delay_s':'中位模型滞后(s)','min_delay_s':'最小(s)','max_delay_s':'最大(s)'}),
      '<p>现有帧语义审计显示：预测目标是clip最后一帧，保存的帧号和时间戳使用相应anchor；未发现中心帧与末帧混淆的证据。直接相关偏移可能包含真实模型反应滞后、时间结构差异或非平稳性，不等于已证实的采集或计算延迟。来源：'+link('direct_model_delay.csv')+'、'+link('temporal_semantics_audit.json')+'。</p>',
      '<h2>模型曲线复现与峰时间复现应分别评价</h2>',
      '<p>下表均为每名参与者内部对可观测阶段平均后，再对参与者等权。所有模型在每个任务与侧别使用相同的输入条件；数值较小仅表示这一描述性指标较小。</p>',
      '<h3>相关曲线平均绝对误差</h3>',table(metric_matrix('participant_equal_curve_mae'),list(metric_matrix('participant_equal_curve_mae').columns)),
      '<p>Task A两侧的 all_ABCD 曲线误差数值最小；Task C两侧的 leave_out 数值最小。这种任务依赖性不支持选出一个统一最优模型。不同任务的相关形状、变异和阶段组成不同，不能把跨任务误差差异直接解释为模型总体能力差异。</p>',
      '<h3>峰位置绝对差 单位秒</h3>',table(metric_matrix('participant_equal_abs_lag_difference_s'),list(metric_matrix('participant_equal_abs_lag_difference_s').columns)),
      '<h3>峰位置有符号差 单位秒</h3>',table(metric_matrix('participant_equal_delta_lag_s'),list(metric_matrix('participant_equal_delta_lag_s').columns)),
      '<p>Task A各模型、两侧的平均signed Δlag均为负，方向与GT–模型直接相关所见的模型相对滞后一致。但这两种统计量并不相等：一个来自对同一PSM的两个相关峰，一个来自GT与模型的直接相关。Task C的signed差随侧别与模型变化，不支持用统一偏移解释全部阶段。</p>',
      '<p><strong>transfer ring右侧的模型–GT差异最一致。</strong>GT组峰约+0.512秒，而模型组峰约−0.244至+0.098秒；五个paired组峰差的探索性范围都在0以下。这提示模型没有完整复现该侧GT的正向峰位置。左侧差异较弱且并非所有范围都排除0。这里是“组曲线峰之差”，不能与上述逐条件绝对差平均混用。</p>',
      table(transfer,['side','model','n_shared_users','gt_peak_lag_s','model_peak_lag_s','delta_group_peak_lag_s','delta_group_peak_lag_resample_low','delta_group_peak_lag_resample_high'],{'side':'侧','model':'模型','n_shared_users':'共同人数','gt_peak_lag_s':'GT组峰(s)','model_peak_lag_s':'模型组峰(s)','delta_group_peak_lag_s':'组峰差(s)','delta_group_peak_lag_resample_low':'范围下限(s)','delta_group_peak_lag_resample_high':'范围上限(s)'}),
      '<p class="source">来源：'+link('paired_summary.csv')+'、'+link('paired_group_summary.csv')+'。主表中的毫秒或秒值是描述性峰统计，不能无条件解释为精确时序测量误差。</p>',
      '<h2>身份参照说明曲线结构与单点 lag 的证据强度不同</h2>',
      '<p>身份参照保持任务、器械侧别和阶段结构，整体置换参与者的完整条件向量。每任务5人时枚举120种映射（包括原映射）；A保留4个全员共有阶段，C保留2个全员共有阶段，因此该参照不代表全部C阶段。</p>',
      '<p>例如Task A左侧，all_ABCD、single_task、spd_finetune的曲线误差经验秩均为1/120≈0.008；同一集合上的绝对lag差经验秩却分别约0.250、0.633、0.242。这说明曲线整体能保留参与者相关结构，而单点峰时间匹配的证据较弱。不能由此说模型“完全随机”，也不能把曲线匹配直接当成精确领先时间复现。</p>',
      table(identity,['task','side','model','n_common_stages','n_permutations','empirical_lower_tail_rank','lag_empirical_lower_tail_rank'],{'task':'任务','side':'侧','model':'模型','n_common_stages':'共有阶段','n_permutations':'映射数','empirical_lower_tail_rank':'曲线误差下尾经验秩','lag_empirical_lower_tail_rank':'绝对lag差下尾经验秩'}),
      '<p>经验秩依赖参与者标签可交换这一假设，属于条件性参照，未作为跨模型、跨侧别正式发现进行解释。模型间的参与者整块标签交换检验经各任务/侧别家族Holm校正后，本次全部p值为1；这表示没有足够证据进行显著排名，不表示模型等效。来源：'+link('identity_reference.csv')+'、'+link('model_comparisons.csv')+'。</p>',
      '<h2>轻度平滑影响较小 搜索窗口影响更明显</h2>',
      '<p>在相同条件集合上，position从3点改为5点平滑，20个任务×侧别×模型汇总的绝对lag差变化中位数约0.007秒、最大约0.059秒；改为不平滑时最大约0.095秒。相应曲线误差最大变化约0.0039和0.0066。这支持当前position描述性结果对轻度平滑较稳定，不需要以低likelihood剔除作为本次结论的前提。</p>',
      '<p>velocity对滤波选择更敏感：SG5改为box3或SG7时，汇总绝对lag差变化中位数约0.132/0.172秒，最大约0.418/0.423秒。应继续把velocity作为补充结果；这里反映分析定义的敏感性，不能单凭这些变化归因为DLC定位错误。</p>',
      table(changes,['signal','branch','median_curve_change','max_curve_change','median_lag_mae_change_s','max_lag_mae_change_s'],{'signal':'信号','branch':'敏感性分支','median_curve_change':'曲线误差变化中位数','max_curve_change':'曲线误差最大变化','median_lag_mae_change_s':'lag MAE变化中位数(s)','max_lag_mae_change_s':'lag MAE最大变化(s)'}),
      '<p><strong>pull through and cinch的边界问题没有被扩窗解决。</strong>右侧GT组峰从±2秒窗口的+2秒移动到±4秒窗口的+4秒，仍然触边；左侧从约−1.927秒移到−4秒，且相关很弱。这是尚未完整定位的搜索区间最大值，不应写成“真实领先4秒”。</p>',
      '<p>将±4秒计算结果重新限制到原±2秒域，lag统计完全恢复、曲线误差差异仅约10⁻¹⁶。这排除了本次扩窗变化由数值内核漂移造成的解释，提示搜索域与峰选择本身是重要敏感性来源。边缘裁剪分支可能改变时间样本；共同条件不等于完全相同的时间mask。来源：'+link('sensitivity_summary.csv')+'、'+link('window4_restricted2_pairs.csv')+'。</p>',
      '<h2>可定位率是操作性诊断 不是生理现象的否定</h2>',
      f'<p>在{len(gt_complete)}个主GT条件中，只有{gt_local}个满足当前严格操作性可定位规则；{len(valid)}条模型配对中，双方均满足规则的只有{both}条。规则同时要求足够重复、内部峰、较窄重采样范围、无明显竞争峰及留一/分半稳定性，阈值尚未经过生理或外部数据验证。因此这一低比例不能解释成“其余条件没有领先”，也未被用来删除主表中的条件。</p>',
      '<p>某些C条件只有1次occurrence，无法估计记录内重复性。即使被操作性规则标为可定位，也仍应检查完整分布：例如User6_1_2左侧pull through and cinch的重采样上端达到+2秒，不能无视窗口限制而宣称得到精确时间。组峰相对稳定与单条记录峰难以定位也并不矛盾，它们是不同层级的统计量。</p>',
      fig(root,'peak_distributions.png','GT峰位置分布示例。多峰、边界与重复不足分别呈现，不能统称为同一种噪声。'),
      '<p>仿真进一步显示，小样本bootstrap峰范围不保证名义覆盖：参与者数3的已知正/零延迟场景覆盖约78%/81%，参与者数5时约92%/95%；独立有色噪声中，仅凭峰范围判方向仍有约11%–15%的误判。它们是有限模拟场景的诊断，不是实际数据错误率的直接估计。来源：'+link('validation_suite/participant_simulations.csv')+'。</p>',
      '<h2>可用于研究结论的表述</h2>',
      '<blockquote>在固定左右器械、保留DLC坐标并进行轻度平滑的分析中，GT gaze与PSM的位置相关呈阶段和侧别特异性。Task A右侧transfer ring显示较一致的正向组峰，而模型在该条件下的组峰相对GT提前移向零或负lag，提示未完整复现GT的领先结构。模型整体相关曲线仍保留一定参与者相关结构，但单点峰位置对窗口、峰形和重复数量敏感。现有样本不足以建立五模型的显著优劣排序。</blockquote>',
      '<p>尚不能据此声称：gaze普遍领先全部器械或阶段；相关峰证明因果或预测能力；边界值就是真实延迟；后验平移后的结果是未经校准模型的表现；未显著差异意味着模型等效。</p>',
      '<h2>修订与核验</h2>',
      '<p>指定排除的阶段已在曲线计算及donor参照之前移除；活动图表、统计表与充分统计缓存按相同范围重新生成。它此前没有进入有效主比较，所以其他阶段主指标不因此次删除而改变。原始视频、gaze、DLC和prediction未修改。</p>',
      '<p>数值核验、阶段移除审计、版本与输入哈希见 '+link('validation.json')+'、'+link('stage_removal_audit.json')+'、'+link('run_manifest.json')+'。完整技术图表入口：'+link('report.html','技术结果报告')+'。本报告的数值来自同目录表格，图像内嵌可离线阅读。</p>'])
    css='body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:38px auto;padding:0 24px;color:#182334;line-height:1.8}h1{font-size:32px;line-height:1.4}h2{font-size:23px;border-top:1px solid #dce3ec;padding-top:22px;margin-top:35px}h3{font-size:18px}.subtitle,.source,figcaption{color:#64748b}.lead{padding:16px 22px;background:#eff6ff;border-left:4px solid #2563eb}.note{padding:12px 18px;background:#f8fafc}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}th,td{padding:7px 10px;border-bottom:1px solid #e2e8f0;text-align:right}th{background:#f1f5f9}td:first-child{text-align:left}img{width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:13px}blockquote{padding:15px 22px;border-left:4px solid #64748b;background:#f8fafc;margin:20px 0}a{color:#1d4ed8}'
    document='<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaze与PSM时序分析报告</title><style>'+css+'</style></head><body>'+''.join(body)+'</body></html>'
    (root/'analysis_report.html').write_text(document,encoding='utf-8')
    evidence=dict(n_primary_conditions=n_conditions,n_model_pairs=len(valid),gt_operationally_localizable=gt_local,both_operationally_localizable=both,
                  delay_medians=delay.set_index('model').median_delay_s.to_dict(),holm_p_min=float(comparisons.p_holm.min()),
                  no_excluded_stage_in_active_statistics=True,interpretation='exploratory; not calibrated confidence or causal inference')
    (root/'analysis_report_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    return root/'analysis_report.html'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(main(a.output))
