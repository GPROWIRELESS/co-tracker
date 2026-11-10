"""Build the Chinese GT84 report from saved, independently auditable tables."""
from pathlib import Path
import argparse
import html
import json
import math
import pandas as pd

METHODS = {'between_participants':'参与者间 Spearman','within_participant':'参与者内秩关联','within_adjusted_order':'参与者内秩关联（调整顺序）'}
SCENARIOS = {'primary':'主分析范围','completed_only':'仅 Completed','both_in_stage':'Both in stage','velocity':'速度信号','exclude_boundary':'排除边界峰','lag_identifiable_exploratory':'可定位时滞探索'}
STAGE_NAMES = {'reach ring':'reach ring','grab ring':'grab ring','transfer ring':'transfer ring','place ring':'place ring','drop ring':'drop ring','reach string':'reach string','threading string through tower':'threading string through tower','wrap around psm':'wrap around psm','pull through and cinch':'pull through and cinch'}
STATUS = {'ok':'可估计','insufficient_users':'有效参与者不足 6 人','no_predictor_variation':'指标无变异','no_score_variation':'Score 无变异','insufficient_users_with_within_variation':'有个体内变异的参与者不足 6 人','insufficient_residual_variation':'残差变异不足','insufficient_residual_degrees_of_freedom':'残差自由度不足','degenerate_cluster_standard_error':'聚类标准误退化','no_eligible_records':'无有效录制'}

def esc(value):return html.escape(str(value),quote=True)

def finite(x):
    try:return math.isfinite(float(x))
    except (TypeError,ValueError):return False

def fmt(x,n=3):
    if not finite(x):return '—'
    x=float(x)
    if x!=0 and abs(x)<.001:return f'{x:.2e}'
    return f'{x:.{n}f}'

def intval(x):return str(int(float(x))) if finite(x) else '—'

def truth(x):return str(x).lower() in ['true','1','1.0']

def table(headers,rows,classes=''):
    return '<div class="table-wrap"><table class="'+classes+'"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def ci(row):
    if not finite(row.get('ci_low')):return '—'
    mark='' if truth(row.get('ci_reliable')) else ' †'
    return f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]{mark}"

def association_table(df,include_method=False,include_scenario=False):
    headers=(['场景'] if include_scenario else [])+['任务','阶段','侧']+(['方法'] if include_method else [])+['录制数','人数','关联系数','95% 重抽样区间','p','家族 q','逐人剔除范围','状态']
    rows=[]
    for r in df.to_dict('records'):
        cells=([esc(SCENARIOS.get(r['scenario'],r['scenario']))] if include_scenario else [])+[esc(r['task']),esc(r['stage']),'左' if r['side']=='l' else '右']
        if include_method:cells.append(esc(METHODS.get(r['method'],r['method'])))
        q=fmt(r.get('q_bh')); q='<strong>'+q+'</strong>' if finite(r.get('q_bh')) and r['q_bh']<.05 else q
        loo=f"[{fmt(r.get('loo_min'))}, {fmt(r.get('loo_max'))}]" if finite(r.get('loo_min')) else '—'
        cells += [intval(r.get('n_recordings')),intval(r.get('n_users')),fmt(r.get('association')),ci(r),fmt(r.get('p')),q,loo,esc(STATUS.get(r['status'],r['status']))]
        rows.append(cells)
    return table(headers,rows)

def figure(root,stem,title,caption):
    rel=f'figures/{stem}.png'
    if not (root/rel).exists():return '<p class="pending">图表待生成：'+esc(title)+'</p>'
    return f'<figure><a href="{rel}"><img src="{rel}" alt="{esc(title)}" loading="lazy"></a><figcaption><strong>{esc(title)}</strong> · {esc(caption)} <a href="figures/{stem}.pdf">PDF</a></figcaption></figure>'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();root=args.root
    a=pd.read_csv(root/'tables/associations.csv');c=pd.read_csv(root/'tables/coverage.csv');scores=pd.read_csv(root/'tables/matched_scores.csv')
    protocol=json.loads((root/'analysis_protocol.json').read_text(encoding='utf-8'))
    prov=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    p=a[(a.scenario=='primary')&(a.predictor=='peak_r')&(a.method=='between_participants')]
    rz=a[(a.scenario=='primary')&(a.predictor=='r_zero')&(a.method=='between_participants')]
    good=p[p.status=='ok'];sig=good[good.q_bh<.05]
    all_sig=a[(a.status=='ok')&(a.q_bh<.05)]
    within=a[(a.scenario=='primary')&(a.predictor=='peak_r')&(a.method=='within_participant')]
    adjusted=a[(a.scenario=='primary')&(a.predictor=='peak_r')&(a.method=='within_adjusted_order')]
    primary_cover=c[(c.scenario=='primary')&(c.predictor=='peak_r')]
    n_scored=int(scores.score_usable.map(truth).sum());n_users=scores.user.nunique()
    score_rows=[]
    for t,g in scores.groupby('task'):
        score_rows.append([esc(t),str(len(g)),str(g.user.nunique()),str(int(g.score_usable.map(truth).sum())),str(int((g.Score==0).sum())),str(int((g.AttemptStatus=='Failed').sum())),esc('；'.join(g.loc[~g.score_usable.map(truth),'content_id']) or '无')])
    lead=(f'主分析中，{len(good)} 个可估计的阶段/侧别条件有 {len(sig)} 个通过预设家族内 BH 校正。' if len(sig) else f'主分析的 {len(good)} 个可估计阶段/侧别条件均未通过预设家族内 BH 校正（q&lt;0.05）。')
    lead+=' 这是这批 NPY gaze、固定左右侧录制的探索性证据；不能据此证明 gaze–PSM 相关强度与评分完全无关。' if not len(sig) else ' 这些结果表示统计关联，不能推断因果关系，也不能解释为个体技能的确定性指标。'
    unadj_sig=within[(within.status=='ok')&(within.q_bh<.05)];adj_sig=adjusted[(adjusted.status=='ok')&(adjusted.q_bh<.05)]
    best=good.sort_values('p').head(1)
    strongest=''
    if len(best):
        r=best.iloc[0];strongest=f"在主分析中，最小 p 对应 {esc(r.task)} / {esc(r.stage)} / {'左' if r.side=='l' else '右'}：Spearman ρ={fmt(r.association)}，95% 重抽样区间 {ci(r)}，p={fmt(r.p)}，q={fmt(r.q_bh)}。这行用于定位结果，完整的全部预设条件列在下表。"
    content=[]
    content.append(f'<header><div class="eyebrow">GT84 · 20261106 NPY gaze · 固定左右侧</div><h1>gaze–PSM 相关强度与评分的关联</h1><p class="subtitle">分析日期 {esc(protocol["analysis_date"])} · 独立补充分析 · 结果不覆盖旧版 max 选侧分析</p></header>')
    content.append('<nav><a href="#summary">结论</a><a href="#data">数据</a><a href="#primary">主结果</a><a href="#within">个体内结果</a><a href="#sensitivity">稳健性</a><a href="#methods">方法</a><a href="#files">文件</a></nav>')
    content.append(f'<section id="summary"><h2>主要结论</h2><div class="callout">{lead}</div><div class="cards"><div><b>{len(scores)}</b><span>原始录制</span></div><div><b>{n_scored}</b><span>有可用总分</span></div><div><b>{n_users}</b><span>独立参与者</span></div><div><b>{len(sig)} / {len(good)}</b><span>主分析家族内 q&lt;0.05</span></div></div><p>{strongest}</p>')
    content.append(f'<p>同一参与者内，主范围的 peak_r 未调整模型有 {len(unadj_sig)} 个家族内 q&lt;0.05 条件；调整任务顺序与尝试次数后有 {len(adj_sig)} 个。个体间与个体内结果回答不同问题，分别报告，不将其当成相互独立的重复发现。参与者内推断仅有 {n_users} 个聚类，属于近似推断。</p><p>补充及敏感性分析出现 {len(all_sig)} 行家族内 q&lt;0.05 结果，主要涉及重复出现的 Task A 右侧 transfer ring 个体内结果，以及 Task C 右侧 reach string 的速度零时滞相关。它们不等于 {len(all_sig)} 个独立发现，详见后文；没有跨全部家族的整体 FDR 声明。</p></section>')
    content.append('<section id="data"><h2>数据来源、匹配与可用性</h2><p>本次使用 20261106 gt84 的已保存逐录制指标。gaze 来自各 User 的 <code>npy/&lt;record&gt;_gaze.npy</code>，接受两个坐标均有限且位于归一化闭区间 [0,1] 的点，不根据 stored valid 或眼动 CSV 的 Status 筛选；左右 PSM 分开分析。原始 xcorr 未重新计算，也未修改。</p>')
    content.append(table(['任务','录制数','参与者数','可用 Score','零分数','Failed 数','无可用 Score'],score_rows))
    content.append('<p>匹配键为 <code>User{UserID}_{TaskNum}_{Attempt}</code>，同时核对 task、user、TaskNum、Attempt、UserType 与 content_index。84 条录制名单与旧版一致，Score 和完成状态也与旧匹配表一致。<code>User8_7_2</code> 为 NotAttempted 且 Score 为空；保留审计记录，排除评分关联，未填零。失败尝试中已有的零分保留；Completed 中也有零分，因此仅 Completed 的敏感性分析不能消除 Task C 的地板效应。</p>')
    content.append('<p>Score 是整次 attempt 的总分，并非阶段评分。各阶段沿用同一录制的 Score 不会增加独立样本。84 条录制来自 11 人；每个分析条件内先处理同一人的重复录制，不能按 84 个独立参与者解读。</p>')
    content.append(figure(root,'01_score_distribution','评分分布与完成状态','显示真实零分、缺失评分和完成状态；这是阶段筛选前的覆盖。'))
    content.append(table(['项目','旧版 score_corr_max_dlc','本次补充'],[
        ['gaze 来源','原始 gaze CSV，Status=2','20261106 NPY 坐标规则'],
        ['器械侧别','每次录制、每阶段选择较高峰的一侧','固定左侧与右侧分别分析'],
        ['强度分析筛选','正的内部峰及旧质量规则','有效 Score、有限指标、完整 lag 支持；保留有效非正相关'],
        ['已有结果的使用','历史分析与审计','只复用身份/Score 核对与源分数疑点标记，不复用旧关联系数']]))
    content.append('<p>旧版曾标记 Task C 部分 TimeRaw 与视频时长差异，以及 User6_1_2 / User8_6_1 的若干评分分量重复。这些标记沿用到匹配审计表，不足以证明身份匹配错误，本次未据此改分、重配或删除录制。总 Score 为本次唯一结局；未把 Time/TimeRaw 或 Eco/EcoRaw 的近似反向映射计为新的验证证据。</p></section>')
    content.append('<section id="primary"><h2>主分析：位置 peak_r 与 Score</h2><p>范围为 <code>variant=primary</code>、position、PSM in stage。A/C、阶段和左右侧各自估计；预设 18 个条件。每位参与者的相关指标用 Fisher-z 平均后反变换，Score 用同一批有效录制的算术平均。图中一点代表一人，关联检验为参与者间 Spearman。</p>')
    content.append(association_table(p))
    content.append('<p class="note">主分析家族：primary / peak_r / between_participants，A/C、全部阶段及左右侧一起校正，仅纳入可计算的 p 值。表中“—”为不可估计或不适用，不能解释为 0 或不显著。† 表示有效 bootstrap 比例低于 97.5%，该区间不可靠。</p>')
    content.append(figure(root,'05_forest_primary_peak_r','peak_r 关联总览','展示主范围下全部条件的参与者间、参与者内及顺序调整后估计；三列属于不同检验家族。'))
    content.append(figure(root,'03_scatter_A_peak_r','Task A 参与者级散点','横轴为参与者平均相关强度，纵轴为同批有效录制的平均 Score；所有预设阶段和左右侧均展示。'))
    content.append(figure(root,'03_scatter_C_peak_r','Task C 参与者级散点','Task C 总分集中于零；图中覆盖随阶段有效性而变化。'))
    content.append('<details><summary>查看主分析各条件的录制覆盖</summary>')
    content.append(table(['任务','阶段','侧','候选录制','有可用 Score','有效录制','人数','有效非正指标','有效零分','边界峰'],[[esc(r['task']),esc(r['stage']),'左' if r['side']=='l' else '右',intval(r['n_total_records']),intval(r['n_score_available']),intval(r['n_eligible_records']),intval(r['n_users']),intval(r['n_nonpositive_metric']),intval(r['n_zero_scores']),intval(r['n_boundary'])] for r in primary_cover.to_dict('records')]))
    content.append('<p><a href="figures/02_primary_coverage.pdf">覆盖图 PDF</a> · <a href="tables/coverage.csv">全部场景覆盖表</a></p></details></section>')
    content.append('<section id="zero"><h2>补充指标：零时滞相关 r_zero</h2><p>峰值相关受到搜索窗口内最大值选择的影响。r_zero 用于检查关联是否依赖峰值搜索；它与 peak_r 回答相近但不相同的问题。其结果为补充分析，并使用独立、预先定义的检验家族。</p>')
    content.append(association_table(rz))
    rr=a[(a.scenario=='primary')&(a.predictor=='r_zero')&(a.task=='A')&(a.stage=='transfer ring')&(a.side=='r')]
    un=rr[rr.method=='within_participant'].iloc[0];ad=rr[rr.method=='within_adjusted_order'].iloc[0]
    content.append(f'<p>Task A / transfer ring / 右侧的位置 r_zero 在未调整的参与者内模型中为 {fmt(un.association)}（q={fmt(un.q_bh)}）；调整顺序后为 {fmt(ad.association)}（q={fmt(ad.q_bh)}）。这表明该补充结果不支持独立于任务顺序的稳定个体内关联。由于 Task A 全部录制均 Completed，Completed-only 重复出现这一结果并不是独立复制。</p>')
    content.append(figure(root,'05_forest_primary_r_zero','r_zero 关联总览','全部固定侧条件与三种方法；不能将不同方法或补充指标的显著行数当成独立发现数。'))
    content.append('</section><section id="within"><h2>同一参与者内的变化与顺序调整</h2><p>参与者内模型先对每个条件的原始指标与 Score 作总体秩转换，再吸收参与者固定效应。每人的全部有效录制总权重为 1。未调整模型描述同一人的相对变化；调整模型另控制 TaskNum 和 Attempt，以检查关联是否可由任务顺序或重复尝试解释。该系数称为参与者内秩关联（调整后为部分秩关联），不是普通 Spearman 系数。</p>')
    within_rows=[]
    for _,r in within.iterrows():
        adj=adjusted[(adjusted.task==r.task)&(adjusted.stage==r.stage)&(adjusted.side==r.side)].iloc[0]
        within_rows.append([esc(r.task),esc(r.stage),'左' if r.side=='l' else '右',intval(r.n_informative_users),fmt(r.association),fmt(r.q_bh),fmt(adj.association),fmt(adj.q_bh),esc(STATUS.get(adj.status,adj.status))])
    content.append(table(['任务','阶段','侧','有个体内变异的人数','未调整系数','未调整 q','调整后系数','调整后 q','调整后状态'],within_rows))
    content.append('<p>Task A 的 Attempt 恒为 1，因此不能提供额外调整信息；实际设计秩用于自由度计算，TaskNum 顺序仍被调整。调整前后系数改变说明结论依赖分析条件，不能证明练习导致了全部关联。</p><p>置信区间由整位参与者重抽样得到，参与者内 bootstrap 保留原全样本秩，是条件性的探索区间。p 值采用参与者聚类 CR1 修正及 t(G−1) 近似；只有 11 人时可能不稳定。逐人剔除则重新排名、重新拟合，以显示对单个参与者的敏感程度。</p>')
    content.append('<p><a href="tables/within_points.csv">全部主范围残差绘图数据</a> · <a href="tables/leave_one_user_out.csv">逐人剔除结果</a> · <a href="figures/all_figures.pdf">查看残差图全集</a></p></section>')
    content.append('<section id="sensitivity"><h2>敏感性分析与结果边界</h2>')
    content.append(table(['场景','规则','解读'],[
        ['仅 Completed','保留已完成的有效录制','改变目标样本；保留真实零分'],
        ['Both in stage','两端时间均属于同一次阶段 occurrence','检查阶段边界定义的影响'],
        ['速度信号','使用已有 velocity 指标','补充信号；不能与位置结果合成为统一结论'],
        ['排除边界峰','排除 peak_at_boundary=True','检查最大值受窗口边界影响；会改变纳入样本'],
        ['逐人剔除','每次删除一位参与者后重估','检验单个参与者影响；不是额外独立假设检验']]))
    family_rows=[]
    for (scenario,predictor,method),g in a[a.scenario!='lag_identifiable_exploratory'].groupby(['scenario','predictor','method'],sort=False):
        valid=g[g.status=='ok']; significant=valid[valid.q_bh<.05]
        family_rows.append([esc(SCENARIOS.get(scenario,scenario)),esc(predictor),esc(METHODS[method]),str(len(valid)),str(len(significant)),fmt(valid.p.min()),fmt(valid.q_bh.min())])
    content.append('<details open><summary>全部强度分析家族的可估计性与校正结果</summary>'+table(['场景','指标','方法','可估计条件','家族 q<0.05 行数','最小 p','最小 q'],family_rows)+'</details>')
    content.append('<p>每个 scenario × predictor × method 单独建立 BH 家族，跨 A/C、全部阶段和左右侧。没有跨所有家族的整体 FDR 声明。不同家族显著行数不能相加作为独立发现数，敏感性分析也不能替代预设主分析。Bootstrap 百分位区间与置换检验/聚类 t 检验并非同一种推断，区间是否包含零不能替代 p 值及 BH 校正后的结论。</p>')
    velocity=a[(a.scenario=='velocity')&(a.predictor=='r_zero')&(a.method=='between_participants')&(a.task=='C')&(a.stage=='reach string')&(a.side=='r')].iloc[0]
    content.append(f'<p>速度补充分析中，Task C / reach string / 右侧 r_zero 的参与者间 Spearman ρ={fmt(velocity.association)}，区间 {ci(velocity)}，p={fmt(velocity.p)}，家族 q={fmt(velocity.q_bh)}；逐人剔除范围 [{fmt(velocity.loo_min)}, {fmt(velocity.loo_max)}]。这是特定信号、阶段和侧别的补充关联，不改变位置 peak_r 主分析未通过校正的结论。</p>')
    if len(all_sig):
        content.append('<details><summary>查看所有家族内 q&lt;0.05 行（含补充与敏感性）</summary><p>这些行仅用于完整列示，不代表全局校正后的独立发现；指标见结果表。</p>')
        rows=[]
        for r in all_sig.to_dict('records'):
            rows.append([esc(SCENARIOS[r['scenario']]),esc(r['predictor']),esc(r['task']),esc(r['stage']),'左' if r['side']=='l' else '右',esc(METHODS[r['method']]),fmt(r['association']),ci(r),fmt(r['p']),fmt(r['q_bh'])])
        content.append(table(['场景','指标','任务','阶段','侧','方法','系数','95% 区间','p','家族 q'],rows)+'</details>')
    else:content.append('<p>本次全部预设强度分析家族均无 q&lt;0.05 的结果行。无显著结果不构成零关联的证明。</p>')
    content.append('<p><a href="tables/family_summary.csv">家族统计表</a> · <a href="tables/family_significant_results.csv">家族内显著结果表</a> · <a href="figures/all_figures.pdf">敏感性森林图见图表全集</a></p>')
    lag=a[a.scenario=='lag_identifiable_exploratory'];lagc=c[(c.scenario=='lag_identifiable_exploratory')&(c.predictor=='signed_lag_s')]
    content.append(f'<h3>时滞探索</h3><p>时滞只纳入原 GT84 规则标为 lag_identifiable 的录制，再要求可用 Score 与完整 lag 支持。各条件最多 {intval(lagc.n_users.max())} 名参与者；{len(lag[lag.status=="ok"])} 个方法/条件组合可估计。人数或变异不足时明确标为不可估计，不放宽门槛获取 p 值。正时滞在原配对定义中表示较早 gaze 对应当前 PSM，但不能解释为因果领先。</p><p><a href="tables/lag_exploratory_associations.csv">时滞探索结果与不可估计原因</a></p></section>')
    content.append('<section id="methods"><h2>方法细节与局限</h2><ol><li><strong>纳入规则：</strong>读取新版 primary 分支、GT、固定左右侧。要求有限 Score、Completed/Failed 状态、有限指标以及完整 lag 支持。相关强度分析保留零/负相关，不要求正峰、内部峰或峰位可定位。DLC likelihood、跳变等只作来源质量标记，不增加排除阈值。</li><li><strong>参与者间统计：</strong>相关强度在参与者内以 Fisher z 等权平均，Score 在相同录制上求均值；对参与者汇总计算 Spearman，至少 6 人。双侧 p 值通过 19,999 次参与者评分置换及 +1 修正得到。参与者级 bootstrap 10,000 次，每次重抽后重新计算 Spearman 秩。</li><li><strong>参与者内统计：</strong>总体秩转换、参与者固定效应、每人总权重相同；至少 6 位参与者同时具有指标和 Score 的个体内变异。调整 TaskNum/Attempt 的模型使用实际设计秩。CR1 聚类标准误与 t(G−1) 属小样本近似；bootstrap 整人重抽，保留原始全样本秩。所报区间为探索性百分位区间，不是生理时延的置信区间。</li><li><strong>多重比较：</strong>每个场景、指标、方法独立校正，家族覆盖 A/C、阶段与左右侧。不可估计条件仍列出，但不把缺失 p 值当 0 或 1 混入 BH。该选择控制可估计家族范围，不声明跨全部探索的整体 FDR。</li><li><strong>有限样本与缺失：</strong>仅 11 人；零分集中、阶段缺失及不同条件采用不同录制集合，会限制可比性与检验功效。Completed-only 与排除边界峰会改变样本。没有进行因果建模或个体评分预测验证。</li><li><strong>最大峰选择：</strong>peak_r 是固定搜索窗内的最大有符号相关，本身受搜索选择影响。固定左右侧减少了旧版再次选择器械侧的影响，但未消除最大 lag 选择；因此配套报告 r_zero 与边界峰敏感性。</li></ol>')
    content.append('<h3>计算核验与来源保护</h3>')
    validation_path=root/'validation/independent_validation.json'
    if validation_path.exists():
        validation=json.loads(validation_path.read_text(encoding='utf-8'))
        content.append(f'<p>独立核验结论：<strong>{"通过" if validation.get("passed") else "未通过，请检查记录"}</strong>。核验针对评分匹配、筛选、参与者汇总、关联点估计、CR1 标准误、BH 及来源指纹；不等同于独立重跑全部 Monte Carlo 抽样。</p><p><a href="validation/independent_validation.json">独立核验结果 JSON</a></p>')
        content.append('<details><summary>核验摘要</summary><pre>'+esc(json.dumps(validation,ensure_ascii=False,indent=2))+'</pre></details>')
    else:content.append('<p class="pending">独立核验正在生成，发布前将重建此报告并链接验证记录。</p>')
    unchanged=prov.get('input_unchanged',{})
    content.append(f'<p>分析器记录了 {len(unchanged)} 个输入及旧版关键结果文件的 SHA-256，计算结束后全部一致：<strong>{"是" if unchanged and all(unchanged.values()) else "请检查 provenance"}</strong>。源指标与原评分未修改。运行使用 Python {esc(str(prov.get("python","" )).split()[0])}、NumPy {esc(prov.get("numpy",""))}、SciPy {esc(prov.get("scipy",""))}、pandas {esc(prov.get("pandas",""))}；随机种子 {protocol["seed"]}。完整参数见 <a href="analysis_protocol.json">分析协议</a>，文件指纹见 <a href="provenance.json">来源记录</a>。</p></section>')
    content.append('<section id="files"><h2>交付文件</h2>')
    content.append(table(['文件','内容'],[
        ['<a href="tables/associations.csv">associations.csv</a>','全部场景、指标、方法和预设条件的结果，包括不可估计原因'],
        ['<a href="tables/primary_peak_r_associations.csv">primary_peak_r_associations.csv</a>','预设主分析家族的结果'],
        ['<a href="tables/matched_scores.csv">matched_scores.csv</a>','84 条录制、原 Score、完成状态、匹配与源评分疑点标记'],
        ['<a href="tables/matched_record_metrics.csv">matched_record_metrics.csv</a>','逐录制指标与评分长表，逐场景/指标纳入理由'],
        ['<a href="tables/participant_summaries.csv">participant_summaries.csv</a>','每个条件的参与者汇总及纳入录制列表'],
        ['<a href="tables/coverage.csv">coverage.csv</a>','各条件的候选、可评分、有效录制及人数'],
        ['<a href="tables/leave_one_user_out.csv">leave_one_user_out.csv</a>','逐人剔除后重新估计的系数'],
        ['<a href="tables/within_points.csv">within_points.csv</a>','主范围参与者内/顺序调整模型的残差图数据'],
        ['<a href="figures/all_figures.pdf">all_figures.pdf</a>','全部 PNG/PDF 图表对应的合并 PDF'],
        ['<a href="figure_manifest.json">figure_manifest.json</a>','图表清单与各图输入'],
        ['<a href="analysis_protocol.json">analysis_protocol.json</a> / <a href="provenance.json">provenance.json</a>','分析口径、软件、随机种子及来源指纹'],
        ['<a href="REPRODUCE.txt">REPRODUCE.txt</a>','复现步骤、独立输出目录和核验说明'],
        ['<a href="inputs/record_summary.csv">inputs/record_summary.csv</a>','新 GT84 逐录制指标快照；同目录保留评分、索引及清单输入快照']]))
    manifest_path=root/'figure_manifest.json'
    if manifest_path.exists():
        figures=json.loads(manifest_path.read_text(encoding='utf-8')).get('figures',[])
        content.append('<details><summary>全部单页图表</summary>'+table(['图','PNG','PDF'],[[esc(f['title']),f'<a href="figures/{esc(f["stem"])}.png">PNG</a>',f'<a href="figures/{esc(f["stem"])}.pdf">PDF</a>'] for f in figures])+'</details>')
    content.append('<p class="note">CSV 采用 UTF-8 BOM，便于 Excel 打开。空白数值表示不可用；分析没有插补缺失评分。报告与图表使用同一组已保存统计结果，绘图不重新选择样本或重算显著性。</p></section><footer>GT84 · 20261106 数据版本 · 2026-11-09 补充评分关联分析</footer>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#20313c;font:15px/1.8 "Microsoft YaHei","Segoe UI",Arial,sans-serif}main{max-width:1240px;margin:auto;padding:38px 34px 80px;background:white}header{border-top:6px solid #256f85;padding:28px 0 18px}.eyebrow{color:#256f85;font-size:13px;font-weight:700;letter-spacing:1px}h1{font-size:32px;line-height:1.4;letter-spacing:-.5px;margin:10px 0}.subtitle{color:#657681;margin:8px 0}nav{display:flex;flex-wrap:wrap;gap:9px 22px;border-block:1px solid #dce4e8;padding:12px 0;margin:10px 0 28px}a{color:#176981;text-decoration:none}a:hover{text-decoration:underline}section{margin:34px 0 46px;scroll-margin-top:15px}h2{font-size:23px;line-height:1.4;border-left:4px solid #378399;padding-left:12px;margin:0 0 18px}h3{font-size:18px;margin-top:25px}p{margin:12px 0}.callout{background:#edf5f7;border-left:4px solid #378399;padding:18px 20px;font-size:17px;line-height:1.8}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}.cards>div{padding:18px;border:1px solid #dce4e8;border-radius:7px;background:#fafcfd}.cards b{display:block;font-size:27px;color:#236b80}.cards span{color:#5d707b;font-size:13px}.table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #dce4e8;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6}th{background:#edf2f5;white-space:nowrap;font-weight:600;color:#284754;text-align:left;padding:10px 11px;border-bottom:1px solid #cddbe2}td{padding:9px 11px;border-bottom:1px solid #e7edef;vertical-align:top}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcfd}td strong{color:#9e4b24}code{background:#eef2f4;padding:2px 5px;border-radius:3px;font-size:12px;overflow-wrap:anywhere}figure{margin:24px 0;border:1px solid #dae4e8;border-radius:7px;overflow:hidden}figure img{display:block;width:100%;height:auto}figcaption{background:#f7f9fa;padding:11px 15px;font-size:13px;color:#5f737e}.note{font-size:13px;color:#627681}.pending{background:#fff4dd;padding:12px 15px}details{margin:18px 0;padding:12px 15px;background:#f8fafb;border:1px solid #dfe7eb;border-radius:6px}summary{cursor:pointer;font-weight:600;color:#285f72}pre{font:12px/1.5 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}li{margin:12px 0}footer{border-top:1px solid #dce4e8;padding-top:20px;color:#778791;font-size:12px}@media(max-width:700px){main{padding:20px 16px}h1{font-size:25px}.cards{grid-template-columns:repeat(2,1fr)}h2{font-size:20px}}@media print{body,main{background:white}main{max-width:none;padding:0}nav{display:none}section{break-before:auto}h2,h3{break-after:avoid}figure{break-inside:avoid}details{background:white}details>*{display:block!important}table{font-size:10px}a{color:#20313c}.cards{grid-template-columns:repeat(4,1fr)}footer{margin-top:30px}}'''
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT84 gaze–PSM corr–score 分析 | 20261106</title><style>'+css+'</style></head><body><main>'+''.join(content)+'</main></body></html>'
    (root/'report.html').write_text(document,encoding='utf-8')
    print(json.dumps({'report':str(root/'report.html'),'primary_estimable':len(good),'primary_q_lt_005':len(sig),'within_primary_q_lt_005':len(unadj_sig),'adjusted_primary_q_lt_005':len(adj_sig),'all_family_significant_rows':len(all_sig)},ensure_ascii=False))

if __name__=='__main__':main()
