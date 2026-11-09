"""Regenerate Chinese HTML, text and Markdown from coordinate-only results."""
from pathlib import Path
from collections import Counter
import argparse,csv,html,json,re,hashlib
from urllib.parse import unquote,urlsplit
import numpy as np
from bs4 import BeautifulSoup,NavigableString

ROOT=Path(r'D:\GitHub_Project\co-tracker\outputs\novice_transition_A_seed42_20261105_analysis_20261106')
NAMES={'gt':'GT','current_transition':'Current','future500_transition':'Future 500','future1000_transition':'Future 1000','future1500_transition':'Future 1500'}
LABEL={'lead':'支持领先','lag':'支持滞后','synchronous':'接近同步','undetermined':'证据不足','undefined':'未定义'}
SIDE={'l':'左','r':'右'}

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pick(data,**k):return [r for r in data if all(str(r.get(a))==str(b) for a,b in k.items())]
def one(data,**k):
    a=pick(data,**k);assert len(a)==1,(k,len(a));return a[0]
def number(v):
    try:return float(v)
    except (TypeError,ValueError):return float('nan')
def fmt(v,d=2):return f'{number(v):.{d}f}' if np.isfinite(number(v)) else '—'
def table(headers,data):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in data)+'</tbody></table></div>'
def mdtable(headers,data):
    cell=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in data])
def figure(path,caption):return f'<figure><img src="../{path}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
def p(text):return '<p>'+text+'</p>'
def lagrow(r):return f"{fmt(number(r['peak_lag_s'])*1000,0)} [{fmt(number(r['peak_lag_resample_low'])*1000,0)}, {fmt(number(r['peak_lag_resample_high'])*1000,0)}]"

def markdown(node):
    if isinstance(node,NavigableString):return re.sub(r'\s+',' ',str(node))
    if node.name in ['head','script','style']:return ''
    if node.name=='table':
        data=[[c.get_text(' ',strip=True) for c in r.find_all(['th','td'],recursive=False)] for r in node.find_all('tr')]
        assert data and all(len(r)==len(data[0]) for r in data)
        return '\n\n'+mdtable(data[0],data[1:])+'\n\n'
    if node.name=='img':
        path=(ROOT/'report'/node['src']).resolve();assert path.is_file()
        return '\n\n!['+node.get('alt','图表')+']('+path.as_posix()+')\n\n'
    text=''.join(markdown(c) for c in node.children)
    if node.name in ['h1','h2','h3']:return '\n\n'+'#'*int(node.name[-1])+' '+text.strip()+'\n\n'
    if node.name in ['strong','b']:return '**'+text.strip()+'**'
    if node.name=='code':return '`'+text.strip()+'`'
    if node.name=='a':
        ref=node.get('href','#')
        if ref.startswith('../'):ref=ref[3:]
        return '['+text.strip()+']('+ref+')'
    if node.name=='li':return '\n- '+text.strip()+'\n'
    if node.name in ['p','figure','figcaption','section','ul','ol']:return '\n\n'+text.strip()+'\n\n'
    return text

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.root.resolve()
    meta=read('input_audit/metadata.json');native=read('input_audit/native_audit.json');timeline=read('input_audit/timeline_summary.json')
    gm=rows('target_locking/group_metrics.csv');ge=rows('target_locking/group_errors.csv');gs=rows('gaze_psm_lead/group_summary.csv')
    loo=rows('gaze_psm_lead/leave_one_user_group_summary.csv');tl_loo=rows('target_locking/leave_one_participant_out.csv')
    findings=read('target_locking/findings.json');comparison=read('input_audit/comparison.json')
    models=list(meta['models']);sources=['gt']+models
    assert {r['scenario'] for r in gm}=={'main','r25','r100'},'Stale valid-filter scenario remains'
    native_valid=0;native_total=0
    for rec in meta['records']:
        with np.load(ROOT/'input_audit/native'/(rec['content_id']+'.npz'),allow_pickle=False) as z:
            count=int(np.sum(z['gaze_valid']));native_valid+=count;native_total+=len(z['gaze_valid'])
            assert count==one(native,content_id=rec['content_id'])['gaze_coordinate_valid_n']
    coordinate_n=sum(r['gaze_coordinate_valid_n'] for r in native)
    assert native_valid==coordinate_n
    spec=dict(axis='anchor',branch='probability_map',scenario='main',stage='whole')
    def metric(m,field,source='model',branch='probability_map',scenario='main',axis='anchor'):
        return one(gm,axis=axis,branch=branch,scenario=scenario,stage='whole',model=m,source=source,metric=field)
    def error(m,field,scenario='main'):
        return one(ge,axis='anchor',branch='probability_map',scenario=scenario,stage='whole',model=m,metric=field)
    gt=number(metric(models[0],'L_target','GT')['estimate'])
    best=max(models,key=lambda m:number(metric(m,'L_target')['estimate']))
    bestval=number(metric(best,'L_target')['estimate'])
    lower=all(number(error(m,'delta_L_target')['ci_high'])<0 for m in models)
    anchor_n=int(findings['common_anchor_n']);anchor_s=number(findings['common_anchor_seconds'])
    primary=pick(gs,variant='anchor__primary',mode='psm_stage')
    whole=pick(primary,stage='whole');wp=pick(whole,signal='position');wv=pick(whole,signal='velocity')
    leads=pick(primary,direction_0p10='lead');position_leads=pick(wp,direction_0p10='lead')
    gt_whole=[one(wp,source='gt',side=s) for s in ['l','r']]
    target_rows=[['GT',fmt(gt*100),fmt(number(metric(models[0],'L_psm','GT')['estimate'])*100),fmt(metric(models[0],'C','GT')['estimate']),'参照','不适用']]
    summary_target=[];decode_rows=[]
    for m in models:
        t=metric(m,'L_target');d=error(m,'delta_L_target');b=error(m,'BS_target')
        diff=f"{fmt(number(d['estimate'])*100)} [{fmt(number(d['ci_low'])*100)}, {fmt(number(d['ci_high'])*100)}]"
        target_rows.append([NAMES[m],fmt(number(t['estimate'])*100),fmt(number(metric(m,'L_psm')['estimate'])*100),fmt(metric(m,'C')['estimate']),diff,fmt(b['estimate'],4)])
        summary_target.append([NAMES[m],fmt(number(t['estimate'])*100),diff,fmt(b['estimate'],4)])
        decode_rows.append([NAMES[m]]+[fmt(number(metric(m,'L_target',branch=b)['estimate'])*100) for b in ['probability_map','decoded_expectation','decoded_argmax']])
    whole_rows=[];whole_position_rows=[]
    for source in sources:
        lr=[one(wp,source=source,side=s) for s in ['l','r']]
        whole_position_rows.append([NAMES[source],lagrow(lr[0]),LABEL.get(lr[0]['direction_0p10'],lr[0]['direction_0p10']),lagrow(lr[1]),LABEL.get(lr[1]['direction_0p10'],lr[1]['direction_0p10'])])
    for signal in ['position','velocity']:
        for side in ['l','r']:
            for source in sources:
                r=one(whole,signal=signal,side=side,source=source)
                whole_rows.append(['位置' if signal=='position' else '速度',SIDE[side],NAMES[source],fmt(number(r['peak_lag_s'])*1000,0),fmt(r['peak_r'],3),f"{fmt(number(r['peak_lag_resample_low'])*1000,0)}～{fmt(number(r['peak_lag_resample_high'])*1000,0)}",LABEL.get(r['direction_0p10'],r['direction_0p10'])])
    local_rows=[];within_rows=[];sensitivity_rows=[];stable_count=0
    for r in leads:
        criteria={k:r[k] for k in ['mode','stage','side','signal','source']}
        l=pick(loo,variant='anchor__primary',**criteria);nlead=sum(x['direction_0p10']=='lead' for x in l)
        stable_count+=bool(l) and nlead==len(l)
        local_rows.append([NAMES[r['source']],r['stage'],SIDE[r['side']],'位置' if r['signal']=='position' else '速度',lagrow(r),fmt(r['peak_r'],3),f'{nlead}/{len(l)}'])
        if r['stage']=='whole':
            within_rows.append([NAMES[r['source']],r['stage'],SIDE[r['side']],lagrow(r),'不适用','whole不定义within_stage'])
        else:
            within_criteria=criteria|{'mode':'within_stage'}
            w=one(gs,variant='anchor__primary',**within_criteria)
            within_rows.append([NAMES[r['source']],r['stage'],SIDE[r['side']],lagrow(r),lagrow(w),LABEL.get(w['direction_0p10'],w['direction_0p10'])])
        variants=['position_unsmoothed','primary','position_smooth5','trim_edges','window4'] if r['signal']=='position' else ['primary','velocity_box3','velocity_sg7','velocity_sg11','trim_edges','window4']
        settings=[one(gs,variant='anchor__'+v,**criteria) for v in variants]
        sensitivity_rows.append([NAMES[r['source']],r['stage'],SIDE[r['side']],', '.join(variants),f"{sum(x['direction_0p10']=='lead' for x in settings)}/{len(settings)}"])
    counts=[]
    for s in sources:
        c=Counter(r['direction_0p10'] for r in pick(primary,source=s));counts.append([NAMES[s]]+[c[k] for k in ['lead','synchronous','lag','undetermined']])
    targetlead=[];target_ranges=[]
    for m in models:
        mr=[]
        for side in ['l','r']:
            q=pick(gs,variant='target_'+m+'__primary',mode='psm_stage',stage='whole',signal='position',side=side)
            a=one(q,source=m);b=one(q,source='gt');mr.append(a)
            targetlead.append([NAMES[m],SIDE[side],lagrow(a),LABEL.get(a['direction_0p10'],a['direction_0p10']),lagrow(b),LABEL.get(b['direction_0p10'],b['direction_0p10'])])
        if meta['models'][m]>0:target_ranges.append(f"{NAMES[m]} 左/右为 {fmt(number(mr[0]['peak_lag_s'])*1000,0)}/{fmt(number(mr[1]['peak_lag_s'])*1000,0)} ms")
    all_wp_uncertain=all(r['direction_0p10']=='undetermined' for r in wp)
    pos_reason=Counter(r['group_reliability_reasons'] for r in wp if r['direction_0p10']=='undetermined')
    pos_summary='所有合并位置曲线均未通过可靠领先规则。' if all_wp_uncertain else f"合并位置的 {len(wp)} 个条件中有 {len(position_leads)} 个被分类为领先；逐项判定见表。"
    if all_wp_uncertain and all('broad_near_peak' in r['group_reliability_reasons'] for r in wp):pos_summary+='共同原因是近峰过宽，不能仅凭正峰位或正重采样范围认定可精确定位的领先。'
    key1=f'GT 的目标区域占用为 {gt*100:.2f}%；模型中 {NAMES[best]} 的目标概率最高，为 {bestval*100:.2f}%。'
    key2='四模型的目标占用差及其参与者配对重采样区间均低于零。' if lower else '各模型与GT的配对差及其区间见表，部分区间没有完全位于零以下。'
    key3=f"四阶段合并的 GT 位置峰为左 {fmt(number(gt_whole[0]['peak_lag_s'])*1000,0)} ms、右 {fmt(number(gt_whole[1]['peak_lag_s'])*1000,0)} ms。{pos_summary}主分析 {len(primary)} 个来源×条件组合中有 {len(leads)} 个满足探索性 lead 规则，其中 {stable_count} 个在逐一剔除每位参与者后仍为 lead。"
    policy=f'本目录已按新规则原地重算：GT来自各User的 <code>npy/*_gaze.npy</code>，归一化坐标x、y均有限且位于闭区间[0,1]即保留，<strong>不使用valid或Status筛选</strong>。有效原生点为 {native_valid:,}/{native_total:,}。NaN/Inf作为缺失点；valid只保留为来源诊断。'
    policy+='原始模型预测、视频帧和时间轴沿用原来源，未重新训练；训练日志中的future_valid、验证样本数等是原训练诊断，不是本次分析GT筛选。原状态子集敏感性已移除，保留半径和时序预处理敏感性。'
    parts=[]
    def section(title,body):parts.append(f'<section><h2>{title}</h2>{body}</section>')
    section('主要结果与本次数据口径',p(policy)+p('<b>目标锁定：</b>'+key1+' '+key2)+p('<b>Lead PSM：</b>'+key3)+p('这些结果来自验证集上的探索性比较，验证集参与过checkpoint选择。预测horizon不等于实际测得的器械提前量。'))
    sample_rows=[[r['content_id'],r['participant'],r['native_n'],one(timeline,content_id=r['content_id'])['prediction_n']] for r in meta['records']]
    model_rows=[[NAMES[m],m,meta['models'][m],comparison[m]['best_checkpoint'].split('/')[-1]] for m in models]
    total_anchors=sum(int(r['prediction_n']) for r in timeline)
    section('1．模型、样本与共同支持',table(['来源','模型目录','horizon / ms','checkpoint'],model_rows)+table(['录像','参与者','原生帧数','每模型预测锚点'],sample_rows)+p(f"每模型共有 {total_anchors:,} 个相同锚点。Target locking 主分析共同有效锚点为 {anchor_n:,} 个，时间权重合计 {anchor_s:.2f} 秒。不同分析的几何和运动支持要求不同，不能共用一个分母。")+p('每录像先汇总，参与者内对录像等权，再参与者间等权。User5的两段录像先在该参与者内汇总，不当作两个人。')+p('Gaze坐标来自每个User的NPY；video_ts.csv按原始Frame_Number提供精确录制时间。原始眼动CSV仅用于来源诊断。分析包含reach、grab、transfer、place ring四阶段，排除drop ring、none和空阶段。'))
    best_decode=[number(metric(best,'L_target',branch=b)['estimate'])*100 for b in ['probability_map','decoded_expectation','decoded_argmax']]
    section('2．Target locking：概率分配与坐标解码',table(['来源','目标占用 %','PSM占用 %','目标偏好 C','目标占用−GT 百分点 [95%重采样范围]','目标Brier'],target_rows)+p('模型值为热图落入区域的概率质量；GT为注视点落入区域的时间占比。Target为有效标注点均值周围50 px圆盘；PSM为左右DLC点50 px圆盘并集。T/P/O/N分别表示目标独占、PSM独占、重叠、其他。目标占用=T+O，PSM占用=P+O，C=T/(T+P)，须结合独占占用D=T+P及绝对占用解读。')+p('32×40热图按网格格内均匀概率质量的像素面积积分。Brier以GT区域二元事件为参照，越低越好；不等于完整gaze分布误差。范围来自10,000次参与者配对bootstrap，属于探索性估计。')+figure('target_locking/figures/tl_anchor_occupancy.png','共同锚点支持上的GT与四模型区域占用。')+figure('target_locking/figures/tl_anchor_paired_differences.png','模型减去同录像GT的参与者配对差。')+figure('target_locking/figures/tl_stage_heatmap.png','分阶段区域占用。')+table(['模型','热图目标概率 %','期望点目标占用 %','argmax目标占用 %'],decode_rows)+p(f"以{NAMES[best]}为例，热图目标概率为{best_decode[0]:.2f}%，期望点目标占用为{best_decode[1]:.2f}%，argmax目标占用为{best_decode[2]:.2f}%；GT为{gt*100:.2f}%。不同表示回答不同问题，不能把热图概率当作所有解码方式的命中率。总体占用接近也不证明逐帧预测正确或统计等效。")+figure('figures/tl_decoding_occupancy_zh.png','同一热图的概率质量、期望点与argmax点占用。')+figure('target_locking/figures/tl_decoding_brier.png','不同输出表示下的区域事件误差。'))
    vr=[number(r['peak_r']) for r in wv]
    section('3．Model gaze与GT gaze是否领先PSM',p('正lag定义为corr(gaze(t), PSM(t+lag))，表示gaze在前。模型轨迹取热图期望坐标；主分析将预测放在输入锚点时刻，固定左右器械分别比较。argmax仅用于空间占用补充。')+table(['信号','PSM侧','来源','峰位 ms','峰值r','95%重采样范围 ms','判定'],whole_rows)+p(pos_summary)+p(f'合并速度曲线峰值r为{min(vr):.4f}–{max(vr):.4f}。同步或领先是峰位置分类，不自动表示强运动耦合。')+p('组判定要求至少3名参与者、完整lag支持、正相关峰、非边界或竞争峰、0.01近峰宽及重采样范围宽均不超过0.4秒、边界抽样频率不超过5%。主同步带为±100 ms。通过峰定位要求后，范围下限>100 ms才分类为lead。5,000次参与者重采样范围没有校准为生理时延置信区间。')+figure('figures/lead_whole_curves_zh.png','四阶段合并的位置和速度互相关，左右PSM分别计算。')+figure('figures/lead_anchor_position_curves.png','位置曲线及峰形。')+figure('figures/lead_anchor_velocity_curves.png','速度曲线，独立SG5滤波后差分。')+figure('figures/lead_paired_peak_differences.png','模型与GT的配对组峰位差。'))
    section('4．分阶段判定及未来时间轴',table(['来源','支持领先','接近同步','支持滞后','证据不足'],counts)+p(f'每个来源有{len(primary)//len(sources)}个主条件，来源×条件共{len(primary)}个；这些条件相关，不能当成独立受试者。以下表仅列主配置中分类为lead的条件，逐人剔除以当前新结果重新统计。')+table(['来源','阶段','侧','信号','峰位 [范围] ms','峰值r','逐人剔除仍为lead'],local_rows)+table(['来源','阶段','侧','psm_stage峰位 [范围] ms','within_stage峰位 [范围] ms','within_stage判定'],within_rows)+p('psm_stage只约束PSM属于当前阶段，允许较早或较晚阶段的gaze配对；within_stage要求同一次连续阶段。主条件若在within_stage不再支持领先，不能将其解释为阶段内部稳定提前。')+table(['模型','侧','目标时间轴模型lag [范围] ms','模型判定','同期GT lag [范围] ms','GT判定'],targetlead)+p('；'.join(target_ranges)+'。')+p('将Future h预测放在t+h后，是与目标时刻同期GT和PSM的关系；锚点时间分析则描述输入时刻给出的预测与后续PSM的关系。两种lag不能混称，标称500/1000/1500 ms不能当作实测提前量。不同horizon有效支持不同，不能把目标时间均值直接作共同支持排名。')+p('预测为未来屏幕坐标，未做镜头运动配准；推理和显示延迟未测量，输入锚点时刻不是经过测量的实际显示时刻。')+figure('figures/lead_target_time_whole_curves.png','名义预测目标时间轴的位置曲线。')+figure('target_locking/figures/tl_target_time_secondary.png','各horizon与自身同期GT配对的空间占用。'))
    sensitivity=[]
    for scenario in ['r25','r100']:
        for m in models:
            r=error(m,'delta_L_target',scenario);sensitivity.append([scenario,NAMES[m],fmt(number(r['estimate'])*100),f"[{fmt(number(r['ci_low'])*100)}, {fmt(number(r['ci_high'])*100)}]"])
    lvals=[number(r['estimate'])*100 for r in pick(tl_loo,**spec,metric='delta_L_target')]
    native_gt=number(one(rows('target_locking/gt_native_reference.csv'),stage='whole',metric='L_target')['estimate'])
    sampling=pick(rows('gaze_psm_lead/gt_sampling_paired_group_summary.csv'),signal='position',mode='psm_stage')
    changed=sum(abs(number(r['delta_group_peak_lag_s']))>1e-12 for r in sampling)
    maxchange=max(abs(number(r['delta_group_peak_lag_s'])) for r in sampling)
    reference=rows('gaze_psm_lead/unpaired_reference.csv')
    ref_min=min(int(r['valid_draws']) for r in reference);ref_max=max(int(r['valid_draws']) for r in reference)
    maxclock=max(number(r['clock_delta_max_abs_ms']) for r in timeline)
    future_error=max(number(r['training_label_actual_horizon_error_max_ms']) for r in read('input_audit/prediction_audit.json'))
    section('5．稳健性与时间审计',p('Target locking保留25/100 px半径敏感性，所有分支均使用相同的坐标有效性规则，不再按valid设置子集。')+table(['半径分支','模型','目标占用差 pp','95%重采样范围 pp'],sensitivity)+p(f'逐一剔除参与者后的{len(lvals)}个主目标占用差为{min(lvals):.2f}至{max(lvals):.2f}个百分点。完整原生GT参考目标占用为{native_gt*100:.2f}%，共同模型锚点GT为{gt*100:.2f}%；原生参考不替代配对GT分母。')+p('位置预处理为41 Hz网格上的3点居中平均；速度从未平滑网格独立SG5后差分。主搜索窗为±2秒，±4秒作为敏感性；阶段条件至少2秒配对，whole至少5秒。最大插值间隙0.1秒，不外推。Lead中DLC有限低置信度、图外点和跳点仅标记；Target locking的PSM几何仍须有效并落在图像范围内，这与GT的valid字段无关。')+p('主lead条件在平滑、滤波边缘及搜索窗敏感性中的保留情况如下。')+table(['来源','阶段','侧','配置','仍为lead / 配置数'],sensitivity_rows)+p(f'GT完整原生与稀疏模型锚点的配对比较中，{len(sampling)}个位置条件有{changed}个组峰位变化，最大绝对变化{maxchange*1000:.1f} ms；具体完整支持与区间见采样配对表。')+p(f'训练PTS与录制时间轴累计差的最大绝对值为{maxclock:.2f} ms，保存future标签实际horizon偏差最大为{future_error:.2f} ms。所有来源按原视频帧映射到同一录制时间；这些审计值不能自动当作主分析gaze–PSM相对错位。41 Hz插值不会恢复stride=2没有预测的独立时间信息。')+p(f'异用户gaze参考共{len(reference)}个条件，有效参考次数为{ref_min}–{ref_max}；其超越比例只作描述性参照，不是正式领先检验的p值。')+figure('figures/input_time_audit.png','新坐标有效性规则下的预测锚点覆盖及独立时钟审计。'))
    links=[('目标锁定主指标','target_locking/group_metrics.csv'),('目标锁定配对误差','target_locking/group_errors.csv'),('目标锁定覆盖','target_locking/coverage.csv'),('Lead组结果','gaze_psm_lead/group_summary.csv'),('Lead配对组差','gaze_psm_lead/paired_group_summary.csv'),('逐人剔除','gaze_psm_lead/leave_one_user_group_summary.csv'),('异用户参考','gaze_psm_lead/unpaired_reference.csv'),('采样参照配对','gaze_psm_lead/gt_sampling_paired_group_summary.csv'),('同条件敏感性','gaze_psm_lead/sensitivity_matched_summary.csv'),('原生输入审计','input_audit/native_audit.json'),('输入验证','verification/independent_input_validation.json'),('结果验证','verification/independent_result_validation.json'),('分析MD','analysis.md'),('总结MD','summary.md')]
    section('6．数据、报告与解释范围','<ul>'+''.join('<li><a href="../'+f+'">'+name+'</a></li>' for name,f in links)+'</ul>'+p('本次直接更新当前目录中的分析输入、数值、图表和报告，没有备份该目录旧结果。原始数据集NPY、训练模型预测和其他分析目录不作为输出覆盖目标。数值核心和参数快照保存在code_snapshot及分支目录。')+p('4名参与者、5段验证录像不足以支持强泛化；该验证集已用于选模。圆盘区域不是完整物体或器械尖端mask；离线居中滤波和互相关不证明因果或精确生理提前。'))
    css='body{font:16px/1.8 "Microsoft YaHei",sans-serif;color:#20313c;background:#f5f3ee;margin:0}main{max-width:1180px;margin:auto;padding:36px}section{background:white;padding:28px;margin:22px 0;border:1px solid #dce5e8;border-radius:9px}h1{font-size:32px}h2{color:#185e6a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dce5e8;text-align:left}th{background:#e9f1f1}.table-wrap{overflow-x:auto}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{color:#5d6a72;font-size:14px}a{color:#116b80}code{font-size:0.92em}'
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task A coordinate-only重算报告</title><style>'+css+'</style></head><body><main><h1>Task A新模型与GT：目标锁定及领先PSM分析</h1><p>2026-11-06 · 归一化坐标有效性筛选 · 原地重算</p>'+''.join(parts)+'</main></body></html>'
    report=ROOT/'report';report.mkdir(exist_ok=True)
    (report/'analysis_report_zh.html').write_text(doc,encoding='utf-8')
    soup=BeautifulSoup(doc,'html.parser')
    (report/'analysis_report_zh.txt').write_text(soup.find('main').get_text('\n',strip=True),encoding='utf-8')
    analysis=re.sub(r'\n{3,}','\n\n',markdown(soup.find('main'))).strip()+'\n'
    (ROOT/'analysis.md').write_text(analysis,encoding='utf-8')
    summary=f'''# Task A transition模型与GT：重算结果总结

结果日期：2026-11-06。完整方法、全部图表与敏感性结果见 [analysis.md](analysis.md)。

**本目录已原地重算。GT从各User的NPY读取，只要求归一化x、y有限且均在[0,1]，不按valid或Status筛选。** 原生有效点为{native_valid:,}/{native_total:,}；NaN/Inf作为缺失点。所有主分支和半径敏感性遵守该规则，原valid子集分支已移除。原训练日志的future_valid等字段仍是训练诊断，不参与本轮GT有效性筛选。

## 目标锁定

{key1} {key2} 共同有效锚点为{anchor_n:,}个，观测时间权重合计{anchor_s:.2f}秒。先录像内时间加权，再参与者内、参与者间等权；以下范围来自10,000次参与者配对重采样。

{mdtable(['模型','热图目标概率 %','模型−GT 百分点 [95%重采样范围]','目标Brier'],summary_target)}

GT为真实注视点落入目标圆盘的时间占比，模型热图为该区域的概率质量。两者不等同于所有坐标解码的命中率：

{mdtable(['模型','热图概率 %','期望点占用 %','argmax占用 %'],decode_rows)}

占用接近不能证明统计等效、逐帧预测正确或完整概率分布正确。

![不同表示下的目标锁定]({(ROOT/'figures/tl_decoding_occupancy_zh.png').as_posix()})

## Gaze是否领先PSM

正lag表示gaze在PSM之前。模型轨迹取热图期望坐标，固定左右侧分别比较；主表将预测放在输入锚点时刻。下表为四阶段合并的位置结果，单位ms，方括号为5,000次参与者重采样范围。

{mdtable(['来源','左PSM lag [范围]','左判定','右PSM lag [范围]','右判定'],whole_position_rows)}

{pos_summary} 位置峰为正不能单独证明稳定领先；速度曲线峰值r仅为{min(vr):.4f}–{max(vr):.4f}。

主分析{len(primary)}个来源×条件组合中有{len(leads)}个满足探索性lead规则：

{mdtable(['来源','阶段','侧','信号','峰位 [范围] ms','峰值r','逐人剔除仍lead'],local_rows)}

{mdtable(['来源','阶段','侧','psm_stage lag [范围]','within_stage lag [范围]','within_stage判定'],within_rows)}

psm_stage允许其他阶段的gaze配对，within_stage限定同一次连续阶段。未通过within_stage领先规则的结果，不能推广为阶段内部稳定提前。统计条件彼此相关，不能当作独立受试者。

![四阶段合并的位置与速度互相关]({(ROOT/'figures/lead_whole_curves_zh.png').as_posix()})

## 未来时间轴与限制

将future预测放在名义目标时刻t+h后，模型合并位置峰：{'；'.join(target_ranges)}。完整方向判定及同期GT参照见详细报告。训练标称500/1000/1500 ms不是测得的器械提前量，各horizon支持不同，不能直接按均值作共同支持排名。

本批为5段验证录像、4名参与者，验证集参与过checkpoint选择；User5的两段录像先在参与者内汇总。分析包括reach、grab、transfer、place ring四阶段，排除drop ring。圆盘不是完整对象mask，未做镜头运动配准，实际显示延迟未测量。所有范围和方向规则为离线探索性描述。

依据：[目标占用](target_locking/group_metrics.csv) · [配对误差](target_locking/group_errors.csv) · [Lead组结果](gaze_psm_lead/group_summary.csv) · [逐人剔除](gaze_psm_lead/leave_one_user_group_summary.csv) · [输入审计](input_audit/native_audit.json)。
'''
    (ROOT/'summary.md').write_text(summary,encoding='utf-8')
    (ROOT/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=report/analysis_report_zh.html"><a href="report/analysis_report_zh.html">中文报告</a> · <a href="analysis.md">analysis.md</a> · <a href="summary.md">summary.md</a>',encoding='utf-8')
    result=dict(target_gt=gt,best_target_model=best,best_target_probability=bestval,all_models_target_interval_below_gt=lower,main_lead_conditions=len(primary),lead_count=len(leads),whole_lead_rows=whole_rows,whole_position_lead_count=len(position_leads),leave_one_user_stable_lead_count=stable_count,native_coordinate_valid=native_valid,native_total=native_total,valid_column_used_for_selection=False,gaze_validity='finite normalized xy within inclusive [0,1]',common_anchor_n=anchor_n,common_anchor_seconds=anchor_s)
    (report/'executive_findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    links_checked=0
    for file in [ROOT/'analysis.md',ROOT/'summary.md']:
        contents=file.read_text(encoding='utf-8');assert '\ufffd' not in contents
        assert '仍要求' not in contents and '未重算' not in contents and 'gaze3' not in contents
        for ref in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',contents):
            if ref.startswith(('http:','https:','#','data:')):continue
            path=Path(unquote(ref.split('#')[0]));path=path if path.is_absolute() else ROOT/path
            assert path.exists(),(file.name,ref);links_checked+=1
    print(json.dumps(result|dict(status='passed',links_checked=links_checked),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
