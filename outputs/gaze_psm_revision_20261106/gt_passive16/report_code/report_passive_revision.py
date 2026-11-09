"""Offline Chinese report and thirteen figures for passive GT gaze and source PSM."""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes
import argparse
import csv
import html
import json
import textwrap
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {'PSM in stage':'#2364a5', 'Both in stage':'#c16a21'}
MODE = {'PSM in stage':'PSM in stage', 'Both in stage':'Both in stage'}
SIDE = {'l':'Left instrument', 'r':'Right instrument'}
STATUS = {'missing_stage':'未观察到阶段', 'insufficient_users':'观察者不足', 'insufficient_support':'支持不足', 'window_limited':'搜索窗限制',
          'positive_peak_resampling_range':'重抽样范围为正',
          'negative_peak_resampling_range':'重抽样范围为负',
          'direction_uncertain':'方向不确定'}
STAGES = {'A':['reach ring','grab ring','transfer ring','place ring','drop ring'],
          'C':['reach string','wrap around psm','threading string through tower','pull through and cinch']}
VARIANT = {'primary':'Primary','position_unsmoothed':'No smoothing',
           'position_smooth5':'Boxcar 5','velocity_box3':'Boxcar 3',
           'velocity_sg7':'SG 7','velocity_sg11':'SG 11',
           'trim_edges':'Trim edges','window4':'Window +/-4 s'}


def read_csv(path):
    if not Path(path).exists(): return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(mode_rows(csv.DictReader(f)))


def read_json(path):
    return normalize_modes(json.loads(Path(path).read_text(encoding='utf8'))) if Path(path).exists() else {}


def num(value):
    try: return float(value)
    except (TypeError,ValueError): return np.nan


def boolean(value):
    return str(value).lower() in ('true','1','yes')


def fmt(value, digits=3):
    x = num(value)
    return f'{x:.{digits}f}' if np.isfinite(x) else '—'


def esc(value): return html.escape(str(value))


def _key(row):
    return tuple(row.get(k,'') for k in ('task','side','signal','stage','mode'))


def stage_order(rows, task=None, finite_only=False):
    tasks = [task] if task is not None else list(STAGES)
    pairs = {(name, stage) for name in tasks for stage in STAGES.get(name, [])}
    pairs.update((row['task'], row['stage']) for row in rows
                 if task is None or row['task'] == task)
    return sorted(pairs, key=lambda pair: (
        pair[0],
        STAGES.get(pair[0], []).index(pair[1]) if pair[1] in STAGES.get(pair[0], []) else 100,
        pair[1]))


def label(pair):
    short={'threading string through tower':'threading string','pull through and cinch':'pull through / cinch',
           'wrap around psm':'wrap around'}
    return pair[0]+' · '+short.get(pair[1],pair[1])


def _save(fig, output, stem):
    folder=Path(output)/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(stem+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return stem


def plot_curves(output, summaries, curves, signal):
    rows=[r for r in summaries if r.get('variant')=='primary' and r['signal']==signal and r['side'] in SIDE]
    lookup=defaultdict(list)
    for r in curves:
        if r.get('variant')=='primary' and r['signal']==signal and r['side'] in SIDE:
            lookup[_key(r)].append(r)
    summary={_key(r):r for r in rows};made=[]
    for task in STAGES:
        stages=stage_order(rows,task,True)
        if not stages: continue
        fig,axes=plt.subplots(len(stages),2,figsize=(12.5,2.55*len(stages)+1),squeeze=False,sharex=True)
        for i,(_,stage) in enumerate(stages):
            limits=[]
            for side in SIDE:
                for mode in MODE:
                    for r in lookup.get((task,side,signal,stage,mode),[]):
                        limits.extend(num(r.get(k)) for k in ('r','r_resample_low','r_resample_high'))
            limits=np.asarray(limits);limits=limits[np.isfinite(limits)]
            span=max(float(np.ptp(limits)),.03) if len(limits) else .1
            ymin,ymax=(float(limits.min())-.08*span,float(limits.max())+.1*span) if len(limits) else (-.1,.1)
            for j,side in enumerate(SIDE):
                ax=axes[i,j];support=[]
                for mode in MODE:
                    k=(task,side,signal,stage,mode);data=sorted(lookup.get(k,[]),key=lambda x:num(x['lag_s']))
                    sr=summary.get(k,{})
                    support.append(f"{MODE[mode]}: observers={sr.get('n_stable_users','0')}, records={sr.get('n_complete_records','0')}")
                    if not data:continue
                    x=np.array([num(r['lag_s']) for r in data]); y=np.array([num(r['r']) for r in data])
                    lo=np.array([num(r.get('r_resample_low')) for r in data]);hi=np.array([num(r.get('r_resample_high')) for r in data])
                    if not np.isfinite(y).any():continue
                    ax.plot(x,y,c=COLORS[mode],lw=1.8,label=MODE[mode])
                    ax.fill_between(x,lo,hi,color=COLORS[mode],alpha=.14,lw=0)
                    px,py=num(sr.get('peak_lag_s')),num(sr.get('peak_r'))
                    if np.isfinite(px) and np.isfinite(py):
                        ax.plot(px,py,'D' if boolean(sr.get('peak_at_boundary')) else 'o',color=COLORS[mode],ms=4)
                if not ax.lines:
                    ax.text(.5,.55,'Insufficient support',transform=ax.transAxes,ha='center',color='#64748b')
                ax.axvline(0,color='#8491a1',ls=':',lw=.8);ax.axhline(0,color='#bac2cb',lw=.7)
                ax.set_xlim(-2,2);ax.set_ylim(ymin,ymax);ax.grid(alpha=.15)
                ax.set_title(f'{stage}\n{SIDE[side]}',fontsize=10,pad=8)
                ax.text(.02,.04,'\n'.join(support),transform=ax.transAxes,fontsize=7,color='#475569',va='bottom',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=1.5))
                if j==0:ax.set_ylabel('Correlation r')
                if i==len(stages)-1:ax.set_xlabel('Lag (s) · positive means gaze earlier')
        handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=MODE[m]) for m in MODE]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
        fig.suptitle(f'Task {task} | {signal.capitalize()} | Fixed physical instruments',fontsize=14,y=.995)
        fig.text(.5,.007,'Shaded bands show exploratory observer percentile resampling ranges.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.025,1,.935),h_pad=1.2)
        made.append(_save(fig,output,f'primary_{signal}_curves_{task}'))
    return made


def plot_lags(output, summaries, signal):
    rows=[r for r in summaries if r.get('variant')=='primary' and r['signal']==signal and r['side'] in SIDE]
    stages=stage_order(rows,finite_only=True)
    if not stages:return None
    lookup={_key(r):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(12.5,max(4.8,.66*len(stages)+1.8)),sharey=True)
    for j,side in enumerate(SIDE):
        ax=axes[j]
        for i,(task,stage) in enumerate(stages):
            for mode,offset in [('PSM in stage',-.12),('Both in stage',.12)]:
                row=lookup.get((task,side,signal,stage,mode),{});peak=num(row.get('peak_lag_s'))
                if not np.isfinite(peak):continue
                y=i+offset;lo,hi=num(row.get('peak_lag_resample_low')),num(row.get('peak_lag_resample_high'))
                if np.isfinite(lo) and np.isfinite(hi):ax.plot([lo,hi],[y,y],c=COLORS[mode],lw=2,alpha=.7)
                ax.scatter(peak,y,color=COLORS[mode],marker='D' if boolean(row.get('peak_at_boundary')) else 'o',s=30,zorder=3)
        if not ax.collections:
            ax.text(.5,.5,'Insufficient support',transform=ax.transAxes,ha='center',color='#64748b')
        ax.axvline(0,c='#718096',ls=':',lw=1);ax.set_xlim(-2.12,2.12);ax.grid(axis='x',alpha=.2)
        ax.set_title(SIDE[side]);ax.set_xlabel('Group peak lag (s) · positive means gaze earlier')
        ax.set_yticks(range(len(stages)),[label(s) for s in stages]);ax.set_ylim(len(stages)-.5,-.5)
    handles=[Line2D([0],[0],color=COLORS[m],marker='o',label=MODE[m]) for m in MODE]
    handles.append(Line2D([0],[0],color='#526173',marker='D',ls='none',label='Boundary point estimate'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False,fontsize=9)
    fig.suptitle(signal.capitalize()+' peak locations and exploratory resampling ranges',fontsize=14,y=.995)
    fig.text(.5,.01,'Bars show the 2.5th–97.5th observer resampling percentiles.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.88))
    return _save(fig,output,f'primary_{signal}_peak_lags')


def plot_sensitivity(output, summaries, signal):
    primary=[r for r in summaries if r.get('variant')=='primary' and r['signal']==signal and r['mode']=='PSM in stage' and r['side'] in SIDE]
    stages=stage_order(primary,finite_only=True)
    if not stages:return None
    variants=['position_unsmoothed','position_smooth5','trim_edges','window4'] if signal=='position' else ['velocity_box3','velocity_sg7','velocity_sg11','trim_edges','window4']
    if not variants:return None
    lookup={(r.get('variant'),*_key(r)):r for r in summaries}
    matrices=[]
    for side in SIDE:
        a=np.full((len(stages),len(variants)),np.nan)
        for i,(task,stage) in enumerate(stages):
            base=num(lookup.get(('primary',task,side,signal,stage,'PSM in stage'),{}).get('peak_lag_s'))
            for j,v in enumerate(variants):
                value=num(lookup.get((v,task,side,signal,stage,'PSM in stage'),{}).get('peak_lag_s'))
                a[i,j]=value-base
        matrices.append(a)
    finite=np.concatenate([a.ravel() for a in matrices]);finite=finite[np.isfinite(finite)]
    bound=max(.2,float(abs(finite).max())) if len(finite) else 1
    fig,axes=plt.subplots(1,2,figsize=(12.5,max(4.8,.52*len(stages)+1.5)),sharey=True)
    for ax,side,a in zip(axes,SIDE,matrices):
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e5e7eb')
        im=ax.imshow(a,cmap=cmap,vmin=-bound,vmax=bound,aspect='auto')
        ax.set_title(SIDE[side]);ax.set_xticks(range(len(variants)),[VARIANT[v] for v in variants],rotation=30,ha='right',fontsize=9)
        ax.set_yticks(range(len(stages)),[label(s) for s in stages])
        for i in range(len(stages)):
            for j in range(len(variants)):
                text=f'{a[i,j]:+.2f}' if np.isfinite(a[i,j]) else 'NA'
                ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if np.isfinite(a[i,j]) and abs(a[i,j])>.6*bound else '#172033')
    fig.suptitle(signal.capitalize()+' sensitivity | peak lag minus primary (s) | PSM in stage',fontsize=13,y=.995)
    fig.tight_layout(rect=(0,.04,.93,.94))
    cax=fig.add_axes([.945,.22,.015,.56]);fig.colorbar(im,cax=cax,label='Change in peak lag (s)')
    fig.text(.5,.01,'Gray cells indicate insufficient support. Sensitivity branches follow the fixed analysis plan.',ha='center',fontsize=8)
    return _save(fig,output,f'{signal}_sensitivity_lag_change')


def plot_spatial(output, rows):
    stages=stage_order(rows)
    if not stages:return None
    lookup={(r['task'],r['side'],r['stage'],r['mode'],r['at']):r for r in rows}
    fig,axes=plt.subplots(2,2,figsize=(12.5,max(8,1.15*len(stages)+2.4)),sharey=True)
    for ri,(field,xlab) in enumerate([('distance_median_px','Mean of recording-median distances (px)'),('proximity_50px','Fraction within 50 pixels')]):
        values=np.array([num(r.get(field)) for r in rows]);values=values[np.isfinite(values)]
        right=max(float(values.max())*1.08,.01) if len(values) else 1.
        for j,side in enumerate(SIDE):
            ax=axes[ri,j]
            for i,(task,stage) in enumerate(stages):
                for mode,offset in [('PSM in stage',-.12),('Both in stage',.12)]:
                    a=num(lookup.get((task,side,stage,mode,'zero'),{}).get(field));b=num(lookup.get((task,side,stage,mode,'group_peak'),{}).get(field))
                    if np.isfinite(a) and np.isfinite(b):
                        ax.plot([a,b],[i+offset,i+offset],color=COLORS[mode],lw=1.3)
                        ax.scatter(a,i+offset,c='white',edgecolors=COLORS[mode],s=26,zorder=3)
                        ax.scatter(b,i+offset,c=COLORS[mode],marker='D',s=24,zorder=3)
            if not ax.lines:
                ax.text(.5,.5,'Insufficient support',transform=ax.transAxes,ha='center',color='#64748b')
            ax.set_title(SIDE[side]);ax.set_xlabel(xlab,fontsize=9);ax.grid(axis='x',alpha=.2)
            ax.set_yticks(range(len(stages)),[label(s) for s in stages]);ax.set_ylim(len(stages)-.5,-.5)
            ax.set_xlim(0,right)
    handles=[Line2D([0],[0],color=COLORS[m],label=MODE[m]) for m in MODE]
    handles += [Line2D([0],[0],color='#526173',marker='o',markerfacecolor='white',ls='none',label='Zero lag'),Line2D([0],[0],color='#526173',marker='D',ls='none',label='Group peak lag')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Spatial correspondence | zero lag versus common group peak',y=.995,fontsize=14)
    fig.tight_layout(rect=(0,.025,1,.925))
    fig.text(.5,.006,'Viewing records are equal within observer and observers are equal across group. The 50 px radius is exploratory.',ha='center',fontsize=8)
    return _save(fig,output,'spatial_zero_vs_peak')


def plot_group_bootstrap(output, signal):
    index=read_json(Path(output)/'group_bootstrap_index.json')
    path=Path(output)/'group_bootstrap_peak_lags.npz'
    if not isinstance(index,list):index=[]
    selected=[r for r in index if r.get('variant')=='primary' and r.get('signal')==signal and r.get('side') in SIDE]
    lookup={_key(r):r for r in selected};made=[]
    with (np.load(path,allow_pickle=False) if path.exists() else nullcontext({})) as data:
        for task in STAGES:
            stages=stage_order(selected,task)
            if not stages:continue
            fig,axes=plt.subplots(len(stages),2,figsize=(12.5,2.1*len(stages)+1),sharex=True,squeeze=False)
            for i,(_,stage) in enumerate(stages):
                for j,side in enumerate(SIDE):
                    ax=axes[i,j]
                    for mode in MODE:
                        row=lookup.get((task,side,signal,stage,mode))
                        if not row or row['token'] not in data:continue
                        values=np.asarray(data[row['token']],float);values=values[np.isfinite(values)]
                        if not len(values):continue
                        ax.hist(values,bins=np.linspace(-2.05,2.05,42),weights=np.ones(len(values))/len(values),
                                histtype='step',color=COLORS[mode],linewidth=1.5,label=MODE[mode])
                    if not ax.patches:
                        ax.text(.5,.55,'No supported bootstrap draws',transform=ax.transAxes,ha='center',fontsize=8,color='#64748b')
                    ax.axvline(0,color='#8491a1',ls=':',lw=.8);ax.set_xlim(-2.05,2.05);ax.grid(alpha=.15)
                    ax.set_title(f'{stage}\n{SIDE[side]}',fontsize=10)
                    if j==0:ax.set_ylabel('Fraction of draws')
                    if i==len(stages)-1:ax.set_xlabel('Group peak lag in observer resampling (s)')
            handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=MODE[m]) for m in MODE]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
            fig.suptitle(f'Task {task} | {signal.capitalize()} | Resampled group peak distribution',fontsize=14,y=.995)
            fig.text(.5,.008,'Each outline uses fractions summing to one. Boundary accumulation and separated peaks are retained.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.03,1,.93),h_pad=1.2)
            made.append(_save(fig,output,f'group_bootstrap_{signal}_{task}'))
    return made


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(stem,caption):
    if not stem:return ''
    return f'<figure><a href="figures/{stem}.png"><img loading="lazy" src="figures/{stem}.png" alt="{esc(caption)}"></a><figcaption>{esc(caption)} · <a href="figures/{stem}.pdf">PDF</a> · <a href="figures/{stem}.png">PNG</a></figcaption></figure>'


def group_table(rows):
    sorted_rows=sorted(rows,key=lambda r:(stage_order(rows).index((r['task'],r['stage'])),r['side'],r['mode']))
    return table(['任务 / 阶段','器械','mode','观察者 / 观看记录','峰值 r','r(0)','峰增益','峰位 / 秒','重抽样 2.5%–97.5%','边界频率','解释'],[
        [r['task']+' / '+r['stage'],SIDE.get(r['side'],r['side']),r['mode'],r.get('n_stable_users','')+' / '+r.get('n_complete_records',''),
         fmt(r.get('peak_r')),fmt(r.get('r_zero')),fmt(r.get('peak_gain_over_zero')),
         fmt(r.get('peak_lag_s')), '['+fmt(r.get('peak_lag_resample_low'))+', '+fmt(r.get('peak_lag_resample_high'))+']',
         fmt(r.get('bootstrap_boundary_frequency')),STATUS.get(r.get('lag_interpretation', r.get('status')),r.get('lag_interpretation',r.get('status','')))] for r in sorted_rows])


def _display_primary(groups):
    rows = [row for row in groups
            if row.get('variant') == 'primary' and row.get('side') in SIDE]
    lookup = {_key(row): row for row in rows}
    for task, stages in STAGES.items():
        for stage in stages:
            for signal in ('position', 'velocity'):
                for side in SIDE:
                    for mode in MODE:
                        key = (task, side, signal, stage, mode)
                        if key not in lookup:
                            rows.append(dict(task=task, side=side, signal=signal,
                                             stage=stage, mode=mode, variant='primary',
                                             n_stable_users='NA', n_complete_records='NA',
                                             lag_interpretation='missing_stage'))
    return rows


def _audit_table(value):
    if isinstance(value, dict):
        rows = [[key, json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item]
                for key, item in value.items()]
        return table(['字段', '审计记录'], rows)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        headers = list(dict.fromkeys(key for item in value for key in item))
        return table(headers, [[json.dumps(item.get(key), ensure_ascii=False)
                              if isinstance(item.get(key), (dict, list)) else item.get(key, '')
                              for key in headers] for item in value])
    return '<p>' + esc(json.dumps(value, ensure_ascii=False)) + '</p>'


def _csv_table(rows):
    if not rows:
        return '<p>未记录</p>'
    headers = list(dict.fromkeys(key for row in rows for key in row))
    return table(headers, [[row.get(key, '') for key in headers] for row in rows])


def _mapping_notice(metadata):
    warning = metadata.get('mapping_warning', {})
    if not warning:
        return '', False
    if not isinstance(warning, dict):
        return '<section class="mapping-note"><h2>输入映射说明</h2><p>' + esc(warning) + '</p></section>', False
    unresolved = boolean(warning.get('unresolved', False))
    row_mapping = boolean(warning.get('user_authorised_row_mapping', False))
    default_title = '按用户指定逐行对应' if row_mapping else '输入映射说明'
    title = warning.get('title', '播放时间映射的解释条件' if unresolved else default_title)
    detail = warning.get('message', warning.get('summary', warning.get('detail', '')))
    style = 'mapping-warning' if unresolved else 'mapping-note'
    text = '<section class="' + style + '"><h2>' + esc(title) + '</h2>'
    if unresolved:
        text += ('<p><b>当前结果以现有 NPY 帧映射与 source-content-clock 为条件。</b>'
                 '既有 NPY 与实际显示帧时间的对应尚未通过完整核验。'
                 '数值方法一致性与实际播放对应关系分别审计。'
                 '全部实际分析分支按现有时间轴完成。</p>')
    elif row_mapping:
        text += ('<p><b>按用户指定按行对应，时间轴沿用精确 video_ts。</b>'
                 '现有 NPY 第 i 行对应视频第 i 帧与 DLC 第 i 行。'
                 '原 VID_FRAME 差异继续保留审计，不作为剔除条件。按行对应已由用户明确接受。</p>')
    if detail:
        text += '<p>' + esc(detail) + '</p>'
    text += '<details><summary>映射审计状态</summary>' + _audit_table(warning) + '</details></section>'
    return text, unresolved


def _results_overview(primary, record_rows, reference, conditional):
    parts = ['<section id="results"><h2>数据结果概览</h2>']
    if conditional:
        parts.append('<p>以下数值均以现有 NPY 与来源内容时钟的映射为条件。</p>')
    rows = []
    for signal in ('position', 'velocity'):
        selected = [row for row in primary if row.get('signal') == signal]
        finite = [row for row in selected if np.isfinite(num(row.get('peak_lag_s')))]
        lag = [num(row.get('peak_lag_s')) for row in finite]
        status = {name: sum(row.get('lag_interpretation') == name for row in selected)
                  for name in ('window_limited', 'positive_peak_resampling_range',
                               'negative_peak_resampling_range', 'direction_uncertain')}
        rows.append([signal, len(finite),
                     f'{min(lag):.3f} 至 {max(lag):.3f}' if lag else '未记录',
                     status['window_limited'], status['positive_peak_resampling_range'],
                     status['negative_peak_resampling_range'], status['direction_uncertain'],
                     len(selected) - len(finite)])
    parts.append(table(['signal', '有可用峰组', '组峰位范围 / 秒', '窗口限制',
                        '重抽样范围全正', '重抽样范围全负', '方向不确定', '预定组缺少支持'], rows))
    position = [row for row in primary if row.get('signal') == 'position'
                and np.isfinite(num(row.get('peak_lag_s')))]
    if position:
        peak_r = [num(row.get('peak_r')) for row in position]
        peak_r = [value for value in peak_r if np.isfinite(value)]
        if peak_r:
            parts.append(f'<p>Position 主分支有 {len(position)} 个组具备可用峰，'
                         f'峰高 r 范围为 {min(peak_r):.3f} 至 {max(peak_r):.3f}。'
                         '窗口限制计数同时保留点估计触边与重抽样范围触边的情形。</p>')
    primary_records = [row for row in record_rows if row.get('variant') == 'primary']
    if primary_records:
        identifiable = sum(boolean(row.get('lag_identifiable')) for row in primary_records)
        supported = sum(boolean(row.get('full_lag_support')) for row in primary_records)
        parts.append(f'<p>记录层主分支共 {len(primary_records)} 条记录×阶段×侧×mode 曲线，'
                     f'其中 {supported} 条具有完整 lag 支持，{identifiable} 条通过冻结的 lag 可定位规则。'
                     '记录层规则同时考察重复 occurrence、峰形、重抽样和稳定性。</p>')
    if reference:
        summary = []
        for scope in sorted({row.get('reference_scope', '') for row in reference}):
            selected = [row for row in reference if row.get('reference_scope', '') == scope]
            valid = [row for row in selected if num(row.get('n_valid')) > 0]
            attempts = [num(row.get('n_valid')) for row in valid]
            tails = [num(row.get('reference_tail_fraction')) for row in valid]
            tails = [value for value in tails if np.isfinite(value)]
            summary.append([scope, len(valid), len(selected),
                            f'{min(attempts):.0f} 至 {max(attempts):.0f}' if attempts else '未记录',
                            f'{min(tails):.4f} 至 {max(tails):.4f}' if tails else '未记录'])
        parts.append(table(['参考范围', '有有效抽样的组', '参考组总数', '每组有效抽样数范围',
                            '描述性尾部比例范围'], summary))
        parts.append('<p>参考统计比较观测最大相关峰与有效 donor 抽样的峰高，'
                     '并保留共享来源视频及抽样失败的条件。</p>')
    parts.append('</section>')
    return ''.join(parts)


def generate_report(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    manifest = read_json(output / 'run_manifest.json')
    validation = read_json(output / 'validation.json')
    metadata = read_json(output / 'passive_metadata_audit.json')
    groups = read_csv(output / 'group_summary.csv')
    curves = read_csv(output / 'group_curves.csv')
    primary = _display_primary(groups)
    plot_groups = [row for row in groups if row.get('variant') != 'primary'] + primary
    records = manifest.get('records', [])
    n_records = len(records)
    observer = lambda row: row.get('observer_user', row.get('user', ''))
    n_observers = len({observer(row) for row in records if observer(row)})
    n_sources = len({row.get('source_content_id', row.get('content_id', ''))
                     for row in records})
    finite = [row for row in primary if np.isfinite(num(row.get('peak_lag_s')))]
    boundary = [row for row in finite if boolean(row.get('peak_at_boundary'))]
    record_rows = read_csv(output / 'record_summary.csv')
    reference = read_csv(output / 'unpaired_reference.csv')
    mapping_notice, conditional = _mapping_notice(metadata)
    images = {}
    for signal in ('position', 'velocity'):
        images[signal + '_curves'] = plot_curves(output, plot_groups, curves, signal)
        images[signal + '_lags'] = plot_lags(output, plot_groups, signal)
        images[signal + '_sensitivity'] = plot_sensitivity(output, plot_groups, signal)
        images[signal + '_bootstrap'] = plot_group_bootstrap(output, signal)
    spatial = read_csv(output / 'spatial_summary.csv')
    images['spatial'] = plot_spatial(output, spatial)
    parts = [f'''<header><p class="eyebrow">PASSIVE GAZE / SOURCE PSM</p>
<h1>观看者 gaze 与所观看视频的左右 PSM</h1>
<p>gaze 属于观看者，stage 与 DLC 器械坐标来自 source_content_id 对应的主动操作视频。
按观看记录计算曲线，再以观察者为重抽样与聚合单位。</p>
<div class="cards"><div><b>{n_records}</b><span>纳入观看记录</span></div>
<div><b>{n_observers}</b><span>观察者</span></div>
<div><b>{n_sources}</b><span>来源视频</span></div>
<div><b>{len(finite)}</b><span>有可用峰的主分析组</span></div>
<div><b>{len(boundary)}</b><span>点估计触及窗口的组</span></div></div></header>''']
    parts.append(mapping_notice)
    navigation = [('results', '结果概览'), ('population', '数据覆盖'), ('methods', '方法'), ('position', 'Position'),
                  ('velocity', 'Velocity'), ('sensitivity', '敏感性'), ('support', '共同支持'),
                  ('spatial', '空间对应'), ('reference', 'Unpaired reference'), ('files', '验证与文件')]
    parts.append('<nav>' + ''.join(f'<a href="#{key}">{title}</a>'
                                  for key, title in navigation) + '</nav>')
    parts.append(_results_overview(primary, record_rows, reference, conditional))
    parts.append('<section id="population"><h2>观看记录与来源映射</h2>')
    sections = [('population_counts', '完整 passive 数据'), ('inclusion_counts', '本次纳入'),
                ('exclusions', '排除记录与原因'), ('task_scope', '任务范围'),
                ('mapping_summary', '帧与来源映射'), ('findings', '本次结果与审计说明')]
    for key, title in sections:
        parts.append(f'<h3>{title}</h3>' + _audit_table(metadata.get(key, '未记录')))
    discrepancies = read_csv(output / 'raw_frame_discrepancies.csv')
    if discrepancies:
        parts.append('<h3>原始记录与现有 NPY 对应帧的差异</h3>'
                     '<p>下表逐条保留差异定义、匹配数和帧差范围，播放对应状态结合上方映射说明阅读。</p>'
                     + _csv_table(discrepancies))
    input_rows = read_csv(output / 'input_record_inventory.csv')
    analysis_rows = read_csv(output / 'analysis_record_inventory.csv')
    for title, rows in [('全部输入记录', input_rows), ('纳入分析的观看记录', analysis_rows)]:
        parts.append(f'<details><summary>{title}</summary>' + _csv_table(rows) + '</details>')
    coverage_rows = read_csv(output / 'stage_coverage.csv')
    parts.append('''<h3>预定阶段覆盖</h3><p>所有 Task A 五个阶段与 Task C 四个阶段均保留。
缺少观察阶段或完整 lag 支持时，图中保留空位并在表格中显示状态。</p>'''.replace('\n', ''))
    if coverage_rows:
        parts.append(_csv_table(coverage_rows))
    else:
        summary = []
        for task, stages in STAGES.items():
            for stage in stages:
                rows = [row for row in primary if row['task'] == task and row['stage'] == stage]
                summary.append([task, stage, len(rows),
                                sum(np.isfinite(num(row.get('peak_lag_s'))) for row in rows)])
        parts.append(table(['任务', '预定阶段', '侧 / signal / mode 组合', '有可用峰组合'], summary))
    parts.append('</section><section id="methods"><h2>方法与计算参数</h2>')
    config = manifest.get('variants', {}).get('primary', {})
    mapping_warning = metadata.get('mapping_warning', {})
    row_mapping = isinstance(mapping_warning, dict) and boolean(mapping_warning.get('user_authorised_row_mapping', False))
    methods = [
        ['分析对象', '观看者 gaze 与所观看来源视频的左右器械。固定左右器械分别作为主分析。'],
        ['GT 有效点', 'NPY gaze_x 与 gaze_y 均有限且位于闭区间 [0,1]，valid 仅作审计。'],
        ['DLC', '所有有限坐标保留。likelihood、画面外点与跳变作 QC 记录。'],
        ['画面坐标', '用户确认 gaze 归一化到 1280×1024 视频画面，像素坐标按画面宽高缩放。画面几何与实际播放时间映射分别审计。'],
        ['时间轴', f"以来源视频逐帧 video_ts 的精确 elapsed time 建立 {config.get('hz', 41)} Hz 网格，最大插值间隙 {config.get('max_gap', .1)} 秒，连续有限段内滤波。"],
        ['Position', f"原始网格坐标的 {config.get('position_window', 3)} 点居中移动平均。"],
        ['Velocity', f"原始网格坐标独立经二阶 SG{config.get('velocity_window', 5)} 滤波，再差分。标签位于区间中点。"],
        ['有效重叠', f"每 lag 合并 occurrence 后至少 {config.get('min_overlap', 2)} 秒，gaze 与 PSM RMS 均至少 {config.get('min_rms', 1)}。"],
        ['lag 定义', '正 lag 比较较早的观看者 gaze 与较晚的来源 PSM。每个 occurrence 在对应有效重叠内单独中心化后合并充分统计量。'],
        ['阶段条件', 'PSM in stage 约束 PSM 所属 occurrence，Both in stage 约束 gaze 与 PSM 同属该 occurrence。'],
        ['聚合与支持', '全 lag 有效的观看记录集合固定。观察者内各观看记录等权 Fisher-z，组内观察者等权。至少三个观察者。'],
        ['重抽样', f"组重抽样 {manifest.get('bootstrap', '未记录')} 次。记录内按相邻 occurrence 的非循环长度二移动块重抽样。"],
        ['搜索范围', '主分支 ±2 秒，敏感性分支 ±4 秒。'],
        ['解释', '相关峰描述观看者与所看操作视频器械轨迹的对应。百分位范围为探索性重抽样范围。']]
    if row_mapping:
        methods.extend([
            ['逐行对应', '用户明确指定现有 NPY 第 i 行、视频第 i 帧与 DLC 第 i 行对应。初始严格播放对应核验要求由该规则覆盖，原 VID_FRAME 差异保留为审计。'],
            ['时间解释', '分析时钟沿用精确 video_ts。实际播放时序尚无经验验证，结果按冻结基线作为探索性 lag 描述。']])
    parts.append(table(['部分', '本次规则'], methods))
    methods_file = output / 'methods_consistency.csv'
    if methods_file.exists():
        parts.append('<details><summary>与冻结基线的方法逐项对照</summary>'
                     + _csv_table(read_csv(methods_file)) + '</details>')
    parts.append('</section>')
    for signal, title in [('position', 'Position 主结果'), ('velocity', 'Velocity 补充结果')]:
        rows = [row for row in primary if row['signal'] == signal]
        parts.append(f'''<section id="{signal}"><h2>{title}</h2><p>图中 n 表示观察者数，records 表示观看记录数。
圆点为窗口内峰，菱形表示点估计位于搜索边界。缺少支持的阶段显示 NA 或空位。</p>'''.replace('\n', ''))
        parts.append(figure(images[signal + '_lags'], '组峰位与观察者重抽样范围'))
        for stem in images[signal + '_curves']:
            parts.append(figure(stem, '完整组曲线与逐 lag 的观察者重抽样范围'))
        parts.append('<details><summary>组峰位重抽样分布</summary>')
        for stem in images[signal + '_bootstrap']:
            parts.append(figure(stem, '保留边界堆积和多峰的组峰位分布'))
        parts.append('</details><details><summary>所有预定阶段的主结果数值</summary>'
                     + group_table(rows) + '</details></section>')
    parts.append('''<section id="sensitivity"><h2>平滑、边缘与搜索窗口敏感性</h2>
<p>Position 保留无平滑、Box3 主分支与 Box5。Velocity 保留 Box3、SG5 主分支、SG7 与 SG11。
另有有限段边缘裁除和 ±4 秒窗口。灰色 NA 单元保留缺少支持的比较。</p>'''.replace('\n', ''))
    for signal in ('position', 'velocity'):
        parts.append(figure(images[signal + '_sensitivity'], signal + ' 峰位相对主分支的变化，单位秒'))
    selected = read_csv(output / 'gt_selected_group_summary.csv')
    parts.append('''<p>GT 选择侧按记录、阶段、signal 与 mode 的带符号最大峰选侧，平局选左侧。
选择结果受 gaze 参与选侧的过程影响。</p>'''.replace('\n', ''))
    if selected:
        parts.append('<details><summary>GT 选择侧补充结果</summary>' + group_table(selected) + '</details>')
    parts.append('''</section><section id="support"><h2>两种 mode 的共同支持</h2>
<p>两种 mode 使用同一完整 lag 有效记录子集，再按观察者聚合。保留共同支持前后的覆盖变化。</p>'''.replace('\n', ''))
    matched = read_csv(output / 'matched_mode_summary.csv')
    lookup = {_key(row): row for row in primary}
    support_rows = []
    for row in matched:
        before = lookup.get(_key(row), {})
        support_rows.append([row.get('task', '') + ' / ' + row.get('stage', ''),
                             SIDE.get(row.get('side'), row.get('side', '')),
                             row.get('signal', ''), row.get('mode', ''),
                             str(before.get('n_stable_users', '—')) + ' → ' + str(row.get('n_stable_users', '—')),
                             str(before.get('n_complete_records', '—')) + ' → ' + str(row.get('n_complete_records', '—')),
                             fmt(before.get('peak_lag_s')) + ' → ' + fmt(row.get('peak_lag_s'))])
    parts.append(table(['阶段', '器械', 'signal', 'mode', '观察者', '观看记录', 'lag / 秒'], support_rows))
    contrasts = read_csv(output / 'stage_contrasts.csv')
    if contrasts:
        parts.append('<details><summary>共同观察者的阶段峰位差</summary><p>差值为阶段 B 减阶段 A，保留边界频率与共同观察者数。</p>'
                     + _csv_table(contrasts) + '</details>')
    parts.append('''</section><section id="spatial"><h2>空间对应与跨边界配对</h2>
<p>观看者 gaze 与来源 PSM 的距离在各观看记录内汇总，再按观察者分层平均。
25、50 与 100 px 接近比例为描述性指标。跨边界比例描述所用配对样本。</p>'''.replace('\n', ''))
    parts.append(figure(images['spatial'], '零 lag 与组峰位处的空间对应'))
    parts.append('''</section><section id="reference"><h2>Unpaired reference</h2>
<p>同 task、不同观察者的 gaze 与目标 PSM 按经过时间配对。
先等概率抽 donor 观察者，再等概率抽其观看记录。观看同一来源视频的不同观察者继续互为允许 donor。
该规则保留共享刺激结构，参考峰反映此条件下的关联强度。失败抽样逐次记录，持续时间覆盖子集另算其自身观测统计量。</p>'''.replace('\n', ''))
    parts.append(_audit_table(metadata.get('shared_stimulus', '共享来源视频的不同观察者按冻结规则保留为 donor')))
    if reference:
        headers = ['阶段', '侧', 'signal / mode', '参考范围', '观测峰 r', '有效 / 请求', '尾部比例', '描述性 BH', '状态']
        rows = [[row.get('task', '') + ' / ' + row.get('stage', ''), row.get('side', ''),
                 row.get('signal', '') + ' / ' + row.get('mode', ''), row.get('reference_scope', ''),
                 fmt(row.get('observed_peak_r')), str(row.get('n_valid', '')) + ' / ' + str(row.get('n_requested', '')),
                 fmt(row.get('reference_tail_fraction'), 4), fmt(row.get('reference_bh_descriptive'), 4),
                 row.get('status', '')] for row in reference]
        parts.append(_csv_table(read_csv(output / 'unpaired_donor_coverage.csv')))
        parts.append('<details><summary>参考峰比较与有效次数</summary>' + table(headers, rows) + '</details>')
    else:
        parts.append('<p>参考结果未记录。</p>')
    parts.append('<p>参考尾部比例比较最大相关峰高。描述性 BH 保留全部预定 A/C 阶段的比较族。</p></section>')
    parts.append('<section id="files"><h2>验证、逐点来源与文件</h2>')
    mapping_status = '未通过完整核验，按现有 NPY 有条件解释' if conditional else '见映射审计'
    if row_mapping:
        mapping_status = '按用户指定的逐行对应规则接受，实际播放差异保留审计'
    checks = [['输入文件未改变', validation.get('input_unchanged', '未记录')],
              ['DLC 质量规则删除点', validation.get('dlc_quality_deleted', '未记录')],
              ['原生字段与计算支持审计通过', validation.get('all_support_audits_pass', '未记录')],
              ['数值计算核的一致性', validation.get('same_input_kernel_validation', {}).get('passed', '未记录')],
              ['本次对齐规则与播放核查', mapping_status],
              ['独立核最大绝对误差', validation.get('same_input_kernel_validation', {}).get('maximum_error', '未记录')]]
    parts.append(table(['检查', '结果'], checks))
    parts.append(_audit_table(metadata.get('method_validation', '未记录')))
    parts.append('<details><summary>完整验证记录</summary>' + _audit_table(validation) + '</details>')
    files = [
        ('group_summary.csv', '各分支组峰位、峰高与观察者支持'), ('group_curves.csv', '完整组曲线'),
        ('record_summary.csv', '逐观看记录峰形与 occurrence 诊断'), ('record_curves.jsonl', '逐记录完整曲线'),
        ('preprocessing_qc.csv', '原生与网格预处理 QC'), ('support_audit.csv', '时间轴与支持审计'),
        ('side_selection.csv', 'GT 选择侧规则与赢家'), ('aggregation_support_comparison.csv', '支持规则对照'),
        ('stage_contrasts.csv', '共同观察者的阶段差'), ('stage_coverage.csv', '全部预定阶段覆盖'),
        ('group_bootstrap_index.json', '组重抽样索引'), ('group_bootstrap_peak_lags.npz', '组峰位分布'),
        ('unpaired_reference.csv', 'Unpaired 参考结果'), ('unpaired_pairings.csv', '每次 donor 配对'),
        ('unpaired_donor_coverage.csv', 'Donor 覆盖'), ('input_record_inventory.csv', '完整输入记录清单'),
        ('analysis_record_inventory.csv', '纳入记录清单'), ('passive_metadata_audit.json', '完整 passive 元数据审计'),
        ('raw_frame_discrepancies.csv', '原始记录与现有 NPY 的帧差明细'),
        ('methods_consistency.csv', '冻结方法逐项对照'), ('methods_consistency.md', '方法对照说明'),
        ('run_manifest.json', '运行参数与代码指纹'), ('input_manifest.json', '输入指纹'), ('validation.json', '独立验证')]
    files = [(name, meaning) for name, meaning in files if (output / name).exists()]
    parts.append(table(['文件', '内容'], files))
    parts.append('<ul>' + ''.join(f'<li>{esc(meaning)} <a href="{name}">{name}</a></li>'
                                 for name, meaning in files) + '</ul>')
    parts.append('''<p>point_provenance 保存原生、网格与速度时间轴及逐点 QC。
occurrence_statistics 保存充分统计量，bootstrap_peaks 保存记录内重抽样。
code_snapshot 与 report_code 保存本次执行和绘图源码。</p></section>'''.replace('\n', ''))
    css = '''body{margin:0;background:#f4f7fa;color:#172335;font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1320px;margin:auto;padding:32px}header{background:#112c46;color:white;padding:36px;border-radius:12px}
h1{font-size:30px;line-height:1.3}h2{color:#173d5a}h3{font-size:17px}.eyebrow{letter-spacing:.12em;color:#add8ee}
.cards{display:flex;gap:14px;flex-wrap:wrap}.cards div{background:#20415f;padding:12px 18px;border-radius:8px}
.cards b{font-size:26px;display:block}.cards span{font-size:12px;color:#c5d5e2}nav{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}
section{background:white;border:1px solid #e0e7ed;border-radius:12px;padding:26px;margin:20px 0}a{color:#126091;text-decoration:none}
.mapping-warning{border:2px solid #b91c1c;background:#fff1f2}.mapping-warning h2{color:#991b1b}
.mapping-note{border:2px solid #2563eb;background:#eff6ff}.mapping-note h2{color:#1e40af}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:9px;border-bottom:1px solid #e1e8ef;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#eaf0f5}tr:nth-child(even){background:#f8fafc}figure{margin:24px 0}figure img{width:100%;height:auto;border:1px solid #e5e9ee}
figcaption,footer{font-size:12px;color:#57687a}summary{cursor:pointer;padding:12px;background:#edf2f7;margin:12px 0}
@media(max-width:760px){main{padding:12px}header,section{padding:18px}h1{font-size:24px}}'''
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Passive gaze 与来源 PSM</title><style>' + css + '</style><main>'
                + ''.join(parts) + '<footer>图表由本目录的 passive 分析结果生成。</footer></main></html>')
    (output / 'report.html').write_text(document, encoding='utf-8')
    readme = f'''# Passive gaze 与来源 PSM

[report.html](report.html) 包含中文离线报告与十三张 PNG/PDF 图。

本次纳入 {n_records} 条观看记录、{n_observers} 位观察者与 {n_sources} 个来源视频。
gaze 取自观看者 NPY，stage 和 DLC 取自 source_content_id 对应的操作视频。

固定左右器械分别分析。41 Hz，最大插值间隙 0.1 秒，主 lag 窗口 ±2 秒。
实际 pooled min_overlap 为 {config.get('min_overlap', 2)} 秒。
Position 使用 Box3，velocity 从原始网格坐标独立做二阶 SG5 后差分。
组内按观察者聚合与重抽样。完整 lag 支持规则与预定敏感性分支见 run_manifest.json。

Unpaired reference 使用同 task、不同观察者的 donor，保留共享来源视频的配对。
预定 A 五阶段与 C 四阶段均展示，缺失支持保留 NA。
数据覆盖、排除原因与来源映射见 passive_metadata_audit.json 和输入清单。
验证见 validation.json，冻结方法对照见 methods_consistency.csv。
'''
    if conditional:
        readme += ('\n当前结果以现有 NPY 帧映射与 source-content-clock 为条件。'
                   '既有 NPY 与实际显示帧时间的对应尚未通过完整核验。'
                   '数值方法一致性通过与完整输入映射分开报告。\n')
    elif row_mapping:
        readme += ('\n按用户指定采用 NPY 第 i 行、视频第 i 帧与 DLC 第 i 行逐行对应，'
                   '时间轴沿用精确 video_ts。原 VID_FRAME 差异保留审计。'
                   '实际播放时序尚无经验验证，lag 按冻结基线作探索性描述。\n')
    (output / 'README.md').write_text(readme, encoding='utf-8')
    return dict(report=str(output / 'report.html'),
                figures=[path.name for path in sorted((output / 'figures').glob('*.png'))],
                n_records=n_records, n_observers=n_observers, n_sources=n_sources,
                n_primary_supported_groups=len(finite), n_predefined_primary_groups=len(primary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    print(json.dumps(generate_report(parser.parse_args().output), ensure_ascii=False, indent=2))


