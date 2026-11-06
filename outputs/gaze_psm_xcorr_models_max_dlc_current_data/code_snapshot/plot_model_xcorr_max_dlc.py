"""Descriptive paired comparisons for independently selected maximum-xcorr DLC PSMs.

Reads comparison_results.json only. No score data are read or analyzed.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

MODELS = ['all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot']
SOURCES = ['gt', *MODELS]
LABELS = {'gt': 'Recorded gaze', 'all_ABCD': 'All tasks', 'single_task': 'Single task',
          'leave_out': 'Leave task out', 'spd_finetune': 'SPD fine-tuned', 'spd_zero_shot': 'SPD zero-shot'}
SHORT = {'gt': 'GT', 'all_ABCD': 'All', 'single_task': 'Single', 'leave_out': 'Leave-out',
         'spd_finetune': 'SPD-FT', 'spd_zero_shot': 'SPD-zero'}
COLORS = dict(zip(SOURCES, ['#111827', '#1672b8', '#df8b17', '#329563', '#cc4b53', '#9168ad']))
MODES = ['psm_stage', 'within_stage']
MODE_LABELS = {'psm_stage': 'PSM-stage anchored', 'within_stage': 'Within occurrence'}
STAGES = {'A': ['reach ring', 'grab ring', 'transfer ring', 'place ring', 'drop ring'],
          'C': ['reach string', 'wrap around psm', 'pull through and cinch', 'threading string through tower']}
STAGE_LABELS = {'reach ring': 'Reach ring', 'grab ring': 'Grab ring', 'transfer ring': 'Transfer ring',
                'place ring': 'Place ring', 'drop ring': 'Drop ring', 'reach string': 'Reach string',
                'wrap around psm': 'Wrap around PSM', 'pull through and cinch': 'Pull through and cinch',
                'threading string through tower': 'Thread string through tower'}
SCOPES = ['own_eligible', 'common5', 'common5_same_gt_side']
SCOPE_LABELS = {'own_eligible': 'Each model\'s eligible pairs', 'common5': 'Same eligible pairs for all five models',
                'common5_same_gt_side': 'Common five-model pairs with all models selecting the GT side'}
CONDITION = ('content_id', 'task', 'signal', 'stage', 'mode')
GROUP = ('task', 'signal', 'stage', 'mode')


def isfinite(value):
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def yes(value):
    return value is True or str(value).lower() == 'true'


def values(items):
    return np.asarray([np.nan if x is None else x for x in items], dtype=float)


def key(row, fields=CONDITION):
    return tuple(row.get(k) for k in fields)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def source_key(row):
    return (*key(row), row.get('side'))


def prepare(data):
    curves = {s: {source_key(c): c for c in data['models'][s].get('record_curves', [])} for s in SOURCES}
    rows = []
    band = float(data['config'].get('synchronous_band_s', 1/float(data['config'].get('hz', 41.))))
    direction = lambda lag: None if not isfinite(lag) else 'lead' if lag > band else 'lag' if lag < -band else 'synchronous'
    seen = set()
    for original in data.get('paired_records', []):
        if original.get('model') not in MODELS or original.get('side') != 'max' or original.get('mode') not in MODES:
            continue
        row = dict(original)
        identity = (row['model'], *key(row))
        if identity in seen:
            raise ValueError(f'Duplicate model-condition row: {identity}')
        seen.add(identity)
        gt = curves['gt'].get(source_key(row), {})
        model = curves[row['model']].get(source_key(row), {})
        row['gt_selected_side'] = row.get('gt_selected_side', gt.get('selected_side'))
        row['model_selected_side'] = row.get('model_selected_side', model.get('selected_side'))
        defined_sides = row['gt_selected_side'] in ['l', 'r'] and row['model_selected_side'] in ['l', 'r']
        row['same_selected_side'] = (row['gt_selected_side'] == row['model_selected_side']) if defined_sides else None
        row['eligible'] = yes(row.get('eligible'))
        for suffix in ['lag_s', 'peak_r', 'r_zero']:
            gt_field = 'gt_peak_lag_s' if suffix == 'lag_s' else 'gt_'+suffix
            model_field = 'model_peak_lag_s' if suffix == 'lag_s' else 'model_'+suffix
            delta_field = 'delta_'+suffix
            if isfinite(row.get(gt_field)) and isfinite(row.get(model_field)):
                row[delta_field] = float(row[model_field])-float(row[gt_field])
        row['abs_lag_error_s'] = abs(float(row['delta_lag_s'])) if isfinite(row.get('delta_lag_s')) else None
        row['gt_direction'] = direction(row.get('gt_peak_lag_s'))
        row['model_direction'] = direction(row.get('model_peak_lag_s'))
        row['direction_agreement'] = (row['gt_direction'] == row['model_direction']) if row['gt_direction'] is not None and row['model_direction'] is not None else None
        rows.append(row)
    eligible_sets = [{key(r) for r in rows if r['model'] == model and r['eligible']} for model in MODELS]
    common = set.intersection(*eligible_sets) if eligible_sets else set()
    same_sets = [{key(r) for r in rows if r['model'] == model and r['eligible'] and r['same_selected_side'] is True} for model in MODELS]
    common_same = set.intersection(*same_sets) if same_sets else set()
    for row in rows:
        row['in_common5'] = key(row) in common
        row['in_common5_same_gt_side'] = key(row) in common_same
    return rows, common, common_same


def selected(rows, scope):
    if scope == 'own_eligible':
        return [r for r in rows if r['eligible']]
    field = 'in_common5' if scope == 'common5' else 'in_common5_same_gt_side'
    return [r for r in rows if r[field]]


def participant_mean(rows, metric):
    by_user = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool):
            value = float(value)
        if isfinite(value):
            by_user[row['user']].append(float(value))
    return float(np.mean([np.mean(v) for v in by_user.values()])) if by_user else None


def summary_metrics(total, part):
    answer = dict(n_total_conditions=len(total), n_eligible_conditions=sum(r['eligible'] for r in total),
                  n_conditions=len(part), n_videos=len({r['content_id'] for r in part}),
                  n_participants=len({r['user'] for r in part}),
                  participants=';'.join(sorted({r['user'] for r in part})))
    for field, name in [('abs_lag_error_s', 'mean_abs_max_lag_difference_s'),
                        ('delta_lag_s', 'mean_signed_max_lag_difference_s'),
                        ('delta_peak_r', 'mean_delta_peak_r'), ('delta_r_zero', 'mean_delta_r_zero'),
                        ('direction_agreement', 'direction_agreement'), ('same_selected_side', 'side_agreement')]:
        answer['participant_equal_'+name] = participant_mean(part, field)
    lags = [float(r['abs_lag_error_s']) for r in part if isfinite(r.get('abs_lag_error_s'))]
    answer['condition_median_abs_max_lag_difference_s'] = float(np.median(lags)) if lags else None
    for field in ['direction_agreement', 'same_selected_side']:
        defined = [r[field] for r in part if r.get(field) is not None]
        answer['condition_'+field] = float(np.mean(defined)) if defined else None
    return answer


def summaries(root, rows, common, common_same):
    outputs = {}
    for filename, fields in [('paired_summary.csv', ('model', *GROUP)),
                             ('paired_overview.csv', ('model', 'task', 'signal', 'mode'))]:
        bundles = defaultdict(list)
        for row in rows:
            bundles[key(row, fields)].append(row)
        result = []
        for grouping, total in sorted(bundles.items()):
            for scope in SCOPES:
                result.append(dict(zip(fields, grouping), scope=scope, **summary_metrics(total, selected(total, scope))))
        write_csv(root/filename, result)
        outputs[filename] = result
    write_csv(root/'own_eligible_pairs.csv', selected(rows, 'own_eligible'))
    write_csv(root/'common5_pairs.csv', selected(rows, 'common5'))
    write_csv(root/'common5_same_gt_side_pairs.csv', selected(rows, 'common5_same_gt_side'))
    write_csv(root/'common5_conditions.csv', [dict(zip(CONDITION, k)) for k in sorted(common)])
    write_csv(root/'common5_same_gt_side_conditions.csv', [dict(zip(CONDITION, k)) for k in sorted(common_same)])
    write_csv(root/'common5'/'position_psm_stage_summary.csv', [
        row for row in outputs['paired_overview.csv']
        if row['signal'] == 'position' and row['mode'] == 'psm_stage'
        and row['scope'] in ['common5', 'common5_same_gt_side']])
    return outputs


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15, linewidth=.65)
    ax.set_axisbelow(True)


def handles():
    return [Line2D([], [], color=COLORS[s], lw=2.2 if s == 'gt' else 1.5, label=LABELS[s]) for s in SOURCES]


def ordered_stages(rows, task):
    actual = {r['stage'] for r in rows if r.get('task') == task and r.get('mode') in MODES}
    return [s for s in STAGES[task] if s in actual] + sorted(actual-set(STAGES[task]))


class FigureWriter:
    def __init__(self, root):
        self.root = root
        (root/'figures').mkdir(exist_ok=True)
        self.book = PdfPages(root/'figures'/'comparison_report.pdf')
        self.items = []

    def save(self, fig, relative, kind, **meta):
        stem = self.root/'figures'/relative
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix('.png'), dpi=160, facecolor='white')
        fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
        self.book.savefig(fig, facecolor='white')
        self.items.append(dict(kind=kind, png=str(stem.with_suffix('.png').relative_to(self.root)),
                               pdf=str(stem.with_suffix('.pdf').relative_to(self.root)), **meta))
        plt.close(fig)
        print(f'Figure {len(self.items):02d}: {relative}', flush=True)

    def close(self):
        self.book.close()
        counts = {kind: sum(f['kind'] == kind for f in self.items) for kind in sorted({f['kind'] for f in self.items})}
        payload = dict(figure_count=len(self.items), png_count=len(self.items), individual_pdf_count=len(self.items),
                       combined_pdf='figures/comparison_report.pdf', combined_pdf_pages=len(self.items), kinds=counts,
                       source_policy='Each source independently selects one DLC PSM by largest signed peak; a pair may compare different PSMs.',
                       figures=self.items)
        (self.root/'comparison_figure_manifest.json').write_text(json.dumps(clean(payload), indent=2), encoding='utf-8')


def plot_groups(data, writer):
    lookup = {s: {key(c, GROUP): c for c in data['models'][s].get('group_curves', [])}
              for s in SOURCES}
    all_curves = [c for source in data['models'].values() for c in source.get('group_curves', [])]
    for task in ['A', 'C']:
        for signal in ['position', 'velocity']:
            selected_curves = [c for c in all_curves if c['task'] == task and c['signal'] == signal]
            stages = ordered_stages(selected_curves, task)
            if not stages:
                continue
            height = 2.45*len(stages)+1.65
            fig, axes = plt.subplots(len(stages), 2, figsize=(13, height), squeeze=False, sharex=True, sharey=True)
            support = []
            for row, stage in enumerate(stages):
                for col, mode in enumerate(MODES):
                    ax = axes[row, col]; style(ax)
                    ax.axvline(0, color='#aaaaaa', lw=.8); ax.axhline(0, color='#aaaaaa', lw=.8)
                    counts = []
                    for source in SOURCES:
                        curve = lookup[source].get((task, signal, stage, mode))
                        n = curve.get('n_stable_users', 0) if curve else 0
                        counts.append(f'{SHORT[source]}={n}')
                        support.append(dict(source=source, task=task, signal=signal, stage=stage, mode=mode, stable_users=n))
                        if curve:
                            ax.plot(values(curve['lag_s']), values(curve['r']), color=COLORS[source],
                                    lw=2.2 if source == 'gt' else 1.4, alpha=.95)
                    ax.set_title(f'{STAGE_LABELS.get(stage,stage)} | {MODE_LABELS[mode]}', fontsize=10.5)
                    ax.text(.015,.025,'Stable users: '+', '.join(counts), transform=ax.transAxes, fontsize=7.5, color='#555555')
                    ax.set_ylim(-1.03,1.03)
                    if col == 0:
                        ax.set_ylabel('2-D correlation r')
                    if row == len(stages)-1:
                        ax.set_xlabel('Lag (s); positive = gaze leads')
            fig.suptitle(f'Task {task}: {signal} xcorr with independently selected DLC PSM', fontsize=14.5, y=.987)
            fig.legend(handles=handles(), loc='upper center', bbox_to_anchor=(.5,.953), ncol=3, frameon=False, fontsize=9)
            fig.text(.5,.02,'Each source retains its selected side\'s whole curve. Sides and stable-user support may differ between sources.', ha='center', fontsize=8.5, color='#555555')
            fig.subplots_adjust(left=.07,right=.985,top=1-1.5/height,bottom=max(.085,.8/height),hspace=.35,wspace=.12)
            writer.save(fig, f'group_curves/task_{task}_{signal}', 'group_curves', task=task, signal=signal, support=support)


def primary(rows, scope):
    return [r for r in selected(rows, scope) if r['signal']=='position' and r['mode']=='psm_stage']


def plot_lag_scatter(data, rows, writer):
    limit=float(data['config'].get('max_lag',2.))
    for scope in ['own_eligible','common5']:
        subset=primary(rows,scope)
        fig,axes=plt.subplots(2,3,figsize=(13.5,9),sharex=True,sharey=True)
        counts=[]
        for ax,model in zip(axes.flat,MODELS):
            style(ax);ax.plot([-limit,limit],[-limit,limit],ls='--',color='#999999',lw=1)
            ax.axvline(0,color='#cccccc',lw=.7);ax.axhline(0,color='#cccccc',lw=.7)
            part=[r for r in subset if r['model']==model and isfinite(r.get('gt_peak_lag_s')) and isfinite(r.get('model_peak_lag_s'))]
            task_counts=[]
            for task,marker,color in [('A','o','#1672b8'),('C','^','#df8b17')]:
                rr=[r for r in part if r['task']==task];task_counts.append(f'{task}={len(rr)}')
                ax.scatter([r['gt_peak_lag_s'] for r in rr],[r['model_peak_lag_s'] for r in rr],
                           s=35,marker=marker,color=color,alpha=.75,edgecolor='white',linewidth=.35)
            ax.set_title(f'{LABELS[model]} | n={len(part)}\n'+', '.join(task_counts),fontsize=10.5)
            ax.set_xlabel('GT selected-side peak lag (s)');ax.set_ylabel('Model selected-side peak lag (s)')
            ax.set_xlim(-limit-.08,limit+.08);ax.set_ylim(-limit-.08,limit+.08);ax.set_aspect('equal',adjustable='box')
            counts.append(dict(model=model,n_conditions=len(part),n_participants=len({r['user'] for r in part})))
        axes.flat[-1].axis('off')
        axes.flat[-1].legend(handles=[Line2D([],[],marker='o',ls='',color='#1672b8',label='Task A'),
                                     Line2D([],[],marker='^',ls='',color='#df8b17',label='Task C'),
                                     Line2D([],[],ls='--',color='#999999',label='Equal peak lag')],loc='center',frameon=False)
        fig.suptitle('Independent maximum-xcorr lag comparison\n'+SCOPE_LABELS[scope],fontsize=15,y=.985)
        fig.text(.5,.022,'Position, PSM-stage anchored. A point compares selected-side statistics; GT and model need not select the same instrument.',ha='center',fontsize=8.5,color='#555555')
        fig.tight_layout(rect=(.01,.055,.99,.91))
        writer.save(fig,f'paired_comparison/lag_scatter_{scope}','lag_scatter',scope=scope,counts=counts)


def plot_differences(rows,writer):
    for scope in ['own_eligible','common5']:
        subset=primary(rows,scope)
        for metric,ylabel,title in [('delta_lag_s','Model - GT selected-side peak lag (s)','Peak-lag differences'),
                                    ('delta_peak_r','Model - GT selected-side peak correlation','Peak-correlation differences')]:
            plotted=[abs(float(r[metric])) for r in subset if isfinite(r.get(metric))]
            bound=max(.05,max(plotted,default=0.)*1.12)
            fig,axes=plt.subplots(1,2,figsize=(13.5,6.2),sharey=True)
            counts=[]
            for ax,task in zip(axes,['A','C']):
                style(ax);ax.axhline(0,color='#777777',lw=1)
                labels=[]
                for j,model in enumerate(MODELS,1):
                    rr=[r for r in subset if r['task']==task and r['model']==model and isfinite(r.get(metric))]
                    vals=np.asarray([r[metric] for r in rr],dtype=float)
                    if len(vals):
                        bp=ax.boxplot([vals],positions=[j],widths=.5,patch_artist=True,showfliers=False,manage_ticks=False,
                                      medianprops={'color':'#111111','lw':1.4},whiskerprops={'color':COLORS[model]},capprops={'color':COLORS[model]})
                        bp['boxes'][0].set(facecolor=COLORS[model],edgecolor=COLORS[model],alpha=.18)
                        offsets=np.linspace(-.16,.16,len(vals)) if len(vals)>1 else np.zeros(1)
                        ax.scatter(j+offsets,vals,s=24,color=COLORS[model],alpha=.8,edgecolor='white',linewidth=.3,zorder=3)
                    labels.append(LABELS[model].replace(' ','\n',1)+f'\nn={len(vals)}')
                    counts.append(dict(task=task,model=model,n_conditions=len(rr),n_participants=len({r['user'] for r in rr})))
                ax.set_xticks(range(1,6),labels,fontsize=8.5);ax.set_xlim(.5,5.5);ax.set_ylim(-bound,bound)
                ax.set_title(f'Task {task}',fontsize=12);ax.set_ylabel(ylabel)
            fig.suptitle(title+'\n'+SCOPE_LABELS[scope],fontsize=15,y=.985)
            fig.text(.5,.025,'Each dot is one recording-stage pair. Independently selected sides may differ. Boxes show descriptive spread, not confidence intervals.',ha='center',fontsize=8.5,color='#555555')
            fig.subplots_adjust(left=.075,right=.99,top=.81,bottom=.19,wspace=.15)
            writer.save(fig,f'paired_comparison/{metric}_{scope}','paired_differences',scope=scope,metric=metric,counts=counts)


def side_summary(rows):
    output=[]
    bundles=defaultdict(list)
    for row in rows:
        bundles[(row['model'],row['task'],row['signal'],row['mode'])].append(row)
    for identity,all_rows in sorted(bundles.items()):
        for scope in ['all_defined_selections','own_eligible','common5','common5_same_gt_side']:
            candidates=all_rows if scope=='all_defined_selections' else selected(all_rows,scope)
            part=[r for r in candidates if r['gt_selected_side'] in ['l','r'] and r['model_selected_side'] in ['l','r']]
            entry=dict(zip(('model','task','signal','mode'),identity),scope=scope,n_candidate_conditions=len(candidates),
                       n_defined_conditions=len(part),n_missing_sides=len(candidates)-len(part),n_participants=len({r['user'] for r in part}),
                       condition_agreement=float(np.mean([r['same_selected_side'] for r in part])) if part else None,
                       participant_equal_agreement=participant_mean(part,'same_selected_side'))
            for gt in ['l','r']:
                for model in ['l','r']:
                    entry[f'gt_{gt}_model_{model}']=sum(r['gt_selected_side']==gt and r['model_selected_side']==model for r in part)
            output.append(entry)
    return output


def plot_side_selection(rows,writer):
    table=side_summary(rows)
    write_csv(writer.root/'side_selection_concordance.csv',table)
    for task in ['A','C']:
        parts=[r for r in table if r['task']==task and r['signal']=='position' and r['mode']=='psm_stage' and r['scope']=='all_defined_selections']
        lookup={r['model']:r for r in parts}
        fig,axes=plt.subplots(2,3,figsize=(12.8,8.5))
        maximum=max((max(r[k] for k in ['gt_l_model_l','gt_l_model_r','gt_r_model_l','gt_r_model_r']) for r in parts),default=1)
        for ax,model in zip(axes.flat,MODELS):
            row=lookup.get(model,{});matrix=np.asarray([[row.get('gt_l_model_l',0),row.get('gt_l_model_r',0)],
                                                       [row.get('gt_r_model_l',0),row.get('gt_r_model_r',0)]])
            ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(maximum,1))
            for (y,x),value in np.ndenumerate(matrix):
                ax.text(x,y,str(int(value)),ha='center',va='center',fontsize=16,color='white' if value>maximum*.6 else '#111111')
            ax.set_xticks([0,1],['L','R']);ax.set_yticks([0,1],['L','R']);ax.set_xlabel('Model-selected PSM');ax.set_ylabel('GT-selected PSM')
            agreement=row.get('condition_agreement');text=f'{100*agreement:.1f}%' if isfinite(agreement) else 'NA'
            ax.set_title(f'{LABELS[model]}\nAgreement {text} | n={row.get("n_defined_conditions",0)}',fontsize=10.5)
        ax=axes.flat[-1];style(ax)
        rates=[100*lookup[m]['condition_agreement'] if m in lookup and isfinite(lookup[m].get('condition_agreement')) else 0. for m in MODELS]
        ax.barh(range(5),rates,color=[COLORS[m] for m in MODELS],height=.6)
        ax.set_yticks(range(5),[SHORT[m] for m in MODELS],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('Same-side selections (%)')
        for y,(model,rate) in enumerate(zip(MODELS,rates)):
            n=lookup.get(model,{}).get('n_defined_conditions',0)
            ax.text(rate+1.5,y,f'{rate:.0f}%' if n else 'NA',va='center',fontsize=8)
        ax.set_title('Selection agreement',fontsize=11)
        fig.suptitle(f'Task {task}: selected DLC instrument agreement',fontsize=15,y=.985)
        fig.text(.5,.025,'Position, PSM-stage anchored. Counts use all defined L/R selections before peak-eligibility filtering. Selection does not identify the operated instrument.',ha='center',fontsize=8,color='#555555')
        fig.tight_layout(rect=(.01,.06,.99,.94),h_pad=2.0,w_pad=2.)
        writer.save(fig,f'side_selection/task_{task}_position','side_selection',task=task,counts=parts)


def plot_paired_groups(data, writer):
    all_rows = data.get('paired_groups', [])
    for task in ['A', 'C']:
        rows = [r for r in all_rows if r['task'] == task and r['signal'] == 'position'
                and r['mode'] == 'psm_stage' and r['side'] == 'max']
        stages = ordered_stages(rows, task)
        if not stages:
            continue
        lookup = {(r['model'], r['stage']): r for r in rows}
        height = max(6.5, len(stages)*1.05+2.4)
        fig, axes = plt.subplots(1, 2, figsize=(14.5, height), sharey=True)
        for ax, metric, label in zip(axes, ['delta_lag_s', 'delta_peak_r'],
                                    ['Group peak-lag difference (model - GT), s',
                                     'Group peak-correlation difference (model - GT)']):
            style(ax); ax.axvline(0, color='#777777', lw=1.)
            endpoints = []
            for i, stage in enumerate(stages):
                for j, model in enumerate(MODELS):
                    row = lookup.get((model, stage), {})
                    y = i+(j-2)*.14
                    value, low, high = row.get(metric), row.get(metric+'_ci_low'), row.get(metric+'_ci_high')
                    n = row.get('n_shared_users', 0)
                    if isfinite(value):
                        endpoints.append(float(value))
                        if isfinite(low) and isfinite(high):
                            ax.plot([low, high], [y, y], color=COLORS[model], lw=1.3)
                            endpoints.extend([float(low), float(high)])
                        ax.scatter([value], [y], s=26, marker='o' if row.get('status') == 'ok' else 'x',
                                   color=COLORS[model], zorder=3)
                        if metric == 'delta_lag_s':
                            ax.annotate(f'n={n}', (high if isfinite(high) else value, y), xytext=(4, 0),
                                        textcoords='offset points', va='center', fontsize=7, color=COLORS[model])
                    elif metric == 'delta_lag_s':
                        ax.text(.99, y, f'NA (n={n})', transform=ax.get_yaxis_transform(), ha='right',
                                va='center', fontsize=7, color=COLORS[model])
            if endpoints:
                lower, upper = min(0., min(endpoints)), max(0., max(endpoints))
                span = max(upper-lower, .05)
                ax.set_xlim(lower-.06*span, upper+.22*span)
            ax.set_yticks(range(len(stages)), [STAGE_LABELS.get(s, s) for s in stages], fontsize=9)
            ax.set_ylim(len(stages)-.5, -.5)
            ax.set_xlabel(label, fontsize=10)
        fig.suptitle(f'Task {task}: paired group-curve differences', fontsize=15, y=.985)
        fig.legend(handles=handles()[1:], loc='upper center', bbox_to_anchor=(.5,.94),
                   ncol=5, frameon=False, fontsize=9)
        draws = data['config'].get('bootstrap', 1000)
        fig.text(.5, .025, f'Position, PSM-stage anchored. Same complete-lag participants within each model/GT pair; {draws:,} paired participant bootstrap draws.\n'
                 'Lines: unadjusted exploratory 95% intervals. Numbers: shared participants. Cross: nonpositive or boundary group peak.\n'
                 'Independent maximum-xcorr selection may use different PSMs. Group-peak difference is not recording-level lag error; pairs may use different participants.',
                 ha='center', fontsize=8, color='#555555')
        fig.subplots_adjust(left=.20, right=.985, top=.87, bottom=.18, wspace=.13)
        writer.save(fig, f'paired_group_contrasts/task_{task}_position', 'paired_group_contrasts',
                    task=task, counts=[{k:r.get(k) for k in ['model','stage','n_shared_users','status']} for r in rows])


def plot_records(data,writer):
    lookup={s:{key(c):c for c in data['models'][s].get('record_curves',[]) if c['signal']=='position'} for s in SOURCES}
    gtrows=data['models']['gt'].get('record_curves',[])
    for clip in sorted({r['content_id'] for r in gtrows}):
        rows=[r for r in gtrows if r['content_id']==clip and r['signal']=='position'];task=rows[0]['task']
        stages=ordered_stages(rows,task);height=2.45*len(stages)+1.65
        fig,axes=plt.subplots(len(stages),2,figsize=(13,height),squeeze=False,sharex=True,sharey=True)
        selections=[]
        for i,stage in enumerate(stages):
            for j,mode in enumerate(MODES):
                ax=axes[i,j];style(ax);ax.axvline(0,color='#aaaaaa',lw=.8);ax.axhline(0,color='#aaaaaa',lw=.8)
                sides=[];has_curve=False
                for source in SOURCES:
                    curve=lookup[source].get((clip,task,'position',stage,mode));side=curve.get('selected_side') if curve else None
                    sides.append(f'{SHORT[source]}:{side.upper() if side in ["l","r"] else "-"}')
                    selections.append(dict(source=source,stage=stage,mode=mode,selected_side=side))
                    if curve:
                        lag,r=values(curve['lag_s']),values(curve['r']);has_curve|=np.isfinite(r).any()
                        ax.plot(lag,r,color=COLORS[source],lw=2.2 if source=='gt' else 1.4,alpha=.95)
                        if np.isfinite(r).any():
                            peak=int(np.nanargmax(r));boundary=peak in [0,len(r)-1]
                            ax.plot(lag[peak],r[peak],marker='^' if boundary else 'o',markersize=4,color=COLORS[source])
                ax.set_title(f'{STAGE_LABELS.get(stage,stage)} | {MODE_LABELS[mode]}',fontsize=10.5)
                ax.text(.015,.025,'Selected side: '+', '.join(sides),transform=ax.transAxes,fontsize=7.5,color='#555555')
                if not has_curve:
                    ax.text(.5,.5,'No supported correlation',transform=ax.transAxes,ha='center',color='#777777')
                ax.set_ylim(-1.03,1.03)
                if j==0:ax.set_ylabel('2-D correlation r')
                if i==len(stages)-1:ax.set_xlabel('Lag (s); positive = gaze leads')
        fig.suptitle(f'{clip} | Task {task}: independently selected maximum-xcorr DLC PSM',fontsize=14,y=.987)
        fig.legend(handles=handles(),loc='upper center',bbox_to_anchor=(.5,.953),ncol=3,frameon=False,fontsize=9)
        fig.text(.5,.02,'Selected instruments may differ across sources and modes. Dots mark signed maxima; triangles mark search boundaries, not reliable timing.',ha='center',fontsize=8.2,color='#555555')
        fig.subplots_adjust(left=.07,right=.985,top=1-1.5/height,bottom=max(.085,.8/height),hspace=.35,wspace=.12)
        writer.save(fig,f'record_curves/{clip}_position','record_curves',content_id=clip,task=task,selections=selections)


def write_notes(root,summary,common,common_same):
    text='''# Maximum-xcorr DLC model comparison: figure and table notes

The main analysis follows `gaze_psm_xcorr_max_dlc`: each gaze source independently selects the left or right DLC instrument with the largest signed correlation peak for each recording, signal, stage and mode. The winner's entire curve is retained. This is not a pointwise maximum envelope. GT and a model can select different instruments.

`position` with `psm_stage` is the primary descriptive comparison. `velocity` and `within_stage` results remain available. Positive lag means gaze leads PSM. Peak-lag differences compare independently selected maximum-xcorr statistics and should not automatically be interpreted as timing error against the same physical instrument.

## Tables

- `paired_summary.csv`: task × stage × signal × mode × model summaries.
- `paired_overview.csv`: task × signal × mode × model summaries across stages. Conditions are first averaged within each participant, then participants receive equal weight for fields prefixed `participant_equal_`.
- `common5/position_psm_stage_summary.csv`: the primary common-five comparison and the all-five-same-GT-side sensitivity summary, distinguished by `scope`.
- Every summary contains `own_eligible`, `common5`, and `common5_same_gt_side` scopes. `own_eligible` uses each model's eligible subset. `common5` fixes exactly the same eligible recording-stage conditions across all five models. `common5_same_gt_side` further requires every model to select the same instrument as GT.
- `common5_pairs.csv` and `common5_same_gt_side_pairs.csv` are long tables retaining all five models' values. The corresponding `_conditions.csv` files list unique condition identities.
- `side_selection_concordance.csv` gives selection confusion counts and agreement. `all_defined_selections` includes defined L/R selections before peak-eligibility filtering; other scopes are listed separately.
- `participant_equal_mean_abs_max_lag_difference_s` is the participant-weighted mean of absolute recording-stage max-lag differences. `condition_median_abs_max_lag_difference_s` is a condition-level median. These are different estimands and are named explicitly.
- Direction agreement uses the saved synchronous band. A small negative and positive lag within that band both count as synchronous.

## Figures

- Four group overlays: Task A/C × position/velocity. Rows are all stages; columns are PSM-stage and within-occurrence modes. Group overlays show each source's own stable-user count and may not share identical support.
- Two lag-agreement scatter figures: own eligible and common five-model support.
- Four difference figures: peak lag and peak correlation, each on own eligible and common five-model support.
- Two selected-side confusion/selection-agreement figures: Task A/C, position PSM-stage, all defined selections before eligibility filtering.
- Two paired group-curve difference figures: shared complete-lag participants within each model/GT pair, with paired participant-bootstrap 95% intervals. These support sets can differ between model pairs. Crosses mark nonpositive or boundary group peaks; unestimable rows are marked NA. These exploratory intervals are not adjusted for multiple comparisons.
- Ten per-video position curve figures. Each panel records every source's selected instrument. Boundary peaks use triangle markers.
- PNG and PDF versions are in `figures/`. `figures/comparison_report.pdf` combines the comparison figures. `comparison_figure_manifest.json` records the complete list.

Repeated stages from the same person are not independent observations. Boxes and scatter points describe the observed sample; no independent-condition confidence interval or score correlation is computed by this script. Group-difference intervals use the saved paired participant bootstrap. The same-side subset may be small and is a sensitivity analysis, not a replacement for the original independent-selection method.
'''
    text+=f'\nUnique common-five conditions across tasks/signals/modes: {len(common)}. Common-five same-GT-side conditions: {len(common_same)}.\n'
    (root/'comparison_notes.md').write_text(text,encoding='utf-8')
    overview=summary['paired_overview.csv']
    core=[r for r in overview if r['signal']=='position' and r['mode']=='psm_stage']
    (root/'comparison_summary.json').write_text(json.dumps(clean(dict(primary_overview=core,common5_conditions=len(common),
                                                                    common5_same_gt_side_conditions=len(common_same))),indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();root=args.output
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    data=json.loads((root/'comparison_results.json').read_text(encoding='utf-8'))
    for source in SOURCES:
        if source not in data['models']:raise ValueError(f'Missing source: {source}')
        for curve in data['models'][source].get('record_curves',[]):
            if curve.get('side')!='max' or curve.get('mode') not in MODES:
                raise ValueError('This script expects maximum-side stage curves only.')
    rows,common,common_same=prepare(data)
    summary=summaries(root,rows,common,common_same)
    write_notes(root,summary,common,common_same)
    if not args.summary_only:
        writer=FigureWriter(root)
        try:
            plot_groups(data,writer)
            plot_lag_scatter(data,rows,writer)
            plot_differences(rows,writer)
            plot_side_selection(rows,writer)
            plot_paired_groups(data,writer)
            plot_records(data,writer)
        finally:
            writer.close()
    snapshot=root/'code_snapshot';snapshot.mkdir(exist_ok=True)
    destination=snapshot/Path(__file__).name
    if destination.resolve()!=Path(__file__).resolve():shutil.copy2(__file__,destination)
    print(json.dumps(dict(paired_rows=len(rows),common5_conditions=len(common),common5_same_gt_side_conditions=len(common_same))),flush=True)


if __name__=='__main__':main()
