"""Independent checks of exported paired aggregation and bootstrap results."""
from functools import lru_cache
import argparse
import json
import traceback
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from validate_passive16 import WORK, OUTPUT, check, same, CHECKS


@lru_cache(None)
def resample_weights(n):
    return np.random.default_rng(np.random.SeedSequence([42, n])).multinomial(n, np.full(n, 1/n), size=10000)


def interval(v):
    return cached_interval(tuple(float(x) for x in v))


@lru_cache(maxsize=200000)
def cached_interval(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return dict(ci_low=np.nan, ci_high=np.nan, bootstrap_defined=0, bootstrap_undefined=0)
    samples = resample_weights(len(v)) @ v / len(v)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return dict(ci_low=lo, ci_high=hi, bootstrap_defined=10000, bootstrap_undefined=0)


def verify_aggregate(out, model=False):
    prefix = 'model_' if model else ''
    raw = pd.read_csv(out/('model_pair_metrics.csv' if model else 'pair_metrics.csv'))
    content = pd.read_csv(out/(prefix+'content_stage_summary.csv'))
    persons = pd.read_csv(out/(prefix+'participant_stage_summary.csv'))
    summary = pd.read_csv(out/(prefix+'group_stage_summary.csv'))
    context = (['analysis_branch', 'model'] if model else [])+['task', 'scenario', 'stage']
    metrics = [c[7:] for c in raw if c.startswith('active_') and 'passive_'+c[7:] in raw and not c.endswith('reason') and c in content]
    check(prefix+'required_core_metrics_exported', {'L_T', 'L_P', 'L_O', 'L_N', 'L_target', 'L_gripper', 'C', 'D', 'B', 'C_excess'} <= set(metrics))
    c_index = content.set_index(context+['participant', 'content_id'])
    p_index = persons.set_index(context+['participant'])
    s_index = summary.set_index(context)
    check(prefix+'unique_content_rows', not c_index.index.has_duplicates)
    check(prefix+'unique_participant_rows', not p_index.index.has_duplicates)
    check(prefix+'unique_group_rows', not s_index.index.has_duplicates)
    for ctx, group in raw.groupby(context, sort=True, dropna=False):
        ctx = ctx if isinstance(ctx, tuple) else (ctx,)
        group_row = s_index.loc[ctx]
        for metric in metrics:
            domains = ['active', 'passive']+(['model'] if model and 'model_'+metric in group else [])
            cols = [domain+'_'+metric for domain in domains]
            numeric = group[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
            paired = group[np.isfinite(numeric).all(1)].copy()
            by_content = paired.groupby(['participant', 'content_id'], sort=True)[cols].mean()
            by_content['delta_'+metric] = by_content['passive_'+metric]-by_content['active_'+metric]
            for key, expected in by_content.iterrows():
                actual = c_index.loc[ctx+key]
                for domain in domains+['delta']:
                    col = domain+'_'+metric
                    same(prefix+'/'.join(map(str, ctx+key))+'/'+col, actual[col], expected[col], atol=1e-8)
            by_person = by_content.groupby('participant', sort=True).mean()
            for person, expected in by_person.iterrows():
                actual = p_index.loc[ctx+(person,)]
                for domain in domains+['delta']:
                    col = domain+'_'+metric
                    same(prefix+'/'.join(map(str, ctx+(person,)))+'/'+col, actual[col], expected[col], atol=1e-8)
            for domain in domains+['delta']:
                col = domain+'_'+metric
                vals = by_person[col].to_numpy() if col in by_person else np.array([])
                tag = prefix+'/'.join(map(str, ctx))+'/'+col
                same(tag+'/mean', group_row[col], vals.mean() if len(vals) else np.nan, atol=1e-8)
                for name, value in interval(vals).items():
                    same(tag+'/'+name, group_row[col+'_'+name], value, atol=1e-8)
            for name, expected in [('n_participants', len(by_person)), ('n_contents', len(by_content)),
                                   ('n_pairs', len(paired)), ('n_observers', paired.observer_id.nunique())]:
                same(prefix+'/'.join(map(str, ctx))+'/'+name+'_'+metric, group_row[name+'_'+metric], expected)
    for frame, name in [(content, 'content'), (persons, 'participant'), (summary, 'group')]:
        ac = [c for c in frame if c.startswith('active_')]
        check(prefix+name+'_every_active_column_has_passive', all('passive_'+c[7:] in frame for c in ac))
    long = pd.read_csv(out/(prefix+'group_stage_summary_long.csv'))
    for key, g in long.groupby(context+['metric'], dropna=False):
        check(prefix+'long_has_both_domains/'+str(key), {'active', 'passive', 'delta'} <= set(g.domain))


def verify_correlations(out):
    points = pd.read_csv(out/'performance_participants.csv')
    corrs = pd.read_csv(out/'performance_correlations.csv')
    for row in corrs.itertuples():
        p = points[(points.task == row.task) & (points.scenario == row.scenario) & (points.stage == row.stage) & (points.metric == row.metric)]
        same('performance_n/'+str((row.task, row.scenario, row.stage, row.metric)), row.n_participants, len(p))
        for domain in ['active', 'passive']:
            rho = spearmanr(p.score, p[domain]).statistic if len(p) >= 3 and p.score.nunique() > 1 and p[domain].nunique() > 1 else np.nan
            same('performance_rho/'+str((row.task, row.scenario, row.stage, row.metric, domain)), getattr(row, domain+'_rho'), rho)
        same('performance_delta_rho/'+str((row.task, row.scenario, row.stage, row.metric)), row.delta_rho, row.passive_rho-row.active_rho)


def verify_bss(out):
    content = pd.read_csv(out/'model_content_stage_summary.csv')
    bss = pd.read_csv(out/'model_brier_skill.csv')
    for row in bss.itertuples():
        g = content[(content.analysis_branch == row.analysis_branch) & (content.model == row.model) &
                    (content.task == row.task) & (content.scenario == row.scenario) & (content.stage == row.stage)]
        cols = [f'{domain}_BS_{row.event}{suffix}' for domain in ['active', 'passive'] for suffix in ['', '_'+row.reference]]
        g = g[np.isfinite(g[cols].to_numpy(float)).all(1)]
        p = g.groupby('participant', sort=True)[cols].mean()
        tag = 'BSS/'+str((row.analysis_branch, row.model, row.task, row.scenario, row.stage, row.event, row.reference))
        same(tag+'/n_participants', row.n_participants, len(p))
        for domain in ['active', 'passive']:
            loss = p[f'{domain}_BS_{row.event}'].to_numpy()
            baseline = p[f'{domain}_BS_{row.event}_{row.reference}'].to_numpy()
            point = 1-loss.mean()/baseline.mean() if len(p) and baseline.mean() > 0 else np.nan
            same(tag+'/'+domain, getattr(row, domain+'_BSS'), point)
            if len(p) >= 2 and baseline.mean() > 0:
                numerator = resample_weights(len(p)) @ loss
                denominator = resample_weights(len(p)) @ baseline
                v = 1-numerator[denominator > 0]/denominator[denominator > 0]
                lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
                defined = len(v)
            else:
                lo, hi, defined = np.nan, np.nan, 0
            same(tag+'/'+domain+'_ci_low', getattr(row, domain+'_ci_low'), lo)
            same(tag+'/'+domain+'_ci_high', getattr(row, domain+'_ci_high'), hi)
            same(tag+'/'+domain+'_bootstrap_defined', getattr(row, domain+'_bootstrap_defined'), defined)
        same(tag+'/delta', row.delta_BSS, row.passive_BSS-row.active_BSS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['human', 'model', 'all'], default='all')
    args = parser.parse_args()
    out = OUTPUT
    report = {'scope': 'independent paired hierarchy, means, 10000-draw percentile CIs, Spearman point estimates and BSS'}
    try:
        if (out/'statistics_manifest.json').exists():
            manifest = json.loads((out/'statistics_manifest.json').read_text(encoding='utf-8'))
            check('bootstrap_42_10000', manifest['bootstrap_seed'] == 42 and manifest['bootstrap_replicates'] == 10000)
            check('bootstrap_source_operator', manifest['bootstrap_unit'] == 'source_operator')
        else:
            manifest = json.loads((out/'run_config.json').read_text(encoding='utf-8'))
            check('declared_bootstrap_42_10000', manifest['bootstrap_seed'] == 42 and manifest['bootstrap_n'] == 10000)
        if args.part in ('human', 'all'):
            verify_aggregate(out)
            verify_correlations(out)
        if args.part in ('model', 'all'):
            verify_aggregate(out, model=True)
            verify_bss(out)
        report['status'] = 'passed' if not any(not c['passed'] for c in CHECKS) else 'failed'
    except Exception:
        check('statistics_validation_exception', False, traceback.format_exc())
        report['status'] = 'failed'
    report.update(check_count=len(CHECKS), failed_checks=[c for c in CHECKS if not c['passed']],
                  successful_check_count=sum(c['passed'] for c in CHECKS),
                  check_storage='All failures retained; successful assertions counted.')
    path = WORK/'validation'/('independent_'+args.part+'_statistics_validation.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(CHECKS), 'failures': len(report['failed_checks'])}), flush=True)
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
