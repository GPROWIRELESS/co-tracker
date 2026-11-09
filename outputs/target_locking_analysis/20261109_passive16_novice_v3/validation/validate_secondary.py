"""Independent checks of round contrasts and leave-one-operator correlations."""
import json
import traceback
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from validate_passive16 import WORK, OUTPUT, CHECKS, check, same
from validate_statistics import interval


def main():
    out = OUTPUT
    report = {'scope': 'round contrasts, source-operator pairing, leave-one-out Spearman'}
    try:
        rounds = pd.read_csv(out/'participant_round_summary.csv')
        changes = pd.read_csv(out/'repetition_contrasts.csv')
        content = pd.read_csv(out/'content_stage_summary.csv')
        for _, row in rounds.iterrows():
            p = content[(content.task == row.task) & (content.scenario == row.scenario) & (content.stage == row.stage) &
                        (content.participant == row.participant) & (content['round'] == row['round'])]
            for column in rounds:
                if not column.startswith('active_'):
                    continue
                metric = column[7:]
                cols = ['active_'+metric, 'passive_'+metric]
                use = p[np.isfinite(p[cols].to_numpy(float)).all(1)]
                for domain in ['active', 'passive']:
                    same('round/'+str((row.task, row.scenario, row.stage, row.participant, row['round'], domain, metric)),
                         row[domain+'_'+metric], use[domain+'_'+metric].mean())
                same('round_delta/'+str((row.task, row.scenario, row.stage, row.participant, row['round'], metric)),
                     row['delta_'+metric], use['passive_'+metric].mean()-use['active_'+metric].mean())
        for row in changes.itertuples():
            g = rounds[(rounds.task == row.task) & (rounds.scenario == row.scenario) & (rounds.stage == row.stage)]
            first = g[g['round'] == row.round_from].set_index('participant')
            last = g[g['round'] == row.round_to].set_index('participant')
            ids = first.index.intersection(last.index).sort_values()
            a, b = first.loc[ids], last.loc[ids]
            cols = ['active_'+row.metric, 'passive_'+row.metric]
            good = np.isfinite(a[cols].to_numpy(float)).all(1) & np.isfinite(b[cols].to_numpy(float)).all(1)
            a, b = a.loc[good], b.loc[good]
            tag = 'contrast/'+str((row.task, row.scenario, row.stage, row.round_from, row.round_to, row.metric))
            same(tag+'/n_participants', row.n_participants, len(a))
            for domain in ['active', 'passive', 'delta']:
                values = b[domain+'_'+row.metric].to_numpy()-a[domain+'_'+row.metric].to_numpy()
                same(tag+'/'+domain+'_change', getattr(row, domain+'_change'), values.mean() if len(values) else np.nan)
                for name, expected in interval(values).items():
                    same(tag+'/'+domain+'_'+name, getattr(row, domain+'_'+name), expected)
        points = pd.read_csv(out/'performance_participants.csv')
        loo = pd.read_csv(out/'performance_leave_one_out.csv')
        for row in loo.itertuples():
            p = points[(points.task == row.task) & (points.scenario == row.scenario) & (points.stage == row.stage) &
                       (points.metric == row.metric) & (points.participant != row.held_out)]
            tag = 'LOO/'+str((row.task, row.scenario, row.stage, row.metric, row.held_out))
            same(tag+'/n_retained', row.n_retained, len(p))
            for domain in ['active', 'passive']:
                rho = spearmanr(p.score, p[domain]).statistic if len(p) >= 3 and p.score.nunique() > 1 and p[domain].nunique() > 1 else np.nan
                same(tag+'/'+domain+'_rho', getattr(row, domain+'_rho'), rho)
    except Exception:
        check('secondary_validation_exception', False, traceback.format_exc())
    report.update(status='passed' if not any(not c['passed'] for c in CHECKS) else 'failed',
                  check_count=len(CHECKS), failed_checks=[c for c in CHECKS if not c['passed']],
                  successful_check_count=sum(c['passed'] for c in CHECKS),
                  check_storage='All failures retained; successful assertions counted.')
    (WORK/'validation'/'independent_secondary_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(CHECKS), 'failures': len(report['failed_checks'])}), flush=True)
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
