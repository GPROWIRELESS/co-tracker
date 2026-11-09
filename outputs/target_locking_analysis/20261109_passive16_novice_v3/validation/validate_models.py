"""Independent model occupancy/Brier checks on exported frame-level predictions."""
from pathlib import Path
import json
import sys
import traceback
import numpy as np
import pandas as pd

from validate_passive16 import WORK, OUTPUT, FROZEN, STAGES, SCENARIOS, check, same, CHECKS, independent_metrics, divide


def main():
    out = OUTPUT
    report = {'scope': 'independent occupancy, common support, event Brier and hard-label temporal checks'}
    try:
        sys.path.insert(0, str(FROZEN))
        import tl_geometry as geo
        rows = pd.read_csv(out/'model_pair_metrics.csv')
        manifest = pd.read_csv(out/'pair_manifest.csv')
        expected_contents = {'User12_3_1', 'User7_1_1', 'User12_5_2'}
        check('model_content_subset', set(rows.content_id) == expected_contents)
        check('model_1500_metric_rows', len(rows) == 1500)
        check('model_exact_five_models', set(rows.model) == {'all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot'})
        check('model_exact_four_branches', set(rows.analysis_branch) == {'probability_mc', 'decoded_expectation', 'decoded_argmax', 'decoded_sample'})
        reproduction = pd.read_csv(out/'model_reproduction.csv')
        check('model_frozen_prediction_reproduction_passes', reproduction.passed.all())
        same('model_frozen_prediction_max_error', reproduction.max_abs_error, np.zeros(len(reproduction)), atol=1e-12)
        for pair_id, group in rows.groupby('pair_id'):
            frames = dict(np.load(out/'frames'/f'{pair_id}.npz', allow_pickle=False))
            model_frames = dict(np.load(out/'model_frames'/f'{pair_id}.npz', allow_pickle=False))
            dt = frames['dt_s']
            stage = frames['stage']
            included = frames['included']
            task = group.task.iloc[0]
            for scenario in SCENARIOS:
                valid = frames['valid_'+scenario] & model_frames['model_common_valid']
                same(f'{pair_id}/{scenario}/model_frame_common_support', model_frames['valid_'+scenario], valid)
                astate = model_frames['active_state_'+scenario]
                pstate = model_frames['passive_state_'+scenario]
                for domain, actual in [('active', astate), ('passive', pstate)]:
                    same(f'{pair_id}/{scenario}/{domain}_restricted_human_states', actual,
                         np.where(valid, frames[domain+'_state_'+scenario], -1))
                area = frames['area_'+scenario]
                for st in ['whole']+STAGES[task]:
                    mask = valid & (included if st == 'whole' else stage == st)
                    seconds = dt[mask].sum()
                    sm = {d: independent_metrics(state, mask, dt, area)
                          for d, state in [('active', astate), ('passive', pstate)]}
                    for _, row in group[(group.scenario == scenario) & (group.stage == st)].iterrows():
                        tag = f'{pair_id}/{scenario}/{st}/{row.analysis_branch}/{row.model}/'
                        prob = row.analysis_branch == 'probability_mc'
                        if prob:
                            q = model_frames[f'q_{row.model}_{scenario}']
                            same(tag+'four_region_probability_conservation', q[valid].sum(1), np.ones(valid.sum()))
                            share = dt[mask] @ q[mask]/seconds if seconds else np.full(4, np.nan)
                            n, t, p, o = share
                            mm = dict(L_N=n, L_T=t, L_P=p, L_O=o, L_target=t+o, L_gripper=p+o,
                                      D=t+p, C=divide(t, t+p), B=t-p)
                        else:
                            rule = row.analysis_branch.removeprefix('decoded_')
                            state = model_frames[f'state_{rule}_{row.model}_{scenario}']
                            q = np.full((len(state), 4), np.nan)
                            q[state >= 0] = np.eye(4)[state[state >= 0]]
                            mm = independent_metrics(state, mask, dt, area)
                            mm.update(geo.temporal(np.where(mask, state, -1), dt, frames['occurrence'], frames['break_before']))
                        for name, value in mm.items():
                            same(tag+'model_'+name, row['model_'+name], value, atol=1e-8)
                        for domain in sm:
                            for name, value in sm[domain].items():
                                same(tag+domain+'_'+name, row[domain+'_'+name], value, atol=1e-8)
                                same(tag+'delta_'+name, row['delta_'+name], sm['passive'][name]-sm['active'][name], atol=1e-8)
                            same(tag+domain+'_common_n', row[domain+'_valid_n'], mask.sum())
                            same(tag+domain+'_common_s', row[domain+'_valid_s'], seconds)
                        for region, bit in [('target', 1), ('gripper', 2)]:
                            pred = q[mask, bit]+q[mask, 3]
                            for domain, state in [('active', astate), ('passive', pstate)]:
                                observed = (state[mask] & bit) > 0
                                bs = np.sum(dt[mask]*(pred-observed)**2)/seconds if seconds else np.nan
                                same(tag+domain+'_independent_BS_'+region, row[domain+'_BS_'+region], bs, atol=1e-10)
                                for metric in ('L_target', 'L_gripper', 'C'):
                                    same(tag+domain+'_error_delta_'+metric, row[domain+'_delta_'+metric], row['model_'+metric]-row[domain+'_'+metric])
        report['status'] = 'passed' if not any(not c['passed'] for c in CHECKS) else 'failed'
    except Exception:
        check('model_validation_exception', False, traceback.format_exc())
        report['status'] = 'failed'
    report.update(check_count=len(CHECKS), failed_checks=[c for c in CHECKS if not c['passed']],
                  successful_check_count=sum(c['passed'] for c in CHECKS),
                  check_storage='All failures retained; successful assertions counted.')
    target = WORK/'validation'/'independent_model_validation.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(CHECKS), 'failures': len(report['failed_checks'])}), flush=True)
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
