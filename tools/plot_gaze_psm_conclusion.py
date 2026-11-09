"""Create one all-user Task A/C conclusion figure from saved position estimates."""
from gaze_psm_modes import normalize_modes
import argparse
import json
from pathlib import Path
import sys
from textwrap import fill

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gaze_psm_figures import BLUE, ORANGE, _interval, _limits, _save


STAGES = {'A': ['reach ring', 'grab ring', 'transfer ring', 'place ring', 'drop ring'],
          'C': ['reach string', 'wrap around psm', 'threading string through tower',
                'pull through and cinch']}
COLORS = {'l': BLUE, 'r': ORANGE}


def _summary(curve):
    fields = ('task', 'side', 'stage', 'mode', 'n_stable_users', 'stable_users',
              'peak_lag_s', 'peak_lag_ci_low', 'peak_lag_ci_high', 'peak_r',
              'peak_at_boundary', 'boundary_frequency')
    row = {key: curve.get(key) for key in fields}
    r = np.asarray(curve['r'], dtype=float)
    lag = np.asarray(curve['lag_s'], dtype=float)
    zero = np.flatnonzero(np.isclose(lag, 0., atol=1e-12, rtol=0))
    row['zero_r'] = float(r[zero[0]]) if len(zero) and np.isfinite(r[zero[0]]) else None
    return row


def _direction(curve):
    low, high, lag = (curve[key] for key in ('peak_lag_ci_low', 'peak_lag_ci_high', 'peak_lag_s'))
    if low is None or high is None:
        return 'timing unavailable'
    if low > 0:
        return f'gaze leads {abs(lag)*1000:.0f} ms'
    if high < 0:
        return f'gaze follows {abs(lag)*1000:.0f} ms'
    return 'direction unclear'


def _timing_axis(ax):
    ax.axvline(0, color='.35', linestyle=':', linewidth=1.2)
    ax.grid(axis='x', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlabel('Lag (ms); positive = gaze leads', fontsize=11)


def generate_conclusion(output):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    results = normalize_modes(json.loads((output/'results.json').read_text(encoding='utf-8')))
    positions = [g for g in results['group_curves']
                 if g['signal'] == 'position' and g['task'] in STAGES]
    whole = {(g['task'], g['side']): g for g in positions if g['mode'] == 'whole'}
    order = [(task, side) for task in ('A', 'C') for side in ('l', 'r')]
    if any(key not in whole or whole[key].get('peak_lag_s') is None for key in order):
        raise ValueError('Supported whole-task position results are required for both tasks and instruments')
    staged = {(g['task'], g['stage'], g['side']): g
              for g in positions if g['mode'] == 'PSM in stage'}
    contrasts = {c['side']: c for c in results['contrasts']
                 if c.get('dimension') == 'task' and c.get('signal') == 'position'
                 and c.get('mode') == 'whole' and c.get('condition_a') == 'A'
                 and c.get('condition_b') == 'C'}
    users = sorted({user for group in whole.values() for user in group['stable_users']})
    record_ids = {task: sorted({r['content_id'] for r in results['record_curves']
                               if r['task'] == task and r['signal'] == 'position'
                               and r['mode'] == 'whole'}) for task in STAGES}
    summaries = [_summary(whole[key]) for key in order]
    figure_path = output/'figures'/'all_users_task_A_C_conclusion.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({'font.size': 11, 'savefig.facecolor': 'white'}):
        fig = plt.figure(figsize=(16, 11.5))
        timing = fig.add_axes([.14, .58, .24, .24])
        correlation = fig.add_axes([.66, .58, .22, .24])
        stage_axes = {'A': fig.add_axes([.16, .16, .29, .25]),
                      'C': fig.add_axes([.67, .16, .24, .25])}
        try:
            labels = [f"Task {task} / {'Left' if side == 'l' else 'Right'}" for task, side in order]
            lag_values = []
            for index, key in enumerate(order):
                row, group = summaries[index], whole[key]
                y = len(order)-1-index
                lag, low, high = [row[name]*1000 for name in
                                  ('peak_lag_s', 'peak_lag_ci_low', 'peak_lag_ci_high')]
                _interval(timing, lag, low, high, y, COLORS[key[1]], row['peak_at_boundary'])
                timing.text(1.04, y, f'{lag:+.0f} [{low:+.0f}, {high:+.0f}] ms',
                            transform=timing.get_yaxis_transform(), va='center', fontsize=10.5)
                lag_values.extend([lag, low, high])
                # Offset the markers within each category so nearly identical
                # zero/peak correlations remain visible without magnifying r.
                r0, peak = row['zero_r'], row['peak_r']
                correlation.plot([r0, peak], [y+.11, y-.11], color='.6', linewidth=1.2)
                correlation.plot(r0, y+.11, 'o', markerfacecolor='white',
                                 markeredgecolor='.4', markersize=7)
                correlation.plot(peak, y-.11, 'D', color=COLORS[key[1]], markersize=6)
                correlation.text(1.04, y, f'{r0:.3f} → {peak:.3f}',
                                 transform=correlation.get_yaxis_transform(), va='center', fontsize=11)
            for ax in (timing, correlation):
                ax.set_yticks(range(3, -1, -1), labels)
                ax.set_ylim(-.55, 3.55)
            _timing_axis(timing)
            timing.set_xlim(*_limits([lag_values], zero=True, minimum=100, padding=.15))
            timing.set_title('A  Whole-task timing (95% CI)', loc='left', fontsize=14, pad=15)
            timing._gaze_comparison_group = 'all_users:whole_task_timing'
            correlation.set_xlim(0, max(.6, max(row['peak_r'] for row in summaries)*1.15))
            correlation.set_xlabel('Combined X/Y position correlation r', fontsize=11)
            correlation.set_title('B  Zero lag → group-best lag', loc='left', fontsize=14, pad=15)
            correlation.grid(axis='x', alpha=.18)
            correlation.spines[['top', 'right']].set_visible(False)
            correlation.tick_params(labelsize=11)
            correlation._gaze_comparison_group = 'all_users:zero_vs_group_peak'

            for task, ax in stage_axes.items():
                stages = [stage for stage in STAGES[task] if (task, stage, 'l') in staged
                          or (task, stage, 'r') in staged]
                stage_labels = []
                for index, stage in enumerate(stages):
                    y = len(stages)-1-index
                    groups = [staged.get((task, stage, side)) for side in ('l', 'r')]
                    counts = sorted({g['n_stable_users'] for g in groups if g is not None})
                    n = '/'.join(map(str, counts))
                    stage_labels.append(fill(stage, width=22)+f' (n={n})')
                    supported = False
                    for side, offset in (('l', .12), ('r', -.12)):
                        group = staged.get((task, stage, side))
                        if group is None or group.get('peak_lag_s') is None:
                            continue
                        supported = True
                        lag, low, high = [group[name]*1000 for name in
                                          ('peak_lag_s', 'peak_lag_ci_low', 'peak_lag_ci_high')]
                        _interval(ax, lag, low, high, y+offset, COLORS[side], group['peak_at_boundary'])
                    if not supported:
                        ax.text(0, y, 'Insufficient stage support', ha='center', va='center',
                                fontsize=9.5, color='.45')
                ax.set_yticks(range(len(stages)-1, -1, -1), stage_labels, fontsize=10.5)
                ax.set_ylim(-.6, len(stages)-.4)
                window = max(abs(float(x)) for g in positions for x in g['lag_s'])*1000
                ax.set_xlim(-window*1.10, window*1.10)
                ax.set_xticks([-window, -window/2, 0, window/2, window])
                _timing_axis(ax)
                ax.set_title(f"{'C' if task == 'A' else 'D'}  Task {task} stages (exploratory)",
                             loc='left', fontsize=14, pad=15)
                ax._gaze_comparison_group = 'all_users:stage_timing'

            fig.suptitle('Gaze–PSM timing: one conclusion across Task A and Task C',
                         fontsize=19, y=.985)
            fig.text(.5, .948, f'{len(users)} novice users · {sum(map(len, record_ids.values()))} recordings '
                     f"(A: {len(record_ids['A'])}; C: {len(record_ids['C'])}) · X/Y position analysis",
                     ha='center', fontsize=12, color='.35')
            headline = (f"Right PSM — Task A: {_direction(whole[('A', 'r')])}; "
                        f"Task C: {_direction(whole[('C', 'r')])}.")
            left_direction = ('Left PSM: direction unclear in both tasks.'
                              if all(_direction(whole[(task, 'l')]) == 'direction unclear' for task in STAGES)
                              else '; '.join(f"Task {task} left: {_direction(whole[(task, 'l')])}" for task in STAGES))
            fig.text(.5, .902, headline+'\n'+left_direction, ha='center', va='center',
                     fontsize=13, linespacing=1.5, color='.15',
                     bbox=dict(boxstyle='round,pad=.6', facecolor='#edf3f7', edgecolor='none'))
            legend = [Line2D([], [], color=BLUE, marker='o', linestyle='-', label='Left PSM'),
                      Line2D([], [], color=ORANGE, marker='o', linestyle='-', label='Right PSM'),
                      Line2D([], [], color='.4', marker='o', markerfacecolor='white', linestyle='none', label='Zero lag'),
                      Line2D([], [], color='.3', marker='D', linestyle='none', label='Group-best lag')]
            fig.legend(handles=legend, loc='center', bbox_to_anchor=(.5, .49),
                       ncol=4, frameon=False, fontsize=12)
            changes = []
            for side in ('r', 'l'):
                contrast = contrasts.get(side)
                if contrast is not None:
                    value, low, high = [contrast[key]*1000 for key in ('difference_s', 'ci_low', 'ci_high')]
                    changes.append(f"{'Right' if side == 'r' else 'Left'} PSM: {value:+.0f} "
                                   f'[{low:+.0f}, {high:+.0f}] ms')
            fig.text(.5, .075, 'Paired task change (C − A), 95% CI:  '+';   '.join(changes),
                     ha='center', fontsize=12)
            fig.text(.5, .019, 'Users receive equal weight after within-user recording aggregation; bars = participant-bootstrap 95% CIs.\n'
                     'Stage results are exploratory (unadjusted); triangles mark search boundaries. '
                     'Panel B uses the peak of the group curve, not separately optimized individual peaks.',
                     ha='center', fontsize=9.5, color='.3', linespacing=1.5)
            _save(fig, figure_path)
        finally:
            plt.close(fig)

    payload = {'users': users, 'record_ids_by_task': record_ids, 'whole_task': summaries,
               'stages': [_summary(group) for group in staged.values()],
               'paired_task_contrasts': [{key: value for key, value in c.items()
                                          if key != 'bootstrap_difference_s'} for c in contrasts.values()],
               'weighting': 'Equal-weight recording Fisher-z means within users, then equal-weight user Fisher-z means',
               'source': 'results.json; saved group_curves and contrasts; no estimates recalculated'}
    figure_path.with_suffix('.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return figure_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gaze_psm_xcorr'))
    args = parser.parse_args(argv)
    try:
        path = generate_conclusion(args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f'Saved {path}, matching PDF, and source summary JSON.')


if __name__ == '__main__':
    main()
