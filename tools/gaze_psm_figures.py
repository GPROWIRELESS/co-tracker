"""Comparable, data-scaled views of saved gaze/PSM estimates.

No estimates are recalculated here. Shared scales follow the comparison:
whole tasks share a scale per signal; stages share a scale across instruments
and conditioning modes in each row. Individual trajectories have their own
diagnostic figures so they cannot hide group uncertainty in the main figures.
"""
import json
from pathlib import Path

import numpy as np


BLUE = '#2166ac'
ORANGE = '#d47716'
GREEN = '#287d55'
PURPLE = '#8052a0'
CURVE_FIELDS = ('r', 'r_ci_low', 'r_ci_high', 'null_r_low', 'null_r_high',
                'r_fixed_records', 'r_primary_fixed_record_users')


def _matches(curve, **conditions):
    return all(curve.get(key) == value for key, value in conditions.items())


def _values(value):
    a = np.asarray(value, dtype=float).ravel()
    return a[np.isfinite(a)]


def _finite(curve, field='r'):
    return bool(len(_values(curve.get(field, []))))


def _task_order(curves):
    return sorted({str(c['task']) for c in curves})


def _limits(values, *, zero=False, minimum=.01, bounds=None, padding=.10):
    arrays = [_values(v) for v in values]
    data = np.concatenate([a for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.array([0.])
    low, high = float(data.min()), float(data.max())
    if zero:
        low, high = min(0., low), max(0., high)
    span = max(high-low, minimum)
    center = (high+low)/2
    low, high = center-span*(.5+padding), center+span*(.5+padding)
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return low, high


def _curve_limits(curves, signal):
    return _limits([c.get(f, []) for c in curves for f in CURVE_FIELDS],
                   zero=True, minimum=.02 if signal == 'position' else .008,
                   bounds=(-1, 1))


def _share_y(axes, limits, group):
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.set_ylim(*limits)
        ax._gaze_comparison_group = group


def _decorate(ax, ylabel='Position correlation r'):
    from matplotlib.ticker import MaxNLocator
    ax.axvline(0, color='.45', linewidth=.9, linestyle=':', zorder=0)
    ax.set_xlabel('Lag (s); positive = gaze leads')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines[['top', 'right']].set_visible(False)


def _note(ax, text):
    ax.text(0, 1.025, text, transform=ax.transAxes, fontsize=8.5,
            va='bottom', color='.28')


def _figure_legend(fig, axes, *, y=.925, columns=3):
    handles = {}
    for ax in np.asarray(axes, dtype=object).ravel():
        hh, ll = ax.get_legend_handles_labels()
        for handle, label in zip(hh, ll):
            handles.setdefault(label, handle)
    if handles:
        fig.legend(handles.values(), handles, loc='upper center',
                   bbox_to_anchor=(.5, y), ncol=columns, frameon=False, fontsize=9)


def _save(fig, path):
    """Save publication-size raster/vector artifacts and inspectable axis metadata."""
    path = Path(path)
    try:
        partial = bool(json.loads((path.parent.parent/'config.json').read_text()).get('partial_record', False))
    except (OSError, ValueError, TypeError):
        partial = False
    if partial:
        fig.text(.5, .51, 'PARTIAL RECORD / SMOKE TEST', ha='center', va='center',
                 fontsize=25, rotation=24, color='firebrick', alpha=.16, zorder=1000)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    manifest = path.parent/'axes.json'
    current = json.loads(manifest.read_text()) if manifest.exists() else {}
    current[path.name] = [{'title': ax.get_title(), 'xlim': list(ax.get_xlim()),
                           'ylim': list(ax.get_ylim()), 'ylabel': ax.get_ylabel(),
                           'comparison_group': getattr(ax, '_gaze_comparison_group', None)}
                          for ax in fig.axes]
    manifest.write_text(json.dumps(current, indent=2), encoding='utf-8')


def _curve_panel(ax, groups, users, conditions):
    # Individual curves are available in novice_curves_*.png, at their full range.
    selected = [g for g in groups if _matches(g, **conditions)]
    if not selected:
        return
    group = selected[0]
    lag = np.asarray(group['lag_s'], dtype=float)
    ax.set_xlim(lag.min(), lag.max())
    ax.axhline(0, color='.5', linewidth=.7, zorder=0)
    if not _finite(group):
        ax.set_facecolor('#f5f5f5')
        ax.text(.5, .5, 'Insufficient support\nNo group estimate', ha='center',
                va='center', transform=ax.transAxes, color='.4')
        _note(ax, f"n = {group.get('n_stable_users', 0)}; {group.get('status', '')}")
        return
    if _finite(group, 'null_r_low'):
        ax.fill_between(lag, group['null_r_low'], group['null_r_high'],
                        color='.65', alpha=.24, label='Unpaired reference (95%)')
    if _finite(group, 'r_ci_low'):
        ax.fill_between(lag, group['r_ci_low'], group['r_ci_high'],
                        color=BLUE, alpha=.17, label='Participant bootstrap (95%)')
    ax.plot(lag, group['r'], color=BLUE, linewidth=2.1, label='Group mean')
    peak, r = group.get('peak_lag_s'), group.get('peak_r')
    if peak is not None and np.isfinite(peak):
        ax.plot(peak, r, marker='>' if group.get('peak_at_boundary') and peak > 0 else
                '<' if group.get('peak_at_boundary') else 'o', color=BLUE, markersize=6, clip_on=False)
    if group.get('record_support_changes') and _finite(group, 'r_fixed_records'):
        ax.plot(lag, group['r_fixed_records'], color=PURPLE, linestyle=':', linewidth=1.4,
                label='Fixed-record sensitivity')
    _note(ax, f"n = {group.get('n_stable_users', 0)}; peak {peak*1000:+.0f} ms; "
          f"r = {r:.3f}" + ('; BOUNDARY' if group.get('peak_at_boundary') else ''))


def _whole_figures(plt, directory, groups, users):
    whole = [g for g in groups if g.get('mode') == 'whole']
    tasks = _task_order(whole)
    for signal in sorted({g['signal'] for g in whole}):
        selected = [g for g in whole if g['signal'] == signal]
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(12, 3.8*len(tasks)))
        for row, task in enumerate(tasks):
            for col, side in enumerate(('l', 'r')):
                ax = axes[row, col]
                _curve_panel(ax, groups, users, dict(task=task, side=side, signal=signal,
                                                    stage='__whole__', mode='whole'))
                _decorate(ax, f'{signal.capitalize()} correlation r')
                ax.set_title(f'Task {task} / PSM {side.upper()}', pad=30, fontsize=12)
        _share_y(axes, _curve_limits(selected, signal), f'whole:{signal}')
        fig.suptitle(f'Whole-task {signal} correlation', fontsize=16, y=.995)
        _figure_legend(fig, axes, y=.955)
        fig.text(.5, .005, 'Same y scale across all panels; group estimates and complete uncertainty ranges. '
                 'Individual curves: novice_curves figure.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .91), h_pad=3, w_pad=2.5)
        _save(fig, directory/f'whole_task_{signal}.png')
        plt.close(fig)


def _stage_figures(plt, directory, groups, users):
    staged = [g for g in groups if g.get('mode') == 'psm_stage']
    for task in _task_order(staged):
        for signal in sorted({g['signal'] for g in staged if g['task'] == task}):
            stages = sorted({g['stage'] for g in staged if _matches(g, task=task, signal=signal)})
            fig, axes = plt.subplots(len(stages), 2, squeeze=False,
                                     figsize=(12.5, max(4, 3.7*len(stages))))
            for row, stage in enumerate(stages):
                selected = [g for g in groups if _matches(g, task=task, signal=signal, stage=stage)]
                for col, side in enumerate(('l', 'r')):
                    ax = axes[row, col]
                    conditions = dict(task=task, side=side, signal=signal, stage=stage, mode='psm_stage')
                    _curve_panel(ax, groups, users, conditions)
                    within = [g for g in selected if _matches(g, side=side, mode='within_stage')]
                    if within and _finite(within[0]):
                        curve = within[0]
                        ax.plot(curve['lag_s'], curve['r'], color=ORANGE, linestyle='--',
                                linewidth=1.7, label='Within occurrence')
                        if _finite(curve, 'r_ci_low'):
                            ax.fill_between(curve['lag_s'], curve['r_ci_low'], curve['r_ci_high'],
                                            color=ORANGE, alpha=.09, label='Within-occurrence CI (95%)')
                        if curve.get('record_support_changes') and _finite(curve, 'r_fixed_records'):
                            ax.plot(curve['lag_s'], curve['r_fixed_records'], color=PURPLE,
                                    linestyle='-.', linewidth=1.2, label='Within: fixed-record sensitivity')
                        ax.text(.98, .035, f"Within n = {curve['n_stable_users']}",
                                transform=ax.transAxes, ha='right', fontsize=8, color=ORANGE)
                    _decorate(ax, f'{signal.capitalize()} correlation r')
                    ax.set_title(f'{stage} / PSM {side.upper()}', pad=30, fontsize=12)
                _share_y(axes[row], _curve_limits(selected, signal), f'stage:{task}:{signal}:{stage}')
            fig.suptitle(f'Task {task}: {signal} / stages anchored to PSM time', fontsize=16, y=.997)
            _figure_legend(fig, axes, y=.976, columns=3)
            fig.text(.5, .004, 'Same y scale for left/right and modes WITHIN each row; '
                     'different stages may use different y scales. Full lag window retained.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .015, 1, .943), h_pad=3.5, w_pad=2.5)
            _save(fig, directory/f'stages_{task}_{signal}.png')
            plt.close(fig)


def _spatial_figures(plt, directory, groups):
    from matplotlib.ticker import PercentFormatter
    selected = [g for g in groups if g.get('mode') == 'whole' and g.get('signal') == 'position']
    tasks = _task_order(selected)
    if not tasks:
        return
    for field, label, filename in [('distance_median_px', 'Mean novice median distance (px)', 'spatial_distance.png'),
                                    ('proximity', 'Mean novice proximity (%)', 'spatial_proximity.png')]:
        factor = 100 if field == 'proximity' else 1
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(12, 3.6*len(tasks)))
        limits = _limits([np.asarray(g[field], dtype=float)*factor for g in selected],
                         zero=field == 'proximity', minimum=.1,
                         bounds=(0, 100) if field == 'proximity' else (0, np.inf))
        for row, task in enumerate(tasks):
            for col, side in enumerate(('l', 'r')):
                ax = axes[row, col]
                group = next((g for g in selected if _matches(g, task=task, side=side)), None)
                if group is not None:
                    ax.plot(group['lag_s'], np.asarray(group[field], dtype=float)*factor,
                            color=GREEN, linewidth=2)
                    peak = group.get('peak_lag_s')
                    if peak is not None and np.isfinite(peak):
                        ax.axvline(peak, color=BLUE, linestyle='--', linewidth=1, label='Correlation peak')
                        index = int(np.argmin(abs(np.asarray(group['lag_s'])-peak)))
                        value = group[field][index]*factor
                        ax.plot(peak, value, 'o', color=BLUE, markersize=5)
                        _note(ax, f"n = {group['n_stable_users']}; value at peak = {value:.2f}" +
                              ('%' if factor == 100 else ' px'))
                    ax.set_xlim(group['lag_s'][0], group['lag_s'][-1])
                _decorate(ax, label)
                if factor == 100:
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                ax.set_title(f'Task {task} / PSM {side.upper()}', pad=30)
        _share_y(axes, limits, f'spatial:{field}')
        fig.suptitle('Spatial correspondence at each lag', fontsize=16, y=.995)
        _figure_legend(fig, axes, y=.955)
        fig.text(.5, .005, 'Same y scale across all panels. '
                 'Small changes are shown separately in spatial_changes.png.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .91), h_pad=3)
        _save(fig, directory/filename)
        plt.close(fig)
    # Absolute comparisons and tiny within-trajectory changes need separate views.
    fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(12, 3.6*len(tasks)))
    for col, (field, factor, ylabel) in enumerate([
            ('distance_median_px', 1, 'Distance change from zero lag (px)'),
            ('proximity', 100, 'Proximity change (percentage points)')]):
        changes = []
        for row, task in enumerate(tasks):
            ax = axes[row, col]
            for side, color in [('l', BLUE), ('r', ORANGE)]:
                group = next((g for g in selected if _matches(g, task=task, side=side)), None)
                if group is None:
                    continue
                lag = np.asarray(group['lag_s'], dtype=float)
                values = np.asarray(group[field], dtype=float)
                delta = (values-values[np.argmin(abs(lag))])*factor
                changes.append(delta)
                ax.plot(lag, delta, color=color, linewidth=1.8, label=f'PSM {side.upper()}')
                peak = group.get('peak_lag_s')
                if peak is not None and np.isfinite(peak):
                    i = np.argmin(abs(lag-peak))
                    ax.plot(peak, delta[i], 'o', color=color)
                ax.set_xlim(lag.min(), lag.max())
            ax.axhline(0, color='.5', linewidth=.8)
            _decorate(ax, ylabel)
            ax.set_title(f'Task {task}', pad=12)
        _share_y(axes[:, col], _limits(changes, zero=True, minimum=.05), f'spatial_change:{field}')
    fig.suptitle('Spatial changes relative to zero lag', fontsize=16, y=.995)
    _figure_legend(fig, axes, y=.95)
    fig.text(.5, .005, 'Same y scale within each column. Dots mark correlation peaks; '
             'these are descriptive changes, without confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .91), h_pad=2.5)
    _save(fig, directory/'spatial_changes.png')
    plt.close(fig)


def _record_figures(plt, directory, records):
    from tools.gaze_psm_report import _summary
    selected = [c for c in records if c.get('mode') == 'whole']
    tasks = _task_order(selected)
    for signal in sorted({c['signal'] for c in selected}):
        estimates = [(c, _summary(c)) for c in selected if c['signal'] == signal]
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(12, 3.5*len(tasks)))
        people = sorted({c['user'] for c, _ in estimates})
        colors = {user: plt.get_cmap('tab20')(i % 20) for i, user in enumerate(people)}
        lags = [c['lag_s'] for c, _ in estimates]
        for row, task in enumerate(tasks):
            for col, side in enumerate(('l', 'r')):
                ax = axes[row, col]
                for curve, s in estimates:
                    if not _matches(curve, task=task, side=side) or not np.isfinite(s['peak_lag_s']):
                        continue
                    color = colors[curve['user']]
                    ax.scatter(s['peak_lag_s'], s['peak_r'], s=35,
                               facecolors='none' if s['peak_at_boundary'] else color,
                               edgecolors=color, linewidths=1, alpha=.8)
                _decorate(ax, 'Recording peak correlation r')
                ax.set_xlim(*_limits(lags, minimum=.1, padding=.04))
                ax.set_title(f'Task {task} / PSM {side.upper()}', pad=12)
        _share_y(axes, _limits([s['peak_r'] for _, s in estimates], zero=True,
                              minimum=.02, bounds=(-1, 1)), f'record_peaks:{signal}')
        fig.suptitle(f'{signal.capitalize()} recording peaks (diagnostic)', fontsize=16, y=.995)
        fig.text(.5, .005, 'Same x/y scales across panels; colors = novices; hollow markers = search boundaries. '
                 'Records are not independent participants.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .95), h_pad=2.5)
        _save(fig, directory/f'record_peaks_{signal}.png')
        plt.close(fig)


def _paired_figures(plt, directory, users):
    selected = [u for u in users if u.get('mode') == 'whole' and u.get('task') in ('A', 'C')]
    for signal in sorted({u['signal'] for u in selected}):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), squeeze=False)
        all_lags = []
        zoom = (-.8, .8)
        for col, side in enumerate(('l', 'r')):
            people = sorted({u['user'] for u in selected})
            n = outside = 0
            for person in people:
                pair = {u['task']: u for u in selected if _matches(u, user=person, side=side, signal=signal)}
                if set(pair) != {'A', 'C'} or any(u.get('peak_lag_s') is None for u in pair.values()):
                    continue
                values = [pair[task]['peak_lag_s'] for task in ('A', 'C')]
                if not np.isfinite(values).all():
                    continue
                n += 1
                all_lags.extend(values)
                outside += sum(v < zoom[0] or v > zoom[1] for v in values)
                color = ORANGE if any(pair[t].get('peak_at_boundary') for t in ('A', 'C')) else '.4'
                for row in range(2):
                    axes[row, col].plot([0, 1], values, 'o-', color=color, alpha=.65,
                                        linewidth=1.1, markersize=5)
            for row in range(2):
                ax = axes[row, col]
                ax.axhline(0, color='.4', linestyle=':', linewidth=.8)
                ax.set_xticks([0, 1], ['Task A', 'Task C'])
                ax.set_xlim(-.15, 1.15)
                ax.set_ylabel('Novice peak lag (s)')
                ax.grid(alpha=.18)
                ax.set_title(f'PSM {side.upper()}: {n} paired novices / ' + ('full range' if row == 0 else 'detail'), pad=25)
            _note(axes[1, col], f'{outside} endpoint(s) outside detail; all retained in full-range panel')
        _share_y(axes[0], _limits([all_lags], zero=True, minimum=.1), f'paired:{signal}:full')
        _share_y(axes[1], zoom, f'paired:{signal}:detail')
        fig.suptitle(f'{signal.capitalize()} novice peaks: full data and common detail window', fontsize=15, y=.995)
        fig.text(.5, .005, 'Same y scale within each row; positive = gaze leads. Orange includes boundary peaks. '
                 'These individual peaks differ from the paired group contrast.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .95), h_pad=3)
        _save(fig, directory/f'paired_novices_{signal}.png')
        plt.close(fig)


def _bootstrap_figures(plt, directory, groups):
    from matplotlib.ticker import PercentFormatter
    selected = [g for g in groups if g.get('mode') == 'whole']
    tasks = _task_order(selected)
    for signal in sorted({g['signal'] for g in selected}):
        curves = [g for g in selected if g['signal'] == signal]
        peaks = [_values(g.get('bootstrap_peak_lag_s', [])) for g in curves]
        if not any(len(v) for v in peaks):
            continue
        step = min(np.median(np.diff(g['lag_s'])) for g in curves if len(g['lag_s']) > 1)
        low, high = _limits(peaks, zero=True, minimum=step, padding=0)
        edges = (np.arange(np.floor(low/step), np.ceil(high/step)+2)-.5)*step
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(12, 3.7*len(tasks)))
        heights = []
        for row, task in enumerate(tasks):
            for col, side in enumerate(('l', 'r')):
                ax = axes[row, col]
                group = next((g for g in curves if _matches(g, task=task, side=side)), None)
                values = _values(group.get('bootstrap_peak_lag_s', [])) if group else np.array([])
                if len(values):
                    weights = np.full(len(values), 100/len(values))
                    counts, _, _ = ax.hist(values, bins=edges, weights=weights, color=BLUE, alpha=.72)
                    heights.extend(counts)
                    ax.axvspan(group['peak_lag_ci_low'], group['peak_lag_ci_high'], color=BLUE,
                               alpha=.07, label='Peak-lag interval (95%)')
                    ax.axvline(group['peak_lag_s'], color=BLUE, linestyle='--', linewidth=1.5, label='Observed group peak')
                    _note(ax, f"B = {len(values)}; boundary = {group['boundary_frequency']:.1%}")
                else:
                    ax.text(.5, .5, 'No supported bootstrap estimate', transform=ax.transAxes, ha='center')
                _decorate(ax, 'Bootstrap replicates (%)')
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                ax.set_xlim(edges[0], edges[-1])
                ax.set_title(f'Task {task} / PSM {side.upper()}', pad=30)
        _share_y(axes, (0, max(heights, default=1)*1.12), f'bootstrap:{signal}')
        fig.suptitle(f'{signal.capitalize()} peak-lag bootstrap distribution', fontsize=16, y=.995)
        _figure_legend(fig, axes, y=.955)
        fig.text(.5, .005, 'Common one-sample-width bins and x/y scales; all bootstrap peaks retained. '
                 'Shading marks the peak-lag interval, not a curve band.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .91), h_pad=3)
        _save(fig, directory/f'bootstrap_peak_lags_{signal}.png')
        plt.close(fig)


def _novice_figures(plt, directory, groups, users):
    selected = [u for u in users if u.get('mode') == 'whole']
    tasks = _task_order(selected)
    for signal in sorted({u['signal'] for u in selected}):
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(12, 3.7*len(tasks)))
        curves = [u for u in selected if u['signal'] == signal]
        values = []
        for row, task in enumerate(tasks):
            for col, side in enumerate(('l', 'r')):
                ax = axes[row, col]
                people = [u for u in curves if _matches(u, task=task, side=side)]
                for u in people:
                    ax.plot(u['lag_s'], u['r'], color='.5', linewidth=.8, alpha=.55)
                    values.append(u['r'])
                    ax.set_xlim(u['lag_s'][0], u['lag_s'][-1])
                g = next((g for g in groups if _matches(g, task=task, side=side, signal=signal, mode='whole')), None)
                if g:
                    ax.plot(g['lag_s'], g['r'], color=BLUE, linewidth=2, label='Group mean (stable set)')
                _decorate(ax, f'{signal.capitalize()} correlation r')
                ax.set_title(f'Task {task} / PSM {side.upper()}', pad=12)
        _share_y(axes, _limits(values, zero=True, minimum=.02, bounds=(-1, 1)), f'novice:{signal}')
        fig.suptitle(f'{signal.capitalize()} individual curves (diagnostic)', fontsize=16, y=.995)
        _figure_legend(fig, axes, y=.955)
        fig.text(.5, .005, 'Same y scale across panels; all available novice curves shown. '
                 'Group inference uses its complete-support novice set.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .91), h_pad=2.5)
        _save(fig, directory/f'novice_curves_{signal}.png')
        plt.close(fig)


def _interval(ax, estimate, low, high, y, color, boundary=False, label=None):
    # Hlines support bootstrap percentile intervals not containing the estimate.
    if low is not None and high is not None and np.isfinite([low, high]).all():
        ax.hlines(y, low, high, color=color, linewidth=1.8)
        ax.plot([low, high], [y, y], '|', color=color, markersize=7)
    if estimate is not None and np.isfinite(estimate):
        ax.plot(estimate, y, marker=('>' if estimate > 0 else '<') if boundary else 'o',
                color=color, markersize=6, label=label, clip_on=False)


def _conclusion_figures(plt, directory, groups, contrasts):
    from matplotlib.ticker import PercentFormatter
    selected = [g for g in groups if _matches(g, signal='position', mode='whole') and _finite(g)]
    if not selected:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1.2, 1, 1]})
    selected.sort(key=lambda g: (g['task'], g['side']))
    labels, limits = [], []
    for i, g in enumerate(selected):
        y = len(selected)-1-i
        color = BLUE if g['task'] == 'A' else PURPLE
        estimate, low, high = [g[k]*1000 if g[k] is not None else np.nan for k in
                               ('peak_lag_s', 'peak_lag_ci_low', 'peak_lag_ci_high')]
        _interval(axes[0], estimate, low, high, y, color, g.get('peak_at_boundary'))
        labels.append(f"{g['task']} / {g['side'].upper()} (n={g['n_stable_users']})\n"
                      f'{estimate:+.0f} [{low:+.0f}, {high:+.0f}] ms')
        limits.extend([estimate, low, high])
        axes[2].plot([g['proximity_at_zero']*100, g['proximity_at_peak']*100], [y, y], color='.5')
        axes[2].plot(g['proximity_at_zero']*100, y, 'o', color='.45', label='Zero lag' if i == 0 else None)
        axes[2].plot(g['proximity_at_peak']*100, y, 'D', color=GREEN, markersize=5,
                     label='Group peak lag' if i == 0 else None)
    for ax in (axes[0], axes[2]):
        ax.set_yticks(range(len(selected)-1, -1, -1), labels if ax is axes[0] else
                      [f"{g['task']} / {g['side'].upper()}" for g in selected])
        ax.set_ylim(-.65, len(selected)-.35)
    axes[0].set_xlim(*_limits([limits], zero=True, minimum=100, padding=.18))
    axes[0].axvline(0, color='.4', linestyle=':', linewidth=1)
    axes[0].set_title('A  Whole-task peak lag', loc='left', pad=16)
    axes[0].set_xlabel('Lag (ms); positive = gaze leads')
    pairs = [c for c in contrasts if _matches(c, dimension='task', signal='position', mode='whole')
             and c.get('difference_s') is not None]
    diffs, pair_labels = [], []
    for i, c in enumerate(sorted(pairs, key=lambda c: c['side'])):
        estimate, low, high = [c[k]*1000 if c[k] is not None else np.nan
                               for k in ('difference_s', 'ci_low', 'ci_high')]
        y = len(pairs)-1-i
        _interval(axes[1], estimate, low, high, y, BLUE, c.get('status') == 'boundary_peak')
        pair_labels.append(f"PSM {c['side'].upper()} (n={c['n_users']})\n{estimate:+.0f} [{low:+.0f}, {high:+.0f}] ms")
        diffs.extend([estimate, low, high])
    axes[1].set_yticks(range(len(pairs)-1, -1, -1), pair_labels)
    axes[1].set_ylim(-.65, max(1, len(pairs))-.35)
    axes[1].set_xlim(*_limits([diffs], zero=True, minimum=100, padding=.18))
    axes[1].axvline(0, color='.4', linestyle=':', linewidth=1)
    axes[1].set_title('B  Paired task difference', loc='left', pad=16)
    axes[1].set_xlabel('C minus A peak lag (ms)')
    timing_limits = _limits([limits, diffs], zero=True, minimum=100, padding=.15)
    for ax in axes[:2]:
        ax.set_xlim(*timing_limits)
    if not pairs:
        axes[1].text(.5, .5, 'Paired task contrast unavailable', ha='center',
                     transform=axes[1].transAxes, color='.4')
    axes[2].set_xlim(0, max(g['proximity_at_peak']*100 for g in selected)*1.2)
    axes[2].xaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[2].set_title('C  Spatial proximity (50 px)', loc='left', pad=16)
    axes[2].set_xlabel('Mean novice proximity (%)')
    axes[2].legend(loc='lower right', fontsize=8, frameon=False)
    for i, ax in enumerate(axes):
        ax.grid(axis='x', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax._gaze_comparison_group = f'conclusion:panel{i}'
    fig.suptitle('Gaze / instrument position: timing and spatial correspondence', fontsize=16, y=1.01)
    fig.text(.5, .005, 'Panels A/B share the timing scale. Points = group-curve peaks; bars = bootstrap 95% intervals. '
             'Trajectory timing does not establish fixation arrival at the same target.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95), w_pad=3)
    _save(fig, directory/'conclusion_position.png')
    plt.close(fig)


def _stage_peak_figures(plt, directory, groups):
    selected = [g for g in groups if g.get('signal') == 'position' and g.get('mode') in {'psm_stage', 'within_stage'}]
    tasks = _task_order(selected)
    if not tasks:
        return
    fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(13, 4.5*len(tasks)))
    lag_limits = _limits([g['lag_s'] for g in selected], minimum=.1, padding=.08)
    for row, task in enumerate(tasks):
        stages = sorted({g['stage'] for g in selected if g['task'] == task})
        for col, side in enumerate(('l', 'r')):
            ax = axes[row, col]
            for i, stage in enumerate(stages):
                y = len(stages)-1-i
                for mode, offset, color, label in [('psm_stage', .12, BLUE, 'PSM-stage'),
                                                   ('within_stage', -.12, ORANGE, 'Within occurrence')]:
                    g = next((g for g in selected if _matches(g, task=task, side=side, stage=stage, mode=mode)), None)
                    if g is None or not _finite(g):
                        if mode == 'psm_stage':
                            ax.text(0, y, 'unsupported', ha='center', va='center', color='.5', fontsize=8)
                        continue
                    _interval(ax, g['peak_lag_s'], g['peak_lag_ci_low'], g['peak_lag_ci_high'],
                              y+offset, color, g.get('peak_at_boundary'), label if i == len(stages)-1 else None)
                    ax.text(1.015, y+offset, f"n={g['n_stable_users']}", ha='left',
                            va='center', fontsize=7.5, color=color,
                            transform=ax.get_yaxis_transform(), clip_on=False)
            ax.set_yticks(range(len(stages)-1, -1, -1), stages, fontsize=9)
            ax.set_ylim(-.7, len(stages)-.3)
            ax.set_xlim(*lag_limits)
            ax.axvline(0, linestyle=':', color='.4', linewidth=1)
            ax.grid(axis='x', alpha=.18)
            ax.set_xlabel('Peak lag (s); positive = gaze leads')
            ax.set_title(f'Task {task} / PSM {side.upper()}', pad=14)
            ax._gaze_comparison_group = 'stage_peaks:position'
    fig.suptitle('Position stage peaks and participant-bootstrap intervals (exploratory)', fontsize=15, y=.995)
    _figure_legend(fig, axes, y=.95, columns=2)
    fig.text(.5, .005, 'Same lag scale in every panel; triangles = boundary estimates. '
             'Different modes can have different participant/record support; no multiple-comparison correction.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .91), h_pad=2.5, w_pad=2.5)
    _save(fig, directory/'stage_peak_lags_position.png')
    plt.close(fig)


def generate_figures(output, records, users, groups, contrasts=None):
    """Replot stored estimates only; no statistical files are written."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    directory = Path(output)/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'axes.json').write_text('{}', encoding='utf-8')
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'savefig.facecolor': 'white'}):
        _whole_figures(plt, directory, groups, users)
        _spatial_figures(plt, directory, groups)
        _stage_figures(plt, directory, groups, users)
        _record_figures(plt, directory, records)
        _paired_figures(plt, directory, users)
        _bootstrap_figures(plt, directory, groups)
        _novice_figures(plt, directory, groups, users)
        _conclusion_figures(plt, directory, groups, contrasts or [])
        _stage_peak_figures(plt, directory, groups)
