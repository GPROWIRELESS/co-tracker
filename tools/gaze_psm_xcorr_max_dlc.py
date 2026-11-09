#!/usr/bin/env python3
"""Select one PSM per recording/stage by signed xcorr peak, then aggregate.

Example: python tools/gaze_psm_xcorr_max_dlc.py --workers 8
The source outputs and trajectory caches are read only.
"""
from gaze_psm_modes import normalize_modes
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gaze_psm_group import aggregate
from tools.gaze_psm_pipeline_dlc import _init_analysis_worker, _null_job, save_json
from tools.gaze_psm_report import LAG_FIELDS, _write, generate_outputs

SELECTION_FIELDS = ('content_id', 'user', 'task', 'signal', 'stage', 'mode')
STAGE_MODES = {'PSM in stage', 'Both in stage'}


def select_sides(curves):
    """Keep a winner's whole curve; never form a pointwise maximum envelope."""
    pairs = defaultdict(dict)
    for curve in curves:
        if curve['mode'] not in STAGE_MODES:
            continue
        key = tuple(curve[f] for f in SELECTION_FIELDS)
        side = curve['side']
        if side not in {'l', 'r'} or side in pairs[key]:
            raise ValueError(f'Invalid or duplicate side for {key}: {side}')
        pairs[key][side] = curve
    selected, details = [], []
    for key, sides in sorted(pairs.items()):
        if set(sides) != {'l', 'r'}:
            raise ValueError(f'Missing left/right curve for {key}')
        if not np.array_equal(sides['l']['lag_s'], sides['r']['lag_s']):
            raise ValueError(f'Left/right lag grids differ for {key}')
        peaks, lags = {}, {}
        for side, curve in sides.items():
            r = np.asarray(curve['r'], dtype=float)
            valid = np.isfinite(r)
            index = int(np.argmax(np.where(valid, r, -np.inf))) if valid.any() else None
            peaks[side] = float(r[index]) if index is not None else np.nan
            lags[side] = float(curve['lag_s'][index]) if index is not None else np.nan
        left, right = np.isfinite(peaks['l']), np.isfinite(peaks['r'])
        if not left and not right:
            winner, status = None, 'both_invalid'
        elif not left or not right:
            winner, status = ('l' if left else 'r'), 'single_valid_side'
        elif peaks['l'] == peaks['r']:
            winner, status = 'l', 'tie_left'
        else:
            winner, status = ('l' if peaks['l'] > peaks['r'] else 'r'), 'larger_peak'
        # Both-invalid rows retain the stage's missing support, without inventing
        # a selected instrument. The left row is only a deterministic template.
        row = dict(sides[winner or 'l'])
        row.update(side='max', selected_side=winner, selection_status=status)
        detail = dict(zip(SELECTION_FIELDS, key))
        detail.update(selected_side=winner, selection_status=status,
                      left_peak_r=peaks['l'], right_peak_r=peaks['r'],
                      left_peak_lag_s=lags['l'], right_peak_lag_s=lags['r'])
        selected.append(row)
        details.append(detail)
    return selected, details


def _selected_null_job(pairings):
    return select_sides(_null_job(pairings))[0]


def _hashes(source):
    paths = sorted(p for p in source.rglob('*') if p.is_file())
    result = {}
    for path in paths:
        with path.open('rb') as stream:
            result[path.relative_to(source).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def _load_curves(source):
    # Only record curves are reused: all higher-level estimates are rebuilt.
    results = normalize_modes(json.loads((source/'results.json').read_text()))
    curves = results['record_curves']
    for curve in curves:
        for key in LAG_FIELDS:
            if isinstance(curve.get(key), list):
                curve[key] = np.asarray(curve[key], dtype=float)
    return curves


def _load_caches(source, curves, config):
    records, datasets = {}, {}
    for curve in curves:
        records.setdefault(curve['content_id'], {'content_id': curve['content_id'],
                                                'Task Name': curve['task']})
    # Retain recording order from the original observed results for RNG parity.
    for cid, record in records.items():
        with np.load(source/'cache'/f'{cid}.npz', allow_pickle=False) as z:
            quality = normalize_modes(json.loads(str(z['qc_json'])))
            if quality['content_id'] != cid or quality['task'] != record['Task Name']:
                raise ValueError(f'Cache identity mismatch: {cid}')
            if quality['hz'] != config['hz'] or quality['cache_version'] != config['cache_version']:
                raise ValueError(f'Cache configuration mismatch: {cid}')
            datasets[cid] = {k: z[k] for k in z.files}
    return records, datasets


def _pairings(records, count, seed):
    donors = {}
    for cid, record in records.items():
        by_user = defaultdict(list)
        for other, candidate in records.items():
            if other.split('_')[0] != cid.split('_')[0] and candidate['Task Name'] == record['Task Name']:
                by_user[other.split('_')[0]].append(other)
        donors[cid] = by_user
    rng = np.random.default_rng(seed)
    for _ in range(count):
        replicate = []
        for cid, by_user in donors.items():
            if by_user:
                user = rng.choice(sorted(by_user))
                replicate.append((cid, rng.choice(by_user[user])))
        yield replicate


def generate_figures(output, users, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.gaze_psm_figures import (_curve_panel, _decorate, _figure_legend,
                                       _save, _curve_limits)

    directory = output/'figures'
    directory.mkdir(exist_ok=True)
    note = ('One PSM selected per recording/stage by largest signed xcorr peak; '
            'selection does not identify the operated instrument.')
    for task in sorted({g['task'] for g in groups}):
        for signal in sorted({g['signal'] for g in groups if g['task'] == task}):
            subset = [g for g in groups if g['task'] == task and g['signal'] == signal]
            stages = sorted({g['stage'] for g in subset})
            modes = sorted({g['mode'] for g in subset})
            fig, axes = plt.subplots(len(stages), len(modes), squeeze=False,
                                     figsize=(7*len(modes), 3.9*len(stages)+1))
            for i, stage in enumerate(stages):
                limits = _curve_limits([g for g in subset if g['stage'] == stage], signal)
                for j, mode in enumerate(modes):
                    ax = axes[i, j]
                    _curve_panel(ax, subset, users, dict(task=task, signal=signal,
                                 stage=stage, mode=mode, side='max'))
                    _decorate(ax, f'{signal.capitalize()} correlation r')
                    ax.set_ylim(*limits)
                    ax.set_title(f'{stage} / {mode}', pad=30, fontsize=11)
            fig.suptitle(f'Task {task}: {signal}, maximum-xcorr PSM per stage', y=.995)
            _figure_legend(fig, axes, y=.97, columns=3)
            fig.tight_layout(rect=(0, .045, 1, .94), h_pad=3)
            fig.text(.5, .008, note, ha='center', fontsize=8, wrap=True)
            _save(fig, directory/f'stages_{task}_{signal}.png')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, max(4, .9*len(stages)+2)))
            for j, mode in enumerate(modes):
                color = ('#2166ac', '#d47716')[j % 2]
                labeled = False
                for i, stage in enumerate(stages):
                    group = next(g for g in subset if g['stage'] == stage and g['mode'] == mode)
                    peak = group.get('peak_lag_s', np.nan)
                    if peak is None or not np.isfinite(peak):
                        ax.text(.02, i + (j-(len(modes)-1)/2)*.22,
                                f"{mode}: insufficient support (n={group['n_stable_users']})",
                                transform=ax.get_yaxis_transform(), ha='left',
                                va='center', fontsize=7, color=color)
                        continue
                    y = i + (j-(len(modes)-1)/2)*.22
                    low, high = np.asarray([group['peak_lag_ci_low'], group['peak_lag_ci_high']],
                                           dtype=float)
                    if np.isfinite([low, high]).all():
                        ax.plot([low, high], [y, y], color=color, linewidth=2)
                    ax.plot(peak, y, 'D' if group['peak_at_boundary'] else 'o', color=color,
                            label=mode if not labeled else None)
                    labeled = True
                    ax.annotate(f" n={group['n_stable_users']}", (peak, y),
                                xytext=(4, 6), textcoords='offset points', fontsize=8)
            ax.set_yticks(range(len(stages)), stages)
            ax.set_ylim(len(stages)-.5, -.5)
            ax.axvline(0, color='.5', linestyle=':')
            ax.set_xlim(subset[0]['lag_s'][0], subset[0]['lag_s'][-1])
            ax.set_xlabel('Group peak lag (s); positive = gaze leads; bars = bootstrap 95% CI')
            ax.set_title(f'Task {task}: {signal} stage peak lags (exploratory)')
            ax.grid(alpha=.15)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
            fig.tight_layout(rect=(0, .08, 1, 1))
            fig.text(.5, .015, note, ha='center', fontsize=8)
            _save(fig, directory/f'stage_peak_lags_{task}_{signal}.png')
            plt.close(fig)
    save_json(directory/'figure_data.json', [{k: g.get(k) for k in
              ('task', 'signal', 'stage', 'mode', 'peak_r', 'peak_lag_s',
               'peak_lag_ci_low', 'peak_lag_ci_high', 'n_stable_users', 'status')}
              for g in groups])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/gaze_psm_xcorr_dlc'))
    parser.add_argument('--output', type=Path, default=Path('outputs/gaze_psm_xcorr_max_dlc'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--shuffles', type=int, default=100)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or min(args.bootstrap, args.shuffles) < 0:
        parser.error('workers must be positive; bootstrap and shuffles must be nonnegative')
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('source and output must be separate, non-nested directories')
    before = _hashes(source)
    config = normalize_modes(json.loads((source/'config.json').read_text()))
    curves = _load_curves(source)
    selected, details = select_sides(curves)
    if not selected:
        raise ValueError('No stage curves in source results')
    records, datasets = _load_caches(source, curves, config) if args.shuffles else ({}, {})
    settings = argparse.Namespace(**config)
    config.update(source=str(source), output=str(output), side_selection='record_stage_signed_peak',
                  bootstrap=args.bootstrap, shuffles=args.shuffles, workers=args.workers,
                  no_plots=args.no_plots)
    print(f'{len(selected)} selected stage curves from {len(curves)} source curves', flush=True)
    with ExitStack() as stack:
        if args.shuffles and args.workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('fork'),
                initializer=_init_analysis_worker, initargs=(records, datasets, settings)))
        else:
            pool = None
            _init_analysis_worker(records, datasets, settings)

        def references():
            pairings = iter(_pairings(records, args.shuffles, config['seed']))
            completed = 0
            while completed < args.shuffles:
                # Bound queued work and memory to one batch of workers.
                batch = [next(pairings) for _ in range(min(args.workers, args.shuffles-completed))]
                results = pool.map(_selected_null_job, batch) if pool else map(_selected_null_job, batch)
                for result in results:
                    completed += 1
                    print(f'Unpaired selected-side reference [{completed}/{args.shuffles}]', flush=True)
                    yield result

        users, groups, contrasts = aggregate(selected, bootstrap=args.bootstrap,
            seed=config['seed'], min_users=config['min_users'], baseline_replicates=references())
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'config.json', config)
    save_json(output/'results.json', dict(record_curves=selected, user_curves=users,
                                         group_curves=groups, contrasts=contrasts))
    qc = normalize_modes(json.loads((source/'quality.json').read_text()))
    save_json(output/'quality.json', qc)
    generate_outputs(output, selected, users, groups, contrasts, qc, make_plots=False)
    _write(output/'side_selection.csv', details)
    if not args.no_plots:
        generate_figures(output, users, groups)
    after = _hashes(source)
    if before != after:
        raise RuntimeError('Source files changed during analysis')
    save_json(output/'validation.json', dict(source_unchanged=True, source_sha256=after,
        n_record_curves=len(selected), n_user_curves=len(users), n_groups=len(groups),
        selected_sides=dict(Counter(d['selected_side'] or 'missing' for d in details)),
        selection_status=dict(Counter(d['selection_status'] for d in details))))
    print(f'Saved {len(groups)} stage groups to {output}; source files unchanged', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
