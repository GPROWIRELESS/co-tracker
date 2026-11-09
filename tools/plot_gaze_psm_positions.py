"""Plot synchronized, smoothed gaze and PSM positions from trajectory caches."""
from gaze_psm_modes import normalize_modes
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gaze_psm_figures import BLUE, ORANGE, _limits, _save


def load_trajectory(path, *, include_stages=False):
    """Read only the arrays needed for the time plots; never rewrite a cache."""
    keys = ('time_s', 'gaze_l', 'psm_l', 'gaze_r', 'psm_r',
            'resolution_l', 'resolution_r')
    if include_stages:
        keys += ('stage', 'occurrence')
    try:
        with np.load(path, allow_pickle=False) as cache:
            data = {key: cache[key] for key in keys}
            qc = normalize_modes(json.loads(str(cache['qc_json'])))
        time = data['time_s']
        if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
                or not (np.diff(time) > 0).all()):
            raise ValueError('time_s must contain at least two increasing, finite timestamps')
        if include_stages:
            for key in ('stage', 'occurrence'):
                if data[key].shape != time.shape:
                    raise ValueError(f'{key} must have the same shape as time_s')
        for side in ('l', 'r'):
            for source in ('gaze', 'psm'):
                key = f'{source}_{side}'
                if data[key].shape != (len(time), 2):
                    raise ValueError(f'{key} must have shape ({len(time)}, 2)')
            resolution = data[f'resolution_{side}']
            if (resolution.shape != (2,) or not np.isfinite(resolution).all()
                    or not (resolution > 0).all()):
                raise ValueError(f'resolution_{side} must contain positive width and height')
        if not isinstance(qc, dict) or not qc.get('task'):
            raise ValueError('qc_json must contain task metadata')
        if qc.get('content_id') != path.stem:
            raise ValueError('qc_json content_id does not match the cache filename')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{path.stem}: {exc}') from exc
    return data, qc


def stage_names(data):
    valid = ((data['occurrence'] >= 0)
             & ~np.isin(data['stage'], ('', 'none', 'nan', 'null')))
    return list(dict.fromkeys(data['stage'][valid].tolist()))


def stage_runs(data, stage):
    """Return separate contiguous occurrences, including breaks in occurrence IDs."""
    labels, occurrences = data['stage'], data['occurrence']
    edges = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1])
                                   | (occurrences[1:] != occurrences[:-1]))+1,
                  len(labels)]
    runs = [(start, end) for start, end in zip(edges[:-1], edges[1:])
            if labels[start] == stage and occurrences[start] >= 0]
    if not runs:
        raise ValueError(f'No labeled samples for stage {stage!r}')
    return runs


def occurrence_intervals(data, stage):
    """Lay out occurrence windows with narrow blank gaps and original time bounds."""
    time = data['time_s']
    step = float(np.median(np.diff(time)))
    runs = stage_runs(data, stage)
    durations = np.array([max(float(time[end-1]-time[start]), step) for start, end in runs])
    gap = min(.012, .10/max(len(runs)-1, 1)) if len(runs) > 1 else 0.
    widths = (1-gap*(len(runs)-1)) * (.8/len(runs) + .2*durations/durations.sum())
    cursor = 0.
    intervals = []
    for ordinal, ((start, end), width) in enumerate(zip(runs, widths), start=1):
        paired = {side: int((np.isfinite(data[f'gaze_{side}'][start:end]).all(axis=1)
                            & np.isfinite(data[f'psm_{side}'][start:end]).all(axis=1)).sum())
                  for side in ('l', 'r')}
        intervals.append({'ordinal': ordinal, 'occurrence_id': int(data['occurrence'][start]),
                          'video_start_s': float(time[start]), 'video_end_s': float(time[end-1]),
                          'display_start': float(cursor), 'display_end': float(cursor+width),
                          'samples': int(end-start),
                          'paired_samples_l': paired['l'], 'paired_samples_r': paired['r']})
        cursor += float(width)+gap
    return intervals


def stage_view(data, stage, *, compress_time=False, paired_only=False):
    """Select a stage, optionally masking unpaired samples and using separate time windows."""
    runs = stage_runs(data, stage)
    # A separator is needed even if two occurrence IDs meet at adjacent samples.
    indices = np.concatenate([np.r_[np.arange(start, end), -1] for start, end in runs])[:-1]
    separator = indices < 0
    view = {}
    for key in ('time_s', 'gaze_l', 'psm_l', 'gaze_r', 'psm_r'):
        values = data[key][indices].copy()
        values[separator] = np.nan
        view[key] = values
    offsets = np.cumsum([0] + [end-start+1 for start, end in runs[:-1]])
    singletons = [int(offset) for offset, (start, end) in zip(offsets, runs) if end-start == 1]
    if compress_time:
        for offset, (start, end), interval in zip(offsets, runs, occurrence_intervals(data, stage)):
            duration = interval['video_end_s']-interval['video_start_s']
            fraction = ((data['time_s'][start:end]-data['time_s'][start])/duration
                        if duration > 0 else np.full(end-start, .5))
            view['time_s'][offset:offset+end-start] = (interval['display_start'] + fraction
                    * (interval['display_end']-interval['display_start']))
    if paired_only:
        for side in ('l', 'r'):
            paired = (np.isfinite(view[f'gaze_{side}']).all(axis=1)
                      & np.isfinite(view[f'psm_{side}']).all(axis=1))
            view[f'gaze_{side}'][~paired] = np.nan
            view[f'psm_{side}'][~paired] = np.nan
    return view, len(runs), singletons


def stage_slug(stage):
    return re.sub(r'[^a-z0-9]+', '_', stage.lower()).strip('_') or 'stage'


def load_position_lags(output, mode='PSM in stage'):
    """Select signed position-correlation maxima from the saved record curves."""
    results = normalize_modes(json.loads((Path(output)/'results.json').read_text(encoding='utf-8')))
    peaks = {}
    for curve in results['record_curves']:
        if curve.get('signal') != 'position' or curve.get('mode') != mode:
            continue
        r, lags = np.asarray(curve['r'], dtype=float), np.asarray(curve['lag_s'], dtype=float)
        key = (curve['content_id'], curve['stage'], curve['side'])
        if key in peaks:
            raise ValueError(f'Duplicate saved lag curve: {key}')
        supported = np.isfinite(r) & np.isfinite(lags)
        index = int(np.argmax(np.where(supported, r, -np.inf))) if supported.any() else None
        zero = np.flatnonzero(np.isclose(lags, 0., atol=1e-12, rtol=0))
        peaks[key] = {'lag_s': float(lags[index]) if index is not None else None,
                      'peak_r': float(r[index]) if index is not None else None,
                      'zero_r': float(r[zero[0]]) if len(zero) and np.isfinite(r[zero[0]]) else None,
                      'peak_at_boundary': index in (0, len(r)-1) if index is not None else False,
                      'search_low_s': float(lags.min()), 'search_high_s': float(lags.max()),
                      'status': 'ok' if index is not None else 'no_supported_curve'}
    return peaks


def align_gaze(data, peaks, *, mode='PSM in stage'):
    """At PSM time t, show gaze(t-lag); never wrap or interpolate missing data."""
    time = data['time_s']
    step = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), step, rtol=1e-6, atol=1e-9):
        raise ValueError('Lag comparison requires the uniformly sampled trajectory cache')
    aligned = dict(data)
    shifts = {}
    target = np.arange(len(time))
    for side in ('l', 'r'):
        lag = peaks[side]['lag_s']
        shifted = np.full_like(data[f'gaze_{side}'], np.nan)
        if lag is not None:
            shift = int(round(lag/step))
            if not np.isclose(lag/step, shift, atol=1e-6, rtol=0):
                raise ValueError(f'PSM {side.upper()}: saved lag is incompatible with the cache time grid')
            source = target-shift
            inside = (source >= 0) & (source < len(time))
            destination = target[inside]
            origin = source[inside]
            if mode == 'Both in stage':
                same = ((data['occurrence'][destination] >= 0)
                        & (data['occurrence'][destination] == data['occurrence'][origin]))
                destination, origin = destination[same], origin[same]
            shifted[destination] = data[f'gaze_{side}'][origin]
            shifts[side] = shift
        else:
            shifts[side] = None
        aligned[f'gaze_{side}'] = shifted
    return aligned, shifts


def lag_comparison_limits(original, aligned, stage):
    views = [stage_view(data, stage, paired_only=True)[0] for data in (original, aligned)]
    return [_limits([view[f'{source}_{side}'][:, row] for view in views
                     for source in ('gaze', 'psm') for side in ('l', 'r')],
                    minimum=50, padding=.08,
                    bounds=(0, max(float(original[f'resolution_{side}'][row]) for side in ('l', 'r'))))
            for row in range(2)]


def lag_comparison_directory(output, mode):
    return Path(output)/'figures'/'position_traces_by_stage'/'lag_comparison'/mode


def _plot_trajectory(plt, output, path, config, data, qc, *, stage=None, comparison=None):
    staged = stage is not None
    view, occurrences, singletons = (stage_view(data, stage, compress_time=True, paired_only=True)
                                     if staged else (data, 0, []))
    intervals = occurrence_intervals(data, stage) if staged else []
    time = view['time_s']
    finite_time = time[np.isfinite(time)]
    low, high = float(finite_time[0]), float(finite_time[-1])
    if staged:
        low, high = 0., 1.
    if low == high:
        half_step = float(np.median(np.diff(data['time_s'])))/2
        low, high = low-half_step, high+half_step
    fig, axes = plt.subplots(2, 2, figsize=(16, 10 if staged else 8.5), sharex=True, sharey='row')
    try:
        markers = dict(marker='.', markersize=3, markevery=singletons) if singletons else {}
        for row, coordinate in enumerate(('X', 'Y')):
            for col, side in enumerate(('l', 'r')):
                ax = axes[row, col]
                if staged:
                    paired = np.isfinite(view[f'gaze_{side}']).all(axis=1)
                    isolated = np.flatnonzero(paired & ~np.r_[False, paired[:-1]]
                                              & ~np.r_[paired[1:], False]).tolist()
                    markers = dict(marker='.', markersize=4, markevery=isolated) if isolated else {}
                    for number, interval in enumerate(intervals):
                        left, right = interval['display_start'], interval['display_end']
                        if number % 2 == 0:
                            ax.axvspan(left, right, color='#e9eef3', alpha=.4, zorder=0)
                        if number:
                            ax.axvspan(intervals[number-1]['display_end'], left,
                                       color='white', zorder=1)
                        ax.plot([left, right], [0, 0], transform=ax.get_xaxis_transform(),
                                color='.25', linewidth=.8, clip_on=False)
                    if row == 0:
                        for interval in intervals:
                            center = (interval['display_start']+interval['display_end'])/2
                            ax.text(center, 1.025, f"#{interval['ordinal']}",
                                    transform=ax.get_xaxis_transform(), ha='center',
                                    fontsize=10, color='.35')
                if not staged or paired.any():
                    ax.plot(time, view[f'gaze_{side}'][:, row], color=BLUE,
                            linewidth=1.25 if staged else .75, alpha=.95 if staged else .85,
                            label='Gaze', **markers)
                    ax.plot(time, view[f'psm_{side}'][:, row], color=ORANGE,
                            linewidth=1.65 if staged else 1.05,
                            linestyle='--' if staged else '-', label='PSM centroid', **markers)
                if not staged or row == 0:
                    title = f"{'Left' if side == 'l' else 'Right'} PSM" if staged else f'PSM {side.upper()} / {coordinate} position'
                    if comparison:
                        peak = comparison['peaks'][side]
                        lag = 0. if comparison['kind'] == 'zero_lag' else peak['lag_s']
                        title += (f' / {lag*1000:+.0f} ms' if lag is not None else ' / lag unavailable')
                        if comparison['kind'] == 'optimal_lag' and peak['peak_at_boundary']:
                            title += ' (boundary)'
                    ax.set_title(title, pad=32 if staged else 10,
                                 fontsize=14 if comparison else 16 if staged else 12)
                ax.set_ylabel(f'{coordinate} position (px)', fontsize=14 if staged else 10)
                ax.set_xlim(low, high)
                ax.grid(axis='y' if staged else 'both', alpha=.2, zorder=0)
                ax.spines[['top', 'right']].set_visible(False)
                if staged:
                    ax.tick_params(labelsize=12)
                    ax.spines['bottom'].set_visible(False)
                    ax._gaze_time_reference = 'video_time_s_in_separate_occurrence_windows'
                if comparison:
                    ax._gaze_alignment = {'comparison': comparison['kind'],
                                          'lag_s': 0. if comparison['kind'] == 'zero_lag'
                                          else comparison['peaks'][side]['lag_s'],
                                          'lag_steps': 0 if comparison['kind'] == 'zero_lag'
                                          else comparison['shifts'][side],
                                          'mode': comparison['mode'],
                                          'optimal_lag_s': comparison['peaks'][side]['lag_s'],
                                          'peak_at_boundary': comparison['peaks'][side]['peak_at_boundary']}
                ax._gaze_comparison_group = (f'position_traces:{path.stem}:{coordinate.lower()}'
                                             + (f':{stage}' if stage is not None else ''))
                missing = [label for key, label in ((f'gaze_{side}', 'gaze'),
                                                   (f'psm_{side}', 'PSM'))
                           if not np.isfinite(view[key][:, row]).any()]
                if missing:
                    message = 'No paired gaze / PSM samples' if staged else f"No valid {' / '.join(missing)} samples"
                    if (comparison and comparison['kind'] == 'optimal_lag'
                            and comparison['peaks'][side]['lag_s'] is None):
                        message = 'No supported optimal lag'
                    ax.text(.02, .96, message,
                            transform=ax.transAxes, va='top', fontsize=11 if staged else 9,
                            color='.35')
            maximum = max(float(data[f'resolution_{side}'][row]) for side in ('l', 'r'))
            limits = (_limits([view[f'{source}_{side}'][:, row]
                               for source in ('gaze', 'psm') for side in ('l', 'r')],
                              minimum=50, padding=.08, bounds=(0, maximum))
                      if staged else (0, maximum))
            if comparison:
                limits = comparison['limits'][row]
            axes[row, 0].set_ylim(*limits)
        for ax in axes[1]:
            ax.set_xlabel('Video time (s); shortened gaps' if staged else 'Time since video start (s)',
                          fontsize=13 if staged else 10)
            if staged:
                positions, labels = [], []
                for interval in intervals:
                    positions.extend([interval['display_start'], interval['display_end']])
                    labels.extend([f"{interval['video_start_s']:.2f}", f"{interval['video_end_s']:.2f}"])
                ax.set_xticks(positions, labels, rotation=90,
                              fontsize=8.5 if occurrences > 16 else 10)
                ax.tick_params(axis='x', pad=5, length=3)
                for index, label in enumerate(ax.get_xticklabels()):
                    label.set_ha('left' if index % 2 == 0 else 'right')
                    label.set_va('top')
                    label.set_rotation_mode('default')
        if staged:
            for ax in axes[0]:
                ax.tick_params(axis='x', bottom=False)
        title = f"{path.stem} / Task {qc['task']} / Gaze and PSM position over time"
        if staged:
            title = f"{path.stem} / Task {qc['task']}\n{stage.capitalize()} / {occurrences} occurrence{'s' if occurrences != 1 else ''}"
        if comparison:
            title += ' / ' + ('Zero lag' if comparison['kind'] == 'zero_lag' else 'Optimal lag')
        fig.suptitle(title, fontsize=18 if staged else 16, y=.985)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if staged and not handles:
            handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(.5, .91 if staged else .945),
                   ncol=2, frameon=False, fontsize=13 if staged else 10)
        smoothing = (f"{config['smooth_ms']:g} ms smoothing setting"
                     if config.get('smooth_ms') is not None else 'smoothing duration unavailable')
        footer = (f'Cached smoothed trajectories; {smoothing}. '
                  'Original synchronized times; gaps preserved. Image Y increases downward.')
        if staged:
            footer = ('Only simultaneous gaze / PSM samples are shown; all other samples are blank.\n'
                      'Each occurrence has its own time scale; tick labels show original video start / end times. '
                      f'{smoothing}.')
        if comparison:
            anchoring = ('PSM in stage anchored; shifted gaze may come from an adjacent stage.'
                         if comparison['mode'] == 'PSM in stage' else 'Gaze and PSM remain in the same stage occurrence.')
            footer = ('Only valid gaze / PSM pairs; separate time scales per occurrence; labels = original video start / end times.\n'
                      'Positive lag = gaze leads; optimal alignment shows gaze(t - lag) versus PSM(t). '
                      'The saved 2D position lag is shared by X and Y.\n' + anchoring)
        fig.text(.5, .015, footer,
                 ha='center', fontsize=10 if staged else 9, color='.3')
        if staged:
            fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.80,
                                hspace=.24, wspace=.18)
        else:
            fig.tight_layout(rect=(0, .045, 1, .91), h_pad=2.5, w_pad=2.5)
        directory = output/'figures'
        filename = f'position_traces_{path.stem}'
        if stage is not None:
            directory /= 'position_traces_by_stage'
            filename += f'__{stage_slug(stage)}'
        if comparison:
            directory = lag_comparison_directory(output, comparison['mode'])
            filename += f"__{comparison['kind']}"
        destination = directory/f'{filename}.png'
        _save(fig, destination, partial_record=bool(qc.get('partial_record', False)
                                                   or config.get('partial_record', False)))
        return destination
    finally:
        plt.close(fig)


def plot_record(plt, output, path, config):
    data, qc = load_trajectory(path)
    return _plot_trajectory(plt, output, path, config, data, qc)


def plot_positions(output, records=None, *, by_stage=False, compare_lags=False, lag_mode='PSM in stage'):
    """Save four-panel PNG/PDF pairs for recordings or their labeled stages."""
    output = Path(output)
    if compare_lags and not by_stage:
        raise ValueError('--compare-lags requires --by-stage')
    available = {path.stem: path for path in sorted((output/'cache').glob('*.npz'))}
    if not available:
        raise ValueError(f'No trajectory caches found in {output / "cache"}')
    selected = list(dict.fromkeys(records)) if records else list(available)
    missing = [cid for cid in selected if cid not in available]
    if missing:
        raise ValueError(f"Trajectory caches not found: {', '.join(missing)}")
    config_path = output/'config.json'
    config = normalize_modes(json.loads(config_path.read_text(encoding='utf-8'))) if config_path.exists() else {}
    if not isinstance(config, dict):
        raise ValueError(f'{config_path} must contain a JSON object')

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    directory = output/'figures'
    if by_stage:
        directory /= 'position_traces_by_stage'
    if compare_lags:
        directory = lag_comparison_directory(output, lag_mode)
    directory.mkdir(parents=True, exist_ok=True)
    occurrence_manifest_path = directory/'occurrences.json'
    occurrence_manifest = (normalize_modes(json.loads(occurrence_manifest_path.read_text(encoding='utf-8')))
                           if by_stage and occurrence_manifest_path.exists() else {})
    peaks = load_position_lags(output, lag_mode) if compare_lags else {}
    lag_manifest_path = directory/'lag_comparisons.json'
    lag_manifest = (normalize_modes(json.loads(lag_manifest_path.read_text(encoding='utf-8')))
                    if compare_lags and lag_manifest_path.exists() else {})
    paths = []
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'savefig.facecolor': 'white', 'path.simplify': False}):
        for index, cid in enumerate(selected, start=1):
            if by_stage:
                data, qc = load_trajectory(available[cid], include_stages=True)
                stages = stage_names(data)
                if not stages:
                    raise ValueError(f'{cid}: no labeled stage samples in the trajectory cache')
                for stage in stages:
                    if compare_lags:
                        selected_peaks = {side: peaks[(cid, stage, side)] for side in ('l', 'r')}
                        aligned, shifts = align_gaze(data, selected_peaks, mode=lag_mode)
                        context = {'mode': lag_mode, 'peaks': selected_peaks, 'shifts': shifts,
                                   'limits': lag_comparison_limits(data, aligned, stage)}
                        filenames = {}
                        for kind, trajectories in (('zero_lag', data), ('optimal_lag', aligned)):
                            paths.append(_plot_trajectory(plt, output, available[cid], config,
                                trajectories, qc, stage=stage, comparison={**context, 'kind': kind}))
                            occurrence_manifest[paths[-1].name] = occurrence_intervals(trajectories, stage)
                            filenames[kind] = paths[-1].name
                        lag_manifest[f'{cid}__{stage_slug(stage)}'] = {
                            'content_id': cid, 'stage': stage, 'mode': lag_mode,
                            'source': 'results.json: record_curves, signal=position',
                            'definition': 'gaze(t - lag) versus PSM(t); positive lag = gaze leads',
                            'shared_xy_lag': True, 'peaks': selected_peaks, 'shifts': shifts,
                            'files': filenames}
                    else:
                        paths.append(_plot_trajectory(plt, output, available[cid], config,
                                                      data, qc, stage=stage))
                        occurrence_manifest[paths[-1].name] = occurrence_intervals(data, stage)
                count = len(stages)*(2 if compare_lags else 1)
                print(f'[{index}/{len(selected)}] {cid}: saved {count} stage PNG/PDF pairs',
                      flush=True)
            else:
                paths.append(plot_record(plt, output, available[cid], config))
                print(f'[{index}/{len(selected)}] Saved {paths[-1]} and PDF', flush=True)
    if by_stage:
        occurrence_manifest_path.write_text(json.dumps(occurrence_manifest, indent=2), encoding='utf-8')
    if compare_lags:
        lag_manifest_path.write_text(json.dumps(lag_manifest, indent=2), encoding='utf-8')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gaze_psm_xcorr'),
                        help='Analysis directory containing cache/ and config.json')
    parser.add_argument('--records', nargs='+', metavar='ID',
                        help='Recording IDs to plot; default: all cached recordings')
    parser.add_argument('--by-stage', action='store_true',
                        help='Save each labeled stage separately in figures/position_traces_by_stage/')
    parser.add_argument('--compare-lags', action='store_true',
                        help='With --by-stage, save matching zero-lag and optimal-lag figures')
    parser.add_argument('--lag-mode', choices=('PSM in stage', 'Both in stage'), default='PSM in stage',
                        help='Saved stage correlation mode supplying the optimal lag (default: PSM in stage)')
    args = parser.parse_args(argv)
    try:
        paths = plot_positions(args.output, args.records, by_stage=args.by_stage,
                               compare_lags=args.compare_lags, lag_mode=args.lag_mode)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f'Saved {len(paths)} position-over-time PNG/PDF pairs in {paths[0].parent}.')


if __name__ == '__main__':
    main()
