#!/usr/bin/env python3
"""Relate selected-PSM stage peaks to all five performance score fields.

Run from the repository root: python tools/analyze_score_corr_max.py
Reads existing selected curves and scores; no trajectory extraction or re-selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from tools import analyze_score_corr as base
from tools.analyze_raw_score_corr import validate_results
from tools import plot_peak_correlation_vs_score as plots

OUTCOMES = ('Score', 'Time', 'Eco', 'TimeRaw', 'EcoRaw')
SELECTION_KEY = ('content_id', 'user', 'task', 'signal', 'stage', 'mode')


def save_json(path, value):
    def ready(v):
        if isinstance(v, dict):
            return {str(k): ready(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [ready(x) for x in v]
        if isinstance(v, np.generic):
            return ready(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    path.write_text(json.dumps(ready(value), indent=2, allow_nan=False)+'\n')


def selection_lookup(source):
    rows = base.read_csv(source/'side_selection.csv')
    lookup = {tuple(r[k] for k in SELECTION_KEY): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate selected-PSM metadata')
    return lookup


def attach_selection(records, lookup):
    """Join by recording/stage identity, never by score or chosen instrument."""
    if len(records) != len(lookup):
        raise ValueError('Stage records and selection metadata do not match')
    for row in records:
        if row['side'] != 'max' or row['mode'] not in ('psm_stage', 'within_stage'):
            raise ValueError('Expected selected-PSM stage curves only')
        detail = lookup[tuple(row[k] for k in SELECTION_KEY)]
        side = detail['selected_side']
        if side not in ('', 'l', 'r'):
            raise ValueError('Invalid selected_side')
        expected = base.number(detail[('left' if side == 'l' else 'right')+'_peak_r']) if side else np.nan
        if not np.isclose(row['peak_r'], expected, atol=1e-10, equal_nan=True):
            raise ValueError('Selected peak differs from selection metadata')
        row.update({k: detail[k] for k in ('selected_side', 'selection_status',
                    'left_peak_r', 'right_peak_r', 'left_peak_lag_s', 'right_peak_lag_s')})


def overview(results, phases, outcome, signal, scenario):
    phases = sorted([p for p in phases if p['signal'] == signal and p['mode'] == 'psm_stage'],
                    key=base.group_order)
    lookup = {(r['method'], base.key(r), r['predictor']): r
              for r in results if r['scenario'] == scenario}
    fig, axes = base.plt.subplots(1, 3, figsize=(15, 9), sharey=True)
    titles = ('Between participants', 'Within participant', 'Within participant\nadjusted for task order + attempt')
    for ax, method, title in zip(axes, base.METHODS, titles):
        matrix = np.full((len(phases), 3), np.nan)
        for i, phase in enumerate(phases):
            for j, predictor in enumerate(base.PREDICTORS):
                row = lookup[(method, base.key(phase), predictor)]
                value = row.get('association', np.nan)
                matrix[i, j] = value
                label = (f"{value:+.2f}"+(' *' if row.get('q_bh', 1) < .05 else '')
                         if np.isfinite(value) else ('excluded' if not phase['group_usable'] else 'low N'))
                ax.text(j, i, label, ha='center', va='center', fontsize=9,
                        color='white' if np.isfinite(value) and abs(value) > .65 else 'black')
        cmap = base.plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#eeeeee')
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect='auto')
        ax.set_xticks(range(3), ('Signed lag', 'Absolute lag', 'Peak r'))
        ax.set_yticks(range(len(phases)), [base.phase_label(p).replace('(MAX)', '(selected PSM)') for p in phases])
        ax.set_title(title)
    fig.suptitle(f'{outcome} vs selected-PSM stage features ({signal})\n'
                 '* BH q < 0.05 within outcome/scenario/method; all three predictors and stages', fontsize=13)
    fig.subplots_adjust(left=.25, right=.89, top=.88, bottom=.16, wspace=.08)
    fig.colorbar(im, cax=fig.add_axes([.92, .23, .015, .55]), label='Rank association')
    fig.text(.5, .035, 'One PSM chosen per recording/stage by largest signed xcorr peak, before score matching.\n'
             'Positive signed lag = gaze leads. Scores describe whole attempts. Missing outcomes are excluded.\n'
             'Selected PSM indicates stronger correlation, not necessarily the operated instrument.', ha='center', fontsize=9)
    return fig


def protected_hashes(paths):
    files = set()
    for path in paths:
        if path.is_dir():
            files.update(p for p in path.rglob('*') if p.is_file())
        elif path.is_file():
            files.add(path)
    return {str(p): base.sha256(p) for p in sorted(files)}


def validate_scenarios(results, people, records, outcome):
    checks = {}
    for scenario in sorted({r['scenario'] for r in results}):
        # Independent validation must use the same sensitivity-specific attempts.
        checks[scenario] = validate_results(
            [r for r in results if r['scenario'] == scenario],
            [p for p in people if p['scenario'] == scenario],
            base.scenario_rows(records, scenario), outcome)
    return checks


def make_report(output, coverage, results, settings, n_recordings):
    lines = ['# Performance associations with maximum-xcorr PSM stage peaks', '',
             'Each recording/stage uses its previously selected PSM. Score matching never changes that selection. '
             'Position is primary and velocity is secondary; whole-task curves are not included.', '',
             '| Outcome | Available recordings before stage filtering | Successful tests, including sensitivities | BH q < 0.05 |',
             '|---|---:|---:|---:|']
    for outcome in OUTCOMES:
        rr = [r for r in results if r['outcome'] == outcome]
        lines.append(f"| {outcome} | {coverage[outcome]} / {n_recordings} | {sum(r['status']=='ok' for r in rr)} | "
                     f"{sum(r.get('q_bh', 1)<.05 for r in rr)} |")
    lines += ['', '## Methods', '',
        'Outcomes are supplied Score, Time, Eco, TimeRaw and EcoRaw values, without sign reversal or imputation. '
        'Failed attempts with supplied zero scores remain in primary analyses. Available-score counts are before stage exclusions.', '',
        'Eligible groups have status ok, positive peak correlation and at least three stable users. '
        'Eligible recordings have usable outcomes and quality, positive interior peaks and complete lag support. '
        f"Tests require {settings['min_users']} qualifying participants; within-participant tests require within-user variation.", '',
        'Between-participant summaries use mean outcomes, median signed lag, median absolute lag and Fisher-z mean peak r '
        'from exactly the same eligible recordings. Within-participant analyses use equal participant weights and rank-based '
        'fixed effects, with an additional model adjusting for task order and attempt. '
        'Positive signed lag means gaze leads PSM.', '',
        f"Uncertainty uses {settings['bootstrap']:,} whole-participant bootstrap draws. Between-participant p values use "
        f"{settings['permutations']:,} two-sided participant score permutations with a +1 correction. "
        'Within-participant p values use the existing participant-cluster CR1 t approximation.', '',
        'BH families pool all three predictors and stages separately within each outcome, scenario and method. '
        'Outcomes, methods and sensitivity scenarios are not jointly corrected. '
        'Total Score includes within-stage, completed-only and stronger-peaks sensitivities. '
        'Other outcomes use primary position and secondary velocity analyses.', '',
        'Selected PSM means the instrument with higher signed correlation, not an independently identified operated instrument. '
        'All associations are exploratory; score is measured for the whole attempt, not for each stage.', '',
        '## Results', '',
        'See [all associations](all_associations.csv), [record selection audit](record_phase_audit.csv), '
        '[association overviews](association_overviews.pdf), and the `peak_correlation_vs_*`, '
        '`lag_correlation_vs_*`, and `absolute_lag_correlation_vs_*` figure directories. '
        'Scatter points show participants; Pearson fits are descriptive, while Spearman p/q and confidence intervals '
        'come from the saved inference. Intervals with fewer than 97.5% defined bootstrap draws are labeled unreliable.', '',
        '## Reproduce', '', '```bash',
        f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/analyze_score_corr_max.py "
        f"--input {settings['input']} --index {settings['index']} --output {settings['output']} "
        f"--bootstrap {settings['bootstrap']} --permutations {settings['permutations']} "
        f"--seed {settings['seed']} --min-users {settings['min_users']}", '```', '']
    (output/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs/gaze_psm_xcorr_max'))
    parser.add_argument('--index', type=Path, default=Path('Index'))
    parser.add_argument('--output', type=Path, default=Path('outputs/score_corr_max'))
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--permutations', type=int, default=19999)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--min-users', type=int, default=6)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if min(args.bootstrap, args.permutations) < 100 or args.min_users < 3:
        parser.error('Require bootstrap/permutations >=100 and min-users >=3')
    preserved = [args.input.resolve(), args.index.resolve(),
                 Path('outputs/gaze_psm_xcorr').resolve(), Path('outputs/score_corr').resolve()]
    dest = args.output.resolve()
    if any(dest == p or dest in p.parents or p in dest.parents for p in preserved):
        parser.error('Output must be separate from the input, scores and original output directories')
    before = protected_hashes(preserved)
    lookup = selection_lookup(args.input)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = dict(created_at=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds'),
                      settings=settings, protected_sha256=before,
                      versions=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=base.matplotlib.__version__))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    all_results, all_people, coverage, checks = [], [], {}, {}
    plot_jobs = []
    for outcome in OUTCOMES:
        config, inventory, recordings, records, phases = base.prepare(args, outcome)
        attach_selection(records, lookup)
        coverage[outcome] = sum(r['score_usable'] for r in recordings)
        print(f'{outcome}: {coverage[outcome]}/{len(recordings)} usable outcomes', flush=True)
        scenarios = None if outcome == 'Score' else ('primary', 'velocity_secondary')
        results, people = base.analyze(args, records, phases, outcome, scenarios=scenarios)
        checks[outcome] = validate_scenarios(results, people, records, outcome)
        for row in results+people:
            row['outcome'] = outcome
        all_results.extend(results)
        all_people.extend(people)
        if outcome == 'Score':
            for name, rows in [('associations', results), ('participant_summaries', people),
                ('matched_scores', recordings), ('score_inventory', inventory),
                ('record_phase_audit', records), ('phase_validity', phases)]:
                base.write_csv(output/f'{name}.csv', rows)
        for signal, scenario in [('position', 'primary'), ('velocity', 'velocity_secondary')]:
            if signal == 'position' and outcome == 'Score':
                directory = output
            else:
                directory = output/('velocity_metrics' if signal == 'velocity' else
                                    'component_metrics' if outcome in ('Time', 'Eco') else 'raw_metrics')
                directory.mkdir(exist_ok=True)
                for name, rows in [('associations', results), ('participant_summaries', people),
                    ('matched_outcomes', recordings), ('record_phase_audit', records), ('phase_validity', phases)]:
                    keep = [r for r in rows if r.get('signal', signal) == signal]
                    base.write_csv(directory/f'{name}_{outcome}.csv', keep)
            save_json(directory/'provenance.json', provenance)
            plot_jobs.append((outcome, signal, scenario, results, phases))
    base.write_csv(output/'all_associations.csv', all_results)
    base.write_csv(output/'all_participant_summaries.csv', all_people)
    base.write_csv(output/'side_selection.csv', list(lookup.values()))
    plot_validation = []
    if not args.no_plots:
        figures = output/'figures'
        figures.mkdir(exist_ok=True)
        with PdfPages(output/'association_overviews.pdf') as book:
            for outcome, signal, scenario, results, phases in plot_jobs:
                fig = overview(results, phases, outcome, signal, scenario)
                fig.savefig(figures/f'association_overview_{outcome}_{signal}.png', dpi=180, bbox_inches='tight')
                fig.savefig(figures/f'association_overview_{outcome}_{signal}.pdf', bbox_inches='tight')
                book.savefig(fig, bbox_inches='tight')
                base.plt.close(fig)
                for predictor, prefix in [('peak_r', 'peak_correlation'), ('signed_lag_s', 'lag_correlation'),
                                          ('absolute_lag_s', 'absolute_lag_correlation')]:
                    summary, points, hashes = plots.load_and_check(output, signal, outcome, predictor)
                    label = 'score' if outcome == 'Score' else outcome
                    directory = output/f'{prefix}_vs_{label}'/signal
                    directory.mkdir(parents=True, exist_ok=True)
                    plots.write_csv(directory/'associations.csv', summary)
                    plots.write_csv(directory/'plotted_points.csv', points)
                    saved = plots.make_figures(directory, signal, summary, points, outcome, predictor)
                    assert all(plots.digest(output/name) == sha for name, sha in hashes.items())
                    check = dict(outcome=outcome, signal=signal, predictor=predictor,
                                 n_conditions=len(summary), n_points=len(points), figures=saved,
                                 participant_summaries_and_scipy_checked=True,
                                 spearman_ci_display=plots.ci_display_summary(summary), input_sha256=hashes)
                    save_json(directory/'validation.json', check)
                    plot_validation.append(check)
                    print(f'Plotted {outcome}/{signal}/{predictor}', flush=True)
    make_report(output, coverage, all_results, settings, len(recordings))
    after = protected_hashes(preserved)
    if before != after:
        raise RuntimeError('Protected inputs or existing outputs changed during analysis')
    scripts = [Path(__file__), Path(base.__file__), Path(plots.__file__),
               Path(__file__).with_name('analyze_raw_score_corr.py')]
    snapshot = output/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in scripts:
        (snapshot/path.name).write_bytes(path.read_bytes())
    provenance.update(input_config=config, analysis_source_sha256={str(p): base.sha256(p) for p in scripts},
                      protected_files_unchanged=True, coverage=coverage)
    save_json(output/'provenance.json', provenance)
    # Refresh the provenance reference after adding final run metadata. Numerical
    # input tables have already been verified unchanged during plotting.
    for check in plot_validation:
        for name in check['input_sha256']:
            check['input_sha256'][name] = plots.digest(output/name)
        prefix = {'peak_r': 'peak_correlation', 'signed_lag_s': 'lag_correlation',
                  'absolute_lag_s': 'absolute_lag_correlation'}[check['predictor']]
        label = 'score' if check['outcome'] == 'Score' else check['outcome']
        save_json(output/f'{prefix}_vs_{label}'/check['signal']/'validation.json', check)
    save_json(output/'validation.json', dict(numerical_checks=base.numerical_checks(),
        independent_validation=checks, coverage=coverage, n_tests=len(all_results),
        statuses=dict(Counter(r['status'] for r in all_results)), n_plot_sets=len(plot_validation),
        selection_metadata_checked=True, protected_files_unchanged=True, all_checks_passed=True))
    print(f'Saved {len(all_results)} association rows to {output}; protected files unchanged', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
