#!/usr/bin/env python3
"""Build matching metrics/ and plots/ comparison layouts without changing inference.

Legacy files remain available for existing scripts and historical provenance.
Run again after regenerating an analysis to refresh the comparison layout.
"""
from __future__ import annotations
from gaze_psm_modes import normalize_modes

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tools import analyze_score_corr as base
from tools import plot_peak_correlation_vs_score as plots

OUTCOMES = ('Score', 'Time', 'Eco', 'TimeRaw', 'EcoRaw', 'PenaltySum', 'PenaltyRawSum')
SIGNALS = ('position', 'velocity')
PREDICTORS = ('peak_r', 'signed_lag_s', 'absolute_lag_s')
TABLES = ('associations', 'participant_summaries', 'matched_outcomes',
          'record_phase_audit', 'phase_validity')
MANAGED = {'metrics', 'plots', 'README.md', 'comparison_index.csv', 'layout.json',
           'layout_validation.json', 'layout_code_snapshot'}
PREFIXES = dict(peak_r='peak_correlation', signed_lag_s='lag_correlation',
                absolute_lag_s='absolute_lag_correlation')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def original_hashes(root):
    return {p.relative_to(root).as_posix(): base.sha256(p)
            for p in sorted(root.rglob('*')) if p.is_file()
            and p.relative_to(root).parts[0] not in MANAGED}


def source_tables(root, signal, outcome):
    names, provenance = plots.analysis_table_paths(root, signal, outcome)
    paths = dict(zip(('participant_summaries', 'associations', 'record_phase_audit'), names))
    parent = Path(names[0]).parent
    legacy_score = parent == Path('.')
    paths['matched_outcomes'] = 'matched_scores.csv' if legacy_score else str(parent/f'matched_outcomes_{outcome}.csv')
    paths['phase_validity'] = 'phase_validity.csv' if legacy_score else str(parent/f'phase_validity_{outcome}.csv')
    return {k: root/v for k, v in paths.items()}, root/provenance


def normalize_tables(root, signal, outcome):
    paths, provenance = source_tables(root, signal, outcome)
    dest = root/'metrics'/signal/outcome
    dest.mkdir(parents=True, exist_ok=True)
    data, checks = {}, {}
    for name in TABLES:
        original = base.read_csv(paths[name])
        rows = [r for r in original if r.get('signal', signal) == signal]
        if not rows:
            raise ValueError(f'No {signal}/{outcome} rows in {paths[name]}')
        base.write_csv(dest/f'{name}.csv', rows)
        # CSV strings must survive filtering/export exactly, including missing
        # values, p/q values, intervals and selected-side metadata.
        if base.read_csv(dest/f'{name}.csv') != rows:
            raise ValueError(f'Table changed while normalizing: {dest/name}')
        data[name] = rows
        checks[name] = dict(source=paths[name].relative_to(root).as_posix(),
                           source_sha256=base.sha256(paths[name]), n_rows=len(rows),
                           normalized_sha256=base.sha256(dest/f'{name}.csv'))
    info = normalize_modes(json.loads(provenance.read_text()))
    save_json(dest/'provenance.json', dict(signal=signal, outcome=outcome,
        settings=info['settings'], original_provenance=provenance.relative_to(root).as_posix(),
        original_provenance_sha256=base.sha256(provenance), tables=checks,
        inference_recomputed=False))
    return data, checks


def unavailable_label(row, phase, outcome):
    """Distinguish absent variation from small N; accept in-memory or CSV rows."""
    if str(phase.get('group_usable', True)).lower() == 'false':
        return 'excluded'
    reason = row.get('unavailable_reason', '')
    labels = {
        'no_outcome_variation': 'no score\nvariation' if outcome == 'Score' else 'no outcome\nvariation',
        'no_predictor_variation': 'no predictor\nvariation',
        'insufficient_users': 'low N',
        'insufficient_within_variation': 'low within\nvariation',
        'insufficient_users_with_within_variation': 'low within\nvariation',
        'within_variation': 'low within\nvariation',
        'insufficient_residual_variation': 'no residual\nvariation',
        'insufficient_residual_degrees_of_freedom': 'low residual df',
    }
    if reason in labels:
        return labels[reason]
    status = row.get('status', '')
    if status.startswith('excluded_group_'):
        return 'excluded'
    return labels.get(status, 'N / variation')


def make_overview(data, outcome, signal):
    scenario = 'primary' if signal == 'position' else 'velocity_secondary'
    phases = sorted([p for p in data['phase_validity'] if p['mode'] in ('whole', 'PSM in stage')],
                    key=base.group_order)
    lookup = {(r['method'], base.key(r), r['predictor']): r
              for r in data['associations'] if r['scenario'] == scenario}
    fig, axes = base.plt.subplots(1, 3, figsize=(15, max(8, .44*len(phases)+3)), sharey=True)
    titles = ('Between participants', 'Within participant', 'Within participant\nadjusted for task order + attempt')
    for ax, method, title in zip(axes, base.METHODS, titles):
        matrix = np.full((len(phases), 3), np.nan)
        for i, phase in enumerate(phases):
            for j, predictor in enumerate(PREDICTORS):
                row = lookup[(method, base.key(phase), predictor)]
                # Old tables may lack the explicit display diagnostic. Recover
                # the verifiable no-outcome-variation case from saved summaries.
                if not row.get('unavailable_reason') and row.get('status') != 'ok':
                    if method == 'between_participants':
                        sample = [p for p in data['participant_summaries']
                                  if p['scenario'] == scenario and base.key(p) == base.key(phase)]
                    else:
                        sample = [r for r in data.get('record_phase_audit', [])
                                  if str(r.get('eligible', False)).lower() == 'true'
                                  and r.get('signal') == signal
                                  and r.get('mode') in ('whole', 'PSM in stage')
                                  and base.key(r) == base.key(phase)]
                    outcomes = np.asarray([base.number(p.get(outcome)) for p in sample])
                    if len(outcomes) and np.all(np.isfinite(outcomes)) and np.ptp(outcomes) == 0:
                        row = {**row, 'unavailable_reason': 'no_outcome_variation'}
                value = base.number(row.get('association'))
                matrix[i, j] = value
                label = (f'{value:+.2f}'+(' *' if base.number(row.get('q_bh')) < .05 else '')
                         if np.isfinite(value) else unavailable_label(row, phase, outcome))
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color='white' if np.isfinite(value) and abs(value) > .65 else 'black')
        cmap = base.plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#eeeeee')
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect='auto')
        ax.set_xticks(range(3), ('Peak r', 'Signed lag', 'Absolute lag'))
        ax.set_yticks(range(len(phases)), [base.phase_label(p).replace('(MAX)', '(selected PSM)') for p in phases])
        ax.set_title(title, fontsize=11)
    fig.suptitle(f'{outcome} vs gaze/PSM features / {signal}\n'
                 '* BH q < 0.05 in the original outcome/scenario/method family', fontsize=13)
    fig.subplots_adjust(left=.25, right=.89, bottom=.13, top=.90, wspace=.08)
    fig.colorbar(im, cax=fig.add_axes([.92, .24, .015, .53]), label='Rank association')
    fig.text(.5, .025, 'Same layout and color scale in score_corr and score_corr_max. Original inference is preserved.\n'
             'Whole-task rows appear only where the source analysis contains them. Missing support is shown explicitly.\n'
             'Scores describe complete attempts. Positive signed lag means gaze leads PSM.', ha='center', fontsize=9)
    return fig


def ensure_plot_set(root, signal, outcome, predictor):
    """Reuse existing plotted artifacts, or generate a missing set from saved inference."""
    label = 'score' if outcome == 'Score' else outcome
    legacy = root/f'{PREFIXES[predictor]}_vs_{label}'/signal
    dest = root/'plots'/signal/outcome/predictor
    summary, points, hashes = plots.load_and_check(root, signal, outcome, predictor)
    existing = all((legacy/name).is_file() for name in ('associations.csv', 'plotted_points.csv', 'axes.json'))
    if existing:
        stored = plots.read_csv(legacy/'associations.csv')
        expected = {plots.group(r): r for r in summary}
        if {plots.group(r) for r in stored} != set(expected):
            raise ValueError(f'Stored figure conditions disagree with analysis: {legacy}')
        for row in stored:
            reference = expected[plots.group(row)]
            for key in ('spearman_rho', 'spearman_permutation_p', 'spearman_q_bh_existing_family'):
                np.testing.assert_allclose(float(row[key]), reference[key], atol=1e-10, equal_nan=True)
        if dest.is_symlink():
            if dest.resolve() != legacy.resolve():
                raise ValueError(f'Conflicting comparison link: {dest}')
        elif dest.exists():
            # A compatibility entry can point back to the real canonical folder.
            if dest.resolve() != legacy.resolve():
                raise ValueError(f'Cannot replace comparison directory with link: {dest}')
        else:
            dest.symlink_to(os.path.relpath(legacy, dest.parent), target_is_directory=True)
        action = 'linked_existing'
    else:
        if dest.is_symlink():
            raise ValueError(f'Missing legacy target for {dest}')
        dest.mkdir(parents=True, exist_ok=True)
        plots.write_csv(dest/'associations.csv', summary)
        plots.write_csv(dest/'plotted_points.csv', points)
        plots.make_figures(dest, signal, summary, points, outcome, predictor)
        save_json(dest/'validation.json', dict(original_statistics_unchanged=True,
            source_hashes=hashes, participant_summaries_and_scipy_checked=True,
            inference_recomputed=False))
        action = 'generated_from_saved_statistics'
    axes = normalize_modes(json.loads((dest/'axes.json').read_text()))
    if not axes['all_checks_passed']:
        raise ValueError(f'Invalid axes audit in {dest}')
    for panel in axes['panels']:
        if not panel['all_points_inside_limits']:
            raise ValueError(f'Plot clips observations in {dest}')
    return dict(path=dest.relative_to(root).as_posix(), action=action,
                n_conditions=len(summary), n_points=len(points), source_hashes=hashes)


def write_readme(root, rows):
    title = 'Selected PSM per recording/stage' if any(r['sides'] == 'max' for r in rows) else 'Separate left/right PSMs'
    lines = ['# Score correlation comparison layout', '', title+'.', '',
        'Use the same relative paths in `score_corr` and `score_corr_max` to compare results.', '',
        '```text', 'metrics/', '  all_associations.csv', '  all_participant_summaries.csv',
        '  position|velocity/', '    Score|Time|Eco|TimeRaw|EcoRaw|PenaltySum|PenaltyRawSum/',
        '      associations.csv', '      participant_summaries.csv', '      matched_outcomes.csv',
        '      record_phase_audit.csv', '      phase_validity.csv', '      provenance.json',
        'plots/', '  association_overviews.pdf', '  position|velocity/',
        '    Score|Time|Eco|TimeRaw|EcoRaw|PenaltySum|PenaltyRawSum/', '      overview.png / overview.pdf',
        '      peak_r/', '      signed_lag_s/', '      absolute_lag_s/', '```', '',
        'Each predictor folder contains the corresponding Task A/C figures, association table, plotted participant points and axis audit. '
        'Whole-task figures are present only for analyses with whole-task data.', '',
        '| Signal | Outcome | Tables | Overview | Peak r | Signed lag | Absolute lag |',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        path = f"{r['signal']}/{r['outcome']}"
        lines.append(f"| {r['signal']} | {r['outcome']} | [CSV](metrics/{path}/) | "
                     f"[PDF](plots/{path}/overview.pdf) | [plots](plots/{path}/peak_r/) | "
                     f"[plots](plots/{path}/signed_lag_s/) | [plots](plots/{path}/absolute_lag_s/) |")
    lines += ['', '## Interpretation and compatibility', '',
        '- Original coefficients, confidence intervals, p/q values, score matching and exclusions are unchanged. No inference was rerun.',
        '- `score_corr` contains left/right PSMs and whole-task conditions; `score_corr_max` contains one selected PSM per stage. These differences are retained.',
        '- Original BH families remain intact. Filtering a table into a signal folder does not recalculate its q values.',
        '- Existing file paths, reports, source snapshots and historical audits remain available. Predictor directories reuse existing plots through relative symbolic links where possible.',
        '- Penalties also use the standard top-level `peak_correlation_vs_PenaltySum/`, `peak_correlation_vs_PenaltyRawSum/` and matching `lag_correlation_vs_*` / `absolute_lag_correlation_vs_*` entries, each with position and velocity folders.',
        '- `metrics` contains filtered copies for comparison. After regenerating numerical results or legacy plots, rerun the organizer to refresh these copies and fill missing plot sets.',
        '- [Original analysis report](report.md), [layout manifest](layout.json), [layout validation](layout_validation.json).', '',
        'Rebuild both layouts from the repository root:', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/organize_score_corr_outputs.py', '```', '']
    (root/'README.md').write_text('\n'.join(lines))


def organize(root, no_plots=False):
    before = original_hashes(root)
    if not before:
        raise ValueError(f'No analysis found in {root}')
    index, table_checks, plot_checks, all_results, all_people = [], {}, [], [], []
    if not no_plots:
        (root/'plots').mkdir(exist_ok=True)
    book = PdfPages(root/'plots'/'association_overviews.pdf') if not no_plots else None
    try:
        for signal in SIGNALS:
            for outcome in OUTCOMES:
                data, checks = normalize_tables(root, signal, outcome)
                table_checks[f'{signal}/{outcome}'] = checks
                all_results.extend(dict(r, outcome=outcome) for r in data['associations'])
                all_people.extend(dict(r, outcome=outcome) for r in data['participant_summaries'])
                index.append(dict(signal=signal, outcome=outcome,
                    sides=','.join(sorted({r['side'] for r in data['associations']})),
                    tables=f'metrics/{signal}/{outcome}', plots=f'plots/{signal}/{outcome}',
                    n_associations=len(data['associations']),
                    n_record_phase_rows=len(data['record_phase_audit'])))
                if no_plots:
                    continue
                dest = root/'plots'/signal/outcome
                dest.mkdir(parents=True, exist_ok=True)
                fig = make_overview(data, outcome, signal)
                fig.savefig(dest/'overview.png', dpi=180, bbox_inches='tight')
                fig.savefig(dest/'overview.pdf', bbox_inches='tight')
                book.savefig(fig, bbox_inches='tight')
                base.plt.close(fig)
                for predictor in PREDICTORS:
                    plot_checks.append(ensure_plot_set(root, signal, outcome, predictor))
                print(f'{root.name}: normalized {signal}/{outcome}', flush=True)
    finally:
        if book is not None:
            book.close()
    base.write_csv(root/'metrics'/'all_associations.csv', all_results)
    base.write_csv(root/'metrics'/'all_participant_summaries.csv', all_people)
    unique = [(r['outcome'], r['scenario'], *base.key(r), r['predictor'], r['method']) for r in all_results]
    if len(set(unique)) != len(unique):
        raise ValueError('Duplicate associations in combined comparison table')
    base.write_csv(root/'comparison_index.csv', index)
    write_readme(root, index)
    after = original_hashes(root)
    if before != after:
        raise RuntimeError('Original analysis files changed during layout normalization')
    save_json(root/'layout.json', dict(schema_version=1,
        created_at=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds'),
        table_path='metrics/{signal}/{outcome}/{table}.csv',
        plot_path='plots/{signal}/{outcome}/{predictor}/', signals=SIGNALS, outcomes=OUTCOMES,
        predictors=PREDICTORS, tables=TABLES, conditions=index, plots=plot_checks,
        original_sha256=before, normalized_tables=table_checks, inference_recomputed=False))
    save_json(root/'layout_validation.json', dict(original_files_unchanged=True,
        normalized_table_values_unchanged=True, unique_combined_associations=True,
        n_conditions=len(index), n_associations=len(all_results), n_plot_sets=len(plot_checks),
        plots_checked_against_saved_statistics=not no_plots, all_checks_passed=True))
    snapshot = root/'layout_code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(plots.__file__)):
        (snapshot/path.name).write_bytes(path.read_bytes())
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+',
                        default=[Path('outputs/score_corr'), Path('outputs/score_corr_max')])
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    layouts = [organize(root, args.no_plots) for root in args.roots]
    relative = [{(r['signal'], r['outcome'], r['tables'], r['plots']) for r in rows} for rows in layouts]
    if any(paths != relative[0] for paths in relative):
        raise RuntimeError('Comparison directory structures differ')
    print('Matching comparison layouts ready; original analysis files unchanged', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
