#!/usr/bin/env python3
"""Expose penalty outputs using the same entry points as Time/TimeRaw/Score.

Compatibility paths are relative links to the calculated tables and figures.
No associations, confidence intervals, p/q values or plot data are recalculated.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_score_corr as base
from tools import plot_peak_correlation_vs_score as plots

OUTCOMES = ('PenaltySum', 'PenaltyRawSum')
SETTINGS = ('bootstrap', 'permutations', 'seed', 'min_users')
TABLES = ('associations', 'participant_summaries', 'matched_outcomes', 'record_phase_audit', 'phase_validity')
PREFIXES = dict(peak_r='peak_correlation', signed_lag_s='lag_correlation',
                absolute_lag_s='absolute_lag_correlation')


def link(source, alias):
    if not source.exists():
        raise FileNotFoundError(source)
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink() or alias.exists():
        if alias.resolve() != source.resolve():
            raise ValueError(f'Existing compatibility path points elsewhere: {alias}')
    else:
        alias.symlink_to(os.path.relpath(source, alias.parent), target_is_directory=source.is_dir())


def configure_tables(root):
    reference = json.loads((root/'provenance.json').read_text())['settings']
    for signal in ('position', 'velocity'):
        source = root/'penalty_metrics'/signal
        settings = json.loads((source/'provenance.json').read_text())['settings']
        if any(settings.get(k) != reference.get(k) for k in SETTINGS):
            raise ValueError(f'Penalty inference settings differ from Score: {root}/{signal}')
        for outcome in OUTCOMES:
            parent = root/('velocity_metrics' if signal == 'velocity' else
                           'component_metrics' if outcome == 'PenaltySum' else 'raw_metrics')
            for table in TABLES:
                link(source/f'{table}_{outcome}.csv', parent/f'{table}_{outcome}.csv')
            link(source/'provenance.json', parent/f'provenance_{outcome}.json')


def configure_plots(root):
    for signal in ('position', 'velocity'):
        for outcome in OUTCOMES:
            for predictor, prefix in PREFIXES.items():
                link(root/'plots'/signal/outcome/predictor, root/f'{prefix}_vs_{outcome}'/signal)


def validate(root):
    reference = json.loads((root/'provenance.json').read_text())['settings']
    checks = []
    for signal in ('position', 'velocity'):
        for outcome in OUTCOMES:
            for predictor, prefix in PREFIXES.items():
                folder = root/f'{prefix}_vs_{outcome}'/signal
                summary, points, hashes = plots.load_and_check(root, signal, outcome, predictor)
                stored = plots.read_csv(folder/'plotted_points.csv')
                lookup = {(*plots.group(p), p['user']): p for p in stored}
                if len(lookup) != len(points):
                    raise ValueError(f'Plotted participant count mismatch in {folder}')
                x = outcome+'_x'
                y = {'peak_r':'peak_correlation_y', 'signed_lag_s':'median_peak_lag_ms_y',
                     'absolute_lag_s':'median_absolute_peak_lag_ms_y'}[predictor]
                for p in points:
                    original = lookup[(*plots.group(p), p['user'])]
                    np.testing.assert_allclose([float(original[x]), float(original[y])], [p[x],p[y]], atol=1e-10)
                    if set(original['content_ids'].split(';')) != set(p['content_ids'].split(';')):
                        raise ValueError('Plot recording membership differs from eligible recordings')
                for name in ('associations.csv', 'axes.json'):
                    if not (folder/name).is_file():
                        raise FileNotFoundError(folder/name)
                panels = json.loads((folder/'axes.json').read_text())['panels']
                for panel in panels:
                    for ext in ('png', 'pdf'):
                        if not (folder/f"{panel['figure']}.{ext}").is_file():
                            raise FileNotFoundError(folder/f"{panel['figure']}.{ext}")
                # Match the standard peak/lag exporters, including confidence
                # interval screening, provenance and lag source snapshots.
                figures = list(dict.fromkeys(p['figure'] for p in panels))
                confidence = plots.ci_display_summary(summary)
                validation = dict(signal=signal, outcome=outcome, predictor=predictor,
                    conditions=len(summary), plotted_points=len(points), figures=figures,
                    participant_inputs_match_eligible_recordings=True,
                    pearson_and_spearman_match_saved_results_and_scipy=True,
                    spearman_ci_display=confidence, source_files_unchanged=True,
                    input_sha256=hashes, script_sha256=plots.digest(Path(plots.__file__)),
                    all_checks_passed=True)
                if predictor != 'peak_r':
                    validation.update(y_units='milliseconds',
                        participant_medians_and_mean_outcomes_match_eligible_recordings=True,
                        participant_summary=('median absolute recording peak lag; mean matched outcome'
                            if predictor == 'absolute_lag_s' else 'median signed recording peak lag; mean matched outcome'))
                    if predictor == 'signed_lag_s':
                        validation['positive_lag'] = 'gaze leads PSM'
                    scripts = [Path(plots.__file__), Path(__file__).with_name('plot_lag_vs_score.py')]
                    snapshot = folder/'code_snapshot'
                    snapshot.mkdir(exist_ok=True)
                    for script in scripts:
                        (snapshot/script.name).write_bytes(script.read_bytes())
                    provenance = dict(outcome=outcome, signal=signal, input_root=str(root.resolve()),
                        input_sha256=hashes, source_sha256={str(p):plots.digest(p) for p in scripts},
                        source_files_unchanged=True, all_checks_passed=True, spearman_ci_display=confidence,
                        booklet=('absolute_peak_lag' if predictor == 'absolute_lag_s' else 'peak_lag')+f'_vs_{outcome}.pdf')
                    (folder/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
                (folder/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
                checks.append(dict(signal=signal, outcome=outcome, predictor=predictor,
                    path=folder.relative_to(root).as_posix(), n_conditions=len(summary), n_points=len(points),
                    input_sha256=hashes, plotted_values_match_statistics=True))
    info = dict(settings={k: reference[k] for k in SETTINGS},
                uses_existing_score_analysis_methods=True, configuration_matches_score=True,
                inference_recomputed=False, plot_sets=checks, all_checks_passed=True)
    (root/'plots'/'penalty_configuration_validation.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def refresh_layout(root, info):
    """Update path/provenance manifests after adding compatibility entries."""
    from tools.organize_score_corr_outputs import original_hashes, normalize_tables, write_readme
    path = root/'layout.json'
    if not path.exists():
        return
    layout = json.loads(path.read_text())
    current = original_hashes(root)
    if any(current.get(name) != sha for name,sha in layout['original_sha256'].items()):
        raise ValueError('Original analysis changed while adding penalty compatibility paths')
    for signal in ('position','velocity'):
        for outcome in OUTCOMES:
            _, checks = normalize_tables(root,signal,outcome)
            layout['normalized_tables'][f'{signal}/{outcome}'] = checks
    layout['original_sha256'] = current
    for check in info['plot_sets']:
        canonical = f"plots/{check['signal']}/{check['outcome']}/{check['predictor']}"
        for entry in layout['plots']:
            if entry['path'] == canonical:
                entry['source_hashes'] = check['input_sha256']
    path.write_text(json.dumps(layout,indent=2)+'\n')
    validation_path = root/'layout_validation.json'
    validation = json.loads(validation_path.read_text())
    validation.update(penalty_configuration_matches_score=True, standard_penalty_paths_present=True)
    validation_path.write_text(json.dumps(validation,indent=2)+'\n')
    write_readme(root,layout['conditions'])
    snapshot = root/'layout_code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for script in (Path(__file__), Path(plots.__file__), Path(__file__).with_name('organize_score_corr_outputs.py')):
        (snapshot/script.name).write_bytes(script.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+',
                        default=[Path('outputs/score_corr'), Path('outputs/score_corr_max')])
    args = parser.parse_args(argv)
    for root in args.roots:
        configure_tables(root)
        configure_plots(root)
        info = validate(root)
        refresh_layout(root,info)
        print(f'{root}: {len(info["plot_sets"])} standard penalty plot folders checked',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
