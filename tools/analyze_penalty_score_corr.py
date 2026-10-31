#!/usr/bin/env python3
"""Add PenaltySum/PenaltyRawSum associations to both score correlation analyses.

Uses the existing recording peaks, eligibility, participant statistics and
inference. Refreshes the matching metrics/ and plots/ layouts for both roots.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_score_corr as base
from tools.analyze_score_corr_max import (attach_selection, selection_lookup,
                                          save_json, validate_scenarios)
from tools.organize_score_corr_outputs import organize, original_hashes
from tools.configure_penalty_outputs import configure_tables, configure_plots, validate, refresh_layout

PENALTIES = ('PenaltySum', 'PenaltyRawSum')


def calculate(args):
    """Write only new penalty-specific artifacts; preserve earlier outcomes."""
    inputs = [args.input/name for name in ('config.json', 'record_summary.csv',
              'record_curves.csv', 'group_summary.csv', 'quality.csv')]
    inputs += [args.index/name for name in ('content_index.csv', 'Task_A_Score.csv', 'Task_C_Score.csv')]
    if (args.input/'side_selection.csv').exists():
        inputs.append(args.input/'side_selection.csv')
        selection = selection_lookup(args.input)
    else:
        selection = None
    before = {str(p): base.sha256(p) for p in inputs}
    output = args.output/'penalty_metrics'
    output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = dict(settings=settings, input_sha256=before,
        created_at=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds'),
        values_used='Supplied PenaltySum and PenaltyRawSum, without sign reversal or imputation',
        multiple_comparisons='BH separately by outcome/scenario/method across all stages and three predictors')
    checks, coverage, all_results, all_people = {}, {}, [], []
    for outcome in PENALTIES:
        config, inventory, matched, records, phases = base.prepare(args, outcome)
        if selection is not None:
            attach_selection(records, selection)
        coverage[outcome] = dict(total=len(matched), usable=sum(r['score_usable'] for r in matched),
            by_task={task: sum(r['score_usable'] and r['task']==task for r in matched) for task in ('A','C')})
        print(f'{args.output.name}: {outcome}: {coverage[outcome]["usable"]}/{len(matched)} usable', flush=True)
        results, people = base.analyze(args, records, phases, outcome,
                                      scenarios=('primary', 'velocity_secondary'))
        checks[outcome] = validate_scenarios(results, people, records, outcome)
        all_results.extend(results)
        all_people.extend(people)
        for signal in ('position', 'velocity'):
            dest = output/signal
            dest.mkdir(exist_ok=True)
            for name, rows in [('associations',results),('participant_summaries',people),
                              ('matched_outcomes',matched),('record_phase_audit',records),
                              ('phase_validity',phases)]:
                base.write_csv(dest/f'{name}_{outcome}.csv',
                               [r for r in rows if r.get('signal',signal)==signal])
            save_json(dest/'provenance.json',provenance)
        base.write_csv(output/f'score_inventory_{outcome}.csv',inventory)
    if not all(base.sha256(Path(p))==sha for p,sha in before.items()):
        raise RuntimeError('Correlation inputs or score files changed during penalty analysis')
    base.write_csv(output/'all_associations.csv',all_results)
    base.write_csv(output/'all_participant_summaries.csv',all_people)
    sources = [Path(__file__),Path(base.__file__),Path(__file__).with_name('analyze_raw_score_corr.py'),
               Path(__file__).with_name('analyze_score_corr_max.py')]
    snapshot = output/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        (snapshot/path.name).write_bytes(path.read_bytes())
    provenance.update(coverage=coverage, correlation_config=config,
                      analysis_source_sha256={str(p):base.sha256(p) for p in sources}, inputs_unchanged=True)
    save_json(output/'provenance.json',provenance)
    save_json(output/'validation.json',dict(independent_checks=checks,coverage=coverage,
        n_associations=len(all_results),inputs_unchanged=True,
        selected_psm_metadata_checked=selection is not None,all_checks_passed=True))
    lines = ['# PenaltySum and PenaltyRawSum associations', '',
        'Uses the supplied penalty columns directly. PenaltySum may be negative; no sign reversal is applied. '
        'Zero is a valid observed value; missing values are excluded and never replaced with zero. '
        'The outcome is measured for the entire attempt.', '',
        'Position is primary and velocity secondary. Peak r, signed peak lag and absolute peak lag are analyzed '
        'between participants, within participants and with task-order/attempt adjustment. '
        'Existing peak, group-support and participant-count eligibility rules are unchanged.', '',
        f'{args.bootstrap:,} participant bootstrap samples; {args.permutations:,} between-participant score permutations; '
        f'minimum {args.min_users} qualifying participants. Within-participant p values use the existing cluster CR1 approximation. '
        'BH correction pools all three predictors and stages separately per outcome, scenario and method; '
        'the two penalty outcomes and prior score outcomes are not jointly corrected.', '',
        '| Outcome | Available A / C | Successful tests | BH q < .05 |', '|---|---:|---:|---:|']
    for outcome in PENALTIES:
        rr = [r for r in all_results if r['outcome']==outcome]
        info=coverage[outcome]
        lines.append(f"| {outcome} | {info['by_task']['A']} / {info['by_task']['C']} | "
                     f"{sum(r['status']=='ok' for r in rr)} | {sum(r.get('q_bh',1)<.05 for r in rr)} |")
    lines += ['', 'Browse the matching `../metrics/<signal>/<outcome>/` and '
              '`../plots/<signal>/<outcome>/` folders, or use the [main navigation](../README.md). '
              'Existing five-outcome analyses remain unchanged.', '', 'Reproduce both roots:', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/analyze_penalty_score_corr.py', '```', '']
    (output/'README.md').write_text('\n'.join(lines))
    return coverage


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=Path('Index'))
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--permutations',type=int,default=19999)
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--min-users',type=int,default=6)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    if min(args.bootstrap,args.permutations)<100 or args.min_users<3:
        parser.error('Require bootstrap/permutations >=100 and min-users >=3')
    jobs=[(Path('outputs/gaze_psm_xcorr'),Path('outputs/score_corr')),
          (Path('outputs/gaze_psm_xcorr_max'),Path('outputs/score_corr_max'))]
    def penalty_artifact(name):
        return (name.startswith('penalty_metrics/') or
                any(Path(name).name.endswith(f'_{outcome}.csv') or
                    Path(name).name == f'provenance_{outcome}.json' for outcome in PENALTIES))
    previous={str(dest):{name:sha for name,sha in original_hashes(dest).items()
                        if not penalty_artifact(name)} for _,dest in jobs}
    for source,dest in jobs:
        calculate(argparse.Namespace(**vars(args),input=source,output=dest))
    for _,dest in jobs:
        current=original_hashes(dest)
        if any(current.get(name)!=sha for name,sha in previous[str(dest)].items()):
            raise RuntimeError(f'Existing analysis changed in {dest}')
        path=dest/'penalty_metrics'/'validation.json'
        validation=json.loads(path.read_text())
        validation['previous_outcomes_unchanged']=True
        save_json(path,validation)
        configure_tables(dest)
        organize(dest,no_plots=args.no_plots)
        if not args.no_plots:
            configure_plots(dest)
            refresh_layout(dest,validate(dest))
        current=original_hashes(dest)
        if any(current.get(name)!=sha for name,sha in previous[str(dest)].items()):
            raise RuntimeError(f'Existing analysis changed while refreshing {dest}')
    print('Both penalty outcomes saved to matching metrics/ and plots/ folders; earlier analyses unchanged',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
