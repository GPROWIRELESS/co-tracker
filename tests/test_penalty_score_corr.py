import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tools import analyze_score_corr as base
from tools.analyze_score_corr_max import validate_scenarios
from tools.plot_peak_correlation_vs_score import analysis_table_paths
from tests.test_score_corr_max import fixture
from tools.configure_penalty_outputs import configure_tables, configure_plots, SETTINGS, TABLES, PREFIXES


class PenaltyAssociationTests(unittest.TestCase):
    def test_standard_table_and_plot_entries_match_other_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            settings=dict(bootstrap=10000,permutations=19999,seed=20261030,min_users=6)
            (root/'provenance.json').write_text(json.dumps(dict(settings=settings)))
            for signal in ('position','velocity'):
                source=root/'penalty_metrics'/signal
                source.mkdir(parents=True)
                (source/'provenance.json').write_text(json.dumps(dict(settings=settings)))
                for outcome in ('PenaltySum','PenaltyRawSum'):
                    for table in TABLES:
                        (source/f'{table}_{outcome}.csv').write_text('value\n0\n')
                    for predictor in PREFIXES:
                        folder=root/'plots'/signal/outcome/predictor
                        folder.mkdir(parents=True)
                        (folder/'task_A_phases.png').write_bytes(b'image')
            for _ in range(2):
                configure_tables(root)
                configure_plots(root)
            for signal in ('position','velocity'):
                for outcome in ('PenaltySum','PenaltyRawSum'):
                    names,provenance=analysis_table_paths(root,signal,outcome)
                    expected='velocity_metrics' if signal=='velocity' else 'component_metrics' if outcome=='PenaltySum' else 'raw_metrics'
                    self.assertTrue(all(p.startswith(expected+'/') for p in names))
                    self.assertTrue(all((root/p).is_file() for p in names))
                    self.assertEqual(json.loads((root/provenance).read_text())['settings'],settings)
                    for predictor,prefix in PREFIXES.items():
                        alias=root/f'{prefix}_vs_{outcome}'/signal
                        self.assertTrue(alias.is_dir())
                        self.assertEqual(alias.resolve(),(root/'plots'/signal/outcome/predictor).resolve())
                        self.assertTrue((alias/'task_A_phases.png').is_file())

    def test_mismatched_penalty_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=10000))))
            source=root/'penalty_metrics/position'
            source.mkdir(parents=True)
            (source/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=100))))
            with self.assertRaisesRegex(ValueError,'settings differ'):
                configure_tables(root)

    def test_penalties_have_separate_tables_for_both_signals(self):
        root=Path('/tmp/penalty-routing')
        for signal in ('position','velocity'):
            for outcome in ('PenaltySum','PenaltyRawSum'):
                names,provenance=analysis_table_paths(root,signal,outcome)
                self.assertTrue(all(p.startswith(f'penalty_metrics/{signal}/') for p in names))
                self.assertEqual(provenance,f'penalty_metrics/{signal}/provenance.json')

    def test_negative_sum_raw_sum_and_zero_are_used_directly(self):
        records,phase=fixture()
        for row in records:
            row['PenaltyRawSum']=row['Score']
            row['PenaltySum']=-row['Score']
            row['Score']=0
        args=argparse.Namespace(seed=42,bootstrap=100,permutations=199,min_users=6)
        estimates={}
        for outcome in ('PenaltySum','PenaltyRawSum'):
            results,people=base.analyze(args,records,[phase],outcome,scenarios=('primary',))
            self.assertTrue(validate_scenarios(results,people,records,outcome)['primary']['all_checks_passed'])
            estimates[outcome]=results
            for person in people:
                rr=[r for r in records if r['user']==person['user']]
                self.assertAlmostEqual(person[outcome],np.mean([r[outcome] for r in rr]))
                self.assertNotIn('Score',person)
        for negative,raw in zip(estimates['PenaltySum'],estimates['PenaltyRawSum']):
            self.assertAlmostEqual(negative['association'],-raw['association'],places=10)
        records[0].update(PenaltySum=0,PenaltyRawSum=0,AttemptStatus='Failed')
        self.assertIn(records[0],base.scenario_rows(records,'primary'))

    def test_missing_penalty_excluded_even_with_valid_total_score(self):
        records,phase=fixture()
        for row in records:
            row['PenaltySum']=-row['Score']
        missing=records[0]
        missing.update(PenaltySum=np.nan,eligible=False)
        people=base.aggregate_participants(base.scenario_rows(records,'primary'),'PenaltySum')
        self.assertTrue(all(missing['content_id'] not in p['content_ids'].split(';') for p in people))
        self.assertTrue(all(np.isfinite(p['PenaltySum']) for p in people))


if __name__=='__main__':
    unittest.main()
