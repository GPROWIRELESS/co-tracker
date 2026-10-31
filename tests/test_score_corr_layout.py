import argparse
import json
from pathlib import Path
import tempfile
import unittest

from tools import analyze_score_corr as base
from tools.organize_score_corr_outputs import (
    OUTCOMES, TABLES, normalize_tables, organize, original_hashes, ensure_plot_set,
)
from tests.test_score_corr_max import fixture


def fake_rows(signal, outcome):
    row = dict(task='A', side='max', signal=signal, stage='grab ring', mode='psm_stage',
               scenario='primary' if signal == 'position' else 'velocity_secondary',
               predictor='peak_r', method='between_participants', association='.123400',
               p='.00200', q_bh='.00400', ci_low='-.1', ci_high='.4', outcome=outcome)
    return {table: [dict(row)] for table in TABLES}


def fake_tree(root):
    position = fake_rows('position', 'Score')
    velocity = fake_rows('velocity', 'Score')
    for name in TABLES:
        target = 'matched_scores' if name == 'matched_outcomes' else name
        base.write_csv(root/f'{target}.csv',position[name]+velocity[name])
    (root/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=100))))
    for signal in ('position','velocity'):
        for outcome in OUTCOMES:
            if signal == 'position' and outcome == 'Score':
                continue
            folder=root/(f'penalty_metrics/{signal}' if outcome in ('PenaltySum','PenaltyRawSum') else
                         'velocity_metrics' if signal=='velocity' else
                         'component_metrics' if outcome in ('Time','Eco') else 'raw_metrics')
            folder.mkdir(parents=True,exist_ok=True)
            for name,rows in fake_rows(signal,outcome).items():
                base.write_csv(folder/f'{name}_{outcome}.csv',rows)
            (folder/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=100))))


class ScoreCorrLayoutTests(unittest.TestCase):
    def test_normalized_tables_filter_signal_without_changing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fake_tree(root)
            data,_=normalize_tables(root,'position','Score')
            self.assertEqual(data['associations'],fake_rows('position','Score')['associations'])
            self.assertEqual(data['associations'][0]['association'],'.123400')
            self.assertEqual(data['associations'][0]['q_bh'],'.00400')
            for table in TABLES:
                self.assertEqual(base.read_csv(root/'metrics/position/Score'/f'{table}.csv'),data[table])

    def test_matching_conditions_and_repeatable_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fake_tree(root)
            before=original_hashes(root)
            first=organize(root,no_plots=True)
            self.assertEqual(original_hashes(root),before)
            combined=(root/'metrics/all_associations.csv').read_bytes()
            second=organize(root,no_plots=True)
            self.assertEqual(first,second)
            self.assertEqual(len(first),2*len(OUTCOMES))
            self.assertEqual((root/'metrics/all_associations.csv').read_bytes(),combined)
            self.assertEqual(len(base.read_csv(root/'metrics/all_associations.csv')),2*len(OUTCOMES))
            self.assertEqual(original_hashes(root),before)
            self.assertTrue(json.loads((root/'layout_validation.json').read_text())['all_checks_passed'])

    def test_missing_plot_generated_from_saved_inference_and_reusable(self):
        records,phase=fixture()
        args=argparse.Namespace(seed=42,bootstrap=100,permutations=199,min_users=6)
        results,people=base.analyze(args,records,[phase],scenarios=('primary',))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,rows in [('associations',results),('participant_summaries',people),
                              ('record_phase_audit',records)]:
                base.write_csv(root/f'{name}.csv',rows)
            (root/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=100))))
            before=original_hashes(root)
            result=ensure_plot_set(root,'position','Score','absolute_lag_s')
            self.assertEqual(result['action'],'generated_from_saved_statistics')
            self.assertEqual(original_hashes(root),before)
            self.assertTrue((root/'plots/position/Score/absolute_lag_s/task_A_phases.pdf').exists())
            again=ensure_plot_set(root,'position','Score','absolute_lag_s')
            self.assertEqual(again['action'],'generated_from_saved_statistics')
            self.assertEqual(original_hashes(root),before)
            legacy=root/'absolute_lag_correlation_vs_score/position'
            legacy.parent.mkdir()
            legacy.symlink_to(root/'plots/position/Score/absolute_lag_s',target_is_directory=True)
            alias_check=ensure_plot_set(root,'position','Score','absolute_lag_s')
            self.assertEqual(alias_check['action'],'linked_existing')

    def test_existing_plots_use_relative_link_and_verify_statistics(self):
        records,phase=fixture()
        args=argparse.Namespace(seed=42,bootstrap=100,permutations=199,min_users=6)
        results,people=base.analyze(args,records,[phase],scenarios=('primary',))
        from tools import plot_peak_correlation_vs_score as plots
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,rows in [('associations',results),('participant_summaries',people),
                              ('record_phase_audit',records)]:
                base.write_csv(root/f'{name}.csv',rows)
            (root/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=100))))
            summary,points,_=plots.load_and_check(root,'position','Score','peak_r')
            legacy=root/'peak_correlation_vs_score/position'
            legacy.mkdir(parents=True)
            plots.write_csv(legacy/'associations.csv',summary)
            plots.write_csv(legacy/'plotted_points.csv',points)
            (legacy/'axes.json').write_text(json.dumps(dict(all_checks_passed=True,panels=[])))
            (root/'plots/position/Score').mkdir(parents=True)
            before=original_hashes(root)
            result=ensure_plot_set(root,'position','Score','peak_r')
            dest=root/result['path']
            self.assertEqual(result['action'],'linked_existing')
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(),legacy.resolve())
            self.assertEqual(original_hashes(root),before)
            summary[0]['spearman_rho']+=.1
            plots.write_csv(legacy/'associations.csv',summary)
            with self.assertRaises(AssertionError):
                ensure_plot_set(root,'position','Score','peak_r')


if __name__ == '__main__':
    unittest.main()
