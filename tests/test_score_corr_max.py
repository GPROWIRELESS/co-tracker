import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_score_corr import synthetic_records
from tools import analyze_score_corr as base
from tools.analyze_score_corr_max import (attach_selection, SELECTION_KEY,
                                           validate_scenarios, overview)
from tools import plot_peak_correlation_vs_score as plots


def fixture():
    records = synthetic_records()
    phase = dict(task='A', side='max', signal='position', stage='grab ring',
                 mode='psm_stage', group_usable=True, group_status='ok')
    for i, row in enumerate(records):
        row.update({k: phase[k] for k in base.GROUP})
        row['peak_r'] = .3+.1*np.tanh(row['signed_lag_s'])
        row['EcoRaw'] = -row['Score']
        row['selected_side'] = 'l' if i % 2 else 'r'
    return records, phase


class ScoreCorrMaxTests(unittest.TestCase):
    def test_selection_join_keeps_mixed_sides_and_rejects_wrong_peak(self):
        records, _ = fixture()
        lookup = {}
        for row in records:
            side = row['selected_side']
            lookup[tuple(row[k] for k in SELECTION_KEY)] = dict(selected_side=side,
                selection_status='larger_peak', left_peak_r=row['peak_r'] if side=='l' else .1,
                right_peak_r=row['peak_r'] if side=='r' else .1,
                left_peak_lag_s=0, right_peak_lag_s=0)
        attach_selection(records, lookup)
        self.assertEqual({r['selected_side'] for r in records}, {'l', 'r'})
        self.assertEqual({r['side'] for r in records}, {'max'})
        records[0]['peak_r'] = .999
        with self.assertRaises(ValueError):
            attach_selection(records, lookup)

    def test_missing_selection_and_both_invalid(self):
        row = dict(content_id='User1_1_1', user='User1', task='A', side='max',
                   signal='position', stage='drop ring', mode='psm_stage', peak_r=np.nan)
        detail = dict(selected_side='', selection_status='both_invalid', left_peak_r='nan',
                      right_peak_r='nan', left_peak_lag_s='nan', right_peak_lag_s='nan')
        attach_selection([row], {tuple(row[k] for k in SELECTION_KEY): detail})
        self.assertEqual(row['selected_side'], '')
        with self.assertRaises(ValueError):
            attach_selection([row], {})

    def test_sensitivity_validation_uses_exact_filtered_attempts(self):
        records, phase = fixture()
        records[0].update(AttemptStatus='Failed', Score=0)
        args = argparse.Namespace(seed=42, bootstrap=100, permutations=199, min_users=6)
        results, people = base.analyze(args, records, [phase], scenarios=(
            'primary', 'completed_only', 'stronger_peaks'))
        checks = validate_scenarios(results, people, records, 'Score')
        self.assertTrue(all(v['all_checks_passed'] for v in checks.values()))
        self.assertIn(records[0], base.scenario_rows(records, 'primary'))
        self.assertNotIn(records[0], base.scenario_rows(records, 'completed_only'))

    def test_named_outcome_missing_records_are_excluded_without_score_fallback(self):
        records, phase = fixture()
        missing = records[0]
        missing.update(EcoRaw=np.nan, eligible=False)
        args = argparse.Namespace(seed=42, bootstrap=100, permutations=199, min_users=6)
        results, people = base.analyze(args, records, [phase], 'EcoRaw', scenarios=('primary',))
        self.assertTrue(validate_scenarios(results, people, records, 'EcoRaw')['primary']['all_checks_passed'])
        for person in people:
            ids = person['content_ids'].split(';')
            self.assertNotIn(missing['content_id'], ids)
            rr = [r for r in records if r['content_id'] in ids]
            self.assertAlmostEqual(person['EcoRaw'], np.mean([r['EcoRaw'] for r in rr]))

    def test_absolute_lag_export_and_selected_psm_single_column(self):
        records, phase = fixture()
        args = argparse.Namespace(seed=42, bootstrap=100, permutations=199, min_users=6)
        results, people = base.analyze(args, records, [phase], scenarios=('primary',))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, rows in [('associations',results),('participant_summaries',people),
                               ('record_phase_audit',records)]:
                base.write_csv(root/f'{name}.csv',rows)
            (root/'provenance.json').write_text(json.dumps(dict(settings=dict(bootstrap=100))))
            summary,points,_ = plots.load_and_check(root,'position','Score','absolute_lag_s')
            for point in points:
                person = next(p for p in people if p['user']==point['user'])
                self.assertAlmostEqual(point['median_absolute_peak_lag_ms_y'],person['absolute_lag_s']*1000)
            plots.make_figures(root,'position',summary,points,'Score','absolute_lag_s')
            axes=json.loads((root/'axes.json').read_text())
            self.assertEqual(len(axes['panels']),1)
            self.assertEqual(axes['panels'][0]['side'],'max')
            self.assertEqual(axes['panels'][0]['n_points'],len(people))
            self.assertFalse((root/'whole_task.pdf').exists())
            self.assertTrue((root/'absolute_peak_lag_vs_score.pdf').exists())

    def test_overview_keeps_unsupported_stages_visible(self):
        records,phase=fixture()
        missing=dict(phase,stage='drop ring',group_status='no_valid_curve',group_usable=False)
        args=argparse.Namespace(seed=42,bootstrap=100,permutations=199,min_users=6)
        results,_=base.analyze(args,records,[phase,missing],scenarios=('primary',))
        fig=overview(results,[phase,missing],'Score','position','primary')
        try:
            for ax in fig.axes[:3]:
                self.assertIn('excluded',[t.get_text() for t in ax.texts])
                self.assertTrue(all('selected PSM' in t.get_text() for t in ax.get_yticklabels()))
        finally:
            base.plt.close(fig)


if __name__ == '__main__':
    unittest.main()
