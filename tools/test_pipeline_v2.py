"""Regression tests of fixed-side eligibility and deterministic sensitivity selection."""
import unittest
import numpy as np

from analyze_model_xcorr_dlc_v2 import SOURCES, pair_records, selected_side_sensitivity
from gaze_psm_reliability_v2 import peak_features


def fixture():
    curves, rows = [], []
    for side in ('l', 'r'):
        for j, source in enumerate(SOURCES):
            meta = dict(variant='primary', content_id='User1_test', user='User1',
                        task='A', side=side, signal='position', stage='stage',
                        mode='PSM in stage', source=source)
            # Every curve has complete support, but all peaks are negative and
            # at a search boundary. Those outcomes must not remove the input.
            curve = dict(meta, lag_s=np.array([-2., 0., 2.]),
                         r=np.array([-.9, -.5, -.2]) + j*.01)
            row = dict(meta, **peak_features(curve), lag_identifiable=False,
                       lag_status='boundary_limited')
            curves.append(curve); rows.append(row)
    return curves, rows


class PipelineTests(unittest.TestCase):
    def test_complete_negative_boundary_unidentifiable_curves_remain_in_main_set(self):
        curves, rows = fixture()
        pairs = pair_records(curves, rows)
        self.assertEqual(len(pairs), 10)
        self.assertTrue(all(p['input_eligible'] and p['common_input'] for p in pairs))
        self.assertTrue(all(p['same_side'] for p in pairs))
        self.assertTrue(all(p['model_peak_r'] < 0 and p['model_boundary'] for p in pairs))
        self.assertTrue(all(not p['gt_localizable'] and not p['model_localizable'] for p in pairs))
        self.assertTrue(all(p['curve_mae'] is not None for p in pairs))

    def test_peak_status_changes_do_not_change_input_eligibility(self):
        curves, rows = fixture()
        before = pair_records(curves, rows)
        for row in rows:
            row.update(lag_identifiable=True, lag_status='interior_stable_exploratory', peak_at_boundary=False)
        after = pair_records(curves, rows)
        self.assertEqual([r['common_input'] for r in before], [r['common_input'] for r in after])
        self.assertEqual([r['curve_mae'] for r in before], [r['curve_mae'] for r in after])

    def test_only_support_missingness_changes_common_input_and_is_side_specific(self):
        curves, rows = fixture()
        for curve in curves:
            if curve['source'] == SOURCES[-1] and curve['side'] == 'l':
                curve['r'][0] = np.nan
        pairs = pair_records(curves, rows)
        left = [p for p in pairs if p['side'] == 'l']
        right = [p for p in pairs if p['side'] == 'r']
        self.assertTrue(all(not p['common_input'] for p in left))
        self.assertTrue(all(p['common_input'] for p in right))
        bad = next(p for p in left if p['model'] == SOURCES[-1])
        self.assertFalse(bad['input_eligible'])
        self.assertIsNone(bad['curve_mae'])
        self.assertIsNotNone(bad['curve_mae_partial'])
        self.assertEqual(bad['n_common_lags'], 2)

    def test_fixed_side_comparison_does_not_follow_model_winning_side(self):
        curves, rows = fixture()
        for curve in curves:
            if curve['side'] == 'r' and curve['source'] != 'gt':
                curve['r'][:] = [.1,.3,.8]
        # Refresh features after deliberately moving all model winners right.
        features=[]
        for curve in curves:
            meta={k:v for k,v in curve.items() if k not in ('r','lag_s')}
            features.append(dict(meta,**peak_features(curve),lag_identifiable=False,lag_status='boundary_limited'))
        paired = pair_records(curves, features)
        self.assertEqual({p['side'] for p in paired}, {'l','r'})
        self.assertEqual(len(paired),10)
        sensitivity=selected_side_sensitivity(curves,features)
        fixed=[r for r in sensitivity if r['selection_rule']=='gt_selected_side']
        independent=[r for r in sensitivity if r['selection_rule']=='independent_max_side_legacy']
        self.assertTrue(all(r['gt_side']=='l' and r['model_side']=='l' for r in fixed))
        self.assertTrue(all(r['gt_side']=='l' and r['model_side']=='r' for r in independent))

    def test_tie_rule_is_left_and_independent_of_row_order(self):
        curves, rows = fixture()
        a=selected_side_sensitivity(curves,rows)
        b=selected_side_sensitivity(list(reversed(curves)),list(reversed(rows)))
        project=lambda records:sorted((r['model'],r['selection_rule'],r['gt_side'],r['model_side']) for r in records)
        self.assertEqual(project(a),project(b))
        self.assertTrue(all(r['gt_side']=='l' and r['model_side']=='l' for r in a))

    def test_single_valid_side_selected(self):
        curves, rows = fixture()
        for curve in curves:
            if curve['side']=='l':
                curve['r'][:]=np.nan
        result=selected_side_sensitivity(curves,rows)
        self.assertTrue(all(r['gt_side']=='r' and r['model_side']=='r' for r in result))


if __name__ == '__main__':
    unittest.main()
