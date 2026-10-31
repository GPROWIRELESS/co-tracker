import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.gaze_psm_group import aggregate, fisher_mean
from tools.gaze_psm_pipeline import save_json
from tools.gaze_psm_xcorr_max import (select_sides, _selected_null_job, _pairings,
                                     main, generate_figures)


def curve(side, r, user='User1', record=None, signal='position', mode='psm_stage'):
    return dict(content_id=record or user+'_1_1', user=user, task='A',
                side=side, signal=signal, stage='reach', mode=mode,
                lag_s=np.array([-1., 0., 1.]), r=np.array(r, dtype=float),
                pairs=np.full(3, 100), occurrences=np.ones(3),
                distance_median_px=np.full(3, 10 if side == 'l' else 90))


class SideSelectionTests(unittest.TestCase):
    def test_winner_cases_and_full_curve_preservation(self):
        cases = [([.8, .1, .1], [.1, .7, .1], 'l', 'larger_peak'),
                 ([.8, .1, .1], [.1, .9, .1], 'r', 'larger_peak'),
                 ([-.9, -.8, -.7], [-.6, -.5, -.4], 'r', 'larger_peak'),
                 ([.5, .1, .1], [.1, .5, .1], 'l', 'tie_left'),
                 ([np.nan]*3, [.1, .5, .1], 'r', 'single_valid_side'),
                 ([.1, .5, .1], [np.nan]*3, 'l', 'single_valid_side'),
                 ([np.nan]*3, [np.nan]*3, None, 'both_invalid')]
        for left, right, winner, status in cases:
            with self.subTest(winner=winner, status=status):
                originals = {s: curve(s, r) for s, r in [('l', left), ('r', right)]}
                rows, details = select_sides(list(originals.values()))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['side'], 'max')
                self.assertEqual(details[0]['selected_side'], winner)
                self.assertEqual(details[0]['selection_status'], status)
                template = originals[winner or 'l']
                for field in ('r', 'lag_s', 'pairs', 'distance_median_px'):
                    np.testing.assert_array_equal(rows[0][field], template[field])
                self.assertEqual(originals['l']['side'], 'l')

    def test_signal_and_mode_selected_independently_and_whole_excluded(self):
        curves = []
        for signal in ('position', 'velocity'):
            for mode in ('psm_stage', 'within_stage', 'whole'):
                for side in ('l', 'r'):
                    wins = (signal == 'position') == (side == 'l')
                    curves.append(curve(side, [.1, .8 if wins else .2, .1],
                                        signal=signal, mode=mode))
        rows, _ = select_sides(curves)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['selected_side'] for r in rows if r['signal'] == 'position'}, {'l'})
        self.assertEqual({r['selected_side'] for r in rows if r['signal'] == 'velocity'}, {'r'})

    def test_reject_missing_duplicate_and_mismatched_grids(self):
        a, b = curve('l', [.1]*3), curve('r', [.2]*3)
        for curves in ([a], [a, a, b]):
            with self.assertRaises(ValueError):
                select_sides(curves)
        b['lag_s'] = np.array([-2., 0., 2.])
        with self.assertRaises(ValueError):
            select_sides([a, b])

    def test_select_before_equal_record_then_equal_user_aggregation(self):
        curves = [curve('l', [.8, .1, .1], record='u1_a'),
                  curve('r', [.1, .7, .1], record='u1_a'),
                  curve('l', [.1, .2, .1], record='u1_b'),
                  curve('r', [.1, .9, .1], record='u1_b'),
                  curve('l', [.1, .6, .1], user='User2'),
                  curve('r', [.1, .3, .1], user='User2')]
        selected, _ = select_sides(curves)
        users, groups, _ = aggregate(selected, bootstrap=20, min_users=2)
        user1 = fisher_mean([[.8, .1, .1], [.1, .9, .1]])
        expected = fisher_mean([user1, [.1, .6, .1]])
        np.testing.assert_allclose(groups[0]['r'], expected)
        self.assertEqual(groups[0]['peak_lag_s'], 0.)
        self.assertEqual(len(users), 2)
        self.assertEqual(len(groups[0]['bootstrap_peak_lag_s']), 20)

    def test_unpaired_reselects_its_own_side(self):
        with patch('tools.gaze_psm_xcorr_max._null_job', return_value=[
                curve('l', [.1, .2, .1]), curve('r', [.7, .1, .1])]):
            rows = _selected_null_job([])
        self.assertEqual(rows[0]['selected_side'], 'r')
        np.testing.assert_allclose(rows[0]['r'], [.7, .1, .1])
        records = {c: {'Task Name': t} for c, t in
                   [('User1_1', 'A'), ('User1_2', 'A'), ('User2_1', 'A'), ('User3_1', 'C')]}
        pairs = list(_pairings(records, 3, 42))
        self.assertEqual(pairs, list(_pairings(records, 3, 42)))
        for rep in pairs:
            for target, donor in rep:
                self.assertNotEqual(target.split('_')[0], donor.split('_')[0])
                self.assertEqual(records[target]['Task Name'], records[donor]['Task Name'])

    def test_cli_exports_missing_stages_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            source.mkdir()
            save_json(source/'config.json', dict(seed=42, min_users=1, opencv_threads=1))
            save_json(source/'quality.json', [])
            save_json(source/'results.json', dict(record_curves=[
                curve('l', [np.nan]*3), curve('r', [np.nan]*3)]))
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            self.assertEqual(main(['--source', str(source), '--output', str(output),
                                  '--shuffles', '0', '--bootstrap', '10']), 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            results = json.loads((output/'results.json').read_text())
            self.assertEqual(results['group_curves'][0]['status'], 'no_valid_curve')
            self.assertTrue((output/'figures/stages_A_position.pdf').exists())
            self.assertTrue(json.loads((output/'validation.json').read_text())['source_unchanged'])

    def test_peak_legend_survives_unsupported_first_stage(self):
        curves = []
        for stage, values in [('a_missing', [np.nan]*3), ('b_valid', [.1, .7, .1])]:
            for mode in ('psm_stage', 'within_stage'):
                for side in ('l', 'r'):
                    row = curve(side, values, mode=mode)
                    row['stage'] = stage
                    curves.append(row)
        rows, _ = select_sides(curves)
        users, groups, _ = aggregate(rows, bootstrap=10, min_users=1)
        # Saved JSON represents unsupported scalar estimates as null.
        for group in groups:
            if not np.isfinite(group['peak_lag_s']):
                group['peak_lag_s'] = None
        captured = []

        def inspect(fig, path):
            if path.name.startswith('stage_peak_lags'):
                captured.extend(fig.axes[0].get_legend_handles_labels()[1])

        with tempfile.TemporaryDirectory() as tmp:
            with patch('tools.gaze_psm_figures._save', side_effect=inspect):
                generate_figures(Path(tmp), users, groups)
        self.assertEqual(set(captured), {'psm_stage', 'within_stage'})


if __name__ == '__main__':
    unittest.main()
