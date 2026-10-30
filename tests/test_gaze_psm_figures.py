"""Verify comparable scales and preservation of uncertainty in figure views."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools import gaze_psm_figures as figures


def group(task='A', side='l', signal='velocity', stage='__whole__', mode='whole', scale=1):
    r = np.array([.005, .02, .04, .02, .005])*scale
    return dict(task=task, side=side, signal=signal, stage=stage, mode=mode,
                lag_s=np.arange(-2., 3.), r=r, r_ci_low=r-.008*scale,
                r_ci_high=r+.012*scale, null_r_low=r*0-.005*scale,
                null_r_high=r*0+.009*scale, r_fixed_records=r.copy(),
                r_primary_fixed_record_users=r.copy(), record_support_changes=False,
                n_stable_users=11, peak_lag_s=0., peak_r=float(r.max()),
                peak_at_boundary=False, status='ok', peak_lag_ci_low=-1.,
                peak_lag_ci_high=1., boundary_frequency=0.,
                bootstrap_peak_lag_s=np.array([-2., -1., 0., 0., 1., 2.]),
                proximity=np.array([.01, .011, .012, .011, .01]),
                distance_median_px=np.array([150., 149., 148., 149., 150.]))


class FigureReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.saved = {}
        self.saver = patch.object(figures, '_save', side_effect=lambda fig, path:
                                 self.saved.update({Path(path).name: fig}))
        self.saver.start()
        self.addCleanup(self.saver.stop)
        self.addCleanup(plt.close, 'all')

    def test_main_curves_share_scale_and_show_complete_intervals(self):
        groups = [group(task, side, scale=i+1) for i, (task, side) in
                  enumerate((('A', 'l'), ('A', 'r'), ('C', 'l'), ('C', 'r')))]
        outlier = {**groups[0], 'user': 'User1', 'r': np.ones(5)*.99}
        figures._whole_figures(plt, self.directory, groups, [outlier])
        axes = self.saved['whole_task_velocity.png'].axes
        limits = axes[0].get_ylim()
        self.assertLess(limits[1]-limits[0], .3)
        for ax, curve in zip(axes, groups):
            self.assertEqual(ax.get_ylim(), limits)
            self.assertLessEqual(limits[0], min(curve['r_ci_low'].min(), curve['null_r_low'].min()))
            self.assertGreaterEqual(limits[1], curve['r_ci_high'].max())
            self.assertEqual(ax.get_xlim(), (-2, 2))
        # The individual outlier is retained in the dedicated diagnostic view.
        figures._novice_figures(plt, self.directory, groups, [outlier])
        self.assertGreaterEqual(self.saved['novice_curves_velocity.png'].axes[0].get_ylim()[1], .99)

    def test_stage_pairs_share_scale_across_modes_but_rows_can_differ(self):
        groups = [group(side=side, stage=stage, mode=mode, scale=scale)
                  for stage, scale in [('small', 1), ('large', 8)]
                  for side in ('l', 'r') for mode in ('psm_stage', 'within_stage')]
        groups[-1]['r_ci_high'] += .1
        figures._stage_figures(plt, self.directory, groups, [])
        axes = self.saved['stages_A_velocity.png'].axes
        self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())
        self.assertEqual(axes[2].get_ylim(), axes[3].get_ylim())
        self.assertNotEqual(axes[0].get_ylim(), axes[2].get_ylim())
        large = next(ax for ax in axes if ax.get_title().startswith('large'))
        self.assertGreaterEqual(large.get_ylim()[1], groups[-1]['r_ci_high'].max())

    def test_missing_stage_retains_search_window_and_explicit_missing_message(self):
        curve = group(stage='absent', mode='psm_stage')
        curve.update(r=np.full(5, np.nan), status='no_valid_curve', n_stable_users=0)
        fig, ax = plt.subplots()
        figures._curve_panel(ax, [curve], [], dict(stage='absent'))
        self.assertEqual(ax.get_xlim(), (-2, 2))
        self.assertTrue(any('No group estimate' in t.get_text() for t in ax.texts))
        self.assertFalse(any(line.get_label() == 'Group mean' for line in ax.lines))

    def test_histograms_preserve_tails_with_common_bins_and_percentage_scale(self):
        from matplotlib.patches import Rectangle
        groups = [group(task, side) for task in ('A', 'C') for side in ('l', 'r')]
        figures._bootstrap_figures(plt, self.directory, groups)
        axes = self.saved['bootstrap_peak_lags_velocity.png'].axes
        for ax in axes:
            self.assertEqual(ax.get_xlim(), axes[0].get_xlim())
            self.assertEqual(ax.get_ylim(), axes[0].get_ylim())
            bars = [p for p in ax.patches if isinstance(p, Rectangle)]
            self.assertAlmostEqual(sum(p.get_height() for p in bars), 100)
            self.assertLess(ax.get_xlim()[0], -2)
            self.assertGreater(ax.get_xlim()[1], 2)

    def test_spatial_percent_view_and_delta_view_keep_comparable_scales(self):
        groups = [group(task, side, signal='position') for task in ('A', 'C') for side in ('l', 'r')]
        figures._spatial_figures(plt, self.directory, groups)
        axes = self.saved['spatial_proximity.png'].axes
        self.assertLess(axes[0].get_ylim()[1], 2.)
        self.assertTrue(all(ax.get_ylim() == axes[0].get_ylim() for ax in axes))
        delta = self.saved['spatial_changes.png'].axes
        self.assertEqual(delta[0].get_ylim(), delta[2].get_ylim())
        self.assertEqual(delta[1].get_ylim(), delta[3].get_ylim())
        self.assertTrue(any(np.allclose(line.get_ydata(), 0) for line in delta[0].lines))


if __name__ == '__main__':
    unittest.main()
