"""Protect interpretation of coefficient intervals in performance scatter plots."""
import tempfile
import unittest
from pathlib import Path

from tools.plot_peak_correlation_vs_score import (
    analysis_table_paths, data_limits, figure_row_limits, spearman_ci_label, spearman_ci_metadata,
)


def displayed_result(low, high, valid):
    source = {"ci_low": low, "ci_high": high, "bootstrap_valid": valid}
    return {"spearman_ci_low": low, "spearman_ci_high": high,
            **spearman_ci_metadata(source, 10000)}


class SpearmanConfidenceDisplayTests(unittest.TestCase):
    def test_small_velocity_correlations_have_tight_zero_based_limits(self):
        low, high = data_limits([.03, .05, .084], correlation=True)
        self.assertEqual(low, 0)
        self.assertGreater(high, .084)
        self.assertLessEqual(high, .1)
        self.assertLess(data_limits([.55], correlation=True)[1], .7)

    def test_lag_limits_keep_zero_without_forced_symmetry(self):
        low, high = data_limits([100, 200, 300], include_zero=True)
        self.assertLess(low, 0)
        self.assertGreater(high, 300)
        self.assertLess(abs(low), 50)
        low, high = data_limits([-300, -200, -100], include_zero=True)
        self.assertLess(low, -300)
        self.assertGreater(high, 0)
        self.assertLess(high, 50)

    def test_score_limits_retain_negative_values_and_constant_data(self):
        low, high = data_limits([-40, 25, 80])
        self.assertLess(low, -40)
        self.assertGreater(high, 80)
        low, high = data_limits([3, 3])
        self.assertLess(low, 3)
        self.assertGreater(high, 3)

    def test_phase_rows_share_limits_without_borrowing_other_phase_outliers(self):
        by_group = {}
        phases = ("reach string", "threading string through tower")
        for stage, maximum in zip(phases, (.18, .55)):
            for side in ("l", "r"):
                by_group[("C", side, "velocity", stage, "psm_stage")] = [
                    {"task": "C", "side": side, "signal": "velocity", "stage": stage,
                     "mode": "psm_stage", "Eco_x": x, "peak_correlation_y": y}
                    for x, y in ((-10, .05), (70, maximum))]
        rows = [[("C", stage, side) for side in ("l", "r")] for stage in phases]
        bounds = figure_row_limits(rows, by_group, "velocity", "Eco", "peak_r")
        self.assertLess(bounds[0]["y"][1], .25)
        self.assertGreater(bounds[1]["y"][1], .55)
        self.assertLess(bounds[0]["x"][0], -10)
        self.assertGreater(bounds[0]["x"][1], 70)

    def test_whole_task_limits_share_extremes_across_all_four_panels(self):
        rows = [[(task, "__whole__", side) for side in ("l", "r")] for task in ("A", "C")]
        by_group = {}
        for task in ("A", "C"):
            for side in ("l", "r"):
                by_group[(task, side, "velocity", "__whole__", "whole")] = [
                    {"task": task, "side": side, "signal": "velocity", "stage": "__whole__",
                     "mode": "whole", "score_x": 50, "median_peak_lag_ms_y": 1900 if task == "C" else -300}]
        bounds = figure_row_limits(rows, by_group, "velocity", "Score", "signed_lag_s", whole_task=True)
        self.assertEqual(bounds[0], bounds[1])
        self.assertLess(bounds[0]["y"][0], -300)
        self.assertGreater(bounds[0]["y"][1], 1900)

    def test_velocity_outcomes_use_separate_analysis_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for outcome in ("Time", "Eco", "TimeRaw", "EcoRaw"):
                names, provenance = analysis_table_paths(root, "velocity", outcome)
                self.assertTrue(all(name.startswith("velocity_metrics/") for name in names))
                self.assertEqual(provenance, "velocity_metrics/provenance.json")
            names, _ = analysis_table_paths(root, "position", "Eco")
            self.assertTrue(all(name.startswith("component_metrics/") for name in names))
            legacy, _ = analysis_table_paths(root, "velocity", "Score")
            self.assertEqual(legacy[1], "associations.csv")
            (root/"velocity_metrics").mkdir()
            (root/"velocity_metrics/associations_Score.csv").touch()
            names, _ = analysis_table_paths(root, "velocity", "Score")
            self.assertEqual(names[1], "velocity_metrics/associations_Score.csv")

    def test_score_threading_intervals_are_not_presented_as_reliable(self):
        for valid, low, high in [(6597, .5808, .8885), (9034, -.0885, .9413)]:
            with self.subTest(valid=valid):
                result = displayed_result(low, high, valid)
                self.assertEqual(result["spearman_ci_status"], "unreliable")
                self.assertEqual(result["spearman_ci_reason"], "too_many_undefined_bootstrap_draws")
                label = spearman_ci_label(result)
                self.assertIn("CI unreliable", label)
                self.assertNotIn("95% CI:", label)

    def test_supported_interval_labels_spearman_explicitly(self):
        result = displayed_result(-.990566, -.399038, 10000)
        self.assertEqual(result["spearman_ci_status"], "supported")
        self.assertEqual(spearman_ci_label(result), "Spearman 95% CI: [-0.99, -0.40]")

    def test_nonfinite_and_reversed_bounds_are_unreliable(self):
        for low, high in [(float("nan"), .5), (-.5, float("inf")), (.8, -.2)]:
            with self.subTest(low=low, high=high):
                result = displayed_result(low, high, 10000)
                self.assertEqual(result["spearman_ci_status"], "unreliable")
                self.assertIn("CI unreliable", spearman_ci_label(result))

    def test_defined_draw_display_threshold_is_explicit(self):
        self.assertEqual(displayed_result(-.2, .8, 9750)["spearman_ci_status"], "supported")
        self.assertEqual(displayed_result(-.2, .8, 9749)["spearman_ci_status"], "unreliable")

    def test_multiple_comparison_result_does_not_censor_pointwise_interval(self):
        result = displayed_result(.070226, .903846, 10000)
        result.update(spearman_permutation_p=.06865, spearman_q_bh_existing_family=.47385)
        self.assertEqual(spearman_ci_label(result), "Spearman 95% CI: [+0.07, +0.90]")


if __name__ == "__main__":
    unittest.main()
