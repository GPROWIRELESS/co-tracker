"""Risk-focused tests of fixed-recipient hierarchical unpaired references."""
from gaze_psm_modes import mode_rows
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "model_code"))
from gt_reference import donor_choices, hierarchical_curve, run_reference
from gaze_psm_reliability_v2 import occurrence_statistics, aggregate_statistics
from gaze_psm_statistics_v2 import group_curves


class ReferenceTests(unittest.TestCase):
    def test_record_then_user_equal_weighting(self):
        records = {"a1": [.9, .7], "a2": [.9, .7], "b": [.1, .2], "c": [.1, .2]}
        users = {"a1": "A", "a2": "A", "b": "B", "c": "C"}
        actual = hierarchical_curve(records, users)
        expected = np.tanh(np.mean(np.arctanh([[.9, .7], [.1, .2], [.1, .2]]), axis=0))
        np.testing.assert_allclose(actual, expected)
        self.assertGreater(abs(actual[0] - np.tanh(np.arctanh(list(records.values())).mean(axis=0))[0]), .05)

    def test_donors_equal_by_person_not_record_count(self):
        records = [{"content_id": "A0", "task": "A", "user": "A"},
                   {"content_id": "B0", "task": "A", "user": "B"}] + [
                   {"content_id": f"C{i}", "task": "A", "user": "C"} for i in range(4)]
        choices, _, _ = donor_choices(records, 20000, seed=123)
        fraction = np.mean(choices["A0"] == "B0")
        self.assertGreater(fraction, .48)
        self.assertLess(fraction, .52)
        self.assertNotIn("A0", choices["A0"])
        again, _, _ = donor_choices(records, 20000, seed=123)
        np.testing.assert_array_equal(choices["A0"], again["A0"])

    def test_missing_donor_is_not_rerolled(self):
        records = [{"content_id": "A0", "task": "A", "user": "A"},
                   {"content_id": "B0", "task": "A", "user": "B"}]
        choices, candidates, coverage = donor_choices(records, 17, coverage=lambda a, b: False)
        self.assertEqual(len(choices["A0"]), 17)
        self.assertTrue(np.all(choices["A0"] == ""))
        self.assertEqual(candidates["A0"], [])
        self.assertTrue(coverage[0]["no_eligible_donor"])

    def test_full_run_outputs_fixed_support_and_subset_observed(self):
        config = dict(hz=10., max_lag=.2, min_overlap=.2, min_rms=0.)
        records, prepared, curves = [], {}, []
        for j, n in enumerate([120, 120, 150, 150, 180]):
            cid = f"User{j+1}_test"
            records.append(dict(content_id=cid, user=f"User{j+1}", task="A"))
            rng = np.random.default_rng(j + 3)
            xy = np.cumsum(rng.normal(size=(n, 2)), axis=0)
            gaze = xy + rng.normal(size=(n, 2))
            sigs = {}
            for signal in ("position", "velocity"):
                nn = n if signal == "position" else n-1
                p = xy if signal == "position" else np.diff(xy, axis=0)*10
                g = gaze if signal == "position" else np.diff(gaze, axis=0)*10
                t = np.arange(nn)/10 + (0 if signal == "position" else .05)
                labels = np.full(nn, "reach ring", dtype="U30")
                # A short late stage tests donor truncation and fixed recipients.
                labels[-15:] = "place ring"
                ids = np.where(labels == "reach ring", 0, 1)
                sigs[signal] = {}
                for side in ("l", "r"):
                    sigs[signal][side] = dict(psm=p, gazes={"gt": g}, time_s=t, stage=labels, occurrence=ids)
                    for stage in ("reach ring", "place ring"):
                        for mode in ("PSM in stage", "Both in stage"):
                            st = occurrence_statistics(g, p, np.where(labels == stage, ids, -1), 10, .2, mode == "Both in stage")
                            c = aggregate_statistics(st, min_overlap=.2, min_rms=0)
                            curves.append(dict(c, variant="primary", source="gt", content_id=cid,
                                               user=f"User{j+1}", task="A", side=side, signal=signal, stage=stage, mode=mode))
            prepared[cid] = {"signals": sigs}
        groups, _ = group_curves(curves, bootstrap=0)
        with tempfile.TemporaryDirectory(prefix="gt_reference_test_") as td:
            summary = run_reference(prepared, records, curves, groups, td, count=31, workers=2, config=config)
            self.assertEqual(len(summary), 32)
            for row in summary:
                self.assertEqual(row["n_requested"], 31)
                self.assertEqual(row["n_valid"] + row["n_failed"], 31)
                self.assertEqual(row["reference_family_size"], 9)
                self.assertFalse(row["formal_leading_p_value"])
            subset = [r for r in summary if r["reference_scope"] == "duration_eligible_subset"]
            self.assertTrue(all("User5_test" in r["excluded_duration_records"] for r in subset))
            self.assertTrue(all(r["n_observed_users"] == 4 for r in subset))
            self.assertTrue(any(r["n_failed"] > 0 for r in summary if r["reference_scope"] == "elapsed_time_model_consistent"))
            with np.load(Path(td)/"unpaired_reference_peaks.npz") as z:
                for row in summary:
                    self.assertEqual(np.isfinite(z[row["draws_key"]+"_peak_r"]).sum(), row["n_valid"])
            with (Path(td)/"unpaired_pairings.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual(sum(1 for _ in mode_rows(csv.DictReader(f))), 31*5*2)


if __name__ == "__main__":
    unittest.main()
