import unittest

import numpy as np

from tools.gaze_psm_group import aggregate, fisher_mean


def curve(user, record, r, task="A", stage="__whole__", mode="whole", distances=None):
    n = len(r)
    return dict(user=user, content_id=record, task=task, stage=stage,
                mode=mode, side="l", signal="position",
                lag_s=np.linspace(-1, 1, n), r=np.array(r, dtype=float),
                pairs=np.full(n, 100), occurrences=np.ones(n),
                gaze_rms=np.full(n, 10.), psm_rms=np.full(n, 10.),
                distance_median_px=np.array(distances if distances is not None else np.full(n, 20.)),
                proximity=np.full(n, .5))


class GroupAggregationTests(unittest.TestCase):
    def test_equal_novice_weight_despite_unequal_attempt_counts(self):
        records = [curve("User1", f"u1_{i}", [.1, .9, .1]) for i in range(4)]
        records.append(curve("User2", "u2", [.1, -.1, .1]))
        users, groups, _ = aggregate(records, bootstrap=0, min_users=2)
        self.assertEqual(len(users), 2)
        expected = fisher_mean(np.array([[.1, .9, .1], [.1, -.1, .1]]))
        np.testing.assert_allclose(groups[0]["r"], expected)
        self.assertEqual(groups[0]["n_stable_users"], 2)
        np.testing.assert_array_equal(groups[0]["n_records"], [5, 5, 5])

    def test_partial_users_are_diagnostic_not_changing_group_composition(self):
        records = [curve("User1", "a", [.1, .6, .1]),
                   curve("User2", "b", [.2, .7, .2]),
                   curve("User3", "c", [np.nan, -.9, .9])]
        _, groups, _ = aggregate(records, bootstrap=10, min_users=2)
        group = groups[0]
        self.assertEqual(group["stable_users"], ["User1", "User2"])
        self.assertEqual(group["excluded_partial_users"], ["User3"])
        np.testing.assert_array_equal(group["n_users"], [2, 3, 3])
        self.assertEqual(group["peak_lag_s"], 0)
        self.assertEqual(group["bootstrap_peak_lag_s"].shape, (10,))

    def test_paired_bootstrap_reuses_same_identities_for_tasks(self):
        records = []
        # Individual maxima differ; A and C are identical within every novice.
        for user, r in [("User1", [.9, .2, .1]), ("User2", [.1, .2, .8]),
                        ("User3", [.1, .7, .1])]:
            for task in ("A", "C"):
                records.append(curve(user, f"{user}_{task}", r, task=task))
        _, _, contrasts = aggregate(records, bootstrap=100, min_users=3)
        self.assertEqual(len(contrasts), 1)
        comparison = contrasts[0]
        self.assertEqual(comparison["dimension"], "task")
        self.assertEqual(comparison["difference_definition"], "condition_b_minus_condition_a")
        self.assertEqual(comparison["difference_s"], 0)
        np.testing.assert_array_equal(comparison["bootstrap_difference_s"], np.zeros(100))

    def test_null_uses_maximum_over_full_search_and_fixed_people(self):
        observed = [curve(u, u, [.1, .8, .1]) for u in ("User1", "User2")]
        low = [curve(u, u, [.1, .2, .1]) for u in ("User1", "User2")]
        high_at_edge = [curve(u, u, [.9, .1, .1]) for u in ("User1", "User2")]
        missing_person = [curve("User1", "User1", [.99, .1, .1])]
        _, groups, _ = aggregate(observed, bootstrap=0, min_users=2,
                                  baseline_replicates=[low, high_at_edge, missing_person])
        group = groups[0]
        self.assertEqual(group["n_null_replicates"], 2)
        np.testing.assert_allclose(group["null_peak_r"], [.2, .9])
        self.assertAlmostEqual(group["null_peak_p"], 2 / 3)

    def test_spatial_metrics_evaluated_at_common_group_peak(self):
        records = [curve("User1", "a", [.1, .8, .1], distances=[100, 10, 200]),
                   curve("User1", "b", [.1, .8, .1], distances=[100, 30, 200]),
                   curve("User2", "c", [.1, .8, .1], distances=[100, 40, 200])]
        _, groups, _ = aggregate(records, bootstrap=0, min_users=2)
        self.assertEqual(groups[0]["peak_lag_s"], 0)
        # User1 mean=20; User2 mean=40; equal novice mean=30 (not 80/3).
        self.assertAlmostEqual(groups[0]["spatial_at_peak_px"], 30)
        self.assertAlmostEqual(groups[0]["spatial_at_zero_px"], 30)

    def test_no_valid_curve_and_insufficient_users(self):
        _, groups, _ = aggregate([curve("User1", "a", [np.nan] * 3)], bootstrap=10)
        self.assertEqual(groups[0]["status"], "no_valid_curve")
        self.assertFalse(np.isfinite(groups[0]["r"]).any())
        _, groups, _ = aggregate([curve("User1", "a", [.1, .8, .1])], bootstrap=10)
        self.assertEqual(groups[0]["status"], "insufficient_users")
        self.assertEqual(groups[0]["n_stable_users"], 1)

    def test_stage_comparisons_stay_within_task_and_conditioning_mode(self):
        records = []
        for user in ("User1", "User2"):
            for task in ("A", "C"):
                for stage in ("reach", "place"):
                    records.append(curve(user, f"{user}_{task}", [.1, .8, .1],
                                         task=task, stage=stage, mode="psm_stage"))
        _, _, contrasts = aggregate(records, bootstrap=0, min_users=2)
        self.assertEqual(len(contrasts), 2)
        self.assertTrue(all(c["dimension"] == "stage" for c in contrasts))
        self.assertEqual({c["task"] for c in contrasts}, {"A", "C"})

    def test_mixed_lag_grids_are_rejected(self):
        a = curve("User1", "a", [.1, .8, .1])
        b = curve("User2", "b", [.1, .8, .1])
        b["lag_s"] = np.array([-.5, 0., .5])
        with self.assertRaisesRegex(ValueError, "different lag grids"):
            aggregate([a, b], bootstrap=0, min_users=2)

    def test_streamed_baseline_iterable_is_consumed_once(self):
        observed = [curve(u, u, [.1, .8, .1]) for u in ("User1", "User2")]
        consumed = []

        def references():
            for index in range(3):
                consumed.append(index)
                yield [curve(u, u, [.1, .2 + .1*index, .1]) for u in ("User1", "User2")]

        _, groups, _ = aggregate(observed, bootstrap=0, min_users=2,
                                  baseline_replicates=references())
        self.assertEqual(consumed, [0, 1, 2])
        self.assertEqual(groups[0]["n_null_replicates"], 3)
        np.testing.assert_allclose(groups[0]["null_peak_r"], [.2, .3, .4])

    def test_eleven_paired_novices_with_33_and_51_attempts(self):
        records = []
        expected_c_z = []
        for i in range(11):
            user = f"User{i + 1}"
            for attempt in range(3):
                records.append(curve(user, f"{user}_A_{attempt}", [.1, .4 + .01*i, .1]))
            n_c = 5 if i < 7 else 4
            person_c = []
            for attempt in range(n_c):
                r = [.1, .2, .4 + .01*i + .005*attempt]
                person_c.append(np.arctanh(r))
                records.append(curve(user, f"{user}_C_{attempt}", r, task="C"))
            expected_c_z.append(np.mean(person_c, axis=0))
        users, groups, contrasts = aggregate(records, bootstrap=100, seed=13, min_users=11)
        self.assertEqual(len(records), 84)
        self.assertEqual(len(users), 22)
        by_task = {g["task"]: g for g in groups}
        self.assertEqual(by_task["A"]["n_stable_users"], 11)
        self.assertEqual(by_task["C"]["n_stable_users"], 11)
        np.testing.assert_array_equal(by_task["A"]["n_records"], [33]*3)
        np.testing.assert_array_equal(by_task["C"]["n_records"], [51]*3)
        np.testing.assert_allclose(by_task["C"]["r"], np.tanh(np.mean(expected_c_z, axis=0)))
        self.assertEqual(contrasts[0]["n_users"], 11)
        np.testing.assert_array_equal(contrasts[0]["bootstrap_difference_s"], np.ones(100))

    def test_fixed_record_sensitivity_detects_equal_count_composition_changes(self):
        records = [curve("User1", "a", [.9, np.nan, np.nan]),
                   curve("User1", "b", [np.nan, .9, np.nan]),
                   curve("User1", "c", [np.nan, np.nan, .9]),
                   curve("User1", "stable", [.2, .3, .2]),
                   curve("User2", "stable2", [.1, .2, .1])]
        users, groups, _ = aggregate(records, bootstrap=0, min_users=2)
        first = next(u for u in users if u["user"] == "User1")
        np.testing.assert_array_equal(first["n_records"], [2, 2, 2])
        self.assertTrue(first["record_support_changes"])
        self.assertEqual(first["stable_record_ids"], ["stable"])
        np.testing.assert_allclose(first["r_fixed_records"], [.2, .3, .2])
        self.assertTrue(groups[0]["record_support_changes"])
        self.assertEqual(groups[0]["n_fixed_record_users"], 2)
        expected = fisher_mean([[.2, .3, .2], [.1, .2, .1]])
        np.testing.assert_allclose(groups[0]["r_fixed_records"], expected)
        np.testing.assert_allclose(groups[0]["r_primary_fixed_record_users"], groups[0]["r"])

    def test_fixed_record_sensitivity_discloses_changed_participant_set(self):
        records = [curve("User1", "a", [.1, .8, np.nan]),
                   curve("User1", "b", [np.nan, .8, .1]),
                   curve("User2", "stable2", [.1, .2, .1])]
        _, groups, _ = aggregate(records, bootstrap=0, min_users=2)
        group = groups[0]
        self.assertEqual(group["n_stable_users"], 2)
        self.assertEqual(group["fixed_record_users"], ["User2"])
        self.assertFalse(np.isfinite(group["r_fixed_records"]).any())


if __name__ == "__main__":
    unittest.main()
