"""Check score association statistics against independent direct calculations."""
import argparse
import json
import unittest

import numpy as np
from scipy import stats

from tools.analyze_score_corr import (
    aggregate_participants, analyze, between_test, bh_adjust, scenario_rows,
    spearman, within_test,
)
from tools.analyze_raw_score_corr import validate_results
from tools.analyze_velocity_score_corr import json_value


def synthetic_records():
    rng = np.random.default_rng(45)
    records = []
    for u in range(8):
        # Unequal recording counts exercise participant weighting.
        for t in range(3 + u % 3):
            lag = rng.normal() + .25*u - .1*t
            records.append({"user": f"User{u}", "TaskNum": t+1,
                            "Attempt": 1+t % 2, "signed_lag_s": lag,
                            "absolute_lag_s": abs(lag), "peak_r": .2+.05*u,
                            "Score": float(round(30+4*u+7*lag+3*t+rng.normal()*4)),
                            "content_id": f"User{u}_{t+1}_{1+t%2}",
                            "eligible": True, "signal": "position", "mode": "psm_stage",
                            "AttemptStatus": "Completed"})
    return records


class ScoreCorrTests(unittest.TestCase):
    def test_significant_result_provenance_serializes_numpy_counts(self):
        row = {"bootstrap_valid": np.int64(10000), "ci_low": np.float64(.25),
               "significant": np.bool_(True)}
        parsed = json.loads(json.dumps({"significant_results": [row]}, default=json_value))
        self.assertEqual(parsed["significant_results"][0]["bootstrap_valid"], 10000)
        self.assertTrue(parsed["significant_results"][0]["significant"])

    def test_spearman_ties_and_constant(self):
        x = [0, 0, 1, 2, 4, 4]
        y = [5, 4, 3, 0, 2, 2]
        self.assertAlmostEqual(spearman(x, y), stats.spearmanr(x, y).statistic)
        self.assertTrue(np.isnan(spearman(np.zeros(6), np.arange(6))))

    def test_bh_missing_values_and_monotonicity(self):
        p = np.array([.8, .004, np.nan, .05, .02])
        q = bh_adjust(p)
        np.testing.assert_allclose(q, [.8, .016, np.nan, 1/15, .04], equal_nan=True)
        order = np.argsort(p[np.isfinite(p)])
        self.assertTrue(np.all(np.diff(q[np.isfinite(p)][order]) >= 0))

    def test_between_bootstrap_and_permutation_reproducible(self):
        x, y = np.arange(8.), np.array([0., 0., 3., 2., 4., 6., 6., 7.])
        first = between_test(x, y, np.random.default_rng(33), 300, 399, 6)
        second = between_test(x, y, np.random.default_rng(33), 300, 399, 6)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["association"], stats.spearmanr(x, y).statistic)
        self.assertGreaterEqual(first["p"], 1/400)
        self.assertEqual(between_test(x[:3], y[:3], np.random.default_rng(1), 100, 100, 6)["status"],
                         "insufficient_users_or_variation")

    def test_participant_features_use_exact_same_recordings(self):
        records = synthetic_records()
        people = aggregate_participants(records)
        self.assertEqual(len(people), 8)
        for p in people:
            rr = [r for r in records if r["user"] == p["user"]]
            self.assertAlmostEqual(p["Score"], np.mean([r["Score"] for r in rr]))
            self.assertAlmostEqual(p["signed_lag_s"], np.median([r["signed_lag_s"] for r in rr]))
            self.assertEqual(set(p["content_ids"].split(";")), {r["content_id"] for r in rr})

    def test_within_matches_full_weighted_dummy_regression(self):
        records = synthetic_records()
        users = sorted({r["user"] for r in records})
        dummy = np.array([[r["user"] == u for u in users] for r in records], dtype=float)
        x = stats.rankdata([r["signed_lag_s"] for r in records])
        y = stats.rankdata([r["Score"] for r in records])
        w = np.array([1/sum(r["user"] == u for r in records) for u in [r["user"] for r in records]])
        root_w = np.sqrt(w)
        for adjusted in (False, True):
            nuisance = dummy
            if adjusted:
                nuisance = np.column_stack((dummy, [r["TaskNum"] for r in records], [r["Attempt"] for r in records]))
            nuisance_w = nuisance*root_w[:, None]
            resid = []
            for value in (x, y):
                residual = value-nuisance @ np.linalg.lstsq(nuisance_w, value*root_w, rcond=None)[0]
                resid.append(residual)
            expected_r = np.sum(w*resid[0]*resid[1])/np.sqrt(np.sum(w*resid[0]**2)*np.sum(w*resid[1]**2))
            full = np.column_stack((x, nuisance))
            beta = np.linalg.lstsq(full*root_w[:, None], y*root_w, rcond=None)[0]
            err = y-full @ beta
            bread = np.linalg.inv(full.T @ (w[:, None]*full))
            cluster = np.array([full[dummy[:, i] == 1].T @ (w[dummy[:, i] == 1]*err[dummy[:, i] == 1]) for i in range(len(users))])
            n, g = len(records), len(users)
            covariance = bread @ (cluster.T @ cluster) @ bread*g/(g-1)*(n-1)/(n-full.shape[1])
            expected_p = 2*stats.t.sf(abs(beta[0]/np.sqrt(covariance[0, 0])), g-1)
            result = within_test(records, "signed_lag_s", adjusted, np.random.default_rng(2), 300, 6)
            self.assertAlmostEqual(result["association"], expected_r, places=10)
            self.assertAlmostEqual(result["rank_slope"], beta[0], places=10)
            self.assertAlmostEqual(result["rank_slope_cluster_se"], np.sqrt(covariance[0, 0]), places=10)
            self.assertAlmostEqual(result["p"], expected_p, places=10)

    def test_within_constant_scores_do_not_create_information(self):
        records = synthetic_records()
        for r in records:
            r["Score"] = 0
        result = within_test(records, "signed_lag_s", True, np.random.default_rng(2), 100, 6)
        self.assertEqual(result["n_informative_users"], 0)
        self.assertEqual(result["status"], "insufficient_users_with_within_variation")

    def test_sensitivity_filters_preserve_supplied_failed_zero(self):
        records = synthetic_records()
        records[0]["AttemptStatus"] = "Failed"
        records[0]["Score"] = 0
        self.assertIn(records[0], scenario_rows(records, "primary"))
        self.assertNotIn(records[0], scenario_rows(records, "completed_only"))
        records[0]["eligible"] = False
        self.assertNotIn(records[0], scenario_rows(records, "primary"))
        records[1]["peak_r"] = .19
        self.assertNotIn(records[1], scenario_rows(records, "stronger_peaks"))

    def test_raw_outcome_is_used_without_falling_back_to_score(self):
        records = synthetic_records()
        for r in records:
            r["TimeRaw"] = r["Score"]
            r["EcoRaw"] = -r["Score"]
            r["Score"] = 0
        self.assertNotIn("Score", aggregate_participants(records, "TimeRaw")[0])
        self.assertEqual(within_test(records, "signed_lag_s", False, np.random.default_rng(3), 100, 6)["status"],
                         "insufficient_users_with_within_variation")
        for adjusted in (False, True):
            time = within_test(records, "signed_lag_s", adjusted, np.random.default_rng(3), 100, 6, "TimeRaw")
            eco = within_test(records, "signed_lag_s", adjusted, np.random.default_rng(3), 100, 6, "EcoRaw")
            self.assertEqual(time["status"], "ok")
            self.assertAlmostEqual(time["association"], -eco["association"], places=10)
            self.assertAlmostEqual(time["p"], eco["p"], places=10)

    def test_component_analysis_uses_named_mean_and_separate_fdr_family(self):
        records = synthetic_records()
        phase = {"task": "A", "side": "r", "signal": "position", "stage": "grab ring",
                 "mode": "psm_stage", "group_usable": True, "group_status": "ok"}
        for r in records:
            r.update({k: phase[k] for k in ("task", "side", "signal", "stage", "mode")})
            r["EcoRaw"] = r["Score"]
            r["Score"] = 0
        args = argparse.Namespace(seed=42, bootstrap=100, permutations=199, min_users=6)
        results, people = analyze(args, records, [phase], "EcoRaw", scenarios=("primary",))
        self.assertEqual(len(results), 9)
        self.assertTrue(all(r["family"].startswith("EcoRaw/") for r in results))
        between = next(r for r in results if r["predictor"] == "signed_lag_s" and r["method"] == "between_participants")
        expected = stats.spearmanr([p["signed_lag_s"] for p in people], [p["EcoRaw"] for p in people]).statistic
        self.assertAlmostEqual(between["association"], expected)
        for person in people:
            rr = [r for r in records if r["user"] == person["user"]]
            self.assertAlmostEqual(person["EcoRaw"], np.mean([r["EcoRaw"] for r in rr]))

    def test_time_eco_scores_are_distinct_from_raw_columns(self):
        records = synthetic_records()
        for r in records:
            r["Time"] = r["Score"]
            r["Eco"] = -r["Score"]
            r["TimeRaw"] = -r["Time"]
            r["EcoRaw"] = -r["Eco"]
            r["Score"] = 0
        for outcome in ("Time", "Eco"):
            people = aggregate_participants(records, outcome)
            for person in people:
                rr = [r for r in records if r["user"] == person["user"]]
                self.assertAlmostEqual(person[outcome], np.mean([r[outcome] for r in rr]))
            for adjusted in (False, True):
                score = within_test(records, "signed_lag_s", adjusted, np.random.default_rng(3), 100, 6, outcome)
                raw = within_test(records, "signed_lag_s", adjusted, np.random.default_rng(3), 100, 6, outcome+"Raw")
                self.assertEqual(score["status"], "ok")
                self.assertAlmostEqual(score["association"], -raw["association"], places=10)

    def test_velocity_validation_uses_velocity_attempts(self):
        records = synthetic_records()
        phase = {"task": "A", "side": "r", "signal": "velocity", "stage": "grab ring",
                 "mode": "psm_stage", "group_usable": True, "group_status": "ok"}
        for r in records:
            r.update({k: phase[k] for k in ("task", "side", "signal", "stage", "mode")})
            r["Eco"] = r["Score"]
            r["peak_r"] = .35+.05*np.sin(r["signed_lag_s"])
        args = argparse.Namespace(seed=42, bootstrap=100, permutations=199, min_users=6)
        results, people = analyze(args, records, [phase], "Eco", scenarios=("velocity_secondary",))
        check = validate_results(results, people, records, "Eco")
        self.assertTrue(check["all_checks_passed"])
        self.assertEqual(check["between_vs_scipy"], 3)
        self.assertEqual(check["within_vs_full_dummy_regression"], 6)


if __name__ == "__main__":
    unittest.main()
