"""Numerical/row-alignment regression checks: python -m unittest -v test_xcorr."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from xcorr_common import (
    HZ, compute_record, correlation_curve, filter_sources, peak_correlation,
    read_path_list, smooth_gaze,
)
from Task_level_record_xcorr import aggregate_tasks
from prepare_xcorr_path_list import prepare
from record_level_xcorr import dataset_relative_path, format_record_output


def direct_curve(gaze, psm, lags, minimum):
    values, counts = [], []
    n = len(gaze)
    for lag in lags:
        if abs(lag) >= n:
            values.append(np.nan)
            counts.append(0)
            continue
        g, p = (gaze[:n-lag], psm[lag:]) if lag >= 0 else (gaze[-lag:], psm[:n+lag])
        valid = np.isfinite(g).all(axis=1) & np.isfinite(p).all(axis=1)
        counts.append(int(valid.sum()))
        if valid.sum() < minimum:
            values.append(np.nan)
            continue
        g, p = g[valid], p[valid]
        g, p = g-g.mean(axis=0), p-p.mean(axis=0)
        denominator = np.sqrt(np.sum(g*g) * np.sum(p*p))
        values.append(np.sum(g*p)/denominator if denominator > 0 else np.nan)
    return np.array(values), np.array(counts)


def write_fixture(folder, n=700, dlc_n=None):
    gaze = np.zeros(n, dtype=[("gaze_x", "f8"), ("gaze_y", "f8"), ("valid", "u1")])
    rng = np.random.default_rng(15)
    gaze["gaze_x"] = rng.uniform(.2, .8, n)
    gaze["gaze_y"] = rng.uniform(.2, .8, n)
    gaze_path = folder / "npy" / "User1_1_1_gaze.npy"
    gaze_path.parent.mkdir(parents=True)
    np.save(gaze_path, gaze)
    dlc_path = folder / "sam" / "User1_1_1" / "User1_1_1_DLC.csv"
    dlc_path.parent.mkdir(parents=True)
    with dlc_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scorer"] + ["model"]*6)
        writer.writerow(["bodyparts"] + ["left_psm"]*3 + ["right_psm"]*3)
        writer.writerow(["coords"] + ["x", "y", "likelihood"]*2)
        for i in range(n if dlc_n is None else dlc_n):
            # Deliberately unrelated first column; align by row, never frame ID.
            writer.writerow([10000+i*2, 200+i%11, 400+i%7, .9, 700+i%13, 300+i%5, .8])
    return {"record_id": "User1_1_1", "task_name": "A",
            "gaze_npy_path": str(gaze_path), "dlc_csv_path": str(dlc_path)}


class NumericalTests(unittest.TestCase):
    def test_fft_matches_direct_overlap_centering_with_missing_samples(self):
        rng = np.random.default_rng(801)
        gaze = rng.normal(size=(900, 2))*[150, 85]+[600, 450]
        psm = rng.normal(size=(900, 2))*[80, 50]+[700, 600]
        gaze[rng.choice(900, 127, replace=False)] = np.nan
        psm[rng.choice(900, 163, replace=False)] = np.nan
        lags, actual, counts = correlation_curve(gaze, psm)
        expected, expected_counts = direct_curve(gaze, psm, lags, 500)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12, equal_nan=True)

    def test_known_positive_and_negative_lag_with_gaps(self):
        rng = np.random.default_rng(91)
        gaze = rng.normal(size=(1100, 2))
        for delay in (17, -23):
            psm = np.full_like(gaze, np.nan)
            if delay > 0:
                psm[delay:] = 2*gaze[:-delay]+[30, 40]
            else:
                psm[:delay] = 2*gaze[-delay:]+[30, 40]
            g = gaze.copy()
            g[200:240] = np.nan
            psm[600:650] = np.nan
            peak = peak_correlation(g, psm)
            self.assertEqual(peak["peak_lag_frames"], delay)
            self.assertAlmostEqual(peak["peak_lag_seconds"], delay/60)
            self.assertAlmostEqual(peak["peak_r"], 1, places=12)

    def test_fewer_than_500_or_constant_has_no_peak(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=(499, 2))
        self.assertEqual(peak_correlation(a, a)["status"], "no_valid_peak")
        constant = np.full((800, 2), [100., 200.])
        self.assertEqual(peak_correlation(constant, constant)["status"], "no_valid_peak")
        self.assertEqual(peak_correlation(np.empty((0, 2)), np.empty((0, 2)))["status"], "no_valid_peak")

    def test_gaze_smoothing_does_not_bridge_missing_runs(self):
        a = np.array([[2,4], [4,8], [8,16], [np.nan,np.nan], [100,200], [np.nan,np.nan]])
        expected = np.array([[3,6], [14/3,28/3], [6,12], [np.nan,np.nan], [100,200], [np.nan,np.nan]])
        np.testing.assert_allclose(smooth_gaze(a), expected, equal_nan=True)

    def test_filter_edges_threshold_and_independent_sides_without_psm_smoothing(self):
        gaze = np.array([[.5,.5], [0,.5], [1,.5], [.5,-.1], [.4,.6], [.5,np.nan]])
        left = np.array([[100,200], [120,200], [0,200], [100,1024], [110,210], [120,220.]])
        ll = np.array([.2, .1999, 1, 1, 1, 1])
        right = np.full((6,2), [500.,600.])
        rl = np.ones(6)
        g, l, r = filter_sources(gaze, left, ll, right, rl)
        np.testing.assert_array_equal(np.isfinite(g).all(axis=1), [1,0,0,0,1,0])
        np.testing.assert_array_equal(np.isfinite(l).all(axis=1), [1,0,0,0,1,1])
        np.testing.assert_array_equal(l[[0,4,5]], left[[0,4,5]])
        np.testing.assert_array_equal(r, right)
        np.testing.assert_allclose(g[0], [640,512])

    def test_fisher_z_uses_original_duration_and_signed_arithmetic_lag(self):
        rows = [dict(record_id="one", task_name="A", psm_side="left", status="ok",
                     duration_seconds=10., peak_r=.2, peak_lag_seconds=1.),
                dict(record_id="two", task_name="A", psm_side="left", status="ok",
                     duration_seconds=30., peak_r=.8, peak_lag_seconds=-.5),
                dict(record_id="bad", task_name="A", psm_side="left", status="no_valid_peak",
                     duration_seconds=90.)]
        result = aggregate_tasks(rows)[0]
        self.assertAlmostEqual(result["fisher_z_weighted_r"], np.tanh((np.arctanh(.2)+3*np.arctanh(.8))/4))
        self.assertAlmostEqual(result["duration_weighted_lag_seconds"], -.125)
        self.assertEqual(result["included_duration_seconds"], 40.)
        self.assertEqual(result["total_duration_seconds"], 130.)
        self.assertEqual(result["records_excluded"], 1)
        rows[0]["peak_r"] = 1.
        self.assertTrue(np.isfinite(aggregate_tasks(rows)[0]["fisher_z_weighted_r"]))
        self.assertEqual(aggregate_tasks(rows)[0]["fisher_r_clip_count"], 1)


class SourceTests(unittest.TestCase):
    def test_row_matching_ignores_frame_ids_and_valid_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = write_fixture(Path(tmp))
            rows = compute_record(entry)
            self.assertTrue(all(r["status"] == "ok" for r in rows))
            self.assertEqual(rows[0]["gaze_valid_rows"], 700)
            self.assertEqual(rows[0]["duration_seconds"], 700/HZ)

    def test_mismatch_missing_and_stale_list_are_reported_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = write_fixture(Path(tmp), dlc_n=699)
            rows = compute_record(entry)
            self.assertTrue(all(r["status"] == "error" for r in rows))
            self.assertIn("Row mismatch", rows[0]["error"])
            entry["dlc_csv_path"] = str(Path(tmp)/"absent.csv")
            self.assertIn("FileNotFoundError", compute_record(entry)[0]["error"])
        with tempfile.TemporaryDirectory() as tmp:
            entry = write_fixture(Path(tmp))
            entry["n_rows"] = 701
            self.assertIn("Stale path list", compute_record(entry)[0]["error"])

    def test_discovery_audits_missing_and_only_lists_matched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = write_fixture(root/"Active_Data"/"User1")
            clips = {"one": dict(gaze_npy_path=str(Path(entry["gaze_npy_path"]).relative_to(root)),
                                  meta=dict(content_id="User1_1_1", task_name="A", user=1,
                                            Score=0, EcoRaw=None, new_field={"value": 2})),
                     "two": dict(gaze_npy_path="Active_Data/User2/npy/User2_1_1_gaze.npy",
                                  meta=dict(content_id="User2_1_1", task_name="C"))}
            index = root/"index.json"
            index.write_text(json.dumps({"clips": clips}), encoding="utf-8")
            matched, audit = prepare(index, root)
            self.assertEqual(len(matched), 1)
            self.assertEqual(len(audit), 2)
            self.assertEqual(matched[0]["n_rows"], 700)
            self.assertEqual(matched[0]["meta"], clips["one"]["meta"])
            self.assertEqual(audit[1]["status"], "error")
            self.assertIn("gaze:", audit[1]["error"])
            self.assertIn("dlc:", audit[1]["error"])
            manifest = root/"pairs.json"
            manifest.write_text(json.dumps(matched), encoding="utf-8")
            self.assertEqual(read_path_list(manifest), matched)
            manifest.write_text(json.dumps(matched*2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_path_list(manifest)


class OutputTests(unittest.TestCase):
    def test_csv_paths_start_at_dataset_and_metadata_is_preserved_for_both_sides(self):
        entry = dict(record_id="one", task_name="C", gaze_npy_path=r"D:\Dataset\dvss\Active_Data\User1\npy\one.npy",
                     dlc_csv_path=r"D:\Dataset\dvss\Active_Data\User1\sam\one\one_DLC.csv",
                     meta=dict(task_name="C", user=1, Score=0, EcoRaw=None, PenaltySum=-144.3,
                               extra={"value": 2}, status="source_meta_status"))
        rows = [dict(record_id=entry["record_id"], task_name="C", psm_side=side, status="ok",
                     gaze_npy_path=entry["gaze_npy_path"], dlc_csv_path=entry["dlc_csv_path"])
                for side in ("left", "right")]
        output, fields = format_record_output([entry], rows)
        self.assertEqual(len(fields), len(set(fields)))
        for row in output:
            self.assertEqual(row["gaze_npy_path"], r"dvss\Active_Data\User1\npy\one.npy")
            self.assertEqual(row["dlc_csv_path"], r"dvss\Active_Data\User1\sam\one\one_DLC.csv")
            self.assertEqual(row["Score"], 0)
            self.assertIsNone(row["EcoRaw"])
            self.assertEqual(row["PenaltySum"], -144.3)
            self.assertEqual(json.loads(row["extra"]), {"value": 2})
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["meta_status"], "source_meta_status")
        self.assertEqual(rows[0]["gaze_npy_path"], entry["gaze_npy_path"])
        self.assertEqual(dataset_relative_path("D:/Dataset/dvss/Active_Data/u/file.npy"),
                         r"dvss\Active_Data\u\file.npy")
        with self.assertRaisesRegex(ValueError, "no dvss"):
            dataset_relative_path(r"D:\unrelated\file.npy")

    def test_empty_output_still_has_metadata_columns(self):
        output, fields = format_record_output([dict(record_id="one", task_name="A", meta={"Score": 5})], [])
        self.assertEqual(output, [])
        self.assertIn("Score", fields)


if __name__ == "__main__":
    unittest.main()
