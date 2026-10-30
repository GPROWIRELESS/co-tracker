"""Small real-file tests for synchronization, extraction, caching and CLI output."""
import contextlib
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import gaze_psm_pipeline as pipeline


class NumericInputTests(unittest.TestCase):
    def test_nanoseconds_scientific_notation_preserves_integer_precision(self):
        self.assertEqual(pipeline.timestamp_ns("1.7514710988026337e+18"),
                         1_751_471_098_802_633_700)
        self.assertEqual(pipeline.timestamp_ns("1.750427084986753001E18"),
                         1_750_427_084_986_753_001)
        self.assertEqual(pipeline.timestamp_ns("1750427084986753001.0"),
                         1_750_427_084_986_753_001)
        for value in ("NaN", "Infinity", "1.5", "bad", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.timestamp_ns(value)

    def test_numeric_status_codes_compare_by_value(self):
        for value in ("2", "2.0", " 2.000 ", "2e0", 2, 2.0):
            self.assertEqual(pipeline.status_code(value), pipeline.status_code("2"))
        self.assertEqual(pipeline.status_code(0), pipeline.status_code("0.0"))
        for value in ("0", "2.1", "", None, "NaN", "Infinity"):
            self.assertNotEqual(pipeline.status_code(value), pipeline.status_code("2"))


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class InterpolationAndSmoothingTests(unittest.TestCase):
    def test_exact_sample_survives_large_neighboring_gap(self):
        times = np.array([0., 1.])
        xy = np.array([[10., 20.], [30., 40.]])
        result = pipeline.interpolate_valid(times, xy, np.array([0., .5, 1.]), .1)
        np.testing.assert_array_equal(result[[0, 2]], xy)
        self.assertTrue(np.isnan(result[1]).all())

    def test_smoothing_does_not_cross_missing_run_or_shift_step(self):
        xy = np.r_[np.zeros((5, 2)), np.full((2, 2), np.nan), np.ones((5, 2))*100]
        result = pipeline.smooth_valid(xy, 10, 300)
        np.testing.assert_array_equal(result[:5], 0.)
        self.assertTrue(np.isnan(result[5:7]).all())
        np.testing.assert_array_equal(result[7:], 100.)
        impulse = np.zeros((11, 2))
        impulse[5] = 9
        smoothed = pipeline.smooth_valid(impulse, 10, 300)
        np.testing.assert_allclose(smoothed[4:7], 3.)
        np.testing.assert_allclose(smoothed[:4], 0.)
        np.testing.assert_allclose(smoothed[7:], 0.)

    def test_velocity_stage_uses_native_interval_midpoint(self):
        xy = np.array([[0., 0.], [1., 2.], [2., 4.]])
        data = {"gaze_l": xy, "psm_l": xy,
                "time_s": np.array([0., .1, .2]),
                "stage": np.array(["reach", "grab", "place"]),
                "occurrence": np.array([0, 1, 2]),
                "native_time_s": np.array([0., .04, .11, .2]),
                "native_stage": np.array(["reach", "grab", "place", "place"]),
                "native_occurrence": np.array([0, 1, 2, 2])}
        g, p, labels, ids = pipeline.curve_inputs(data, "l", "velocity", 10.)
        np.testing.assert_allclose(g, [[10., 20.], [10., 20.]])
        np.testing.assert_allclose(p, g)
        self.assertEqual(labels.tolist(), ["grab", "place"])
        self.assertEqual(ids.tolist(), [1, 2])


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_root = self.root/"Active_Data"
        self.base = self.data_root/"User1"
        self.cid = "User1_1_1"
        self.record = {"content_id": self.cid, "src_skill": "novice", "Task Name": "A"}
        self.index = self.root/"index.csv"
        write_rows(self.index, list(self.record), [self.record])
        self.ns_origin = 1_750_427_084_986_753_000
        # Explicit int64 also prevents intermediate int32 overflow on Windows.
        self.times = np.arange(80, dtype=np.int64)*100_000_000+self.ns_origin
        self.frame_numbers = np.arange(80)+9000
        self.centers = np.c_[30+np.rint(12*np.sin(np.arange(80)*.3)),
                             23+np.rint(8*np.cos(np.arange(80)*.2))]
        self.deleted = 20
        timestamp_rows = [{"Frame_Number": int(frame), "Timestamp(ns)": int(t),
                           "IsDelete": int(i == self.deleted)}
                          for i, (frame, t) in enumerate(zip(self.frame_numbers, self.times))]
        write_rows(self.base/"csv"/f"{self.cid}_video_ts.csv",
                   ["Frame_Number", "Timestamp(ns)", "IsDelete"], timestamp_rows)
        # Deliberately reversed: allocation row order must not determine alignment.
        allocation = [{"Frame_Number": int(frame), "Timestamp(ns)": "1.75043E+18",
                       "gaze_x": float(x), "gaze_y": float(y), "valid": 2,
                       "stage": " Reach   Ring " if i < 40 else "place ring"}
                      for i, (frame, (x, y)) in enumerate(zip(self.frame_numbers, self.centers))]
        write_rows(self.base/"npy"/f"{self.cid}_gaze.csv", list(allocation[0]), allocation[::-1])
        raw = [{"Capture Unix timestamp": int(t), "norm_x": float(x/64),
                "norm_y": float(y/48), "Status": 2}
               for t, (x, y) in zip(self.times, self.centers)]
        write_rows(self.base/"csv"/f"{self.cid}_gaze.csv", list(raw[0]), raw)
        for side in ("l", "r"):
            path = self.base/"sam"/self.cid/"mask"/f"obj_psm_{side}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV mp4v encoder unavailable")
            try:
                for i, (x, y) in enumerate(self.centers.astype(int)):
                    if i == self.deleted:
                        continue
                    image = np.zeros((48, 64, 3), dtype=np.uint8)
                    image[y-2:y+3, x-2:x+3] = 255
                    writer.write(image)
            finally:
                writer.release()

    def args(self, *extra):
        return pipeline.parse_args([
            "--index", str(self.index), "--data-root", str(self.data_root),
            "--output", str(self.root/"output"), "--hz", "10",
            "--smooth-ms", "0", "--max-gap", ".15",
            "--max-lag", ".5", "--min-overlap", "1",
            "--stage-min-overlap", ".3", "--bootstrap", "0",
            "--shuffles", "0", "--min-users", "1", "--no-plots", *extra])

    def test_frame_join_deleted_frame_and_exact_clock(self):
        frames, ns, deleted = pipeline.timeline(self.base, self.cid)
        self.assertEqual(deleted, 1)
        self.assertEqual(int(ns[0]), self.ns_origin)
        self.assertNotIn(int(self.frame_numbers[self.deleted]), frames)
        _, labels, ids = pipeline.stage_frames(self.base, self.cid, frames)
        self.assertEqual(labels[0], "reach ring")
        self.assertEqual(labels[-1], "place ring")
        self.assertEqual(set(ids), {0, 1})
        first_place = np.flatnonzero(labels == "place ring")[0]
        self.assertEqual(frames[first_place], self.frame_numbers[40])

    def test_extraction_scaling_gaps_and_cache_reuse(self):
        args = self.args()
        data, qc = pipeline.extract(self.record, args)
        self.assertEqual(qc["frames"], 79)
        self.assertEqual(qc["deleted_rows"], 1)
        np.testing.assert_allclose(data["gaze_raw_l"], self.centers)
        np.testing.assert_allclose(data["psm_raw_l"][:20], self.centers[:20], atol=.05)
        self.assertTrue(np.isnan(data["psm_raw_l"][20]).all())
        self.assertEqual(data["stage"][40], "place ring")
        self.assertEqual(data["frame_number"][20], self.frame_numbers[21])
        with patch.object(pipeline, "decode_mask", side_effect=AssertionError("cache was not reused")):
            cached, cached_qc = pipeline.extract(self.record, args)
        np.testing.assert_allclose(cached["gaze_l"], data["gaze_l"])
        self.assertEqual(cached_qc, qc)

    def test_pixel_allocation_gaze_uses_frame_join_and_can_flip_y(self):
        args = self.args("--gaze-source", "allocation", "--gaze-units", "pixels", "--flip-gaze-y")
        data, _ = pipeline.extract(self.record, args)
        expected = self.centers.copy()
        expected[:, 1] = 48-expected[:, 1]
        keep = np.arange(80) != self.deleted
        np.testing.assert_allclose(data["gaze_raw_l"][keep], expected[keep])
        self.assertTrue(np.isnan(data["gaze_raw_l"][self.deleted]).all())

    def test_source_and_preprocessing_changes_invalidate_cache_key(self):
        args = self.args()
        before = pipeline.fingerprint(self.record, args)
        args.smooth_ms = 100
        changed = pipeline.fingerprint(self.record, args)
        self.assertNotEqual(changed, before)
        raw_path = self.base/"csv"/f"{self.cid}_gaze.csv"
        with raw_path.open("a") as f:
            f.write("\n")
        self.assertNotEqual(pipeline.fingerprint(self.record, args), changed)

    def test_cached_qc_settings_and_task_metadata_invalidate_key(self):
        args = self.args()
        before = pipeline.fingerprint(self.record, args)
        args.jump_px *= 2
        changed = pipeline.fingerprint(self.record, args)
        self.assertNotEqual(before, changed)
        args.area_ratio *= 2
        changed_again = pipeline.fingerprint(self.record, args)
        self.assertNotEqual(changed, changed_again)
        self.assertNotEqual(changed_again, pipeline.fingerprint({**self.record, "Task Name": "C"}, args))

    def test_scientific_timestamps_and_float_status_restore_full_gaze(self):
        raw_path = self.base/"csv"/f"{self.cid}_gaze.csv"
        rows = pipeline.read_rows(raw_path)
        for row in rows:
            row["Capture Unix timestamp"] = format(Decimal(row["Capture Unix timestamp"]), ".18E")
            row["Status"] = "2.0"
        write_rows(raw_path, list(rows[0]), rows)
        args = self.args()
        data, qc = pipeline.extract(self.record, args)
        np.testing.assert_allclose(data["gaze_raw_l"], self.centers)
        self.assertEqual(qc["accepted_gaze_samples"], len(rows))
        self.assertEqual(qc["l_gaze_valid_fraction"], 1.)

    def test_legacy_cache_reuses_masks_but_recomputes_invalid_gaze(self):
        args = self.args()
        raw_path = self.base/"csv"/f"{self.cid}_gaze.csv"
        rows = pipeline.read_rows(raw_path)
        for row in rows:
            row["Status"] = "2.0"
        write_rows(raw_path, list(rows[0]), rows)
        data, _ = pipeline.extract(self.record, args)
        data["fingerprint"] = np.asarray(pipeline.fingerprint(self.record, args, cache_version=3))
        for side in ("l", "r"):
            data[f"gaze_{side}"][:] = np.nan
            data[f"gaze_raw_{side}"][:] = np.nan
        np.savez_compressed(args.output/"cache"/f"{self.cid}.npz", **data)
        with patch.object(pipeline, "decode_mask", side_effect=AssertionError("native masks should be reused")):
            restored, qc = pipeline.extract(self.record, args)
        np.testing.assert_allclose(restored["gaze_raw_l"], self.centers)
        self.assertTrue(qc["l_native_mask_cache_reused"])
        self.assertEqual(str(restored["fingerprint"]), pipeline.fingerprint(self.record, args))

    def test_all_invalid_gaze_is_an_explicit_error(self):
        raw_path = self.base/"csv"/f"{self.cid}_gaze.csv"
        rows = pipeline.read_rows(raw_path)
        for row in rows:
            row["Status"] = "0.0"
        write_rows(raw_path, list(rows[0]), rows)
        with self.assertRaisesRegex(ValueError, "No valid gaze samples"):
            pipeline.extract(self.record, self.args())

    def test_float_status_in_allocation_is_accepted(self):
        path = self.base/"npy"/f"{self.cid}_gaze.csv"
        rows = pipeline.read_rows(path)
        for row in rows:
            row["valid"] = "2.0"
        write_rows(path, list(rows[0]), rows)
        data, qc = pipeline.extract(self.record, self.args(
            "--gaze-source", "allocation", "--gaze-units", "pixels"))
        keep = np.arange(80) != self.deleted
        np.testing.assert_allclose(data["gaze_raw_l"][keep], self.centers[keep])
        self.assertGreater(qc["l_gaze_valid_fraction"], .9)

    def test_serial_and_parallel_cli_match_including_shuffles(self):
        records = [self.record]
        for user, other, shift in (("User2", "User2_1_1", 7), ("User3", "User3_1_1", 19)):
            for source in self.base.rglob("*"):
                if source.is_file():
                    relative = str(source.relative_to(self.base)).replace(self.cid, other)
                    destination = self.data_root/user/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            gaze_path = self.data_root/user/"csv"/f"{other}_gaze.csv"
            rows = pipeline.read_rows(gaze_path)
            shifted = np.roll(self.centers, shift, axis=0)
            for row, (x, y) in zip(rows, shifted):
                row["norm_x"], row["norm_y"] = x/64, y/48
            write_rows(gaze_path, list(rows[0]), rows)
            records.append({**self.record, "content_id": other})
        write_rows(self.index, list(self.record), records)
        common = ["--index", str(self.index), "--data-root", str(self.data_root),
                  "--hz", "10", "--smooth-ms", "0", "--max-gap", ".15",
                  "--max-lag", ".5", "--min-overlap", "1", "--stage-min-overlap", ".3",
                  "--bootstrap", "5", "--shuffles", "2", "--min-users", "2", "--no-plots"]
        results = []
        for workers in (1, 2):
            output = self.root/f"workers_{workers}"
            with contextlib.redirect_stdout(io.StringIO()):
                code = pipeline.main([*common, "--workers", str(workers), "--output", str(output)])
            self.assertEqual(code, 0)
            results.append(json.loads((output/"results.json").read_text()))
        self.assertEqual(results[0], results[1])

    def test_donor_uses_elapsed_time_missing_padding_and_resolution_scaling(self):
        args = self.args()
        data, _ = pipeline.extract(self.record, args)
        donor = {}
        for side in ("l", "r"):
            donor[f"gaze_{side}"] = data[f"gaze_{side}"][:30]/2
            donor[f"resolution_{side}"] = data[f"resolution_{side}"]/2
        args.signals, args.stages = ["position"], False
        with patch.object(pipeline, "analyze_curve", side_effect=lambda *a, **kw: {}) as mock:
            pipeline.calculate(self.record, data, args, donor=donor)
        self.assertEqual(mock.call_count, 2)
        for side, call in zip(("l", "r"), mock.call_args_list):
            gaze, psm = call.args[:2]
            np.testing.assert_allclose(gaze[:30], data[f"gaze_{side}"][:30])
            self.assertTrue(np.isnan(gaze[30:]).all())
            np.testing.assert_allclose(psm, data[f"psm_{side}"])
            self.assertFalse(call.kwargs["compute_spatial"])

    def test_cli_produces_json_without_plots_or_resampling(self):
        args = self.args()
        argv = ["--index", str(args.index), "--data-root", str(args.data_root),
                "--output", str(args.output), "--hz", "10", "--smooth-ms", "0",
                "--max-gap", ".15", "--max-lag", ".5", "--min-overlap", "1",
                "--stage-min-overlap", ".3", "--bootstrap", "0", "--shuffles", "0",
                "--min-users", "1", "--no-plots"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(argv)
        self.assertEqual(code, 0)
        results = json.loads((args.output/"results.json").read_text())
        self.assertTrue(results["record_curves"])
        self.assertTrue(results["user_curves"])
        self.assertTrue(results["group_curves"])
        whole = next(c for c in results["group_curves"]
                     if c["side"] == "l" and c["signal"] == "position" and c["mode"] == "whole")
        self.assertAlmostEqual(whole["peak_lag_s"], 0.)
        self.assertEqual(whole["n_stable_users"], 1)
        self.assertAlmostEqual(whole["spatial_at_zero_px"], 0., places=2)


if __name__ == "__main__":
    unittest.main()
