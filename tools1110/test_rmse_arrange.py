"""Integration checks for row alignment, target selection and frame pooling."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rmse_arrange_common import (
    Group, Record, arrange, load_record, read_index, target_coordinates,
)


def source_array(n):
    fields = [("gaze_x", "f8"), ("gaze_y", "f8"), ("valid", "u1"), ("stage", "U40")]
    for i in range(1, 6):
        fields += [(f"point_{i}_x", "f8"), (f"point_{i}_y", "f8")]
    fields += [("purple_string_knot_center", "U60")]
    data = np.zeros(n, dtype=fields)
    for i in range(1, 6):
        data[f"point_{i}_x"] = np.nan
        data[f"point_{i}_y"] = np.nan
    data["gaze_x"], data["gaze_y"] = 0.5, 0.5
    return data


class ArrangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, cid, task, data, dlc_rows):
        gaze = self.root / "Active_Data" / "User1" / "npy" / f"{cid}_gaze.npy"
        gaze.parent.mkdir(parents=True, exist_ok=True)
        np.save(gaze, data)
        dlc = gaze.parent.parent / "sam" / cid / f"{cid}_DLC.csv"
        dlc.parent.mkdir(parents=True, exist_ok=True)
        with dlc.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["scorer"] + ["model"] * 6)
            writer.writerow(["bodyparts"] + ["left_psm"] * 3 + ["right_psm"] * 3)
            writer.writerow(["coords"] + ["x", "y", "likelihood"] * 2)
            for i, row in enumerate(dlc_rows):
                writer.writerow([1000 + i] + row)  # Intentionally unrelated frame IDs.
        return dict(user=1, record_id=cid, task_name=task,
                    gaze_npy_path=str(gaze), dlc_csv_path=str(dlc))

    def test_target_filters_points_before_mean_and_center_overrides_per_frame(self):
        data = source_array(3)
        data["point_1_x"], data["point_1_y"] = 500, 512
        data["point_2_x"], data["point_2_y"] = 700, 512
        data["point_3_x"], data["point_3_y"] = -100, 512
        data["purple_string_knot_center"][0] = "[640, 612]"
        for i in (1, 2):
            data[f"point_{i}_x"][2] = 1280  # No strictly inside point in row 2.
        target, audit = target_coordinates(data)
        np.testing.assert_allclose(target[:2], [[640, 612], [600, 512]])
        self.assertTrue(np.isnan(target[2]).all())
        self.assertEqual(audit["target_knot_center_frames"], 1)

    def test_normalized_target_fields_are_scaled_before_filter_and_mean(self):
        data = source_array(2)
        data["point_1_x"], data["point_1_y"] = 0.25, 0.5
        data["point_2_x"], data["point_2_y"] = 0.75, 0.5
        data["purple_string_knot_center"][0] = "[0.5, 0.75]"
        target, _ = target_coordinates(data, "normalized", "normalized")
        np.testing.assert_allclose(target, [[640, 768], [640, 512]])

    def test_pair_masks_are_independent_and_no_valid_code_filter_or_smoothing(self):
        data = source_array(5)
        data["gaze_x"][3] = 0
        data["stage"] = ["grab ring", "none", "transport ring", "drop ring", "grab ring"]
        data["valid"] = 0
        data["point_1_x"], data["point_1_y"] = 640, 512
        data["point_1_x"][4] = np.nan
        dlc = [[643, 516, 0.2, 640, 512, 1] for _ in range(5)]
        dlc[0][3] = 1280
        dlc[1][2] = 0.199
        entry = self.fixture("User1_1_1", "A", data, dlc)
        rows, _ = arrange([entry], "record")
        row = rows[0]
        self.assertEqual(row["n_frames"], 5)
        self.assertEqual(row["gaze_psm_l_n_pairs"], 3)
        self.assertEqual(row["gaze_psm_r_n_pairs"], 3)
        self.assertEqual(row["gaze_target_n_pairs"], 3)
        self.assertAlmostEqual(row["gaze_psm_l_rmse_px"], 5 / math.sqrt(2))
        self.assertEqual(row["gaze_psm_r_rmse_px"], 0)
        self.assertEqual(row["gaze_target_rmse_px"], 0)
        stage_rows, _ = arrange([entry], "user_stage")
        by_stage = {r["stage"]: r for r in stage_rows}
        self.assertEqual(set(by_stage), {"grab ring", "transfer ring"})
        self.assertEqual(by_stage["grab ring"]["n_frames"], 2)
        self.assertEqual(by_stage["grab ring"]["gaze_target_n_pairs"], 1)

    def test_all_four_levels_pool_actual_frames_and_merge_repeated_stages(self):
        a, b = source_array(3), source_array(1)
        a["stage"] = ["grab ring", "none", "grab ring"]
        b["stage"] = "grab ring"
        for data in (a, b):
            data["point_1_x"], data["point_1_y"] = 640, 512
        entries = [
            self.fixture("User1_1_1", "A", a, [[640, 512, 1, 640, 512, 1]] * 3),
            self.fixture("User1_1_2", "A", b, [[646, 520, 1, 646, 520, 1]]),
        ]
        records, _ = arrange(entries, "record")
        self.assertEqual(len(records), 2)
        task, _ = arrange(entries, "task")
        self.assertEqual(len(task), 1)
        self.assertAlmostEqual(task[0]["gaze_psm_l_rmse_px"], 5 / math.sqrt(2))
        self.assertEqual(task[0]["gaze_psm_l_n_pairs"], 4)
        stage, _ = arrange(entries, "user_stage")
        self.assertEqual(len(stage), 2)
        all_stage, _ = arrange(entries, "all_stage")
        self.assertEqual(len(all_stage), 1)
        self.assertAlmostEqual(all_stage[0]["gaze_psm_l_rmse_px"], math.sqrt(100 / 6))
        self.assertEqual(all_stage[0]["n_frames"], 3)

    def test_row_mismatch_is_rejected_before_filtering(self):
        entry = self.fixture("User1_1_1", "A", source_array(2), [[640, 512, 1, 640, 512, 1]])
        with self.assertRaisesRegex(ValueError, "Row mismatch"):
            load_record(entry)

    def test_unknown_stage_blocks_stage_analysis_but_record_uses_all_frames(self):
        data = source_array(1)
        data["stage"] = "unconfirmed new label"
        entry = self.fixture("User1_1_1", "A", data, [[640, 512, 1, 640, 512, 1]])
        with self.assertRaisesRegex(ValueError, "Unrecognized stage"):
            arrange([entry], "user_stage")
        rows, _ = arrange([entry], "record")
        self.assertEqual(rows[0]["n_frames"], 1)
        self.assertTrue(math.isnan(rows[0]["gaze_target_rmse_px"]))

    def test_index_uses_task_name_and_excludes_b_d(self):
        index = self.root / "index.json"
        clips = {}
        for i, task in enumerate(("A", "B", "C", "D")):
            clips[str(i)] = dict(gaze_npy_path=f"User1/npy/User1_{i}_1_gaze.npy",
                                 meta=dict(user=1, task_name=task, content_id=f"User1_{i}_1"))
        index.write_text(json.dumps(dict(clips=clips)), encoding="utf-8")
        entries, skipped = read_index(index, self.root)
        self.assertEqual([e["task_name"] for e in entries], ["A", "C"])
        self.assertEqual(len(skipped), 2)


if __name__ == "__main__":
    unittest.main()
