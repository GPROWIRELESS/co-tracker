"""DLC input checks only: no metric calculations or analysis outputs."""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools import gaze_psm_pipeline_dlc as pipeline


ROOT = Path(__file__).resolve().parents[1]


class DLCInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"clip_DLC.csv"
        # Deliberately reorder columns to verify bodypart/coordinate lookup.
        self.headers = [
            ["scorer", *(["model"]*6)],
            ["bodyparts", "right_psm", "left_psm", "left_psm", "right_psm", "left_psm", "right_psm"],
            ["coords", "likelihood", "y", "likelihood", "x", "x", "y"],
        ]
        self.rows = [[0, .01, 20, .02, 100, 10, 200],
                     [1, .99, 21, .98, 101, 11, 201],
                     [2, .5, 22, .5, 102, 12, 202]]

    def write(self):
        with self.path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(self.headers+self.rows)

    def test_separate_coordinates_and_low_confidence_are_preserved(self):
        self.write()
        points, confidence = pipeline.read_dlc(self.path, expected_frames=3)
        np.testing.assert_array_equal(points["l"], [[10, 20], [11, 21], [12, 22]])
        np.testing.assert_array_equal(points["r"], [[100, 200], [101, 201], [102, 202]])
        np.testing.assert_array_equal(confidence["l"], [.02, .98, .5])
        np.testing.assert_array_equal(confidence["r"], [.01, .99, .5])

    def test_missing_coordinate_invalidates_only_its_instrument(self):
        self.rows[1][5] = "nan"
        self.write()
        points, _ = pipeline.read_dlc(self.path, expected_frames=3)
        self.assertTrue(np.isnan(points["l"][1]).all())
        np.testing.assert_array_equal(points["r"][1], [101, 201])

    def test_partial_read_still_checks_full_record_alignment(self):
        self.write()
        points, _ = pipeline.read_dlc(self.path, expected_frames=3, limit=2)
        self.assertEqual(points["l"].shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "3 DLC rows vs 4 retained timestamps"):
            pipeline.read_dlc(self.path, expected_frames=4, limit=2)

    def test_missing_duplicate_and_offset_frame_indices_are_rejected(self):
        for frame in (0, 2, 9276):
            with self.subTest(frame=frame):
                self.rows[1][0] = frame
                self.write()
                with self.assertRaisesRegex(ValueError, "consecutive from zero"):
                    pipeline.read_dlc(self.path)

    def test_missing_bodypart_and_ambiguous_columns_are_rejected(self):
        for replacement in ("other", "right_psm"):
            with self.subTest(replacement=replacement):
                self.headers[1][5] = replacement
                self.write()
                with self.assertRaises(ValueError):
                    pipeline.read_dlc(self.path)

    def test_video_metadata_never_decodes_frames(self):
        cap = Mock()
        cap.get.side_effect = [1280, 1024, 40.9]
        with patch.object(pipeline.cv2, "VideoCapture", return_value=cap):
            self.assertEqual(pipeline.video_metadata("source.mp4"), (1280, 1024, 40.9))
        cap.read.assert_not_called()
        cap.release.assert_called_once()

    def test_source_files_and_defaults_use_dlc(self):
        args = pipeline.parse_args([])
        self.assertEqual(args.tasks, ["A", "C"])
        self.assertEqual(args.skill, "novice")
        self.assertEqual(args.output, Path("outputs/gaze_psm_xcorr_dlc"))
        files = pipeline.source_files({"content_id": "User1_1_1"}, args)
        base = args.data_root/"User1"
        self.assertIn(base/"sam"/"User1_1_1"/"User1_1_1_DLC.csv", files)
        self.assertIn(base/"stimuli"/"User1_1_1.mp4", files)
        self.assertFalse(any("mask" in path.parts for path in files))


class UnchangedAnalysisTests(unittest.TestCase):
    def test_significant_result_metadata_serializes_numpy_counts(self):
        from tools.analyze_raw_score_corr_dlc import json_value
        metadata = {"significant_results": [{"n_users": np.int64(6),
                                             "association": np.float64(.5),
                                             "eligible": np.bool_(True)}]}
        decoded = json.loads(json.dumps(metadata, default=json_value))
        self.assertEqual(decoded, {"significant_results": [{"n_users": 6,
                                                            "association": .5,
                                                            "eligible": True}]})

    def test_analysis_function_bodies_match_without_executing_metrics(self):
        expected = {
            "gaze_psm_pipeline": ("interpolate_valid", "smooth_valid", "curve_inputs", "calculate",
                                  "_init_analysis_worker", "_observed_job", "_null_job"),
            "gaze_psm_xcorr_max": ("select_sides", "_selected_null_job", "_pairings"),
            "analyze_score_corr": ("between_test", "within_test", "prepare", "scenario_rows", "analyze"),
            "analyze_score_corr_max": ("selection_lookup", "attach_selection", "validate_scenarios"),
            "analyze_raw_score_corr": ("validate_results",),
        }
        for stem, names in expected.items():
            original, copied = [
                {node.name: ast.dump(node, include_attributes=False)
                 for node in ast.parse((ROOT/"tools"/f"{stem}{suffix}.py").read_text()).body
                 if isinstance(node, ast.FunctionDef)}
                for suffix in ("", "_dlc")
            ]
            for name in names:
                with self.subTest(script=stem, function=name):
                    self.assertEqual(original[name], copied[name])


if __name__ == "__main__":
    unittest.main()
