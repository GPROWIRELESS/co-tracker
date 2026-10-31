"""Frame synchronization, mask identity, and navigation for the review player."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import play_gaze_psm as player


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PlayerDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/"Active_Data/User1"
        self.base.mkdir(parents=True)
        self.cid = "User1_1_1"
        self.record = {"content_id": self.cid, "Task Name": "A", "src_skill": "novice"}
        self.index = self.root/"index.csv"
        write_rows(self.index, [self.record])
        self.args = player.parse_args(["--index", str(self.index), "--data-root", str(self.root),
                                       "--max-gap", ".15"])
        self.origin = 1_750_427_084_986_753_001
        self.labels = ["none", "reach", "reach", "grab", "none", "reach", "reach", "place"]
        self.ns = np.arange(8, dtype=np.int64)*100_000_000+self.origin
        write_rows(self.base/"csv"/f"{self.cid}_video_ts.csv", [
            {"Frame_Number": 9000+i, "Timestamp(ns)": int(ns), "IsDelete": int(i == 2)}
            for i, ns in enumerate(self.ns)])
        write_rows(self.base/"npy"/f"{self.cid}_gaze.csv", [
            {"Frame_Number": 9000+i, "stage": label, "gaze_x": .2+i*.01,
             "gaze_y": .3, "valid": 0 if i == 4 else 2,
             "Timestamp(ns)": "1.75043E+18"}  # Rounded allocation time is unusable.
            for i, label in reversed(list(enumerate(self.labels)))])
        write_rows(self.base/"csv"/f"{self.cid}_gaze.csv", [
            {"Capture Unix timestamp": int(ns), "norm_x": .2+i*.01,
             "norm_y": .3, "Status": "2.0"}
            for i, ns in enumerate(self.ns)])
        self.video(self.base/"stimuli"/f"{self.cid}.mp4", range(8))
        for side in ("l", "r"):
            self.video(self.base/"sam"/self.cid/"mask"/f"obj_psm_{side}.mp4",
                       [i for i in range(8) if i != 2], side)

    def video(self, path, indices, side=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
        if not writer.isOpened():
            self.skipTest("OpenCV mp4v encoder unavailable")
        try:
            for i in indices:
                frame = np.full((48, 64, 3), i*20 if side is None else 0, dtype=np.uint8)
                if side is not None:
                    x = 10+i if side == "l" else 40+i
                    frame[18:23, x-2:x+3] = 255
                writer.write(frame)
        finally:
            writer.release()

    def test_deleted_rgb_frames_and_source_mask_identity(self):
        record = player.Recording(self.record, self.args)
        self.addCleanup(record.close)
        self.assertEqual(record.frames.tolist(), [9000, 9001, 9003, 9004, 9005, 9006, 9007])
        self.assertEqual(record.video_indices.tolist(), [0, 1, 3, 4, 5, 6, 7])
        np.testing.assert_allclose(record.gaze[:, 0], np.array([.2, .21, .23, .24, .25, .26, .27])*64)
        self.assertEqual(record.starts, [0, 1, 2, 3, 4, 6])
        for index in (0, 1, 2, 6, 3, 2):  # Sequential, forward seek, and backward seek.
            rgb, masks = record.read(index, self.args)
            original = int(record.video_indices[index])
            self.assertLess(abs(float(rgb.mean())-original*20), 6)
            np.testing.assert_allclose(masks["l"][0], [10+original, 20], atol=.01)
            np.testing.assert_allclose(masks["r"][0], [40+original, 20], atol=.01)

    def test_allocation_matches_frame_ids_and_keeps_invalid_frame_missing(self):
        self.args.gaze_source = "allocation"
        record = player.Recording(self.record, self.args)
        self.addCleanup(record.close)
        np.testing.assert_allclose(record.gaze[2], [.23*64, .3*48])
        self.assertTrue(np.isnan(record.gaze[3]).all())

    def test_raw_gaze_interpolates_at_capture_time_and_omits_long_gaps(self):
        rows = [{"Capture Unix timestamp": int(self.ns[0]), "norm_x": .2, "norm_y": .4, "Status": 2},
                {"Capture Unix timestamp": int(self.ns[2]), "norm_x": .4, "norm_y": .6, "Status": 2}]
        write_rows(self.base/"csv"/f"{self.cid}_gaze.csv", rows)
        self.args.max_gap = .25
        gaze = player.align_gaze(self.base, self.cid, [], self.ns, 64, 48, self.args)
        np.testing.assert_allclose(gaze[1], [.3*64, .5*48])
        self.assertTrue(np.isnan(gaze[3:]).all())
        self.args.max_gap = .1
        gaze = player.align_gaze(self.base, self.cid, [], self.ns, 64, 48, self.args)
        self.assertTrue(np.isnan(gaze[1]).all())
        np.testing.assert_allclose(gaze[2], [.4*64, .6*48])

    def test_mismatched_masks_fail_instead_of_drawing_wrong_frame(self):
        path = self.base/"sam"/self.cid/"mask/obj_psm_r.mp4"
        self.video(path, range(3), "r")
        with self.assertRaisesRegex(ValueError, "r mask has 3 frames"):
            player.Recording(self.record, self.args)

    def test_direct_record_selection_overrides_task_and_skill_filters(self):
        self.args.records, self.args.tasks, self.args.skill = [self.cid], ["C"], "intermediate"
        self.assertEqual(player.select_records(self.args), [self.record])

    @unittest.skipUnless(os.environ.get("PLAYER_GUI_TESTS"), "Set PLAYER_GUI_TESTS=1 under a desktop or Xvfb")
    def test_gui_shortcuts_focused_controls_stage_crossing_and_playback(self):
        app = player.Player(self.args, [self.record])
        self.addCleanup(app.close)
        app.window.update()
        for widget, key, expected in [
            (app.slider, "Right", 1),
            (app.stage_box, "Right", 2),
            (app.record_box, "Left", 1),
            (app.canvas, "greater", 2),
            (app.slider, "less", 1),
            (app.canvas, "period", 2),
            (app.canvas, "comma", 1),
        ]:
            with self.subTest(widget=widget, key=key):
                widget.focus_force()
                widget.event_generate("<KeyPress>", keysym=key)
                app.window.update()
                self.assertEqual(app.frame, expected)
                self.assertFalse(app.playing)
                self.assertEqual(app.stage_box.current(), player.bisect_right(app.recording.starts, expected)-1)
        app.seek(4)
        self.assertEqual(app.stage_box.current(), 4)  # Second reach occurrence.
        app.move_stage(-1)
        self.assertEqual(app.frame, 3)  # Intervening none stage remains navigable.
        app.slider.set(5)
        app.window.update()
        self.assertEqual(app.frame, 5)
        app.toggle_play()
        app.window.update()
        self.assertTrue(app.playing)
        with patch.object(app.window, "after", return_value=None):
            app.tick()
        self.assertEqual(app.frame, 6)
        self.assertFalse(app.playing)
        self.assertEqual(app.stage_box.current(), 5)
        app.show_masks.set(True)
        app.render()
        self.assertTrue(app.canvas.find_all())


class NavigationTests(unittest.TestCase):
    def test_repeated_stages_none_and_recording_boundaries(self):
        labels = np.array(["none", "reach", "reach", "grab", "none", "reach"])
        segments = player.stage_segments(labels)
        self.assertEqual(segments, [(0, 1, "none"), (1, 3, "reach"), (3, 4, "grab"),
                                    (4, 5, "none"), (5, 6, "reach")])
        starts = [s[0] for s in segments]
        self.assertEqual(player.stage_target(starts, 2, 1), 3)
        self.assertEqual(player.stage_target(starts, 2, -1), 0)
        self.assertEqual(player.stage_target(starts, 0, -1), 0)
        self.assertEqual(player.stage_target(starts, 5, 1), 5)

    def test_empty_and_tiny_masks_have_no_center(self):
        mask = np.zeros((20, 20, 3), dtype=np.uint8)
        self.assertIsNone(player.centroid(mask)[0])
        mask[0, 0] = 255
        self.assertIsNone(player.centroid(mask)[0])
        mask[5:10, 5:10] = 255
        point, binary = player.centroid(mask)
        expected_y, expected_x = np.nonzero(binary)
        np.testing.assert_allclose(point, [expected_x.mean(), expected_y.mean()])


if __name__ == "__main__":
    unittest.main()
