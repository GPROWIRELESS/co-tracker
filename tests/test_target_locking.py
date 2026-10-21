import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('label_stages', Path(__file__).resolve().parents[1] / 'tools/label_target_locking.py')
label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label)


class LabelStagesTests(unittest.TestCase):
    def test_repeated_events_are_separate_stages(self):
        rows = [{'event': e} for e in ['none', 'reach', 'reach', 'grab', 'none', 'reach']]
        self.assertEqual(label.stages(rows), [(1, 3), (3, 4), (5, 6)])
        self.assertEqual(label.stages(rows, True)[0], (0, 1))

    def test_index_filter_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = [dict(video_path=f'{skill}{task}', domain='active',
                         meta=dict(src_skill=skill, task_name=task))
                    for skill in ['novice', 'intermediate'] for task in ['A', 'B', 'C']]
            (root/'train.json').write_text(json.dumps(data))
            (root/'test.json').write_text(json.dumps(data))
            (root/'summary.json').write_text('{"train": 6}')
            self.assertEqual([x['video_path'] for x in label.load_items([root])], ['noviceA'])

    def test_save_preserves_original_and_first_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'allocation.csv'
            original = 'Frame_Number,event\n9275,reach\n'
            path.write_text(original)
            fields = ['Frame_Number', 'event', 'point_1_x']
            rows = [dict(Frame_Number='9275', event='reach', point_1_x=42)]
            label.save_csv(path, fields, rows)
            rows[0]['point_1_x'] = 43
            label.save_csv(path, fields, rows)
            self.assertEqual(path.with_suffix('.csv.before_point_labels.bak').read_text(), original)
            with path.open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0], dict(Frame_Number='9275', event='reach', point_1_x='43'))

    def test_resume_uses_manual_seed_only(self):
        app = label.Labeler.__new__(label.Labeler)
        app.args = SimpleNamespace(points=1)
        app.segments = [(0, 2)]
        app.rows = [dict(point_label_mode='manual', point_1_x='10', point_1_y='20'), {}]
        self.assertTrue(app.complete(0))
        app.rows[0]['point_label_mode'] = 'tracked'
        self.assertFalse(app.complete(0))

    def test_local_dataset_path_resolution(self):
        root = Path('/local/dvss')
        for path in ['/dvss/Active_Data/User1/stimuli/User1_1_1.mp4',
                     'dvss/Active_Data/User1/stimuli/User1_1_1.mp4',
                     'Active_Data/User1/stimuli/User1_1_1.mp4',
                     '/scratch/old/dvss/Active_Data/User1/stimuli/User1_1_1.mp4']:
            video, allocation = label.record_paths({'video_path': path}, root)
            self.assertEqual(video, root/'Active_Data/User1/stimuli/User1_1_1.mp4')
            self.assertEqual(allocation, root/'Active_Data/User1/sam/User1_1_1/User1_1_1_allocation.csv')

    def test_relabel_clears_only_current_stage_predictions(self):
        rows = [dict(event='reach', point_label_mode='tracked', point_1_x=3, point_1_y=4,
                     point_1_visible=1) for _ in range(4)]
        label.set_manual_seed(rows, 1, 3, [(10, 20)])
        self.assertEqual(rows[1]['point_label_mode'], 'manual')
        self.assertEqual(rows[1]['point_1_x'], 10)
        self.assertEqual(rows[2]['point_1_x'], '')
        self.assertEqual(rows[0]['point_1_x'], 3)
        self.assertEqual(rows[3]['point_1_x'], 3)
        self.assertEqual(rows[2]['event'], 'reach')

    def test_frame_mapping_handles_gaps_and_bounds(self):
        rows = [dict(event='reach', Frame_Number=n) for n in [9275, 9277, 9278]]
        self.assertEqual(label.frame_indices(rows, 4), [0, 2, 3])
        with self.assertRaises(ValueError):
            label.frame_indices(rows, 3)


class TrackingTests(unittest.TestCase):
    def test_propagation_respects_gaps_and_preserves_manual_seed(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        import track_target_locking as tracker
        rows = [dict(event='reach', Frame_Number=n) for n in [100, 102, 103, 104]]
        label.set_manual_seed(rows, 0, 3, [(10, 20)])
        predictions = [[[99, 99]], [[11, 21]], [[12, 22]], [[13, 23]]]
        visibility = [[True], [True], [False], [True]]
        with patch.object(tracker, 'track_stage', return_value=(predictions, visibility)) as run:
            result = tracker.propagate_stage(rows, [0, 2, 3, 4], 0, 3, 1, 'model', 'cap', 'cpu')
        run.assert_called_once_with('model', 'cap', 0, 3, [(10.0, 20.0)], 'cpu', reverse=False)
        self.assertEqual(result[0], rows[0])
        self.assertEqual(result[1]['point_1_x'], 12)
        self.assertNotIn('point_1_visible', result[1])
        self.assertEqual(result[2]['point_1_x'], 13)
        self.assertEqual(result[3], rows[3])
        self.assertEqual(rows[1]['point_1_x'], '')
        self.assertTrue(tracker.is_tracked(result, 0, 3, 1))
        self.assertFalse(tracker.is_tracked(rows, 0, 3, 1))

    def test_last_seed_tracks_backwards_and_maps_frame_gaps(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        import track_target_locking as tracker
        rows = [dict(event='reach', Frame_Number=n) for n in [100, 102, 103, 104]]
        label.set_manual_seed(rows, 0, 3, [(10, 20)], seed=2)
        predictions = [[[99, 99]], [[11, 21]], [[12, 22]], [[13, 23]]]
        with patch.object(tracker, 'track_stage', return_value=(predictions, None)) as run:
            result = tracker.propagate_stage(rows, [0, 2, 3, 4], 0, 3, 1, 'model', 'cap', 'cpu')
        run.assert_called_once_with('model', 'cap', 0, 3, [(10.0, 20.0)], 'cpu', reverse=True)
        self.assertEqual(result[2], rows[2])
        self.assertEqual(result[0]['point_1_x'], 13)
        self.assertEqual(result[1]['point_1_x'], 11)
        self.assertEqual(result[3], rows[3])
        self.assertTrue(tracker.is_tracked(result, 0, 3, 1))
        self.assertEqual(label.seed_index(result, 0, 3, 1), 2)

    def test_save_removes_existing_visibility_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'allocation.csv'
            path.write_text('event,point_1_visible\nreach,1\n')
            rows = [dict(event='reach', point_1_x=10, point_1_y=20, point_1_visible=1)]
            label.save_csv(path, list(rows[0]), rows)
            with path.open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved, [dict(event='reach', point_1_x='10', point_1_y='20')])
            self.assertEqual(rows[0]['point_1_visible'], 1)

    def test_endpoint_switch_replaces_prior_seed(self):
        rows = [dict(event='reach') for _ in range(3)]
        label.set_manual_seed(rows, 0, 3, [(1, 2)])
        label.set_manual_seed(rows, 0, 3, [(3, 4)], seed=2)
        self.assertEqual(label.seed_index(rows, 0, 3, 1), 2)
        self.assertEqual(rows[0]['point_1_x'], '')
        self.assertEqual(rows[2]['point_1_x'], 3)
        rows[0].update(point_label_mode='manual', point_1_x=1, point_1_y=2)
        with self.assertRaises(ValueError):
            label.seed_index(rows, 0, 3, 1)

    def test_variable_point_counts_propagate_with_maximum_five(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        import track_target_locking as tracker
        for count in range(1, 6):
            with self.subTest(count=count):
                rows = [dict(event='reach') for _ in range(3)]
                points = [(i, i+10) for i in range(count)]
                label.set_manual_seed(rows, 0, 3, points, seed=2)
                self.assertTrue(label.has_seed(rows[2], 5))
                predictions = [points, [(x+1, y+1) for x, y in points],
                               [(x+2, y+2) for x, y in points]]
                with patch.object(tracker, 'track_stage', return_value=(predictions, None)):
                    result = tracker.propagate_stage(rows, [0, 1, 2], 0, 3, 5, None, None, 'cpu')
                self.assertTrue(tracker.is_tracked(result, 0, 3, 5))
                self.assertEqual(result[0][f'point_{count}_x'], count+1)
                self.assertNotIn(f'point_{count+1}_x', result[0])

    def test_relabel_with_fewer_points_clears_old_slots(self):
        rows = [dict(event='reach') for _ in range(3)]
        label.set_manual_seed(rows, 0, 3, [(i, i) for i in range(5)])
        label.set_manual_seed(rows, 0, 3, [(20, 30)])
        self.assertEqual(label.seed_points(rows[0]), [(20.0, 30.0)])
        for i in range(2, 6):
            self.assertEqual(rows[0][f'point_{i}_x'], '')
        with self.assertRaises(ValueError):
            label.set_manual_seed(rows, 0, 3, [])
        with self.assertRaises(ValueError):
            label.set_manual_seed(rows, 0, 3, [(i, i) for i in range(6)])
        self.assertFalse(label.has_seed(dict(point_label_mode='manual', point_1_x=1), 5))

    def test_skip_persists_clears_points_and_resumes_without_tracking(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        import track_target_locking as tracker
        rows = [dict(event='reach') for _ in range(4)]
        label.set_manual_seed(rows, 0, 3, [(10, 20)])
        label.set_stage_skipped(rows, 0, 3)
        self.assertTrue(label.is_skipped(rows, 0, 3))
        self.assertTrue(all(r['point_1_x'] == '' for r in rows[:3]))
        self.assertNotIn('point_label_mode', rows[3])
        app = label.Labeler.__new__(label.Labeler)
        app.args = SimpleNamespace(points=5)
        app.segments, app.rows = [(0, 3)], rows
        self.assertTrue(app.complete(0))
        with patch.object(tracker, 'track_stage') as run:
            result = tracker.propagate_stage(rows, [0, 1, 2, 3], 0, 3, 5, None, None, 'cpu')
        run.assert_not_called()
        self.assertEqual(result, rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'allocation.csv'
            path.write_text('event\nreach\n')
            fields = ['event', 'point_label_mode', 'point_1_x', 'point_1_y']
            label.save_csv(path, fields, rows)
            with path.open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertTrue(label.is_skipped(saved, 0, 3))
        label.set_manual_seed(rows, 0, 3, [(30, 40)], seed=2)
        self.assertFalse(label.is_skipped(rows, 0, 3))
        self.assertEqual(label.seed_index(rows, 0, 3, 5), 2)

    def test_stage_type_skip_applies_to_every_occurrence(self):
        events = ['reach', 'reach', 'grab', 'reach', 'transport', 'grab']
        rows = [dict(event=e) for e in events]
        for start, end in label.stages(rows):
            label.set_manual_seed(rows, start, end, [(10, 20)])
        updated = label.skip_selected_stages(rows, label.stages(rows), {'reach', 'transport'})
        for index, event in enumerate(events):
            self.assertEqual(updated[index].get('point_label_mode') == 'skipped',
                             event in {'reach', 'transport'})
        self.assertEqual(updated[2], rows[2])
        self.assertEqual(updated[5], rows[5])
        self.assertEqual(rows[0]['point_label_mode'], 'manual')
        # The same choice applies to the next recording without needing per-stage clicks.
        another = [dict(event='transport'), dict(event='grab')]
        next_rows = label.skip_selected_stages(another, label.stages(another), {'reach', 'transport'})
        self.assertTrue(label.is_skipped(next_rows, 0, 1))
        self.assertNotIn('point_label_mode', next_rows[1])

    def test_checklist_lists_types_across_all_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = []
            for record, events in [('User1_1_1', ['none', 'reach', 'grab']),
                                   ('User1_2_1', ['reach', 'transport'])]:
                item = dict(video_path=f'/dvss/Active_Data/User1/stimuli/{record}.mp4')
                _, allocation = label.record_paths(item, root)
                allocation.parent.mkdir(parents=True)
                allocation.write_text('Frame_Number,event\n' + ''.join(f'{i},{event}\n' for i, event in enumerate(events)))
                items.append(item)
            self.assertEqual(label.stage_types(items, root), ['grab', 'reach', 'transport'])

    def test_missing_event_excluded_without_losing_valid_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = []
            for record, text in [('ready', '\ufeffFrame_Number,event\n0,reach\n'),
                                 ('missing_event', 'Frame_Number,gaze_allocation\n0,ring\n'),
                                 ('empty', 'Frame_Number,event\n')]:
                item = dict(video_path=f'/dvss/Active_Data/User1/stimuli/{record}.mp4')
                _, allocation = label.record_paths(item, root)
                allocation.parent.mkdir(parents=True)
                allocation.write_text(text)
                items.append(item)
            ready, unavailable = label.labelable_items(items, root)
            self.assertEqual(ready, items[:1])
            self.assertEqual(len(unavailable), 2)
            self.assertIn('missing required columns: event', unavailable[0])
            self.assertEqual(label.stage_types(ready, root), ['reach'])
            with self.assertRaisesRegex(ValueError, 'event column'):
                label.stages([{'Frame_Number': '0'}])


if __name__ == '__main__':
    unittest.main()
