"""Regression tests for coordinate-only gaze selection and native-frame matching."""
import sys
import shutil
from contextlib import contextmanager
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import tl_inputs as inputs
import analyze_target_locking as analysis


@contextmanager
def fixture_directory():
    # Ordinary inherited directory permissions also work under restricted Windows tokens.
    parent = Path(__file__).resolve().parent
    path = parent / ('_gaze_fixture_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().parent == parent and path.name.startswith('_gaze_fixture_')
        shutil.rmtree(path)


class CoordinateOnlyTests(unittest.TestCase):
    def load(self, root, xy, codes, frames=None):
        n = len(xy)
        frames = np.arange(100, 100+n) if frames is None else np.asarray(frames)
        source = np.zeros(n, dtype=[('Frame_Number','i8'), ('Timestamp(ns)','i8'),
            ('gaze_x','f8'), ('gaze_y','f8'), ('valid','f8'), ('stage','U24'),
            ('point_1_x','f8'), ('point_1_y','f8')])
        source['Frame_Number'] = frames
        source['Timestamp(ns)'] = np.arange(n)*24_000_000+1_000_000_000
        source['gaze_x'], source['gaze_y'] = np.asarray(xy).T
        source['valid'] = codes
        source['stage'] = 'reach ring'
        source['point_1_x'], source['point_1_y'] = 50, 40
        base = root/'Active_Data'/'User1'
        (base/'npy').mkdir(parents=True, exist_ok=True)
        path = base/'npy'/'User1_1_1_gaze.npy'
        # Physical row order differs from native frame order deliberately.
        np.save(path, source[::-1])
        pipeline = SimpleNamespace(
            dlc_path=lambda *a: root/'dlc.csv',
            timeline=lambda *a: (frames, source['Timestamp(ns)'], 0),
            read_dlc=lambda *a, **kw: ({'l':np.full((n,2),10.), 'r':np.full((n,2),20.)},
                                     {'l':np.ones(n), 'r':np.ones(n)}),
            video_metadata=lambda *a: (100,80,41.),
            normalized_stage=lambda value: str(value).strip().lower())
        with patch.object(inputs,'_pipeline',return_value=pipeline), \
             patch.object(inputs,'_sha256',return_value='fixture'), \
             patch.object(inputs,'_raw_state_audit',return_value=({},{})):
            return inputs.read_record(dict(content_id='User1_1_1',participant='User1',task='A'),root,root)

    def test_arbitrary_valid_values_and_inclusive_boundaries_are_accepted(self):
        xy = [[0,0], [1,1], [.2,.3], [.4,.5], [.6,.7], [.8,.9], [0,1]]
        with fixture_directory() as tmp:
            d = self.load(Path(tmp), xy, [0,-1,4,np.nan,2,3,999])
        np.testing.assert_array_equal(d['gaze_valid'], np.ones(7,dtype=bool))
        np.testing.assert_allclose(d['gaze'], np.array(xy)*[100,80])
        self.assertEqual(d['audit']['gaze_valid_n'], 7)

    def test_nonfinite_and_out_of_range_coordinates_are_rejected_for_every_code(self):
        xy = [[np.nan,.5], [.5,np.inf], [-.0001,.5], [.5,1.0001], [.5,.5]]
        with fixture_directory() as tmp:
            d = self.load(Path(tmp), xy, [3,3,2,2,0])
        np.testing.assert_array_equal(d['gaze_valid'], [False,False,False,False,True])
        self.assertTrue(np.isnan(d['gaze'][:4]).all())
        np.testing.assert_allclose(d['gaze'][4], [50,40])

    def test_original_frame_numbers_match_unsorted_npy_and_keep_gaps(self):
        with fixture_directory() as tmp:
            d = self.load(Path(tmp), [[.1,.2],[.3,.4],[.5,.6]], [0,4,3], [100,102,107])
        np.testing.assert_array_equal(d['frames'], [100,102,107])
        np.testing.assert_allclose(d['gaze'], [[10,16],[30,32],[50,48]])

    def test_missing_or_duplicate_source_frames_raise(self):
        source=np.array([(100,), (102,), (102,)],dtype=[('Frame_Number','i8')])
        for frames in ([100,103], [100,102]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                inputs._take_frame_rows(source,np.array(frames),Path('fixture.npy'))

    def test_all_current_scenarios_ignore_status_and_keep_geometry_conditions(self):
        xy=np.array([[0,0],[100,80],[50,40],[50,40],[50,40],[101,40],[np.nan,40]],float)
        d=dict(width=100,height=80,gaze_unmasked=xy,smooth_xy=xy,
               included=np.array([1,1,0,1,1,1,1],bool),
               count=np.array([1,1,1,0,1,1,1]),
               gripper_ok=np.array([1,1,1,1,0,1,1],bool),
               valid_code=np.array([0,-1,4,np.nan,3,2,3]))
        for scenario, config in analysis.SCENARIOS.items():
            with self.subTest(scenario=scenario):
                self.assertIsNone(config[2])
                _, selected=analysis.support(d,scenario)
                np.testing.assert_array_equal(selected,[True,True,False,False,False,False,False])
        self.assertNotIn('gaze3',analysis.SCENARIOS)
        self.assertNotIn('legacy123',analysis.SCENARIOS)

    def test_preparation_smooths_coordinate_valid_points_without_status_filter(self):
        with fixture_directory() as tmp:
            root=Path(tmp)
            d=self.load(root,[[.1,.2],[.3,.4],[.5,.6]],[0,-1,4])
            args=SimpleNamespace(data_root=root,repo=root,cache_dir=root)
            with patch.object(inputs,'read_record',return_value=d) as loader, \
                 patch.object(analysis.geo,'smooth_gaze',wraps=analysis.geo.smooth_gaze) as smooth:
                prepared=analysis.prepare({'task':'A'},args)
            self.assertIsNone(loader.call_args.kwargs['gaze_allowed_codes'])
            np.testing.assert_allclose(smooth.call_args.args[1],[[10,16],[30,32],[50,48]])
            self.assertTrue(np.isfinite(prepared['smooth_xy']).all())


if __name__ == '__main__':
    unittest.main()
