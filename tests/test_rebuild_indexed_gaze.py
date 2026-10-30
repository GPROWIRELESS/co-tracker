import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/rebuild_indexed_gaze.py'
spec = importlib.util.spec_from_file_location('rebuild_gaze', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DTYPE = np.dtype([('Frame_Number', '<i4'), ('Timestamp(ns)', '<i8'),
                  ('gaze_x', '<f4'), ('gaze_y', '<f4'), ('valid', 'u1')])


class RebuildGazeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / 'dvss'
        self.npy = self.data / 'Active_Data/User1/npy/sample_gaze.npy'
        self.npy.parent.mkdir(parents=True)
        np.save(self.npy, np.zeros(1, dtype=DTYPE))
        self.csv = self.npy.with_suffix('.csv')
        self.csv.write_text('Frame_Number,Timestamp(ns),gaze_x,gaze_y,valid,stage\n'
                            '1,1750428062481235001,0.75,,3,none\n'
                            '2,1.75043E+18,0.25,0.5,0,none\n')
        self.index = self.base / 'index'
        self.index.mkdir()
        reference = {'gaze_npy_path': '/dvss/Active_Data/User1/npy/sample_gaze.npy'}
        (self.index / 'train.json').write_text(json.dumps([reference, reference]))
        self.manifest = self.base / 'manifest.json'

    def run_script(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--data-root', str(self.data),
                               '--index-dirs', str(self.index), '--manifest', str(self.manifest),
                               *extra], capture_output=True, text=True)

    def test_precision_missing_coordinates_and_overwrite(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        array = np.load(self.npy, allow_pickle=False)
        self.assertEqual(array.dtype, DTYPE)
        self.assertEqual(array['Timestamp(ns)'].tolist(), [1750428062481235001, 1750430000000000000])
        self.assertAlmostEqual(array['gaze_x'][0], 0.75)
        self.assertTrue(np.isnan(array['gaze_y'][0]))
        self.assertEqual(len(json.loads(self.manifest.read_text())['files']), 2)

    def test_dry_run_leaves_files_untouched(self):
        original = self.npy.read_bytes()
        result = self.run_script('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.npy.read_bytes(), original)
        self.assertFalse(self.manifest.exists())

    def test_invalid_integer_aborts_before_overwrite(self):
        original = self.npy.read_bytes()
        self.csv.write_text(self.csv.read_text().replace(',3,none', ',256,none'))
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.npy.read_bytes(), original)
        self.assertFalse(self.manifest.exists())

    @unittest.skipUnless(os.name == 'nt', 'Requires Windows robocopy')
    def test_upload_exact_selection_and_listing_only(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.npy.parent / 'unselected.csv').write_text('do not upload')
        destination = self.base / 'NAS copy'
        command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                   str(ROOT / 'tools/upload_indexed_gaze.ps1'), '-Manifest', str(self.manifest),
                   '-DestinationRoot', str(destination)]
        preview = subprocess.run(command + ['-DryRun'], capture_output=True)
        self.assertEqual(preview.returncode, 0, repr(preview.stderr))
        self.assertFalse(destination.exists())
        copied = subprocess.run(command, capture_output=True)
        self.assertEqual(copied.returncode, 0, repr(copied.stderr))
        files = sorted(p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file())
        self.assertEqual(files, ['Active_Data/User1/npy/sample_gaze.csv',
                                 'Active_Data/User1/npy/sample_gaze.npy'])
        self.assertEqual((destination / files[1]).read_bytes(), self.npy.read_bytes())


if __name__ == '__main__':
    unittest.main()
