"""Check averaging both coordinate dimensions and pooled-frame aggregation."""

import math
import unittest

import numpy as np

from RMSE import rmse


class RMSETests(unittest.TestCase):
    def test_single_frame_averages_both_coordinate_dimensions(self):
        self.assertAlmostEqual(rmse([[0, 0]], [[3, 4]]), 5.0 / math.sqrt(2))

    def test_pooling_frames_is_not_averaging_record_rmse(self):
        gaze = np.zeros((3, 2))
        reference = np.array([[0, 0], [0, 0], [6, 8]])
        self.assertAlmostEqual(rmse(gaze, reference), math.sqrt(100 / 6))
        self.assertNotAlmostEqual(rmse(gaze, reference), 10 / (3 * math.sqrt(2)))

    def test_invalid_values_must_be_filtered_explicitly(self):
        with self.assertRaises(ValueError):
            rmse([[float("nan"), 1]], [[2, 3]])

    def test_shape_mismatch_fails_instead_of_broadcasting(self):
        with self.assertRaises(ValueError):
            rmse([[1, 2], [3, 4]], [[1, 2]])

    def test_empty_pairs_are_unavailable(self):
        self.assertTrue(math.isnan(rmse(np.empty((0, 2)), np.empty((0, 2)))))


if __name__ == "__main__":
    unittest.main()
