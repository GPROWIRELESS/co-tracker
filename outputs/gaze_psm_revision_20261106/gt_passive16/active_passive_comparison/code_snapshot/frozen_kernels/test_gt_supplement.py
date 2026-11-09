"""Independent checks of exported bootstrap draws and spatial pooling support."""
from gaze_psm_modes import normalize_modes
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from gaze_psm_statistics_v2 import group_curves
from gt_supplement import supplement
from run_gt84_revision import spatial_and_boundaries


class SupplementTests(unittest.TestCase):
    def test_exported_bootstrap_draws_exactly_match_group_ranges(self):
        curves = []
        lags = np.linspace(-2, 2, 9)
        for j in range(5):
            for repeat in range(1 + (j == 0)):
                for stage, shift in [("reach", .2), ("place", -.2)]:
                    r = .3 + .2*np.exp(-(lags-shift-(j-2)*.3)**2)
                    curves.append(dict(variant="primary", source="gt", task="A", side="l", signal="position",
                                       stage=stage, mode="PSM in stage", content_id=f"U{j}_{repeat}", user=f"U{j}",
                                       lag_s=lags, r=r))
        # A partial recording must not sneak into either exported estimator.
        curves.append(dict(curves[0], content_id="partial", r=np.array([np.nan]+[.9]*8)))
        groups, _ = group_curves(curves, bootstrap=321, seed=42)
        with tempfile.TemporaryDirectory(prefix="supplement_test_") as td:
            supplement(curves, groups, [], td, count=321)
            index = normalize_modes(json.loads((Path(td)/"group_bootstrap_index.json").read_text()))
            with np.load(Path(td)/"group_bootstrap_peak_lags.npz") as z:
                for group in groups:
                    item = next(i for i in index if i["stage"] == group["stage"])
                    draws = z[item["token"]]
                    low, high = np.percentile(draws, [2.5, 97.5])
                    self.assertEqual(low, group["peak_lag_resample_low"])
                    self.assertEqual(high, group["peak_lag_resample_high"])
                    boundary = np.mean((draws == lags[0]) | (draws == lags[-1]))
                    self.assertEqual(boundary, group["bootstrap_boundary_frequency"])

    def test_spatial_record_medians_then_users_with_same_complete_support(self):
        curves, prepared, records = [], {}, []
        for cid, user, offset, partial in [("a1", "A", 100., False), ("a2", "A", 300., False),
                                           ("b", "B", 10., False), ("c", "C", 20., False),
                                           ("bad", "B", 999., True)]:
            psm = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
            g = psm + [offset, 0.]
            prepared[cid] = {"signals": {"position": {"l": dict(psm=psm, gazes={"gt": g},
                            stage=np.repeat("reach", 8), occurrence=np.zeros(8, dtype=int))}}}
            records.append(dict(content_id=cid, user=user, task="A"))
            curves.append(dict(variant="primary", source="gt", content_id=cid, user=user, task="A", side="l",
                               signal="position", stage="reach", mode="PSM in stage", lag_s=np.array([-1., 0., 1.]),
                               r=np.array([np.nan if partial else .1, .3, .1])))
        groups, _ = group_curves(curves, bootstrap=0)
        rr, ss = spatial_and_boundaries(prepared, records, curves, groups)
        self.assertEqual({r["content_id"] for r in rr}, {"a1", "a2", "b", "c"})
        expected = (200 + 10 + 20)/3
        self.assertEqual(len(ss), 2)
        for row in ss:
            self.assertEqual(row["n_users"], 3)
            self.assertAlmostEqual(row["distance_median_px"], expected)
            self.assertEqual(row["cross_boundary_fraction"], 0.)
            self.assertAlmostEqual(row["proximity_50px"], 2/3)


if __name__ == "__main__":
    unittest.main()
