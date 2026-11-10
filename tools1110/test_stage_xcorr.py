"""Numerical, stage boundary, weighting and metadata regression tests."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stage_xcorr_common import (
    SETTINGS, STAGE_FIELDS, SIDES, aggregate_stage_rows, canonical_stage,
    compute_stage_record, format_stage_output, label_audit, occurrence_curve,
    occurrence_peak, stage_runs,
)
from xcorr_common import write_csv
from test_xcorr import write_fixture


def direct_occurrence_curve(gaze, psm, start, end):
    correlations, counts = [], []
    for lag in range(-120, 121):
        p_indices = np.arange(start, end)
        g_indices = p_indices-lag
        inside = (g_indices >= 0) & (g_indices < len(gaze))
        g, p = gaze[g_indices[inside]], psm[p_indices[inside]]
        valid = np.isfinite(g).all(axis=1) & np.isfinite(p).all(axis=1)
        counts.append(int(valid.sum()))
        if valid.sum() < 2:
            correlations.append(np.nan)
            continue
        g, p = g[valid], p[valid]
        g, p = g-g.mean(axis=0), p-p.mean(axis=0)
        denominator = np.sqrt(np.sum(g*g)*np.sum(p*p))
        correlations.append(np.sum(g*p)/denominator if denominator > 0 else np.nan)
    return np.asarray(correlations), np.asarray(counts)


def sample_row(occurrence, nrows, r, lag, right_ok=True):
    row = dict(record_id="one", task_name="A", stage="reach ring", occurrence_index=occurrence,
               start_row=0, end_row_exclusive=nrows, stage_rows=nrows, duration_seconds=nrows/60,
               record_rows=5000, gaze_window_start_row=0, gaze_window_end_row_exclusive=nrows+120,
               gaze_npy_path=r"dvss\Active_Data\User1\npy\one.npy",
               dlc_csv_path=r"dvss\Active_Data\User1\sam\one\one_DLC.csv", Score=0, EcoRaw=None,
               user=1, **SETTINGS)
    for side in SIDES:
        row.update({f"{side}_status": "ok", f"{side}_error": "", f"{side}_peak_r": r,
                    f"{side}_peak_lag_seconds": lag, f"{side}_peak_lag_frames": round(lag*60),
                    f"{side}_peak_valid_pairs": nrows, f"{side}_peak_at_boundary": abs(lag)==2})
    if not right_ok:
        row["r_psm_status"] = "no_valid_peak"
    return row


class StageNumericalTests(unittest.TestCase):
    def test_fft_matches_direct_with_gaps_and_record_edges(self):
        rng = np.random.default_rng(711)
        g = rng.normal(size=(700, 2))*[90,50]+[650,500]
        p = rng.normal(size=(700, 2))*[120,90]+[700,600]
        g[45:60] = np.nan
        p[400:410] = np.nan
        for start, end in ((0,80), (300,390), (650,700)):
            lags, actual, counts = occurrence_curve(g, p, start, end)
            expected, direct_counts = direct_occurrence_curve(g, p, start, end)
            np.testing.assert_array_equal(lags, np.arange(-120,121))
            np.testing.assert_array_equal(counts, direct_counts)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)

    def test_gaze_outside_stage_is_used_but_psm_outside_is_never_used(self):
        rng = np.random.default_rng(45)
        g = rng.normal(size=(500,2))
        # All correct gaze samples are BEFORE this 60-row occurrence.
        start, end, delay = 220,280,100
        p = np.full_like(g, np.nan)
        p[start:end] = 2*g[start-delay:end-delay]+[30,40]
        peak = occurrence_peak(g,p,start,end)
        self.assertEqual(peak["status"], "ok")
        self.assertEqual(peak["peak_lag_frames"], delay)
        self.assertEqual(peak["peak_valid_pairs"], 60)
        self.assertAlmostEqual(peak["peak_r"], 1, places=12)
        changed = p.copy()
        changed[:start] = 1e8
        changed[end:] = -1e8
        self.assertEqual(occurrence_peak(g,changed,start,end), peak)
        # Correct gaze samples AFTER the occurrence, hence negative lag.
        p[start:end] = g[start+delay:end+delay]
        self.assertEqual(occurrence_peak(g,p,start,end)["peak_lag_frames"], -delay)

    def test_gaze_outside_context_is_not_used(self):
        rng = np.random.default_rng(9)
        g, p = rng.normal(size=(700,2)), rng.normal(size=(700,2))
        a = occurrence_curve(g,p,300,350)
        changed = g.copy()
        changed[:180] = 1e9
        changed[470:] = -1e9
        b = occurrence_curve(changed,p,300,350)
        for left,right in zip(a,b):
            np.testing.assert_array_equal(left,right)

    def test_short_occurrence_is_computed_but_undefined_variance_is_preserved(self):
        rng = np.random.default_rng(3)
        g = rng.normal(size=(400,2))
        peak = occurrence_peak(g,g,150,153)
        self.assertEqual(peak["status"], "ok")
        self.assertEqual(peak["peak_valid_pairs"], 3)
        self.assertEqual(peak["peak_lag_frames"], 0)
        self.assertEqual(occurrence_peak(g,g,150,151)["status"], "no_valid_peak")
        self.assertEqual(occurrence_peak(g,np.ones_like(g),150,250)["status"], "no_valid_peak")

    def test_two_pair_overlap_is_directly_centered_for_numerical_stability(self):
        rng = np.random.default_rng(88)
        g = rng.normal(size=(500,2))*100+[600,500]
        p = np.full_like(g,np.nan)
        p[220:222] = [[600,500],[600.0001,500.0002]]
        lags, actual, counts = occurrence_curve(g,p,200,300)
        direct, expected_counts = direct_occurrence_curve(g,p,200,300)
        np.testing.assert_array_equal(counts,expected_counts)
        np.testing.assert_allclose(actual,direct,rtol=1e-13,atol=1e-13,equal_nan=True)

    def test_segmentation_keeps_repeated_runs_and_aliases(self):
        labels = ["none", "reach ring", "reach ring", "", "reach ring", "drop ring",
                  "transfer ring", "transport ring", "place ring", "grab string"]
        self.assertEqual(list(stage_runs(labels)), [("reach ring",1,1,3), ("reach ring",2,4,5),
                                                  ("transfer ring",1,6,8),("place ring",1,8,9)])
        self.assertEqual(canonical_stage("  WRAP   AROUND PSM "), "wrap around psm")
        self.assertEqual(canonical_stage("grab string"), "")


class StageOutputTests(unittest.TestCase):
    def test_weighting_independent_sides_and_csv_metadata(self):
        rows = [sample_row(1,60,.2,1), sample_row(2,180,.8,-.5,False)]
        fields = STAGE_FIELDS + ["Score","EcoRaw","user"]
        # Use an actual CSV round trip, as the second script does.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.csv"
            write_csv(path,rows,fields)
            with path.open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f)
                result,result_fields=aggregate_stage_rows(list(reader),reader.fieldnames)
        self.assertEqual(len(result),1)
        row=result[0]
        self.assertAlmostEqual(row["l_psm_duration_weighted_r"],.65)
        self.assertAlmostEqual(row["l_psm_fisher_z_weighted_r"],np.tanh((np.arctanh(.2)+3*np.arctanh(.8))/4))
        self.assertAlmostEqual(row["l_psm_duration_weighted_lag_seconds"],-.125)
        self.assertEqual(row["total_duration_seconds"],4)
        self.assertEqual(row["r_psm_included_duration_seconds"],1)
        self.assertEqual(row["r_psm_occurrences_excluded"],1)
        self.assertEqual(row["r_psm_duration_weighted_r"],.2)
        self.assertEqual(row["Score"],"0")
        self.assertEqual(row["EcoRaw"],"")
        self.assertIn("user",result_fields)
        rows[0]["l_psm_peak_r"]=1.
        result,_=aggregate_stage_rows(rows,fields)
        self.assertTrue(np.isfinite(result[0]["l_psm_fisher_z_weighted_r"]))
        self.assertEqual(result[0]["l_psm_fisher_r_clip_count"],1)

    def test_one_summary_row_for_each_observed_record_stage(self):
        rows=[]
        for i,stage in enumerate(("reach ring","grab ring","transfer ring","place ring"),1):
            for occurrence in (1,2):
                row=sample_row(occurrence,60,.3,.5)
                row["stage"]=stage
                rows.append(row)
        result,_=aggregate_stage_rows(rows,STAGE_FIELDS+["Score","EcoRaw","user"])
        self.assertEqual(len(result),4)
        self.assertTrue(all(r["occurrences_total"]==2 for r in result))
        self.assertTrue(all("l_psm_fisher_z_weighted_r" in r and "r_psm_fisher_z_weighted_r" in r for r in result))

    def test_duplicate_or_inconsistent_occurrences_rejected(self):
        row=sample_row(1,60,.2,1)
        fields=STAGE_FIELDS+["Score","EcoRaw","user"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            aggregate_stage_rows([row,row],fields)
        second=sample_row(2,60,.2,1)
        second["Score"]=10
        with self.assertRaisesRegex(ValueError,"metadata"):
            aggregate_stage_rows([row,second],fields)

    def test_all_invalid_occurrences_still_have_a_summary(self):
        row=sample_row(1,60,.2,1)
        row["l_psm_status"]="no_valid_peak"
        result,_=aggregate_stage_rows([row],STAGE_FIELDS+["Score","EcoRaw","user"])
        self.assertEqual(result[0]["l_psm_status"],"no_valid_occurrences")
        self.assertEqual(result[0]["l_psm_fisher_z_weighted_r"],"")

    def test_label_audit_complete_metadata_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry=write_fixture(Path(tmp)/"dvss"/"Active_Data"/"User1",n=700,dlc_n=699)
            gaze=np.load(entry["gaze_npy_path"])
            data=np.empty(len(gaze),dtype=gaze.dtype.descr+[("stage","U30")])
            for key in gaze.dtype.names:
                data[key]=gaze[key]
            data["stage"]="reach ring"
            data["stage"][200:210]="none"
            data["stage"][500:510]="unexpected stage"
            np.save(entry["gaze_npy_path"],data)
            audit,unexpected,missing=label_audit([entry])
            self.assertEqual(unexpected,["unexpected stage"])
            self.assertIn("reach string",missing)
            self.assertTrue(any(r["handling"]=="ignore" for r in audit))
            entry["meta"]={"user":1,"task_name":"A","Score":0,"EcoRaw":None,"custom":{"key":3}}
            rows=compute_stage_record(entry)
            self.assertEqual(len(rows),3)
            self.assertTrue(all(r["l_psm_status"]=="source_error" for r in rows))
            self.assertIn("Row mismatch",rows[0]["l_psm_error"])
            output,fields=format_stage_output([entry],rows)
            self.assertEqual(output[0]["Score"],0)
            self.assertIsNone(output[0]["EcoRaw"])
            self.assertEqual(json.loads(output[0]["custom"]),{"key":3})
            self.assertTrue(output[0]["gaze_npy_path"].startswith("dvss\\Active_Data\\"))
            self.assertIn("user",fields)


if __name__ == "__main__":
    unittest.main()
