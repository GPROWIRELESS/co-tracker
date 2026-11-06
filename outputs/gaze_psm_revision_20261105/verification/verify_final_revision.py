"""Independent delivery verification; does not import the analysis implementation.

Recomputes fixed-record/user aggregation, compares overlapping GT inputs/curves,
and checks reference denominators and model comparisons against exported arrays.
Run only after both analysis branches finish writing validation.json.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

GROUP = ("variant", "task", "side", "signal", "stage", "mode", "source")
RECORD = ("variant", "content_id", "side", "signal", "stage", "mode", "source")
SOURCES = ["gt", "all_ABCD", "single_task", "leave_out", "spd_finetune", "spd_zero_shot"]
SIGNALS = {
    "primary": ["position", "velocity"], "position_unsmoothed": ["position"],
    "position_smooth5": ["position"], "velocity_box3": ["velocity"],
    "velocity_sg7": ["velocity"], "velocity_sg11": ["velocity"],
    "trim_edges": ["position", "velocity"], "window4": ["position", "velocity"],
}


def csvrows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def num(x):
    return float(x) if x not in (None, "", "None", "null") else np.nan


def truth(x):
    return str(x).lower() == "true"


def close(a, b, atol=2e-10):
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=atol, equal_nan=True))


class Verifier:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    def groups(self, directory, record_name="record_curves.jsonl", summary_name="group_summary.csv", long_name="group_curves.csv"):
        records, grouped, shapes = {}, defaultdict(lambda: defaultdict(list)), {}
        duplicates = []
        with (directory/record_name).open(encoding="utf-8") as stream:
            for line in stream:
                c = json.loads(line)
                k = tuple(c[f] for f in RECORD)
                if k in records:
                    duplicates.append(k)
                r, lag = np.asarray(c["r"], float), np.asarray(c["lag_s"], float)
                records[k] = dict(r=r, lag_s=lag, user=c["user"], task=c["task"],
                                  pairs=np.asarray(c.get("pairs", []), float))
                g = tuple(c[f] for f in GROUP)
                shapes.setdefault(g, lag)
                if not close(shapes[g], lag, 0):
                    duplicates.append(("lag_mismatch", g))
                # Primary model v2/GT84 share complete-record support.
                if np.isfinite(r).all():
                    grouped[g][c["user"]].append(np.arctanh(np.clip(r, -1+1e-7, 1-1e-7)))
        expected = {}
        for key, lag in shapes.items():
            users = grouped[key]
            r = np.tanh(np.mean([np.mean(v, axis=0) for v in users.values()], axis=0)) if len(users) >= 3 else np.full(len(lag), np.nan)
            expected[key] = (r, lag, len(users), sum(map(len, users.values())))
        errors = []
        summary_count = 0
        for row in csvrows(directory/summary_name):
            summary_count += 1
            k = tuple(row[f] for f in GROUP)
            if k not in expected:
                errors.append(("unknown_summary", k)); continue
            r, lag, users, complete = expected[k]
            pi = int(np.nanargmax(r)) if np.isfinite(r).any() else None
            peak = r[pi] if pi is not None else np.nan
            tau = lag[pi] if pi is not None else np.nan
            if not (close(peak, num(row.get("peak_r"))) and close(tau, num(row.get("peak_lag_s"))) and
                    users == int(row["n_stable_users"]) and complete == int(row["n_complete_records"])):
                errors.append(("summary", k))
            if row.get("range_method") and "uncalibrated" not in row["range_method"]:
                errors.append(("uncertainty_label", k))
            if truth(row.get("formal_leading_test", False)):
                errors.append(("formal_leading_claim", k))
        positions = defaultdict(int)
        long_count = 0
        for row in csvrows(directory/long_name):
            long_count += 1
            k = tuple(row[f] for f in GROUP)
            if k not in expected:
                errors.append(("unknown_long", k)); continue
            r, lag, n, _ = expected[k]
            i = positions[k]; positions[k] += 1
            if i >= len(lag) or not (close(num(row["lag_s"]), lag[i], 0) and close(num(row["r"]), r[i]) and int(row["n_stable_users"]) == n):
                errors.append(("long", k, i))
        for k, (r, lag, _, _) in expected.items():
            if positions[k] != len(lag): errors.append(("long_count", k))
        self.check(f"{directory.name}: independent group reaggregation", not duplicates and not errors and summary_count == len(expected),
                   record_curves=len(records), groups=len(expected), long_rows=long_count,
                   duplicate_count=len(duplicates), mismatch_count=len(errors), examples=str((duplicates+errors)[:5]))
        return records, expected

    def inputs_and_counts(self, gt, models, gtr, modelr, gtm, mm):
        gt_source, model_source = Path(gtm["source"]), Path(mm["source"])
        gids = [r["content_id"] for r in gtm["records"]]
        mids = mm["parameters"]["records"]
        expected_gt, expected_model = set(), set()
        mismatches, common_masks = [], {}
        input_comparisons = 0
        for cid in gids:
            with np.load(gt_source/"cache"/(cid+".npz"), allow_pickle=False) as z:
                base = {k: z[k] for k in ("frame_number", "native_time_s", "time_s", "native_stage", "native_occurrence", "stage", "occurrence",
                                             "native_psm_l", "native_psm_r", "likelihood_l", "likelihood_r", "gaze_raw_l", "gaze_raw_r")}
            mids_t = (base["time_s"][:-1]+base["time_s"][1:])/2
            ix = np.searchsorted(base["native_time_s"], mids_t, side="right")-1
            ix = ix.clip(0, len(base["native_time_s"])-1)
            labels = base["native_stage"][ix].copy()
            labels[mids_t-base["native_time_s"][ix] > .1] = "none"
            stages = {"position": sorted(set(base["stage"])-{"", "none", "nan", "null"}),
                      "velocity": sorted(set(labels)-{"", "none", "nan", "null"})}
            for variant in gtm["variants"]:
                for signal in SIGNALS[variant]:
                    for side in ("l", "r"):
                        for stage in stages[signal]:
                            for mode in ("psm_stage", "within_stage"):
                                expected_gt.add((variant,cid,side,signal,stage,mode,"gt"))
            if cid not in mids:
                continue
            with np.load(model_source/"cache"/"gt"/(cid+".npz"), allow_pickle=False) as z:
                for field, a in base.items():
                    b = z[field]
                    same = np.array_equal(a,b, equal_nan=True) if np.issubdtype(a.dtype,np.number) else np.array_equal(a,b)
                    input_comparisons += 1
                    if not same: mismatches.append((cid,field))
            for side in ("l", "r"):
                gtmask = np.isfinite(base["gaze_raw_"+side]).all(axis=1)
                mask = gtmask.copy()
                for source in SOURCES[1:]:
                    with np.load(model_source/"cache"/source/(cid+".npz"),allow_pickle=False) as z:
                        mask &= np.isfinite(z["gaze_raw_"+side]).all(axis=1)
                common_masks[cid,side] = bool(np.array_equal(gtmask,mask))
            for variant in mm["variants"]:
                for signal in SIGNALS[variant]:
                    for side in ("l","r"):
                        for stage in stages[signal]:
                            for mode in ("psm_stage","within_stage"):
                                for source in SOURCES:
                                    expected_model.add((variant,cid,side,signal,stage,mode,source))
        self.check("GT84 / model GT native and cached input equality", not mismatches and len(gids)==84 and len(mids)==10,
                   gt_records=len(gids),model_records=len(mids),array_checks=input_comparisons,examples=str(mismatches[:5]))
        self.check("GT84 branch condition inventory", set(gtr)==expected_gt, actual=len(gtr),expected=len(expected_gt),
                   missing=len(expected_gt-set(gtr)),extra=len(set(gtr)-expected_gt))
        self.check("model10 branch/source condition inventory", set(modelr)==expected_model, actual=len(modelr),expected=len(expected_model),
                   missing=len(expected_model-set(modelr)),extra=len(set(modelr)-expected_model))
        compared, skipped, errors = 0, 0, []
        for k,c in modelr.items():
            if k[0]!="primary" or k[-1]!="gt": continue
            if not common_masks[k[1],k[2]]:
                skipped+=1; continue
            compared+=1
            b=gtr[k]
            if not (close(c["r"],b["r"],1e-9) and close(c["pairs"],b["pairs"],0)):
                errors.append(k)
        self.check("primary GT curves match on identical support", not errors and compared>0,
                   compared=compared,skipped_changed_common_gaze_mask=skipped,mismatch_count=len(errors),examples=str(errors[:5]))

    def preprocessing(self, directory, manifest):
        cfg=manifest["variants"]["primary"]
        primary_ok=(cfg["position_window"]==3 and cfg["velocity_window"]==5 and cfg["velocity_filter"]=="savgol" and not cfg["trim_edges"])
        failures=[];n=0
        for row in csvrows(directory/"preprocessing_qc.csv"):
            n+=1
            if row.get("quality_rejected_count") and num(row["quality_rejected_count"])!=0: failures.append((row["content_id"],"quality_deleted"))
            if row.get("filled_quality_hole_fraction") and num(row["filled_quality_hole_fraction"])!=0: failures.append((row["content_id"],"quality_hole_fill"))
        provenance=directory/("point_provenance" if directory.name=="gt84" else "preprocessing_provenance")/"primary"
        npz_count=0;flag_checks=0
        for path in provenance.glob("*.npz"):
            npz_count+=1
            with np.load(path,allow_pickle=False) as z:
                for key in z.files:
                    if "quality_rejected" in key or "filled_quality_hole" in key:
                        flag_checks+=1
                        if np.any(z[key]): failures.append((path.name,key))
        self.check(f"{directory.name}: preprocessing / no quality deletion", primary_ok and not failures and flag_checks>0,
                   qc_rows=n,provenance_files=npz_count,zero_deletion_flag_arrays=flag_checks,examples=str(failures[:5]))

    def references(self, gt, models):
        rows=list(csvrows(gt/"unpaired_reference.csv"));errors=[];checked=0
        with np.load(gt/"unpaired_reference_peaks.npz",allow_pickle=False) as z:
            for row in rows:
                checked+=1
                a=z[row["draws_key"]+"_peak_r"];values=a[np.isfinite(a)]
                if len(a)!=5000 or int(row["n_requested"])!=5000 or len(values)!=int(row["n_valid"]) or int(row["n_failed"])+len(values)!=5000:
                    errors.append((row["draws_key"],"count"))
                expected=(1+np.count_nonzero(values>=num(row["observed_peak_r"])))/(len(values)+1) if len(values) else np.nan
                if not close(expected,num(row["reference_tail_fraction"]),1e-14):errors.append((row["draws_key"],"denominator"))
                if row["status"]!="insufficient_observed_users" and int(row["failure_no_donor"])+int(row["failure_incomplete_curve"])!=int(row["n_failed"]):
                    errors.append((row["draws_key"],"failure_partition"))
                if int(row["reference_family_size"])<9 or truth(row["bh_formal_fdr_control_claimed"]):errors.append((row["draws_key"],"family"))
        self.check("GT84 reference 5000 attempts / denominator / failure partition",not errors,groups=checked,examples=str(errors[:5]))
        draws=defaultdict(list)
        for row in csvrows(models/"unpaired_reference_draws.csv"):
            draws[tuple(row[k] for k in ("task","side","stage","source"))].append(num(row["peak_r"]))
        errors=[];checked=0
        for row in csvrows(models/"unpaired_reference.csv"):
            checked+=1;k=tuple(row[f] for f in ("task","side","stage","source"));a=np.asarray(draws[k])
            if int(row["n_requested"])!=5000 or int(row["n_valid"])!=len(a):errors.append((k,"count"))
            p=(1+np.count_nonzero(a>=num(row.get("observed_peak_r"))))/(len(a)+1) if len(a) else np.nan
            if not close(p,num(row.get("exceedance_fraction_corrected")),1e-14):errors.append((k,"denominator"))
        self.check("model10 reference 5000 attempts / denominator",not errors,groups=checked,examples=str(errors[:5]))

    def model_pairs(self, models, records):
        errors=[];count=0
        for row in csvrows(models/"paired_metrics.csv"):
            count+=1
            expected_rule={"l":"fixed_left","r":"fixed_right"}.get(row["side"])
            if not truth(row["same_side"]) or row["analysis_rule"]!=expected_rule:errors.append((count,"side_rule"))
            k=tuple(row[f] for f in RECORD[:-1]);gt=records[k+("gt",)]["r"];model=records[k+(row["model"],)]["r"]
            lag=records[k+("gt",)]["lag_s"]
            good=np.isfinite(gt)&np.isfinite(model)
            gt_tau=lag[np.nanargmax(gt)] if np.isfinite(gt).any() else np.nan
            m_tau=lag[np.nanargmax(model)] if np.isfinite(model).any() else np.nan
            if not close(num(row["delta_lag_s"]),m_tau-gt_tau,1e-14):errors.append((count,"delta"))
            if good.all() and not close(num(row["curve_mae"]),np.mean(abs(gt-model)),1e-14):errors.append((count,"mae"))
            if truth(row["input_eligible"])!=bool(good.all()):errors.append((count,"eligibility"))
        self.check("model10 fixed-side delta / MAE / eligibility independently recomputed",not errors,comparisons=count,examples=str(errors[:5]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("base",type=Path)
    parser.add_argument("--output",type=Path);args=parser.parse_args();started=time.monotonic()
    gt,models=args.base/"gt84",args.base/"models10"
    for directory in (gt,models):
        if not (directory/"validation.json").exists():raise RuntimeError(f"Analysis has not completed: {directory}")
    v=Verifier()
    gtm=json.loads((gt/"run_manifest.json").read_text(encoding="utf8"));mm=json.loads((models/"run_manifest.json").read_text(encoding="utf8"))
    gtr,gg=v.groups(gt);modelr,mg=v.groups(models)
    v.inputs_and_counts(gt,models,gtr,modelr,gtm,mm)
    v.preprocessing(gt,gtm);v.preprocessing(models,mm)
    v.references(gt,models);v.model_pairs(models,modelr)
    for directory in (gt,models):
        required=["run_manifest.json","input_manifest.json","validation.json","record_summary.csv","group_summary.csv", "record_curves.jsonl","group_curves.csv","preprocessing_qc.csv","report.html"]
        missing=[name for name in required if not (directory/name).is_file() or (directory/name).stat().st_size==0]
        v.check(f"{directory.name}: required deliverables nonempty",not missing,missing=missing)
    result=dict(status="passed" if all(c["passed"] for c in v.checks) else "failed",checks=v.checks,
                elapsed_seconds=time.monotonic()-started,independent_of_analysis_functions=True,
                scientific_limit="Implementation consistency only; no claim of calibrated confidence or physiological latency.")
    path=args.output or args.base/"independent_validation.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(dict(status=result["status"],checks=len(v.checks),output=str(path),seconds=result["elapsed_seconds"])),flush=True)
    if result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()
