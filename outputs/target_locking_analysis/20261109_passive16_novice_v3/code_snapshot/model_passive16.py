"""Evaluate frozen v3 models against matched active and passive gaze references.

Only the pair adapter and output schema are new. Measurement functions are
imported, unchanged, from the frozen v3 code snapshot. No training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "v3"))
import tl_geometry as geo
import tl_inputs as inputs
from tl_common import dump_json, save_rows, stage_masks, summarize
from analyze_target_locking import SCENARIOS, MODELS, DECODERS, onehot, sum_probability, compare

REFERENCE = Path(r"D:\GitHub_Project\co-tracker\outputs\target_locking_analysis\20261105T063658759189Z_v3_full")
META = ["pair_id", "content_id", "observer_id", "participant", "task", "task_num", "attempt", "round", "score", "status"]
REF_NAMES = ("center_point", "uniform_area", "train_heatmap", "train_rate")


def sha(path):
    return inputs._sha256(Path(path))


def prefixed(row, prefix, values):
    for key, value in values.items():
        row[prefix + key] = value


def load_frame(path):
    with np.load(path, allow_pickle=False) as archive:
        f = {name: archive[name] for name in archive.files}
    d = dict(frames=f["frame_number"], timestamp_ns=f["timestamp_ns"],
             dt=f["dt_s"], occurrence=f["occurrence"], breaks=f["break_before"],
             stage=f["stage"], included=f["included"], z=f["target_xy"],
             left=f["left_xy"], right=f["right_xy"],
             width=int(f["resolution"][0]), height=int(f["resolution"][1]),
             gaze_unmasked=f["active_xy"])
    return f, d


def copied_prediction_cache(ref, out, selected):
    dest = out / "model_cache" / "prediction_bundle"
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((ref / "cache" / "prediction_bundle").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
        if meta["content_id"] not in selected:
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        records.append(dict(content_id=meta["content_id"], model=meta["model"],
                            source=str(path), copy=str(target), sha256=sha(path)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE.parent / "output")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    out, ref = args.output, args.reference
    manifest = pd.read_csv(out / "pair_manifest.csv")
    audit = pd.read_csv(ref / "prediction_audit.csv")
    model_sets = audit.groupby("content_id").model.agg(set)
    available = {cid for cid, names in model_sets.items() if names == set(MODELS)}
    selected = manifest.loc[manifest.content_id.isin(available)].copy()
    print(f"Model subset: {len(selected)} pairs, {selected.content_id.nunique()} contents, "
          f"{selected.participant.nunique()} source participants", flush=True)
    if not len(selected):
        dump_json(out / "model_audit.json", dict(status="unavailable", pairs=0))
        return

    # A writable copy avoids modifying the frozen reference cache on cache miss.
    cache_audit = copied_prediction_cache(ref, out, set(selected.content_id))
    baseline_audit = []
    base = out / "model_baselines"
    base.mkdir(parents=True, exist_ok=True)
    for name in ("baseline_parameters.csv", "baseline_training_heatmaps.npz", "baseline_training_members.csv", "baseline_input_audit.json"):
        shutil.copy2(ref / name, base / name)
        baseline_audit.append(dict(name=name, reference=str(ref/name), sha256=sha(ref/name)))
    params = pd.read_csv(base / "baseline_parameters.csv")
    with np.load(base / "baseline_training_heatmaps.npz", allow_pickle=False) as z:
        maps = {key: z[key] for key in z.files}
    refs = {(r.task, r.scenario, r.stage): dict(heatmap=maps[r.map_key],
            rate_target=r.rate_target, rate_gripper=r.rate_gripper)
            for r in params.itertuples()}
    metrics, confusions, diag, prediction_audits, reproduction = [], [], [], [], []
    save_dir = out / "model_frames"
    save_dir.mkdir(parents=True, exist_ok=True)

    for _, pair in selected.iterrows():
        owner = {key: pair[key] for key in META if key in pair.index}
        cid, pid = str(pair.content_id), str(pair.pair_id)
        f, d = load_frame(out / "frames" / f"{pid}.npz")
        d.update(content_id=cid, task=pair.task)
        bundles = {}
        old = np.load(ref / "frames" / f"{cid}.npz", allow_pickle=False)
        for model in MODELS:
            raw = audit.loc[(audit.content_id == cid) & (audit.model == model)].iloc[0]
            entry = raw.to_dict()
            entry["path"] = str(raw["path"])
            print(f"  Verify/load {cid}/{model}", flush=True)
            b = inputs.prediction_bundle(entry, d, trajectory_seed=42,
                        cache_dir=out / "model_cache", include_probability=False)
            if b["samples"].shape[1] != 64:
                raise ValueError("The frozen v3 analysis requires K=64")
            bundles[model] = b
            prediction_audits.append(dict(pair_id=pid, **b["metadata"]))
            for rule, key in DECODERS.items():
                expected = old[f"xy_{rule}_{model}"]
                np.testing.assert_allclose(b[key], expected, atol=1e-12, rtol=0, equal_nan=True)
                reproduction.append(dict(pair_id=pid, content_id=cid, model=model,
                            check="decoder_"+rule, max_abs_error=float(np.nanmax(np.abs(b[key]-expected))), passed=True))
        model_ok = np.logical_and.reduce([b["valid_maps"] & b["valid_samples"].all(axis=1) &
                    np.logical_and.reduce([np.isfinite(b[key]).all(axis=1) for key in DECODERS.values()])
                    for b in bundles.values()])
        np.testing.assert_array_equal(model_ok, old["model_common_valid"])
        frame_out = dict(frame_number=d["frames"], timestamp_ns=d["timestamp_ns"],
                         model_common_valid=model_ok, dt_s=d["dt"], stage=d["stage"])
        q_cache = {}
        for scenario, (rt, rp, _, smooth) in SCENARIOS.items():
            print(f"  Measure {pid}/{scenario}", flush=True)
            valid = f["valid_"+scenario].astype(bool) & model_ok
            geometry = f["geometry_valid"].astype(bool)
            active_xy = f["active_smooth_xy"] if smooth else f["active_xy"]
            passive_xy = f["passive_smooth_xy"] if smooth else f["passive_xy"]
            area = f["area_"+scenario]
            ah = geo.classify(active_xy, d["z"], d["left"], d["right"], rt, rp, valid)
            ph = geo.classify(passive_xy, d["z"], d["left"], d["right"], rt, rp, valid)
            aq, pq = onehot(ah), onehot(ph)
            frame_out["valid_"+scenario] = valid
            frame_out["active_state_"+scenario] = ah
            frame_out["passive_state_"+scenario] = ph
            if (rt, rp) not in q_cache:
                q_cache[rt, rp] = {name: geo.sample_region_probabilities(b["samples_xy"],
                    d["z"], d["left"], d["right"], rt, rp, geometry & model_ok)
                    for name, b in bundles.items()}
            qs = q_cache[rt, rp]
            for model, q in qs.items():
                oldq = old[f"q_{model}_{scenario}"]
                check = np.isfinite(oldq).all(axis=1)
                # The frozen artifact stored float32 probabilities. K=64 values are exact.
                np.testing.assert_allclose(q[check], oldq[check], atol=1e-12, rtol=0)
                reproduction.append(dict(pair_id=pid, content_id=cid, model=model,
                            check="region_probability_"+scenario, max_abs_error=float(np.max(np.abs(q[check]-oldq[check]))) if check.any() else 0., passed=True))
                frame_out[f"q_{model}_{scenario}"] = np.where(valid[:, None], q, np.nan).astype(np.float32)
            center = geo.classify(np.broadcast_to([d["width"]/2, d["height"]/2], active_xy.shape),
                                  d["z"], d["left"], d["right"], rt, rp, valid)
            centerq = onehot(center)
            gt = {}
            for domain, xy in (("active", active_xy), ("passive", passive_xy)):
                values = np.full((len(xy), 4), np.nan)
                ids = np.flatnonzero(valid)
                for start in range(0, len(ids), 512):
                    ix = ids[start:start+512]
                    kernels = geo.gaussian_maps(xy[ix], d["width"], d["height"])
                    values[ix] = geo.map_region_probabilities(kernels, d["z"][ix], d["left"][ix],
                        d["right"][ix], d["width"], d["height"], rt, rp, np.ones(len(ix), bool))
                gt[domain] = values
            for stage, st in stage_masks(d):
                ok = valid & st
                base_owner = dict(owner, scenario=scenario, stage=stage,
                                 common_valid_n=int(ok.sum()), common_valid_s=float(d["dt"][ok].sum()))
                r = refs[pair.task, scenario, stage]
                trainq = geo.map_region_probabilities(r["heatmap"], d["z"], d["left"], d["right"],
                            d["width"], d["height"], rt, rp, ok)
                rateq = np.tile([0, r["rate_target"], r["rate_gripper"], 0], (len(valid), 1))
                refq = dict(center_point=centerq, uniform_area=area, train_heatmap=trainq, train_rate=rateq)
                diagnostic = dict(base_owner)
                for domain in ("active", "passive"):
                    prefixed(diagnostic, domain+"_", sum_probability(gt[domain], ok, area, d))
                diag.append(diagnostic)
                for branch in ("probability_mc", "decoded_expectation", "decoded_argmax", "decoded_sample"):
                    isprob = branch == "probability_mc"
                    a = sum_probability(aq, ok, area, d) if isprob else summarize(ah, ok, area, d)
                    p = sum_probability(pq, ok, area, d) if isprob else summarize(ph, ok, area, d)
                    for model, b in bundles.items():
                        mh = None
                        if isprob:
                            mq = qs[model]
                            mm = sum_probability(mq, ok, area, d)
                        else:
                            rule = branch.removeprefix("decoded_")
                            mh = geo.classify(b[DECODERS[rule]], d["z"], d["left"], d["right"], rt, rp, valid)
                            mq = onehot(mh)
                            mm = summarize(mh, ok, area, d)
                            frame_out[f"state_{rule}_{model}_{scenario}"] = mh
                        row = dict(base_owner, analysis_branch=branch, model=model)
                        for prefix, vals in (("active_", a), ("passive_", p), ("model_", mm)):
                            prefixed(row, prefix, vals)
                        for key in a:
                            if isinstance(a[key], (int, float, np.number)) and key in p:
                                row["delta_"+key] = p[key]-a[key]
                                row["model_minus_active_"+key] = mm[key]-a[key]
                                row["model_minus_passive_"+key] = mm[key]-p[key]
                        matrices = {}
                        for domain, h, vals in (("active", ah, a), ("passive", ph, p)):
                            errors, matrix = compare({}, d, ok, h, mq, vals, mm,
                                                     dict(refq, GT_smoothed=gt[domain]), mh)
                            prefixed(row, domain+"_", errors)
                            matrices[domain] = matrix
                        metrics.append(row)
                        if mh is not None:
                            total = float(d["dt"][ok].sum())
                            totals = {domain: {state: sum(x["seconds"] for x in mat if x["actual"] == state)
                                                for state in "NTPO"}
                                      for domain, mat in matrices.items()}
                            for ai, pi in zip(matrices["active"], matrices["passive"]):
                                assert ai["actual"] == pi["actual"] and ai["predicted"] == pi["predicted"]
                                cr = dict(base_owner, analysis_branch=branch, model=model,
                                          reference_region=ai["actual"],
                                          active_region=ai["actual"], model_region=ai["predicted"])
                                # reference_region is a shared category index: active_*
                                # measures active reference rows, passive_* passive rows.
                                # active_region remains only as a legacy-compatible alias.
                                for domain, cell in (("active", ai), ("passive", pi)):
                                    cr[domain+"_frames"] = cell["frames"]
                                    cr[domain+"_seconds"] = cell["seconds"]
                                    cr[domain+"_joint_share"] = cell["seconds"]/total if total else np.nan
                                    den = totals[domain][cell["actual"]]
                                    cr[domain+"_conditional_share"] = cell["seconds"]/den if den else np.nan
                                confusions.append(cr)
        np.savez_compressed(save_dir / f"{pid}.npz", **frame_out)
        old.close()
    save_rows(metrics, out / "model_pair_metrics.csv")
    save_rows(confusions, out / "model_confusions.csv")
    save_rows(diag, out / "model_gt_smoothed.csv")
    save_rows(prediction_audits, out / "model_prediction_audit.csv")
    save_rows(reproduction, out / "model_reproduction.csv")
    dump_json(out / "model_audit.json", dict(status="complete", pairs=len(selected),
        contents=int(selected.content_id.nunique()), source_participants=int(selected.participant.nunique()),
        observers=int(selected.observer_id.nunique()), pair_ids=selected.pair_id.tolist(),
        models=list(MODELS), branches=["probability_mc", *["decoded_"+x for x in DECODERS]],
        scenarios=list(SCENARIOS), model_metric_rows=len(metrics), confusion_rows=len(confusions),
        reproduction_checks=len(reproduction), reproduction_passed=all(x["passed"] for x in reproduction),
        samples=64, decoder_seed=42, reference_run=str(ref), baseline_files=baseline_audit,
        copied_cache=cache_audit, human_frame_rule="pair common support AND all five models all saved samples and all three decoders valid",
        model_training="none; frozen existing predictions", training_reference="unchanged v3 active training-only baselines",
        GT_smoothed="domain-specific label smoothing diagnostic, not a bound or BSS baseline",
        sample_temporal_caution="fixed-seed independent per-frame saved-K decoding; no joint trajectory identity"))
    print(f"Completed {len(metrics)} model rows, {len(confusions)} confusion cells", flush=True)


if __name__ == "__main__":
    main()
