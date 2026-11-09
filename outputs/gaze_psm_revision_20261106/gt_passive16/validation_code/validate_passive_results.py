"""Independent numerical checks for the frozen GT84 passive adaptation.

The direct correlation oracle below uses explicit pairs and centered arrays.
It does not import the FFT kernel or occurrence sufficient-statistic functions.
All imports from the frozen code are restricted to preprocessing.
"""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes, seed_identity

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
GROUP_FIELDS = ("variant", "task", "side", "signal", "stage", "mode", "source")
BASE = dict(hz=41., max_gap=.1, max_lag=2., min_overlap=2., min_rms=1.,
            position_window=3, velocity_window=5, velocity_filter="savgol", trim_edges=False)
BASELINE = Path(r"D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106\gt84")
WORK = Path(r"C:\Users\GT502\Documents\ChatGPT\JMRR\passive_gt_work")


def direct_curve(gaze, psm, occurrence_ids, hz=41., max_lag=2., both_in_stage=False,
                 min_overlap=2., min_rms=1.):
    """Explicitly center every occurrence's actual valid pairs at every lag."""
    gaze, psm = np.asarray(gaze, float), np.asarray(psm, float)
    ids = np.asarray(occurrence_ids)
    shifts = np.arange(-round(hz * max_lag), round(hz * max_lag) + 1)
    members = [np.flatnonzero(ids == value) for value in np.unique(ids[ids >= 0])]
    finite_g, finite_p = np.isfinite(gaze).all(1), np.isfinite(psm).all(1)
    result = {key: np.zeros(len(shifts), float) for key in ("pairs", "occurrences", "gaze_rms", "psm_rms")}
    result["r"] = np.full(len(shifts), np.nan)
    result["lag_s"] = shifts / hz
    for j, shift in enumerate(shifts):
        count = occurrences = 0
        covariance = gss = pss = 0.
        for target in members:
            donor = target - shift
            inside = (donor >= 0) & (donor < len(gaze))
            u, t = target[inside], donor[inside]
            good = finite_p[u] & finite_g[t]
            if both_in_stage:
                good &= ids[t] == ids[u]
            if np.count_nonzero(good) < 2:
                continue
            g, p = gaze[t[good]], psm[u[good]]
            gc, pc = g - g.mean(0), p - p.mean(0)
            covariance += np.sum(gc * pc)
            gss += np.sum(gc * gc)
            pss += np.sum(pc * pc)
            count += len(g)
            occurrences += 1
        result["pairs"][j], result["occurrences"][j] = count, occurrences
        gr = np.sqrt(gss / count) if count else np.nan
        pr = np.sqrt(pss / count) if count else np.nan
        result["gaze_rms"][j], result["psm_rms"][j] = gr, pr
        denominator = np.sqrt(gss * pss)
        if count >= max(2, int(np.ceil(min_overlap * hz))) and gr >= min_rms and pr >= min_rms and denominator > np.finfo(float).eps:
            result["r"][j] = np.clip(covariance / denominator, -1., 1.)
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(mode_rows(csv.DictReader(stream)))


def read_curves(path, primary_only=False):
    curves = []
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            curve = normalize_modes(json.loads(line))
            if not primary_only or curve["variant"] == "primary":
                for name in ("r", "lag_s", "pairs", "occurrences", "gaze_rms", "psm_rms"):
                    if name in curve:
                        curve[name] = np.asarray(curve[name], float)
                curves.append(curve)
    return curves


def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def group_key(row):
    return tuple(row[name] for name in GROUP_FIELDS)


def fisher(arrays):
    return np.tanh(np.mean(np.arctanh(np.clip(arrays, -1 + 1e-7, 1 - 1e-7)), axis=0))


def stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256(repr(seed_identity((int(seed), key))).encode()).digest()[:8], "little")


def as_number(value):
    return np.nan if value in (None, "") else float(value)


def compare_arrays(actual, expected, label, errors, tolerance=1e-9):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        errors.append({"check": label, "reason": "shape_or_finite_mask_mismatch"})
        return np.inf
    finite = np.isfinite(actual)
    error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.
    if error >= tolerance:
        errors.append({"check": label, "max_abs_error": error, "tolerance": tolerance})
    return error


def compare_scalar(actual, expected, label, errors, tolerance=1e-10):
    return compare_arrays([as_number(actual)], [expected], label, errors, tolerance)


def prepare(caches, frozen):
    sys.path.insert(0, str(frozen))
    from gaze_psm_preprocess_revision import prepare_signals
    return {cid: prepare_signals({"gt": load_cache(path)}, BASE) for cid, path in caches.items()}


def validate_direct(curves, prepared, errors):
    maximum = 0.
    checks = 0
    for curve in curves:
        if curve["variant"] != "primary" or curve["side"] not in ("l", "r"):
            continue
        signal = prepared[curve["content_id"]]["signals"][curve["signal"]][curve["side"]]
        ids = np.where(signal["stage"] == curve["stage"], signal["occurrence"], -1)
        expected = direct_curve(signal["gazes"]["gt"], signal["psm"], ids,
                                both_in_stage=curve["mode"] == "Both in stage")
        label = "/".join(str(curve[name]) for name in ("content_id", "side", "signal", "stage", "mode"))
        maximum = max(maximum, compare_arrays(curve["r"], expected["r"], label, errors))
        compare_arrays(curve["pairs"], expected["pairs"], label + "/pair_counts", errors, .5)
        compare_arrays(curve["occurrences"], expected["occurrences"], label + "/occurrence_counts", errors, .5)
        checks += 1
    return {"primary_curves_checked": checks, "finite_masks_equal": not errors,
            "maximum_absolute_correlation_error": maximum,
            "oracle": "explicit lagged pairs centered separately in every occurrence; no FFT imports",
            "min_overlap_s": 2., "minimum_pairs": 82, "minimum_centered_2d_rms": 1.}


def baseline_regression(baseline=BASELINE):
    manifest = normalize_modes(json.loads((baseline / "run_manifest.json").read_text(encoding="utf8")))
    source = Path(manifest["source"])
    cids = ("User1_1_1", "User1_3_1")
    prepared = prepare({cid: source / "cache" / (cid + ".npz") for cid in cids}, baseline / "code_snapshot")
    curves = [curve for curve in read_curves(baseline / "record_curves.jsonl", True) if curve["content_id"] in cids]
    errors = []
    result = validate_direct(curves, prepared, errors)
    from gaze_psm_preprocess_revision import interpolate_valid
    input_checks = []
    for cid in cids:
        cached = load_cache(source / "cache" / (cid + ".npz"))
        data_root = Path(r"D:\Dataset\dvss\Active_Data") / cid.split("_")[0]
        native = np.load(data_root / "npy" / (cid + "_gaze.npy"), allow_pickle=False)
        clock = [row for row in read_csv(data_root / "csv" / (cid + "_video_ts.csv"))
                 if row.get("IsDelete", "0").strip() in ("", "0", "0.0")]
        frames = np.array([int(row["Frame_Number"]) for row in clock], np.int64)
        ns = np.array([int(Decimal(row["Timestamp(ns)"])) for row in clock], np.int64)
        frame_rows = {int(frame): index for index, frame in enumerate(native["Frame_Number"])}
        gaze = native[[frame_rows[int(frame)] for frame in frames]]
        norm = np.column_stack((gaze["gaze_x"], gaze["gaze_y"])).astype(float)
        good = np.isfinite(norm).all(1) & (norm >= 0).all(1) & (norm <= 1).all(1)
        norm[~good] = np.nan
        elapsed = (ns - ns[0]) / 1e9
        maximum = 0.
        compare_arrays(cached["frame_number"], frames, f"baseline_input_frames/{cid}", errors, .5)
        compare_arrays(cached["native_time_s"], elapsed, f"baseline_input_clock/{cid}", errors, 1e-14)
        for side in ("l", "r"):
            pixels = norm * np.asarray(cached["resolution_" + side], float)
            restored = interpolate_valid(elapsed, pixels, cached["time_s"], .1)
            maximum = max(maximum, compare_arrays(cached["gaze_raw_" + side], restored,
                                                  f"baseline_input_gaze/{cid}/{side}", errors, 1e-9))
        input_checks.append({"content_id": cid, "raw_npy_frame_clock_reconstruction_checked": True,
                             "coordinate_valid_rows": int(good.sum()), "valid_used_as_filter": False,
                             "maximum_grid_gaze_pixel_error": maximum})
    result.update(baseline_version=manifest["version"], representative_records=list(cids),
                  source_cache=str(source), active_input_adapter_checks=input_checks, passed=not errors, errors=errors)
    return result


def selected_curves(curves):
    grouped = defaultdict(list)
    for curve in curves:
        if curve["variant"] == "primary":
            grouped[tuple(curve[name] for name in ("content_id", "user", "task", "signal", "stage", "mode", "source"))].append(curve)
    selected = []
    for candidates in grouped.values():
        usable = [curve for curve in candidates if np.isfinite(curve["r"]).any()]
        if usable:
            winner = max(usable, key=lambda curve: (float(np.nanmax(curve["r"])), curve["side"] == "l"))
            selected.append(dict(winner, side="gt_selected_side"))
    return selected


def validate_groups(output, curves, errors):
    grouped = defaultdict(lambda: defaultdict(list))
    for curve in curves + selected_curves(curves):
        if np.isfinite(curve["r"]).all():
            grouped[group_key(curve)][str(curve["user"])].append(curve)
    summaries = read_csv(output / "group_summary.csv") + read_csv(output / "gt_selected_group_summary.csv")
    long = defaultdict(list)
    for row in read_csv(output / "group_curves.csv") + read_csv(output / "gt_selected_group_curves.csv"):
        long[group_key(row)].append(row)
    peak_index = normalize_modes(json.loads((output / "group_bootstrap_index.json").read_text(encoding="utf8")))
    archived_keys = {group_key(row): row for row in peak_index}
    arrays = load_cache(output / "group_bootstrap_peak_lags.npz")
    people = {}
    max_error = 0.
    bootstrap_checks = 0
    for row in summaries:
        key = group_key(row)
        by_user = grouped[key]
        users = sorted(by_user)
        points = long[key]
        lag = np.array([float(point["lag_s"]) for point in points])
        aa = np.stack([fisher([curve["r"] for curve in by_user[user]]) for user in users]) if users else np.empty((0, len(lag)))
        people[key] = (users, aa, lag)
        expected = fisher(aa) if len(users) >= 3 else np.full(len(lag), np.nan)
        actual = np.array([as_number(point["r"]) for point in points])
        max_error = max(max_error, compare_arrays(actual, expected, f"group/{key}", errors, 1e-11))
        if int(row["n_stable_users"]) != len(users) or int(row["n_complete_records"]) != sum(map(len, by_user.values())):
            errors.append({"check": "fixed_complete_support", "group": key})
        if normalize_modes(json.loads(row["participants_json"])) != users:
            errors.append({"check": "observer_participant_identity", "group": key})
        if any(int(point["n_stable_users"]) != len(users) for point in points):
            errors.append({"check": "all_lag_participant_support", "group": key})
        if len(users) < 3:
            continue
        count = int(row["n_bootstrap"])
        rng = np.random.default_rng(stable_seed(42, (key[:-1], tuple(users))))
        indices = rng.integers(0, len(users), (count, len(users)))
        z = np.arctanh(np.clip(aa, -1 + 1e-7, 1 - 1e-7))
        reps = np.tanh(z[indices].mean(1))
        low, high = np.percentile(reps, [2.5, 97.5], axis=0)
        compare_arrays([as_number(point["r_resample_low"]) for point in points], low, f"group_low/{key}", errors, 1e-11)
        compare_arrays([as_number(point["r_resample_high"]) for point in points], high, f"group_high/{key}", errors, 1e-11)
        peaks = lag[np.argmax(reps, axis=1)]
        lo, hi = np.percentile(peaks, [2.5, 97.5])
        compare_scalar(row["peak_lag_resample_low"], lo, f"peak_low/{key}", errors)
        compare_scalar(row["peak_lag_resample_high"], hi, f"peak_high/{key}", errors)
        compare_scalar(row["bootstrap_boundary_frequency"], np.mean((peaks == lag[0]) | (peaks == lag[-1])), f"peak_boundary/{key}", errors)
        if key[0] == "primary":
            index = archived_keys.get(key)
            if index is None:
                errors.append({"check": "missing_primary_bootstrap_array", "group": key})
            else:
                compare_arrays(arrays[index["token"]], peaks, f"bootstrap_array/{key}", errors, 1e-14)
                if index["users"] != users:
                    errors.append({"check": "bootstrap_observer_identity", "group": key})
        bootstrap_checks += 1
    return {"groups_checked": len(summaries), "bootstrap_groups_recomputed": bootstrap_checks,
            "maximum_group_correlation_error": max_error}, people


def validate_occurrence_bootstrap(output, errors):
    grouped = defaultdict(list)
    for row in read_csv(output / "record_summary.csv"):
        grouped[(row["variant"], row["content_id"])].append(row)
    checked = 0
    for (variant, cid), rows in grouped.items():
        archive = load_cache(output / "bootstrap_peaks" / variant / (cid + ".npz"))
        for row in rows:
            token = row["statistics_key"]
            values = np.asarray(archive[token], float)
            good = np.isfinite(values)
            if int(row["bootstrap_valid_replicates"]) != int(good.sum()):
                errors.append({"check": "occurrence_bootstrap_valid_count", "record": cid, "token": token})
            if good.any():
                low, high = np.percentile(values[good], [2.5, 97.5])
                compare_scalar(row["bootstrap_range_low_s"], low, f"occ_low/{variant}/{cid}/{token}", errors)
                compare_scalar(row["bootstrap_range_high_s"], high, f"occ_high/{variant}/{cid}/{token}", errors)
                limit = 4. if variant == "window4" else 2.
                compare_scalar(row["bootstrap_boundary_fraction"], np.mean(np.abs(values[good]) == limit), f"occ_boundary/{variant}/{cid}/{token}", errors)
            checked += 1
    return {"record_summary_bootstrap_rows_checked": checked}


def validate_contrasts(output, people, errors):
    archive = load_cache(output / "stage_contrast_bootstrap.npz")
    checked = 0
    for row in read_csv(output / "stage_contrasts.csv"):
        ka = ("primary", row["task"], row["side"], row["signal"], row["condition_a"], row["mode"], "gt")
        kb = ("primary", row["task"], row["side"], row["signal"], row["condition_b"], row["mode"], "gt")
        ua, aa, lag = people[ka]
        ub, bb, _ = people[kb]
        users = sorted(set(ua) & set(ub))
        if int(row["n_users"]) != len(users) or normalize_modes(json.loads(row["users"])) != users:
            errors.append({"check": "stage_contrast_common_observers", "conditions": [ka, kb]})
        if len(users) < 3:
            continue
        za = np.arctanh(np.clip(np.stack([aa[ua.index(user)] for user in users]), -1 + 1e-7, 1 - 1e-7))
        zb = np.arctanh(np.clip(np.stack([bb[ub.index(user)] for user in users]), -1 + 1e-7, 1 - 1e-7))
        rng = np.random.default_rng(stable_seed(42, ("contrast", ka, kb, tuple(users))))
        indices = rng.integers(0, len(users), (5000, len(users)))
        ia, ib = za[indices].mean(1).argmax(1), zb[indices].mean(1).argmax(1)
        differences = lag[ib] - lag[ia]
        low, high = np.percentile(differences, [2.5, 97.5])
        compare_arrays(archive[row["draw_key"]], differences, f"contrast_array/{ka}/{kb}", errors, 1e-14)
        compare_scalar(row["resample_low"], low, f"contrast_low/{ka}/{kb}", errors)
        compare_scalar(row["resample_high"], high, f"contrast_high/{ka}/{kb}", errors)
        boundary = np.mean((ia == 0) | (ia == len(lag) - 1) | (ib == 0) | (ib == len(lag) - 1))
        compare_scalar(row["boundary_frequency"], boundary, f"contrast_boundary/{ka}/{kb}", errors)
        checked += 1
    return {"stage_contrast_bootstrap_arrays_recomputed": checked}


def validate_reference(output, curves, records, prepared, errors):
    metadata = {str(row["content_id"]): row for row in records}
    pairings = defaultdict(dict)
    shared_stimulus = total_pairings = 0
    for row in read_csv(output / "unpaired_pairings.csv"):
        scope, target, donor, index = row["reference_scope"], row["target"], row["donor"], int(row["replicate"])
        pairings[scope].setdefault(target, np.full(5000, "", dtype="U128"))[index] = donor
        if donor:
            target_meta, donor_meta = metadata[target], metadata[donor]
            if str(target_meta["user"]) == str(donor_meta["user"]) or target_meta["task"] != donor_meta["task"]:
                errors.append({"check": "unpaired_donor_eligibility", "target": target, "donor": donor})
            shared_stimulus += target_meta.get("source_content_id") == donor_meta.get("source_content_id")
            total_pairings += 1
    donor_draws_checked = 0
    for scope, targets in pairings.items():
        for cid, actual in targets.items():
            target = metadata[cid]
            eligible = defaultdict(list)
            for did, donor in sorted(metadata.items()):
                if donor["task"] != target["task"] or str(donor["user"]) == str(target["user"]):
                    continue
                if scope == "duration_eligible_subset":
                    sufficient = True
                    for signal in ("position", "velocity"):
                        for side in ("l", "r"):
                            tt = prepared[cid]["signals"][signal][side]["time_s"]
                            dt = prepared[did]["signals"][signal][side]["time_s"]
                            sufficient &= len(dt) >= len(tt) and (not len(tt) or dt[-1] >= tt[-1] - 1e-9)
                    if not sufficient:
                        continue
                eligible[str(donor["user"])].append(did)
            people = sorted(eligible)
            expected_draws = np.full(5000, "", dtype="U128")
            seed = 31415 if scope == "duration_eligible_subset" else 42
            derived = int.from_bytes(hashlib.sha256(repr(seed_identity((seed, cid))).encode()).digest()[:8], "little")
            rng = np.random.default_rng(derived)
            if people:
                picks = rng.integers(0, len(people), 5000)
                for index, person in enumerate(people):
                    positions = np.flatnonzero(picks == index)
                    candidates = sorted(eligible[person])
                    within = rng.integers(0, len(candidates), len(positions))
                    expected_draws[positions] = np.asarray(candidates)[within]
            if not np.array_equal(actual, expected_draws):
                errors.append({"check": "hierarchical_donor_sampling_and_seed", "scope": scope, "target": cid})
            donor_draws_checked += 1
    lookup = {(curve["content_id"], curve["side"], curve["signal"], curve["stage"], curve["mode"]): curve
              for curve in curves if curve["variant"] == "primary"}
    tables = {}
    for path in (output / "unpaired_pair_curves").glob("*.npz"):
        data = load_cache(path)
        tables[path.stem] = (list(data["donors"].astype(str)),
                             {tuple(item["signature"]): data[item["key"]]
                              for item in normalize_modes(json.loads(str(data["metadata_json"])))})
    arrays = load_cache(output / "unpaired_reference_peaks.npz")
    lag = arrays["lag_s"]
    checked = 0
    reference_rows = read_csv(output / "unpaired_reference.csv")
    main_groups = {group_key(row): row for row in read_csv(output / "group_summary.csv")}
    for row in reference_rows:
        token = row["draws_key"]
        stored = arrays[token + "_peak_r"]
        stored_indices = arrays[token + "_peak_index"]
        targets = normalize_modes(json.loads(row["target_records"]))
        users = sorted({str(metadata[cid]["user"]) for cid in targets})
        expected = np.full(5000, np.nan)
        indices = np.full(5000, -1, int)
        signature = tuple(row[field] for field in ("side", "signal", "stage", "mode"))
        if len(users) >= 3:
            observed = {cid: lookup[(cid,) + signature]["r"] for cid in targets}
            obs = fisher([fisher([observed[cid] for cid in targets if str(metadata[cid]["user"]) == user]) for user in users])
            compare_scalar(row["observed_peak_r"], float(np.max(obs)), f"reference_observed/{token}", errors)
            if row["reference_scope"] == "elapsed_time_model_consistent":
                compare_scalar(main_groups[group_key(row)]["peak_r"], float(np.max(obs)), f"reference_main_group_observed/{token}", errors)
            group_z = np.zeros((5000, len(lag)))
            valid = np.ones(5000, bool)
            no_donor = np.zeros(5000, bool)
            for user in users:
                cids = [cid for cid in targets if str(metadata[cid]["user"]) == user]
                person_z = np.zeros_like(group_z)
                for cid in cids:
                    donors, table = tables[cid]
                    chosen = pairings[row["reference_scope"]][cid]
                    mapping = {donor: index for index, donor in enumerate(donors)}
                    ii = np.array([mapping.get(donor, -1) for donor in chosen])
                    rr = np.full_like(group_z, np.nan)
                    present = ii >= 0
                    no_donor |= ~present
                    rr[present] = table[signature][ii[present]]
                    valid &= np.isfinite(rr).all(1)
                    person_z += np.where(np.isfinite(rr), np.arctanh(np.clip(rr, -1 + 1e-7, 1 - 1e-7)), 0.)
                group_z += person_z / len(cids)
            reference = np.tanh(group_z / len(users))
            expected[valid] = reference[valid].max(1)
            indices[valid] = reference[valid].argmax(1)
            values = expected[valid]
            tail = (1 + np.count_nonzero(values >= obs.max())) / (len(values) + 1) if len(values) else np.nan
            compare_scalar(row["reference_tail_fraction"], tail, f"reference_tail/{token}", errors)
            if int(row["failure_no_donor"]) != int(no_donor.sum()) or int(row["failure_incomplete_curve"]) != int((~valid & ~no_donor).sum()):
                errors.append({"check": "reference_failure_reason_accounting", "token": token})
        compare_arrays(stored, expected, f"reference_draws/{token}", errors, 1e-11)
        if not np.array_equal(stored_indices, indices):
            errors.append({"check": "reference_peak_indices", "token": token})
        if int(row["n_valid"]) != int(np.isfinite(expected).sum()) or int(row["n_failed"]) != int((~np.isfinite(expected)).sum()):
            errors.append({"check": "reference_attempt_accounting", "token": token})
        checked += 1
    families = defaultdict(list)
    for row in reference_rows:
        families[tuple(row[key] for key in ("reference_scope", "side", "signal", "mode"))].append(row)
    for key, rows in families.items():
        size = max(9, len(rows))
        values = [as_number(row["reference_tail_fraction"]) for row in rows]
        p = np.array([value if np.isfinite(value) else 1. for value in values] + [1.] * (size - len(rows)))
        order = np.argsort(p)
        ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
        adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty(len(p))
        q[order] = np.clip(adjusted, 0, 1)
        for index, row in enumerate(rows):
            expected_q = q[index] if np.isfinite(values[index]) else np.nan
            compare_scalar(row["reference_bh_descriptive"], expected_q, f"reference_BH/{key}/{row['task']}/{row['stage']}", errors)
            if int(row["reference_family_size"]) != size:
                errors.append({"check": "reference_predefined_stage_family", "family": key})
    return {"reference_groups_reconstructed_from_actual_pairings": checked,
            "donor_draw_sequences_reconstructed": donor_draws_checked,
            "actual_pairings_checked": total_pairings, "same_source_donor_pairings": shared_stimulus,
            "descriptive_BH_families_checked": len(families)}


def validate_passive(output, inputs, frozen):
    errors = []
    curves = read_curves(output / "record_curves.jsonl")
    records = normalize_modes(json.loads((output / "run_manifest.json").read_text(encoding="utf8")))["records"]
    prepared = prepare({row["content_id"]: inputs / "cache" / (row["content_id"] + ".npz") for row in records}, frozen)
    identities = {str(row["content_id"]): str(row["user"]) for row in records}
    if len(identities) != len(records):
        errors.append({"check": "unique_recording_identifiers"})
    for row in records:
        recording_id = str(row.get("recording_id", row["content_id"]))
        expected_observer = recording_id.split("_")[0]
        if str(row["user"]) != expected_observer:
            errors.append({"check": "observer_from_passive_recording_identity", "record": recording_id})
        if expected_observer in ("User96", "User97", "User98"):
            errors.append({"check": "intermediate_observer_excluded", "record": recording_id})
        source_id = str(row.get("source_content_id", ""))
        if not source_id or source_id.split("_")[0] in ("User96", "User97", "User98"):
            errors.append({"check": "novice_source_identity", "record": recording_id})
    for curve in curves:
        if str(curve["user"]) != identities[curve["content_id"]]:
            errors.append({"check": "record_observer_identity", "record": curve["content_id"]})
    direct = validate_direct(curves, prepared, errors)
    groups, people = validate_groups(output, curves, errors)
    occurrence = validate_occurrence_bootstrap(output, errors)
    contrasts = validate_contrasts(output, people, errors)
    reference = validate_reference(output, curves, records, prepared, errors)
    spatial = read_csv(output / "spatial_records.csv")
    crossed = [row for row in spatial if row["mode"] == "Both in stage" and as_number(row["cross_boundary_pairs"]) != 0]
    if crossed:
        errors.append({"check": "Both in stage_no_cross_occurrence_pairs", "count": len(crossed)})
    return {"passed": not errors, "direct_primary": direct, "groups": groups,
            "occurrence_bootstrap": occurrence, "stage_contrasts": contrasts,
            "unpaired_reference": reference, "Both in stage_crossing_rows": len(crossed), "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inputs", type=Path, default=WORK / "inputs")
    parser.add_argument("--frozen", type=Path, default=WORK / "frozen")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = baseline_regression() if args.baseline_only else validate_passive(args.output, args.inputs, args.frozen)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
