"""Hierarchical elapsed-time unpaired references for standalone GT gaze/PSM.

The main rule matches model v2: same task, different participant, elapsed-time
alignment, without warping, looping or extrapolation. Donor participants are
sampled equally, then their recordings equally. Every observed complete-lag
record and participant stays fixed for a group; failed draws are not rerolled.
The optional duration subset uses an independently recomputed observed statistic.
All tail fractions and BH values are descriptive, conditional references, not
tests that the lag is positive or physiological.
"""
from __future__ import annotations
from gaze_psm_modes import seed_identity

import csv
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FIELDS = ("task", "side", "signal", "stage", "mode")
EMPTY_STAGES = {"", "none", "nan", "null"}


def _seed(*values):
    return int.from_bytes(hashlib.sha256(repr(seed_identity(values)).encode()).digest()[:8], "little")


def _write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not np.isfinite(value):
                    value = None
                safe[key] = value
            writer.writerow(safe)


def _z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -1 + 1e-7, 1 - 1e-7))


def hierarchical_curve(record_curves, record_users):
    """Equal records per user followed by equal users, with no missing support."""
    by_user = defaultdict(list)
    for cid, r in record_curves.items():
        a = np.asarray(r, float)
        if not np.isfinite(a).all():
            raise ValueError("hierarchical_curve requires complete-lag recordings")
        by_user[record_users[cid]].append(_z(a))
    if not by_user:
        return np.array([], dtype=float)
    return np.tanh(np.mean([np.mean(a, axis=0) for a in by_user.values()], axis=0))


def donor_choices(records, count, coverage=None, seed=42):
    """Return donor IDs per target; user equal then recording equal, once only."""
    records = sorted(records, key=lambda r: r["content_id"])
    choices, candidate_ids, coverage_rows = {}, {}, []
    for target in records:
        cid = target["content_id"]
        users = defaultdict(list)
        for donor in records:
            did = donor["content_id"]
            if donor["task"] != target["task"] or donor["user"] == target["user"]:
                continue
            if coverage is not None and not coverage(cid, did):
                continue
            users[donor["user"]].append(did)
        people = sorted(users)
        candidates = sorted(d for pp in users.values() for d in pp)
        candidate_ids[cid] = candidates
        draws = np.full(count, "", dtype=object)
        rng = np.random.default_rng(_seed(seed, cid))
        if people:
            person_indices = rng.integers(0, len(people), count)
            for index, person in enumerate(people):
                where = np.flatnonzero(person_indices == index)
                rr = sorted(users[person])
                picks = rng.integers(0, len(rr), len(where))
                draws[where] = np.asarray(rr, dtype=object)[picks]
        choices[cid] = draws
        coverage_rows.append(dict(content_id=cid, task=target["task"], user=target["user"],
                                  n_donor_users=len(people), n_donor_records=len(candidates),
                                  donor_users=people, donor_records=candidates,
                                  attempted_draws=count, no_eligible_donor=not bool(people)))
    return choices, candidate_ids, coverage_rows


def _signals(prepared, cid):
    value = prepared[cid]
    return value.get("signals", value)


def _precompute_target(job):
    from gaze_psm_reliability_v2 import occurrence_statistics, aggregate_statistics

    cid, donor_ids, signatures, prepared, directory, config = job
    started = time.monotonic()
    arrays = {}
    meta = []
    hz = float(config.get("hz", 41.))
    max_lag = float(config.get("max_lag", 2.))
    min_overlap = float(config.get("min_overlap", 2.))
    min_rms = float(config.get("min_rms", 1.))
    for si, signature in enumerate(signatures):
        side, signal, stage, mode = signature
        target = _signals(prepared, cid)[signal][side]
        psm = np.asarray(target["psm"], float)
        ids = np.where(np.asarray(target["stage"]) == stage, target["occurrence"], -1)
        curves = []
        for donor in donor_ids:
            source = _signals(prepared, donor)[signal][side]
            original = np.asarray(source["gazes"]["gt"], float)
            n = min(len(psm), len(original))
            if not np.allclose(target["time_s"][:n], source["time_s"][:n], rtol=0, atol=1e-9):
                raise ValueError(f"elapsed-time grids disagree: {cid}/{donor}/{signal}")
            gaze = np.full_like(psm, np.nan)
            gaze[:n] = original[:n]
            stats = occurrence_statistics(gaze, psm, ids, hz, max_lag, mode == "Both in stage")
            curve = aggregate_statistics(stats, min_overlap=min_overlap, min_rms=min_rms)
            curves.append(curve["r"])
        key = f"curve_{si:03d}"
        lags = np.arange(-round(max_lag * hz), round(max_lag * hz) + 1) / hz
        arrays[key] = np.asarray(curves, float).reshape(len(donor_ids), len(lags))
        meta.append(dict(key=key, signature=list(signature)))
    arrays["metadata_json"] = np.asarray(json.dumps(meta))
    arrays["donors"] = np.asarray(donor_ids, dtype="U128")
    arrays["elapsed_seconds"] = np.asarray(time.monotonic() - started)
    np.savez_compressed(Path(directory) / f"{cid}.npz", **arrays)
    table = {tuple(m["signature"]): arrays[m["key"]] for m in meta}
    return cid, donor_ids, table, float(arrays["elapsed_seconds"])


def _group_reference(record_ids, signature, observed, users, choices, tables, donor_ids, count):
    """Vectorized fixed-record/group replicate aggregation; missing poisons draw."""
    lag_count = len(next(iter(observed.values())))
    by_user = defaultdict(list)
    for cid in record_ids:
        by_user[users[cid]].append(cid)
    group_z = np.zeros((count, lag_count), dtype=float)
    valid = np.ones(count, dtype=bool)
    no_donor = np.zeros(count, dtype=bool)
    for user, cids in sorted(by_user.items()):
        person_z = np.zeros_like(group_z)
        for cid in cids:
            index = {d: i for i, d in enumerate(donor_ids[cid])}
            selected = np.fromiter((index.get(str(d), -1) for d in choices[cid]), int, count)
            present = selected >= 0
            no_donor |= ~present
            record_r = np.full_like(group_z, np.nan)
            if present.any():
                record_r[present] = tables[cid][signature][selected[present]]
            finite = np.isfinite(record_r).all(axis=1)
            valid &= finite
            person_z += np.where(np.isfinite(record_r), _z(record_r), 0.)
        group_z += person_z / len(cids)
    if by_user:
        group_r = np.tanh(group_z / len(by_user))
    else:
        group_r = np.full_like(group_z, np.nan)
        valid[:] = False
    peaks = np.full(count, np.nan)
    indices = np.full(count, -1, dtype=np.int32)
    if valid.any():
        peaks[valid] = group_r[valid].max(axis=1)
        indices[valid] = group_r[valid].argmax(axis=1)
    return peaks, indices, valid, no_donor


def _bh_descriptive(rows, stage_counts):
    families = defaultdict(list)
    for row in rows:
        families[(row["reference_scope"], row["side"], row["signal"], row["mode"])].append(row)
    total = sum(stage_counts.values())
    for key, members in families.items():
        pvalues = [float(r["reference_tail_fraction"]) if r["reference_tail_fraction"] is not None else 1. for r in members]
        # All predefined stages stay in the family; absent tests enter as p=1.
        pvalues.extend([1.] * max(0, total - len(members)))
        order = np.argsort(pvalues)
        adjusted = np.minimum.accumulate((np.asarray(pvalues)[order] * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1])[::-1]
        q = np.empty(len(pvalues))
        q[order] = np.clip(adjusted, 0, 1)
        for i, row in enumerate(members):
            row["reference_bh_descriptive"] = float(q[i]) if row["reference_tail_fraction"] is not None else None
            row["reference_family_size"] = len(pvalues)
            row["reference_family"] = "scope + side + signal + mode; all predefined A/C stages"
            row["bh_formal_fdr_control_claimed"] = False


def run_reference(prepared, records, curves, groups, output, count=5000, workers=4,
                  config=None, duration_subset=True, min_users=3):
    """Write fixed-side references; returns summary rows. Does not select sides.

    ``prepared[cid]`` is prepare_signals output, or its ``signals`` dictionary.
    ``records`` holds content_id/task/user. ``curves`` contains primary/gt fixed
    l/r curves. ``groups`` is used to check reconstructed observed statistics.
    Config must match those observed curves; defaults mirror the model kernel.
    """
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    config = dict(config or {})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    directory = out / "unpaired_pair_curves"
    directory.mkdir(exist_ok=True)
    records = [dict(r, user=r.get("user", r["content_id"].split("_")[0])) for r in records]
    user = {r["content_id"]: str(r["user"]) for r in records}
    original = defaultdict(dict)
    signatures = defaultdict(set)
    stage_names = defaultdict(set)
    for c in curves:
        if c.get("variant", "primary") != "primary" or c.get("source", "gt") != "gt" or c["side"] not in ("l", "r"):
            continue
        if c["mode"] not in ("PSM in stage", "Both in stage"):
            continue
        key = tuple(c[k] for k in FIELDS)
        stage_names[c["task"]].add(c["stage"])
        original.setdefault(key, {})
        r = np.asarray(c["r"], float)
        if np.isfinite(r).all():
            original[key][c["content_id"]] = r
            signatures[c["content_id"]].add(key[1:])
    eligible_records = [r for r in records if r["content_id"] in prepared]
    choices, candidates, coverage = donor_choices(eligible_records, count)

    def full_duration(cid, donor):
        for signal in ("position", "velocity"):
            for side in ("l", "r"):
                a = _signals(prepared, cid)[signal][side]["time_s"]
                b = _signals(prepared, donor)[signal][side]["time_s"]
                if len(b) < len(a) or (len(a) and b[-1] < a[-1] - 1e-9):
                    return False
        return True

    duration_choices, duration_candidates, duration_coverage = donor_choices(eligible_records, count, full_duration, seed=31415)
    for row, drow in zip(coverage, duration_coverage):
        row["n_full_duration_donor_users"] = drow["n_donor_users"]
        row["n_full_duration_donor_records"] = drow["n_donor_records"]
    _write(out / "unpaired_donor_coverage.csv", coverage)
    scopes = [("elapsed_time_model_consistent", choices)]
    if duration_subset:
        scopes.append(("duration_eligible_subset", duration_choices))
    with (out / "unpaired_pairings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["reference_scope", "replicate", "target", "donor", "donor_covers_full_target"])
        for scope, draw_map in scopes:
            for target in sorted(draw_map):
                for b, donor in enumerate(draw_map[target]):
                    writer.writerow([scope, b, target, donor, full_duration(target, donor) if donor else False])
    jobs = [(cid, candidates[cid], sorted(ss), prepared, str(directory), config) for cid, ss in sorted(signatures.items())]
    tables, ordered_donors, timings = {}, {}, []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_precompute_target, job): job[0] for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            cid, donors, table, elapsed = future.result()
            tables[cid], ordered_donors[cid] = table, donors
            timings.append(dict(content_id=cid, seconds=elapsed, donors=len(donors), conditions=len(table)))
            if done == 1 or done % 5 == 0 or done == len(jobs):
                print(f"Reference pair curves {done}/{len(jobs)} targets; {time.monotonic()-started:.1f}s", flush=True)
    _write(out / "unpaired_precompute_timing.csv", timings)
    group_lookup = {tuple(g[k] for k in FIELDS): g for g in groups
                    if g.get("variant", "primary") == "primary" and g.get("source", "gt") == "gt"}
    summary, arrays, metadata = [], {}, []
    for scope, draw_map in scopes:
        for key, all_observed in sorted(original.items()):
            keep = sorted(all_observed)
            if scope == "duration_eligible_subset":
                keep = [cid for cid in keep if duration_candidates[cid]]
            observed = {cid: all_observed[cid] for cid in keep}
            people = sorted({user[cid] for cid in keep})
            meta = dict(zip(FIELDS, key))
            row = dict(meta, variant="primary", source="gt", reference_scope=scope,
                       n_requested=count, n_observed_records=len(keep), n_observed_users=len(people),
                       target_records=keep, target_users=people,
                       excluded_duration_records=sorted(set(all_observed)-set(keep)),
                       reference_tail_fraction=None, reference_peak_r_95=None, observed_peak_r=None,
                       n_valid=0, n_failed=count, failure_no_donor=0, failure_incomplete_curve=0,
                       status="insufficient_observed_users", formal_leading_p_value=False,
                       support_policy="fixed observed complete-lag recordings, then equal users; no rerolls")
            token = f"g{len(metadata):04d}"
            if len(people) >= min_users:
                obs = hierarchical_curve(observed, user)
                row["observed_peak_r"] = float(obs.max())
                if scope == "elapsed_time_model_consistent" and key in group_lookup:
                    expected = group_lookup[key].get("peak_r")
                    if expected is not None and np.isfinite(float(expected)) and not np.isclose(float(expected), obs.max(), atol=1e-10, rtol=1e-10):
                        raise ValueError(f"observed reference differs from group primary: {key}")
                peaks, indices, valid, no_donor = _group_reference(keep, key[1:], observed, user, draw_map, tables, ordered_donors, count)
                values = peaks[valid]
                row.update(n_valid=int(valid.sum()), n_failed=int((~valid).sum()),
                           failure_no_donor=int(no_donor.sum()),
                           failure_incomplete_curve=int((~valid & ~no_donor).sum()),
                           reference_tail_fraction=float((1+np.count_nonzero(values >= obs.max()))/(len(values)+1)) if len(values) else None,
                           reference_peak_r_95=float(np.percentile(values, 95)) if len(values) else None,
                           valid_fraction=float(valid.mean()),
                           status="conditional_descriptive_reference" if len(values) else "no_complete_reference_draws")
            else:
                peaks = np.full(count, np.nan)
                indices = np.full(count, -1, np.int32)
            row["draws_key"] = token
            arrays[token + "_peak_r"] = peaks
            arrays[token + "_peak_index"] = indices
            metadata.append(dict(meta, reference_scope=scope, key=token))
            summary.append(row)
    # The planned task family has 5 A stages and 4 C stages. Unknown additions
    # remain included, never silently shrinking the adjustment denominator.
    stage_counts = {"A": max(5, len(stage_names.get("A", []))), "C": max(4, len(stage_names.get("C", [])))}
    _bh_descriptive(summary, stage_counts)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    arrays["lag_s"] = np.arange(-round(config.get("max_lag", 2.)*config.get("hz", 41.)), round(config.get("max_lag", 2.)*config.get("hz", 41.))+1)/config.get("hz", 41.)
    np.savez_compressed(out / "unpaired_reference_peaks.npz", **arrays)
    _write(out / "unpaired_reference.csv", summary)
    (out / "unpaired_reference_method.json").write_text(json.dumps(dict(
        method="same-task different-user elapsed-time unpaired reference",
        attempts_per_group=count, main_rule="matches model v2; no extrapolation; incomplete group draws discarded without replacement",
        donor_sampling="equal eligible donor participants, then equal recordings within selected donor",
        hierarchy="fixed complete observed recordings equally within user; users equally",
        duration_subset="separate observed statistic; only targets with full-duration other-user donors; never substituted for full primary",
        missing_reference="exported missing, not zero or nonsignificant", formal_p_or_fdr_claim=False,
        predefined_stage_counts=stage_counts, selected_side_reference_computed=False,
        elapsed_seconds=time.monotonic()-started), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
