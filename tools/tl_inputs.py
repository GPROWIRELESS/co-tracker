"""Read native novice A/C records and previously exported conditional gaze maps."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

import numpy as np


MODEL_NAMES = ("all_ABCD", "single_task", "leave_out", "spd_finetune", "spd_zero_shot")
PREDICTION_MANIFEST = Path("outputs/gaze_psm_xcorr_models_max_dlc_current_data/prediction_manifest.csv")
CANONICAL_STAGES = {
    "rr": "reach ring", "gr": "grab ring", "tr": "transfer ring", "pr": "place ring",
    "wrap": "wrap around psm", "cinch": "pull through and cinch",
    "thread": "threading string through tower",
}
GAZE_ALLOWED_CODES = (2, 3)
STATE_AUDIT_VERSION = 2
BUNDLE_VERSION = 2
EYE_STATUS_MEANING = {"0": "not tracked", "1": "visible but not reliably tracked",
                      "2": "tracked, quality compromised", "3": "tracked"}
EYE_STATUS_REFERENCE = "https://developer.varjo.com/docs/native/eye-tracking"


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@lru_cache(maxsize=2048)
def _sha256_at_stat(path, size, modified_ns, changed_ns):
    # This is only an in-process cache. Every fresh process hashes the actual file.
    path = Path(path)
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _sha256(path):
    path = Path(path).resolve()
    before = path.stat()
    value = _sha256_at_stat(str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Source changed while hashing {path}")
    return value


def _atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".input-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def _ns_integer(value):
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    value = Decimal(text)
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("Timestamp is not an integer number of nanoseconds")
    return int(value)


def _counts(values):
    values, counts = np.unique(values, return_counts=True)
    return {str(v): int(n) for v, n in zip(values, counts)}


def _projection_evidence(raw, target_xy, valid_code, width, height, matrices):
    """Compare candidate stored calibration transforms; never replace the gaze.

    Parameter provenance is not known. Trying all available participant matrices
    is explicitly diagnostic, not a retrospective recalibration or data correction.
    """
    candidates = []
    for path in matrices:
        try:
            params = _rows(path)[0]
            d, ox, oy = (float(params[k]) for k in ("OptimalDistance", "OptimalOffsetX", "OptimalOffsetY"))
            a = np.array([float(params.get("OptimalAngleAxis"+k, 0)) for k in "XYZ"])
            theta = np.linalg.norm(a)
            rotation = np.eye(3)
            if theta > 1e-12:
                k = a / theta
                cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                rotation += np.sin(theta) * cross + (1-np.cos(theta)) * cross @ cross
            for prefix in ("Left", "Gaze", "Right"):
                cols = [prefix + " Forward " + axis for axis in "XYZ"]
                if not set(cols) <= set(raw.columns):
                    continue
                v = raw[cols].to_numpy(float)
                denom = v @ rotation[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    projected = v * (d / denom)[:, None]
                    xy = np.column_stack((projected @ rotation[:, 0] + width/2 + ox,
                                          -projected @ rotation[:, 1] + height/2 + oy))
                error = np.linalg.norm(xy-target_xy, axis=1)
                ok = np.isfinite(error) & np.isin(valid_code, (2, 3)) & (np.abs(denom) > 1e-8)
                if ok.any():
                    candidates.append(dict(eye=prefix, calibration_path=str(path), n=int(ok.sum()),
                                           median_error_px=float(np.median(error[ok])),
                                           p95_error_px=float(np.percentile(error[ok], 95))))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            candidates.append(dict(calibration_path=str(path), error=str(exc)))
    best = {}
    for eye in ("Left", "Gaze", "Right"):
        rows = [c for c in candidates if c.get("eye") == eye]
        if rows:
            best[eye] = min(rows, key=lambda r: r["median_error_px"])
    return best, candidates


def _raw_state_audit(base, cid, frames, ns, gaze_xy, valid_code, width, height,
                     npy_checksum, audit_cache_dir=None):
    """Nearest original samples are evidence about code/coordinate lineage only."""
    raw_path = base / "csv" / f"{cid}_gaze.csv"
    backup_path = base / "npy" / f"{cid}_gaze.before_event_copy.bak.csv"
    matrices = sorted((base / "cali").glob("*_calibrationmatrix.csv"))
    audit = dict(raw_gaze_path=str(raw_path), raw_gaze_available=raw_path.is_file(),
                 gaze_state_evidence="unverified", gaze_coordinate_lineage="not established",
                 eye_status_meaning=EYE_STATUS_MEANING, eye_status_reference=EYE_STATUS_REFERENCE,
                 raw_state_mapping_is_diagnostic=True,
                 synchronization_scope="shared recorded Unix timeline; exposure/display latency not independently verified")
    arrays = {key: np.full(len(frames), np.nan) for key in
              ("raw_left_status", "raw_right_status", "raw_combined_status", "raw_time_gap_ms")}
    if not raw_path.is_file():
        return arrays, audit
    hashes = {str(p): _sha256(p) for p in [raw_path, *matrices]}
    if backup_path.is_file():
        hashes[str(backup_path)] = _sha256(backup_path)
    signature = dict(version=STATE_AUDIT_VERSION, files=hashes, npy_sha256=npy_checksum,
                     frames_sha256=hashlib.sha256(np.asarray(frames, dtype="<i8").tobytes()).hexdigest(),
                     timestamps_sha256=hashlib.sha256(np.asarray(ns, dtype="<i8").tobytes()).hexdigest(),
                     size=[width, height])
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(audit_cache_dir)/"gaze_state"/f"{cid}_{key}.npz" if audit_cache_dir else None
    if cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            cached_audit = json.loads(str(saved["audit_json"]))
            result = {name: saved[name] for name in arrays}
        if all(a.shape == (len(frames),) for a in result.values()):
            cached_audit["raw_state_audit_cache_hit"] = True
            return result, cached_audit
    import pandas as pd
    names = set(pd.read_csv(raw_path, nrows=0).columns)
    needed = {"Capture Unix timestamp", "Left Status", "Right Status", "Status", "norm_x", "norm_y"}
    needed.update(prefix+" Forward "+axis for prefix in ("Left", "Right", "Gaze") for axis in "XYZ")
    if "Capture Unix timestamp" not in names:
        audit["raw_state_error"] = "Raw gaze lacks Capture Unix timestamp"
        return arrays, audit
    raw = pd.read_csv(raw_path, usecols=sorted(needed & names), dtype={"Capture Unix timestamp": "string"})
    total_raw_rows = len(raw)
    parsed = []
    valid_rows = []
    for i, value in enumerate(raw["Capture Unix timestamp"]):
        try:
            stamp = _ns_integer(value)
            if not np.iinfo(np.int64).min <= stamp <= np.iinfo(np.int64).max:
                raise ValueError("timestamp outside int64")
            parsed.append(stamp)
            valid_rows.append(i)
        except (ValueError, InvalidOperation, TypeError):
            pass
    raw = raw.iloc[valid_rows].copy()
    stamps = np.asarray(parsed, dtype=np.int64)
    if not len(stamps):
        audit["raw_state_error"] = "Raw gaze contains no parseable Unix nanoseconds"
        return arrays, audit
    order = np.argsort(stamps, kind="stable")
    stamps, raw = stamps[order], raw.iloc[order].reset_index(drop=True)
    right = np.searchsorted(stamps, ns).clip(0, len(stamps)-1)
    left = np.maximum(right-1, 0)
    nearest = np.where(np.abs(ns-stamps[left]) <= np.abs(ns-stamps[right]), left, right)
    matched = raw.iloc[nearest].reset_index(drop=True)
    for name in set(matched.columns)-{"Capture Unix timestamp"}:
        matched[name] = pd.to_numeric(matched[name], errors="coerce")
    gaps = (ns-stamps[nearest])/1e6
    tolerance = float(1.5*np.median(np.diff(ns))/1e6)
    within = np.abs(gaps) <= tolerance
    arrays["raw_time_gap_ms"] = gaps
    audit.update(raw_gaze_sha256=hashes[str(raw_path)], raw_state_source_hashes=hashes,
                 raw_gaze_parseable_rows=len(stamps), raw_gaze_unparseable_timestamp_rows=total_raw_rows-len(stamps),
                 raw_gaze_duplicate_timestamps=int(np.sum(np.diff(stamps)==0)),
                 raw_matching="nearest capture Unix timestamp; tie uses earlier sample",
                 raw_match_tolerance_ms=tolerance, raw_match_within_tolerance_n=int(within.sum()),
                 raw_match_gap_abs_median_ms=float(np.median(np.abs(gaps))),
                 raw_match_gap_abs_p95_ms=float(np.percentile(np.abs(gaps),95)),
                 raw_match_gap_abs_max_ms=float(np.max(np.abs(gaps))), raw_state_audit_cache_hit=False)
    for name, col in (("raw_left_status", "Left Status"), ("raw_right_status", "Right Status"),
                      ("raw_combined_status", "Status")):
        if col in matched:
            values = matched[col].to_numpy(float)
            arrays[name] = np.where(within, values, np.nan)
            ok = within & np.isfinite(values)
            n = int(ok.sum())
            matches = int(np.sum(valid_code[ok] == values[ok]))
            audit[name+"_compared_n"] = n
            audit[name+"_match_n"] = matches
            audit[name+"_match_fraction"] = matches/n if n else None
            audit[name+"_confusion"] = _counts(np.array([f"{a:g}->{b:g}" for a,b in zip(valid_code[ok], values[ok])]))
    if audit.get("raw_left_status_match_n") == len(frames):
        audit["gaze_state_evidence"] = "all native valid codes equal nearest raw Left Status within tolerance"
    elif audit.get("raw_left_status_match_fraction", 0) and audit["raw_left_status_match_fraction"] >= .99:
        audit["gaze_state_evidence"] = "at least 99% of comparable native codes equal nearest raw Left Status; see exceptions"
    else:
        audit["gaze_state_evidence"] = "native validity source not resolved by nearest-status comparison"
    target = gaze_xy
    audit["projection_reference_gaze"] = "current NPY gaze"
    if backup_path.is_file():
        backup = pd.read_csv(backup_path, dtype={"Timestamp(ns)": "string"})
        if {"Frame_Number", "Timestamp(ns)", "gaze_x", "gaze_y", "valid"} <= set(backup):
            if not backup.Frame_Number.duplicated().any() and np.isin(frames, backup.Frame_Number).all():
                backup = backup.set_index("Frame_Number").loc[frames]
                bns = np.asarray([_ns_integer(v) for v in backup["Timestamp(ns)"]], dtype=np.int64)
                target = backup[["gaze_x", "gaze_y"]].to_numpy(float)*[width,height]
                audit.update(backup_gaze_path=str(backup_path), backup_gaze_sha256=hashes[str(backup_path)],
                             backup_video_timestamp_match_n=int(np.sum(bns==ns)),
                             backup_valid_code_match_n=int(np.sum(backup.valid.to_numpy()==valid_code)),
                             projection_reference_gaze="before_event_copy backup, by original Frame_Number")
                diff = np.linalg.norm(target-gaze_xy, axis=1)
                audit["current_backup_gaze_changed_over_0_01px_n"] = int(np.sum(diff>.01))
    if {"norm_x", "norm_y"} <= set(matched):
        diff = np.linalg.norm(matched[["norm_x", "norm_y"]].to_numpy(float)*[width,height]-target, axis=1)
        ok = within & np.isfinite(diff) & np.isin(valid_code,(2,3))
        audit["raw_norm_to_npy_median_error_px"] = float(np.median(diff[ok])) if ok.any() else None
    projection_codes = np.where(within, valid_code, -1)
    best, candidates = _projection_evidence(matched, target, projection_codes, width, height, matrices)
    audit["projection_candidate_comparison"] = candidates
    audit["projection_best_per_eye"] = best
    audit["projection_comparison_scope"] = "diagnostic search over stored participant matrices; not evidence of exact historical calibration selection"
    if set(best) == {"Left", "Gaze", "Right"} and all(best["Left"]["median_error_px"] < best[k]["median_error_px"] for k in ("Gaze","Right")):
        audit["gaze_coordinate_lineage"] = "stored-calibration reconstruction favors left-eye projection; final historical NPY generation not located"
    if cache:
        _atomic_npz(cache, **arrays, audit_json=np.asarray(json.dumps(audit, sort_keys=True)))
    return arrays, audit


def records(repo):
    """Join all novice A/C index rows to scores by user, task number and attempt."""
    index = Path(repo) / "Index"
    scores = {}
    for task in ("A", "C"):
        for row in _rows(index / f"Task_{task}_Score.csv"):
            if row["UserType"].strip().lower() != "novice":
                continue
            key = (task, int(row["UserID"]), int(row["TaskNum"]), int(row["Attempt"]))
            if key in scores:
                raise ValueError(f"Duplicate score key {key}")
            scores[key] = row

    output = []
    seen = set()
    task_numbers = defaultdict(set)
    for row in _rows(index / "content_index.csv"):
        task = row["Task Name"].strip()
        if task not in ("A", "C") or row["src_skill"].strip().lower() != "novice":
            continue
        uid, num, attempt = (int(row[k]) for k in ("User Num", "Task Num", "Attempt Num"))
        participant = f"User{uid}"
        cid = row["content_id"].strip()
        if cid != f"{participant}_{num}_{attempt}" or cid in seen:
            raise ValueError(f"Duplicate or inconsistent content index ID: {cid}")
        seen.add(cid)
        score_row = scores.get((task, uid, num, attempt))
        status = score_row["AttemptStatus"].strip() if score_row else "missing"
        score = np.nan
        if status in {"Completed", "Failed"} and score_row["Score"].strip():
            score = float(score_row["Score"])
            if not np.isfinite(score):
                score = np.nan
        output.append(dict(content_id=cid, participant=participant, task=task, task_num=num,
                           attempt=attempt, score=score, status=status))
        task_numbers[participant, task].add(num)
    ordering = {key: {num: i + 1 for i, num in enumerate(sorted(nums))}
                for key, nums in task_numbers.items()}
    for rec in output:
        rec["round"] = ordering[rec["participant"], rec["task"]][rec["task_num"]]
    return sorted(output, key=lambda r: (int(r["participant"][4:]), r["task_num"], r["attempt"]))


def train_ids(repo, task):
    path = Path(repo) / "Index" / "items" / f"novice_active_task{task}" / "train_items.json"
    result = set()
    for entry in json.loads(path.read_text(encoding="utf-8-sig")):
        meta = entry["meta"]
        if meta["task_name"] != task or meta["src_skill"] != "novice" or meta["split"] != "train":
            raise ValueError(f"Unexpected training manifest row in {path}")
        result.add(meta["content_id"])
    return result


@lru_cache(maxsize=2)
def _pipeline(repo):
    # The project tool imports its own 'tools' package using the repo as its root.
    home = str(Path(repo).resolve())
    if home not in sys.path:
        sys.path.insert(0, home)
    module = importlib.import_module("tools.gaze_psm_pipeline_dlc")
    if Path(module.__file__).resolve().parent.parent != Path(home):
        raise ValueError("A different co-tracker checkout is already imported; start a fresh process")
    return module


def _take_frame_rows(source, frames, path):
    ids = np.asarray(source["Frame_Number"], dtype=np.int64)
    if np.array_equal(ids, frames):
        return source
    order = np.argsort(ids)
    sorted_ids = ids[order]
    pos = np.searchsorted(sorted_ids, frames)
    if (np.any(np.diff(sorted_ids) == 0) or np.any(pos >= len(ids))
            or not np.array_equal(sorted_ids[pos], frames)):
        raise ValueError(f"Missing or duplicate retained Frame_Number in {path}")
    return source[order[pos]]


def _points(source, cid):
    names = set(source.dtype.names)
    result = np.full((len(source), 5, 2), np.nan, dtype=float)
    if not {"point_1_x", "point_1_y"} <= names:
        raise ValueError(f"{cid} lacks the primary target slot")
    for slot in range(1, 6):
        x, y = f"point_{slot}_x", f"point_{slot}_y"
        if (x in names) != (y in names):
            raise ValueError(f"{cid} has an incomplete schema for target slot {slot}")
        if x in names:
            result[:, slot - 1, 0] = source[x]
            result[:, slot - 1, 1] = source[y]
    return result


def _audit_knot_center(source, slot_one):
    """Check the optional stored annotation against slot 1 without changing any target."""
    name = "purple_string_knot_center"
    present = name in source.dtype.names
    duplicate = conflict = unmatched = malformed = 0
    if present:
        values = np.char.strip(source[name].astype(str))
        for i in np.flatnonzero(values != ""):
            try:
                pair = np.asarray(json.loads(values[i]), dtype=float)
                if pair.shape != (2,) or not np.isfinite(pair).all():
                    raise ValueError("expected a finite x/y pair")
            except (TypeError, ValueError):
                malformed += 1
                continue
            if not np.isfinite(slot_one[i]).all():
                unmatched += 1
            elif np.abs(pair - slot_one[i]).max() <= 1e-6:
                duplicate += 1
            else:
                conflict += 1
    return dict(knot_center_field_present=present,
                knot_center_stored_n=duplicate+conflict+unmatched+malformed,
                knot_center_duplicate_slot1_n=duplicate, knot_center_conflict_slot1_n=conflict,
                knot_center_without_slot1_n=unmatched, knot_center_malformed_n=malformed,
                knot_center_match_tolerance_px=1e-6)


def read_record(rec, data_root, repo, *, gaze_allowed_codes=GAZE_ALLOWED_CODES, audit_cache_dir=None):
    """Load native timestamps, allocated gaze, raw target slots and bilateral DLC points."""
    pipeline = _pipeline(str(Path(repo).resolve()))
    cid = rec["content_id"]
    active = Path(data_root)
    if active.name.lower() != "active_data":
        active = active / "Active_Data"
    base = active / rec["participant"]
    npy = base / "npy" / f"{cid}_gaze.npy"
    video = base / "stimuli" / f"{cid}.mp4"
    clock = base / "csv" / f"{cid}_video_ts.csv"
    dlc = pipeline.dlc_path(base, cid)
    frames, ns, deleted = pipeline.timeline(base, cid)
    full = np.load(npy, allow_pickle=False, mmap_mode="r")
    if not full.dtype.names:
        raise ValueError(f"{npy}: expected a named field array")
    source = _take_frame_rows(full, frames, npy)
    points, likelihood = pipeline.read_dlc(dlc, expected_frames=len(frames))
    width, height, fps = pipeline.video_metadata(video)
    names = set(source.dtype.names)
    gaze = np.column_stack((source["gaze_x"], source["gaze_y"])).astype(float)
    valid_code = np.asarray(source["valid"])
    codes = tuple(int(c) for c in gaze_allowed_codes)
    if not codes or len(set(codes)) != len(codes) or not set(codes) <= {1, 2, 3}:
        raise ValueError("gaze_allowed_codes must be a nonempty unique subset of (1,2,3)")
    coordinate_valid = np.isfinite(gaze).all(axis=1) & (gaze >= 0).all(axis=1) & (gaze <= 1).all(axis=1)
    gaze_valid = np.isin(valid_code, codes) & coordinate_valid
    gaze *= (width, height)
    gaze_unmasked = gaze.copy()
    gaze_unmasked[~coordinate_valid] = np.nan
    gaze[~gaze_valid] = np.nan
    raw = np.asarray(source["stage" if "stage" in names else "event"], dtype=str)
    raw_values, raw_index = np.unique(raw, return_inverse=True)
    normalized = [pipeline.normalized_stage(value) for value in raw_values]
    stage = np.asarray([CANONICAL_STAGES.get(s, s) for s in normalized])[raw_index]
    occurrence = np.r_[0, np.cumsum(stage[1:] != stage[:-1])].astype(np.int32)
    mode = np.asarray(source["point_label_mode"], dtype=str) if "point_label_mode" in names else np.full(len(ns), "")
    identity_col = next((s for s in ("target_id", "target_identity", "point_1_id") if s in names), None)
    identity_change = np.zeros(len(ns), dtype=bool)
    if identity_col:
        identity = source[identity_col]
        identity_change[1:] = identity[1:] != identity[:-1]
    npy_ns = np.asarray(source["Timestamp(ns)"], dtype=np.int64)
    error_ns = np.abs(npy_ns - ns)
    audit = dict(content_id=cid, npy_path=str(npy), video_ts_path=str(clock), dlc_path=str(dlc),
                 video_path=str(video), npy_sha256=_sha256(npy), video_ts_sha256=_sha256(clock),
                 dlc_sha256=_sha256(dlc), original_npy_rows=len(full), retained_frames=len(ns),
                 deleted_video_ts_rows=int(deleted), npy_video_timestamp_differences=int(np.count_nonzero(error_ns)),
                 npy_nonincreasing_timestamp_steps=int(np.count_nonzero(np.diff(npy_ns) <= 0)),
                 npy_video_timestamp_max_abs_ns=int(error_ns.max()),
                 npy_video_timestamp_median_abs_ns=float(np.median(error_ns)),
                 clock_source="video_ts.csv matched by original Frame_Number",
                 target_id_field=identity_col or "unavailable", video_nominal_fps=float(fps),
                 native_median_hz=float(1e9 / np.median(np.diff(ns))), width=width, height=height)
    audit.update(gaze_allowed_codes=list(codes), valid_code_counts=_counts(valid_code),
                 gaze_coordinate_valid_n=int(coordinate_valid.sum()), gaze_valid_n=int(gaze_valid.sum()),
                 gaze_legacy123_coordinate_valid_n=int(np.sum(coordinate_valid & np.isin(valid_code,(1,2,3)))),
                 gaze_strict3_coordinate_valid_n=int(np.sum(coordinate_valid & (valid_code==3))),
                 gaze_quality_policy="2/3 main; 3 strict sensitivity; 1/2/3 legacy audit; no raw-status replacement")
    raw_state, state_audit = _raw_state_audit(base, cid, frames, ns,
                                            np.column_stack((source["gaze_x"],source["gaze_y"]))*[width,height],
                                            valid_code, width, height, audit["npy_sha256"], audit_cache_dir)
    audit.update(state_audit)
    slots = _points(source, cid)
    audit.update(_audit_knot_center(source, slots[:, 0]))
    return dict(**rec, frames=frames, time=(ns - ns[0]) / 1e9, timestamp_ns=ns,
                gaze=gaze, gaze_valid=gaze_valid, gaze_unmasked=gaze_unmasked,
                gaze_coordinate_valid=coordinate_valid, valid_code=valid_code, slots=slots, **raw_state,
                stage=stage, stage_raw=raw, occurrence=occurrence, label_mode=mode,
                identity_change=identity_change, left=points["l"], right=points["r"],
                likelihood=np.column_stack((likelihood["l"], likelihood["r"])),
                width=width, height=height, video=str(video), audit=audit)


def models(repo, logs, records):
    """Resolve only pinned prediction files for the requested clips under the supplied logs root."""
    manifest = Path(repo) / PREDICTION_MANIFEST
    root = Path(logs).resolve()
    selected = {r["content_id"]: r["task"] for r in records}
    out = {cid: {} for cid in selected}
    for row in _rows(manifest):
        cid = row["content_id"]
        if cid not in selected:
            continue
        model = row["model"]
        if model not in MODEL_NAMES or model in out[cid] or row["task"] != selected[cid]:
            raise ValueError(f"Invalid or duplicate prediction manifest row for {cid}/{model}")
        parts = PurePosixPath(row["prediction_path"].replace("\\", "/")).parts
        relative = Path(*parts[parts.index("logs") + 1:])
        path = (root / relative).resolve()
        path.relative_to(root)
        if path.name != cid + "_predictions.npz":
            raise ValueError(f"Unexpected filename for {cid}: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["prediction_sha256"]):
            raise ValueError(f"Missing or malformed pinned SHA256 for {path}")
        out[cid][model] = dict(**row, path=str(path), source_manifest=str(manifest))
    for cid, group in out.items():
        if group and set(group) != set(MODEL_NAMES):
            raise ValueError(f"Incomplete matched prediction manifest for {cid}")
    return out


def _sample_count(path):
    # Read only the .npy header of the large samples member.
    with zipfile.ZipFile(path) as archive, archive.open("samples.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0 }[version]
        shape, _, _ = reader(stream)
    if len(shape) != 3 or shape[2] != 2:
        raise ValueError(f"Invalid samples array shape {shape} in {path}")
    return shape


def _prediction_metadata(entry, path, checksum, n, k):
    metadata = dict(entry, observed_sha256=checksum, num_samples=int(k), model_frame_match=True,
                    model_timestamp_match=True, observed_frames=n,
                    coordinate_order="x,y", coordinate_scale="normalized inclusive [0,1]; pixels multiply [width,height]",
                    model_grid_hw=[32,40], sample_grid_coordinates="x=j/39,y=i/31",
                    sample_generation="independent per-frame latent candidates and categorical cells",
                    sample_column_temporal_identity=False,
                    sample_timestamp_scope="one frame_index/frame_number/timestamp per N row applies to all K samples; no separate per-sample acquisition times",
                    source_code_reference="DVSS_Pytorch/dvss_pytorch/model/cvae.py:190-228; scripts/predictor.py export loop",
                    source_revision_provenance="current available generator source; export-time revision not recorded in pinned manifest")
    seed_dir = next((part for part in path.parts if re.fullmatch(r"seed_\d+", part)), None)
    metadata["seed_directory"] = int(seed_dir[5:]) if seed_dir else None
    provenance = path.parent / "provenance.json"
    metadata["provenance_path"] = str(provenance) if provenance.is_file() else None
    if provenance.is_file():
        info = json.loads(provenance.read_text(encoding="utf-8-sig"))
        if info.get("num_samples", k) != k or info.get("split", "test") != "test":
            raise ValueError(f"Saved arrays conflict with inference provenance for {path}")
        for key in ("checkpoint", "checkpoint_sha256", "manifest_sha256", "seed", "split", "model_class", "test_stride"):
            metadata[key] = info.get(key)
    else:
        for key in ("checkpoint", "checkpoint_sha256", "manifest_sha256", "seed", "model_class", "test_stride"):
            metadata[key] = None
        metadata["split"] = "test" if path.parent.name.startswith("inference_test_") else None
    return metadata


@lru_cache(maxsize=8)
def _legacy_audit(path, checksum):
    return {(r["content_id"],r["model"]):r for r in _rows(Path(path))}


def _legacy_expected(legacy_cache_dir, entry, recdata, checksum):
    if legacy_cache_dir is None:
        return None, None
    root = Path(legacy_cache_dir)
    if root.name == "frames":
        root = root.parent
    audit_path = root/"prediction_audit.csv"
    frame_path = root/"frames"/f"{recdata['content_id']}.npz"
    if not audit_path.is_file() or not frame_path.is_file():
        return None, "legacy cache or audit missing"
    row = _legacy_audit(str(audit_path), _sha256(audit_path)).get((recdata["content_id"],entry["model"]))
    if not row or row.get("observed_sha256") != checksum or row.get("prediction_sha256") != checksum:
        return None, "legacy prediction SHA does not match verified current source"
    with np.load(frame_path, allow_pickle=False) as old:
        key = "xy_"+entry["model"]
        if key not in old or not np.array_equal(old["frame_number"], recdata["frames"]) or not np.array_equal(old["timestamp_ns"], recdata["timestamp_ns"]):
            return None, "legacy frame or time mismatch"
        if not np.array_equal(old["resolution"], [recdata["width"],recdata["height"]]):
            return None, "legacy resolution mismatch"
        xy = old[key]
        if xy.shape != (len(recdata["frames"]),2):
            return None, "legacy expected coordinate shape mismatch"
    return xy, dict(path=str(frame_path),sha256=_sha256(frame_path),prediction_audit_sha256=_sha256(audit_path))


def _decode_probability(probability, width, height, expected=None):
    n,h,w = probability.shape
    valid = np.zeros(n, dtype=bool)
    expectedxy = np.full((n,2),np.nan) if expected is None else np.asarray(expected,dtype=float).copy()
    argmaxxy = np.full((n,2),np.nan)
    max_mass_error = 0.0
    max_ties = 0
    gx,gy = np.linspace(0,width,w),np.linspace(0,height,h)
    for start in range(0,n,256):
        p = probability[start:start+256]
        mass = np.sum(p,axis=(1,2),dtype=np.float64)
        ok = np.isfinite(p).all(axis=(1,2)) & (p>=0).all(axis=(1,2)) & np.isfinite(mass) & (mass>0)
        valid[start:start+len(p)] = ok
        if not ok.any():
            continue
        p,mass = p[ok],mass[ok]
        max_mass_error = max(max_mass_error,float(np.max(np.abs(mass-1))))
        ids = start+np.flatnonzero(ok)
        if expected is None:
            expectedxy[ids,0] = p.sum(axis=1,dtype=np.float64) @ gx / mass
            expectedxy[ids,1] = p.sum(axis=2,dtype=np.float64) @ gy / mass
        flat = p.reshape(len(p),-1)
        cells = flat.argmax(axis=1)
        max_ties += int(np.sum(np.sum(flat == flat.max(axis=1)[:,None],axis=1)>1))
        argmaxxy[ids] = np.column_stack((gx[cells%w],gy[cells//w]))
    expectedxy[~valid] = np.nan
    return expectedxy,argmaxxy,valid,max_mass_error,max_ties


def prediction_bundle(entry, recdata, *, trajectory_seed=42, cache_dir=None,
                      legacy_cache_dir=None, include_probability=True):
    """Verified saved distributions, normalized samples, and declared point decoders.

    `samples` is the original [N,K,2] normalized array; `samples_xy` is pixel xy.
    `sampledxy` picks one saved K entry independently on every native frame using
    the fixed seed. It is a framewise stochastic decoding, NOT a coherent joint
    scanpath. Occupancy can average the K entries without this extra selection.
    The optional writable cache stores small decodings, never changes a source.
    Full source bytes are SHA-verified at least once in every fresh process.
    """
    path = Path(entry["path"]).resolve()
    checksum = _sha256(path)
    if checksum != entry["prediction_sha256"]:
        raise ValueError(f"Prediction SHA256 mismatch: {path}")
    n = len(recdata["frames"])
    if n != int(entry["prediction_frames"]):
        raise ValueError(f"Prediction manifest frame count mismatch for {path}")
    provenance_path = path.parent/"provenance.json"
    signature = dict(version=BUNDLE_VERSION,source_sha256=checksum,
                     provenance_sha256=_sha256(provenance_path) if provenance_path.is_file() else None,
                     frames_sha256=hashlib.sha256(np.asarray(recdata["frames"],dtype="<i8").tobytes()).hexdigest(),
                     time_sha256=hashlib.sha256(np.asarray(recdata["timestamp_ns"],dtype="<i8").tobytes()).hexdigest(),
                     resolution=[int(recdata["width"]),int(recdata["height"])])
    key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/"prediction_bundle"/f"{key}.npz" if cache_dir else None
    probability = None
    cached = None
    if cache and cache.is_file() and not include_probability:
        with np.load(cache, allow_pickle=False) as data:
            if json.loads(str(data["signature_json"])) == signature:
                cached = {k:data[k] for k in ("samples","expectedxy","argmaxxy","valid_maps")}
                metadata = json.loads(str(data["metadata_json"]))
    if cached is None:
        with np.load(path, allow_pickle=False) as data:
            for name,expected in (("frame_index",np.arange(n)),("frame_number",recdata["frames"]),
                                  ("timestamp_ns",recdata["timestamp_ns"])):
                if not np.array_equal(data[name],expected):
                    raise ValueError(f"Prediction {name} mismatch for {path}")
            probability = data["probability"]
            samples = data["samples"]
            if probability.shape != (n,32,40) or n != int(entry["prediction_frames"]):
                raise ValueError(f"Prediction dimensions mismatch for {path}")
            if samples.ndim != 3 or samples.shape[0] != n or samples.shape[2] != 2 or samples.shape[1] < 1:
                raise ValueError(f"Prediction samples shape mismatch for {path}: {samples.shape}")
            k = samples.shape[1]
            metadata = _prediction_metadata(entry,path,checksum,n,k)
            if "valid_code" in data:
                metadata["export_current_valid_code_difference_n"] = int(np.sum(data["valid_code"] != recdata["valid_code"])) if "valid_code" in recdata else None
            if "target" in data and "gaze_unmasked" in recdata:
                target = np.asarray(data["target"],float)*[recdata["width"],recdata["height"]]
                delta = np.linalg.norm(target-recdata["gaze_unmasked"],axis=1)
                metadata["export_current_gaze_changed_over_0_01px_n"] = int(np.sum(delta>.01))
            old, legacy_info = _legacy_expected(legacy_cache_dir,entry,recdata,checksum)
            expectedxy,argmaxxy,valid_maps,error,ties = _decode_probability(probability,recdata["width"],recdata["height"],old)
            metadata.update(valid_maps=int(valid_maps.sum()),invalid_maps=int((~valid_maps).sum()),
                            max_mass_error=error,argmax_tied_frames=ties,argmax_tie_rule="first maximum in row-major y,x order",
                            expectedxy_origin="verified legacy frame cache" if old is not None else "normalized probability map mean",
                            legacy_cache_evidence=legacy_info)
        cached = dict(samples=samples,expectedxy=expectedxy,argmaxxy=argmaxxy,valid_maps=valid_maps)
        if cache:
            _atomic_npz(cache,**cached,signature_json=np.asarray(json.dumps(signature,sort_keys=True)),
                        metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
        metadata["prediction_bundle_cache_hit"] = False
    else:
        # Re-read the small provenance when decoding cache is reused; the source
        # checksum remains freshly verified once per process, not trusted on stat.
        metadata.update(_prediction_metadata(entry,path,checksum,n,cached["samples"].shape[1]))
        metadata["prediction_bundle_cache_hit"] = True
    samples = cached["samples"]
    if samples.ndim != 3 or samples.shape[0] != n or samples.shape[2] != 2 or samples.shape[1] < 1:
        raise ValueError(f"Prediction cached samples shape mismatch: {path}")
    for name in ("expectedxy","argmaxxy"):
        if cached[name].shape != (n,2):
            raise ValueError(f"Prediction cached {name} shape mismatch: {path}")
    if cached["valid_maps"].shape != (n,):
        raise ValueError(f"Prediction cached validity shape mismatch: {path}")
    sample_ok = np.isfinite(samples).all(axis=2) & (samples>=0).all(axis=2) & (samples<=1).all(axis=2)
    samples_xy = np.asarray(samples,dtype=float)*[recdata["width"],recdata["height"]]
    samples_xy[~sample_ok] = np.nan
    indices = np.random.default_rng(trajectory_seed).integers(0,samples.shape[1],size=n)
    selected = samples_xy[np.arange(n),indices].copy()
    selected[~cached["valid_maps"]] = np.nan
    grid_error = np.abs(samples*np.asarray([39,31])-np.rint(samples*np.asarray([39,31])))
    metadata.update(sampled_trajectory_seed=int(trajectory_seed),sampled_decoder="independent saved-K index per frame, numpy PCG64",
                    numpy_version=np.__version__,invalid_sample_points=int((~sample_ok).sum()),
                    invalid_sample_frames=int((~sample_ok).any(axis=1).sum()),
                    samples_grid_roundoff_max=float(np.max(grid_error[sample_ok])) if sample_ok.any() else None,
                    sample_index_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
    result = dict(**cached,samples_xy=samples_xy,valid_samples=sample_ok,sampledxy=selected,
                  sample_index=indices,metadata=metadata)
    if include_probability:
        result["probability"] = probability
    return result


def model_track(entry, recdata):
    """Backward-compatible expected-point reader; pointwise occupancy is not map mass."""
    bundle = prediction_bundle(entry,recdata,include_probability=False)
    return bundle["expectedxy"],bundle["metadata"]
