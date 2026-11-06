"""Read native novice A/C records and previously exported conditional gaze maps."""
from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import numpy as np


MODEL_NAMES = ("all_ABCD", "single_task", "leave_out", "spd_finetune", "spd_zero_shot")
PREDICTION_MANIFEST = Path("outputs/gaze_psm_xcorr_models_max_dlc_current_data/prediction_manifest.csv")
CANONICAL_STAGES = {
    "rr": "reach ring", "gr": "grab ring", "tr": "transfer ring", "pr": "place ring",
    "wrap": "wrap around psm", "cinch": "pull through and cinch",
    "thread": "threading string through tower",
}


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


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


def read_record(rec, data_root, repo):
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
    gaze_valid = np.isin(valid_code, [1, 2, 3]) & np.isfinite(gaze).all(axis=1)
    gaze_valid &= (gaze >= 0).all(axis=1) & (gaze <= 1).all(axis=1)
    gaze[~gaze_valid] = np.nan
    gaze *= (width, height)
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
    slots = _points(source, cid)
    audit.update(_audit_knot_center(source, slots[:, 0]))
    return dict(**rec, frames=frames, time=(ns - ns[0]) / 1e9, timestamp_ns=ns,
                gaze=gaze, gaze_valid=gaze_valid, valid_code=valid_code, slots=slots,
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


def model_track(entry, recdata):
    """Compute per-frame pixel coordinates from the pinned, unmodified inference maps."""
    from tl_geometry import probability_xy

    path = Path(entry["path"])
    checksum = _sha256(path)
    if checksum != entry["prediction_sha256"]:
        raise ValueError(f"Prediction SHA256 mismatch: {path}")
    n = len(recdata["frames"])
    with np.load(path, allow_pickle=False) as data:
        for key, expected in (("frame_index", np.arange(n)), ("frame_number", recdata["frames"]),
                              ("timestamp_ns", recdata["timestamp_ns"])):
            if not np.array_equal(data[key], expected):
                raise ValueError(f"Prediction {key} mismatch for {path}")
        probability = data["probability"]
        if probability.shape != (n, 32, 40) or n != int(entry["prediction_frames"]):
            raise ValueError(f"Prediction dimensions mismatch for {path}")
        xy = np.full((n, 2), np.nan)
        valid_maps = 0
        max_mass_error = 0.0
        for start in range(0, n, 256):
            chunk = probability[start:start + 256]
            mass = np.sum(chunk, axis=(1, 2), dtype=np.float64)
            good = np.isfinite(chunk).all(axis=(1, 2)) & (chunk >= 0).all(axis=(1, 2)) & (mass > 0)
            if good.any():
                max_mass_error = max(max_mass_error, float(np.max(np.abs(mass[good] - 1))))
                xy[start:start + len(chunk)][good] = probability_xy(chunk[good], recdata["width"], recdata["height"])
            valid_maps += int(good.sum())
    count, k, _ = _sample_count(path)
    if n != count:
        raise ValueError(f"Sample count and prediction count differ for {path}")
    metadata = dict(entry, observed_sha256=checksum, num_samples=int(k), valid_maps=valid_maps,
                    invalid_maps=n-valid_maps, max_mass_error=max_mass_error, model_frame_match=True,
                    model_timestamp_match=True)
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
    return xy, metadata
