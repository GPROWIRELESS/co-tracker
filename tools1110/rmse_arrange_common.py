"""Shared row-aligned loading/grouping for the four RMSE arrange scripts.

All coordinate RMSE calculations are delegated to RMSE.rmse. Group summaries
concatenate the original valid coordinate pairs, never average record RMSEs.
"""

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from RMSE import rmse

WIDTH, HEIGHT, HZ = 1280, 1024, 60
LIKELIHOOD_MIN = 0.2
DEFAULT_INDEX = Path(r"C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\dvss_pytorch\Index\index_novice_active.json")
DEFAULT_ROOT = Path(r"D:\Dataset\dvss")
STAGES = (
    "grab ring", "transfer ring", "place ring", "reach ring",
    "reach string", "wrap around psm", "pull through and cinch",
    "threading string through tower",
)
ALIASES = {"transport ring": "transfer ring"}
IGNORED_STAGES = {"", "none", "drop ring"}
METRICS = ("gaze_psm_l", "gaze_psm_r", "gaze_target")
OUTPUT_NAMES = {
    "record": "record_level.csv", "task": "task_level.csv",
    "user_stage": "user_stage_arrange.csv", "all_stage": "all_stage_arrange.csv",
}


def inside_screen(xy, width=WIDTH, height=HEIGHT):
    xy = np.asarray(xy, dtype=np.float64)
    return (np.isfinite(xy).all(axis=-1)
            & (xy[..., 0] > 0) & (xy[..., 0] < width)
            & (xy[..., 1] > 0) & (xy[..., 1] < height))


def to_pixels(xy, units):
    xy = np.asarray(xy, dtype=np.float64).copy()
    if units == "normalized":
        xy *= [WIDTH, HEIGHT]
    elif units != "pixel":
        raise ValueError(f"Unknown coordinate units: {units}")
    return xy


def normalize_stage(value):
    label = " ".join(str(value).lower().split())
    return ALIASES.get(label, label)


def read_index(index_path, data_root):
    clips = json.loads(Path(index_path).read_text(encoding="utf-8-sig"))["clips"]
    if not isinstance(clips, dict):
        raise ValueError("Index clips must be an object")
    entries, skipped, seen = [], [], set()
    root = Path(data_root).resolve()
    for clip_key, clip in clips.items():
        meta = clip["meta"]
        task = meta["task_name"]
        cid = meta["content_id"]
        if task not in {"A", "C"}:
            skipped.append({"record_id": cid, "task_name": task, "reason": "only_A_C_requested"})
            continue
        if cid in seen:
            raise ValueError(f"Duplicate record_id: {cid}")
        seen.add(cid)
        gaze_path = (root / clip["gaze_npy_path"]).resolve()
        dlc_path = gaze_path.parent.parent / "sam" / cid / f"{cid}_DLC.csv"
        entries.append(dict(user=meta["user"], record_id=cid, task_name=task,
                            gaze_npy_path=str(gaze_path), dlc_csv_path=str(dlc_path),
                            meta=meta.copy()))
    if not entries:
        raise ValueError("No A/C records found in index")
    return entries, skipped


def load_dlc(path):
    values = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = [next(reader, None) for _ in range(3)]
        if any(row is None or len(row) != 7 for row in headers):
            raise ValueError(f"Expected three 7-column DLC header rows: {path}")
        if [r[0].strip() for r in headers] != ["scorer", "bodyparts", "coords"]:
            raise ValueError(f"Unexpected DLC header layout: {path}")
        if [s.strip() for s in headers[2][1:]] != ["x", "y", "likelihood"] * 2:
            raise ValueError(f"Expected two x/y/likelihood triples: {path}")
        if [s.strip() for s in headers[1][1:]] != ["left_psm"] * 3 + ["right_psm"] * 3:
            raise ValueError(f"Expected left_psm followed by right_psm: {path}")
        for line_number, row in enumerate(reader, 4):
            if len(row) != 7:
                raise ValueError(f"Invalid DLC row at {path}:{line_number}")
            values.append([float(v) if v.strip() else np.nan for v in row[1:]])
    values = np.asarray(values, dtype=np.float64).reshape(-1, 6)
    result = []
    for start in (0, 3):
        xy = values[:, start:start + 2].copy()
        likelihood = values[:, start + 2]
        keep = inside_screen(xy) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_MIN)
        xy[~keep] = np.nan
        result.append(xy)
    return result[0], result[1]


def target_coordinates(source, point_units="pixel", center_units="pixel"):
    """Filter each point before averaging; a valid per-frame knot overrides.

    Units are explicit to avoid mistaking small pixel coordinates for normalized
    values. All target fields in the audited current A/C data are pixel units.
    """
    names = set(source.dtype.names or ())
    point_arrays = []
    for i in range(1, 6):
        x, y = f"point_{i}_x", f"point_{i}_y"
        if (x in names) != (y in names):
            raise ValueError(f"Incomplete target coordinate pair: {x}/{y}")
        if x in names:
            point_arrays.append(to_pixels(np.column_stack((source[x], source[y])), point_units))
    n = len(source)
    target = np.full((n, 2), np.nan)
    counts = np.zeros(n, dtype=np.int64)
    rejected_finite = 0
    if point_arrays:
        points = np.stack(point_arrays, axis=1)
        keep = inside_screen(points)
        rejected_finite = int((np.isfinite(points).all(axis=-1) & ~keep).sum())
        counts = keep.sum(axis=1)
        usable = counts > 0
        sums = np.where(keep[..., None], points, 0.0).sum(axis=1)
        target[usable] = sums[usable] / counts[usable, None]
    center_used = np.zeros(n, dtype=bool)
    invalid_center = 0
    if "purple_string_knot_center" in names:
        for row_number, value in enumerate(source["purple_string_knot_center"]):
            raw = str(value).strip()
            if not raw:
                continue
            try:
                center = np.asarray(json.loads(raw), dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Malformed knot center at row {row_number}: {raw!r}") from exc
            if center.shape != (2,):
                raise ValueError(f"Knot center needs two coordinates at row {row_number}: {raw!r}")
            center = to_pixels(center, center_units)
            if bool(inside_screen(center)):
                target[row_number] = center
                center_used[row_number] = True
            else:
                invalid_center += 1
    if not point_arrays and "purple_string_knot_center" not in names:
        raise ValueError("NPY has no target coordinate fields")
    return target, dict(target_valid_frames=int(np.isfinite(target).all(axis=1).sum()),
                        target_point_pairs_rejected_boundary_or_outside=rejected_finite,
                        target_knot_center_frames=int(center_used.sum()),
                        target_invalid_nonempty_centers=invalid_center)


@dataclass
class Record:
    entry: dict
    gaze: np.ndarray
    references: tuple
    stages: np.ndarray
    audit: dict


def load_record(entry, point_units="pixel", center_units="pixel", validate_stages=False):
    source = np.load(entry["gaze_npy_path"], allow_pickle=False, mmap_mode="r")
    required = {"gaze_x", "gaze_y", "stage"}
    if source.ndim != 1 or not required.issubset(source.dtype.names or ()):
        raise ValueError(f"Expected 1-D structured NPY with {sorted(required)}")
    left, right = load_dlc(entry["dlc_csv_path"])
    if len(source) != len(left):
        raise ValueError(f"Row mismatch for {entry['record_id']}: NPY={len(source)}, DLC={len(left)}; no truncation")
    gaze = to_pixels(np.column_stack((source["gaze_x"], source["gaze_y"])), "normalized")
    gaze[~inside_screen(gaze)] = np.nan
    target, target_audit = target_coordinates(source, point_units, center_units)
    stages = np.asarray([normalize_stage(s) for s in source["stage"]])
    stage_counts = dict(sorted(Counter(str(s) for s in stages).items()))
    unexpected = sorted(set(stage_counts) - set(STAGES) - IGNORED_STAGES)
    if validate_stages and unexpected:
        raise ValueError(f"Unrecognized stage labels in {entry['record_id']}: {unexpected}; confirm before analysis")
    audit = dict(**entry, n_frames=len(source), stage_counts=stage_counts,
                 gaze_valid_frames=int(np.isfinite(gaze).all(axis=1).sum()),
                 psm_l_valid_frames=int(np.isfinite(left).all(axis=1).sum()),
                 psm_r_valid_frames=int(np.isfinite(right).all(axis=1).sum()),
                 **target_audit)
    return Record(entry, gaze, (left, right, target), stages, audit)


@dataclass
class Group:
    identity: dict
    users: set = field(default_factory=set)
    records: set = field(default_factory=set)
    n_frames: int = 0
    pairs: dict = field(default_factory=lambda: {name: [] for name in METRICS})

    def add(self, record, selected):
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != (len(record.gaze),):
            raise ValueError("Selection must retain the original record row positions")
        self.users.add(record.entry["user"])
        self.records.add(record.entry["record_id"])
        self.n_frames += int(selected.sum())
        gaze_ok = np.isfinite(record.gaze).all(axis=1)
        for name, reference in zip(METRICS, record.references):
            paired = selected & gaze_ok & np.isfinite(reference).all(axis=1)
            if paired.any():
                self.pairs[name].append((record.gaze[paired], reference[paired]))

    def row(self):
        result = dict(self.identity, n_users=len(self.users), n_records=len(self.records),
                      n_frames=self.n_frames, duration_seconds=self.n_frames / HZ)
        for name in METRICS:
            chunks = self.pairs[name]
            if chunks:
                gaze = np.concatenate([p[0] for p in chunks], axis=0)
                reference = np.concatenate([p[1] for p in chunks], axis=0)
            else:
                gaze = reference = np.empty((0, 2), dtype=np.float64)
            # The only coordinate RMSE implementation is in RMSE.py.
            value = rmse(gaze, reference)
            result[f"{name}_rmse_px"] = value
            result[f"{name}_n_pairs"] = len(gaze)
            result[f"{name}_valid_seconds"] = len(gaze) / HZ
        return result


def arrange(entries, level, point_units="pixel", center_units="pixel"):
    if level not in OUTPUT_NAMES:
        raise ValueError(f"Unknown arrange level: {level}")
    groups, audits = {}, []
    stage_analysis = level in {"user_stage", "all_stage"}
    for number, entry in enumerate(entries, 1):
        record = load_record(entry, point_units, center_units, stage_analysis)
        audits.append(record.audit)
        if level == "record":
            selections = [((entry["record_id"],), {k: entry[k] for k in ("user", "record_id", "task_name")},
                           np.ones(len(record.gaze), dtype=bool))]
        elif level == "task":
            selections = [((entry["task_name"],), {"task_name": entry["task_name"]},
                           np.ones(len(record.gaze), dtype=bool))]
        else:
            selections = []
            for stage in STAGES:
                selected = record.stages == stage
                if not selected.any():
                    continue
                if level == "user_stage":
                    key = (entry["record_id"], stage)
                    identity = {k: entry[k] for k in ("user", "record_id", "task_name")}
                else:
                    key = (entry["task_name"], stage)
                    identity = {"task_name": entry["task_name"]}
                selections.append((key, dict(identity, stage=stage), selected))
        for key, identity, selected in selections:
            if key not in groups:
                groups[key] = Group(identity)
            groups[key].add(record, selected)
        print(f"[{number}/{len(entries)}] {entry['record_id']}: {len(record.gaze)} rows", flush=True)
    rows = [group.row() for group in groups.values()]
    if level in {"record", "user_stage"}:
        by_record = {entry["record_id"]: entry for entry in entries}
        reserved = set(csv_fields(level)) - {"user", "task_name"}
        for row in rows:
            entry = by_record[row["record_id"]]
            meta = entry.get("meta", {})
            collisions = reserved.intersection(meta) | {"gaze_npy_path", "dlc_csv_path"}.intersection(meta)
            if collisions:
                raise ValueError(f"Meta keys conflict with output columns: {sorted(collisions)}")
            row.update(meta)
            row["gaze_npy_path"] = entry["gaze_npy_path"]
            row["dlc_csv_path"] = entry["dlc_csv_path"]
    return rows, audits


def csv_fields(level, entries=None):
    fields = ["user", "record_id", "task_name"] if level in {"record", "user_stage"} else ["task_name"]
    if level in {"user_stage", "all_stage"}:
        fields += ["stage"]
    fields += ["n_users", "n_records", "n_frames", "duration_seconds"]
    for name in METRICS:
        fields += [f"{name}_rmse_px", f"{name}_n_pairs", f"{name}_valid_seconds"]
    if level in {"record", "user_stage"} and entries is not None:
        # First-seen JSON key order, unioned across every included record.
        meta_fields = list(dict.fromkeys(key for entry in entries for key in entry.get("meta", {})))
        # user/task_name already come from meta: move them to the right once.
        fields = [name for name in fields if name not in meta_fields]
        fields += meta_fields + ["gaze_npy_path", "dlc_csv_path"]
    return fields


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else "" if isinstance(value, float) and not math.isfinite(value) else value
                             for key, value in row.items()})


def run(level):
    parser = argparse.ArgumentParser(description=f"Direct pooled-frame pixel RMSE: {level}.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / OUTPUT_NAMES[level])
    parser.add_argument("--point-units", choices=("pixel", "normalized"), default="pixel",
                        help="Units of point_1..5 in the source NPY; current data are pixels")
    parser.add_argument("--center-units", choices=("pixel", "normalized"), default="pixel",
                        help="Units of purple_string_knot_center; current data are pixels")
    args = parser.parse_args()
    entries, skipped = read_index(args.index, args.data_root)
    # Complete loading/validation/computation before replacing any CSV.
    rows, records = arrange(entries, level, args.point_units, args.center_units)
    write_csv(args.output, rows, csv_fields(level, entries))
    audit = dict(level=level, index=str(args.index.resolve()), data_root=str(args.data_root.resolve()),
                 output=str(args.output.resolve()), output_rows=len(rows), hz=HZ, width=WIDTH, height=HEIGHT,
                 formula="sqrt(sum(dx**2 + dy**2) / (2 * n_valid_pairs))",
                 point_units=args.point_units, center_units=args.center_units,
                 likelihood_min=LIKELIHOOD_MIN, gaze_valid_code_filter=False,
                 smoothing=False, alignment="original row position; reject unequal lengths",
                 included_tasks=["A", "C"], stage_whitelist=list(STAGES),
                 stage_aliases=ALIASES, record_task_include_all_stages=True,
                 pair_validity="independent mask for each gaze-reference pair",
                 target_rule="valid knot center first; otherwise mean of strictly inside finite points",
                 skipped_index_records=skipped, records=records)
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows: {args.output.resolve()}")
    print(f"Audit: {audit_path.resolve()}")
