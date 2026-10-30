"""CLI, synchronized trajectory extraction, caching and xcorr orchestration."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from tools.gaze_psm_core import analyze_curve
from tools.gaze_psm_group import aggregate

CACHE_VERSION = 4
EMPTY_STAGES = {"", "none", "nan", "null"}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value is not None and str(value).strip() else np.nan


def timestamp_ns(value):
    """Read integer nanoseconds, including scientific notation, without float."""
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid nanosecond timestamp: {value!r}") from exc
        if not parsed.is_finite() or parsed != parsed.to_integral_value():
            raise ValueError(f"Expected finite integer nanoseconds: {value!r}")
        return int(parsed)


def status_code(value):
    """Numeric status spellings 2, 2.0 and 2e0 must compare equally."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return ("text", text)
    return ("number", parsed) if parsed.is_finite() else None


def normalized_stage(value):
    return " ".join(str(value or "").strip().lower().split())


def timeline(base, cid):
    rows = read_rows(base / "csv" / f"{cid}_video_ts.csv")
    kept = [r for r in rows if str(r.get("IsDelete", "0")).strip() in {"", "0", "0.0"}]
    frames = np.asarray([int(r["Frame_Number"]) for r in kept], dtype=np.int64)
    ns = np.asarray([timestamp_ns(r["Timestamp(ns)"]) for r in kept], dtype=np.int64)
    if len(ns) < 2 or np.any(np.diff(ns) <= 0) or len(np.unique(frames)) != len(frames):
        raise ValueError("Video frame timestamps must increase and frame numbers must be unique")
    return frames, ns, len(rows) - len(kept)


def stage_frames(base, cid, frames):
    rows = read_rows(base / "npy" / f"{cid}_gaze.csv")
    mapping = {}
    for r in rows:
        key = int(r["Frame_Number"])
        if key in mapping:
            raise ValueError(f"Duplicate allocation Frame_Number {key}")
        if "stage" not in r and "event" not in r:
            raise ValueError("Allocation CSV needs a stage or event column")
        mapping[key] = r
    missing = [int(f) for f in frames if int(f) not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} retained video frames have no stage row")
    selected = [mapping[int(f)] for f in frames]
    labels = np.asarray([normalized_stage(r.get("stage", r.get("event", ""))) for r in selected])
    # IDs follow contiguous occurrences, including none, so separated repeats never join.
    ids = np.r_[0, np.cumsum(labels[1:] != labels[:-1])].astype(int)
    ids[np.isin(labels, list(EMPTY_STAGES))] = -1
    return selected, labels, ids


def interpolate_valid(times, xy, grid, max_gap):
    out = np.full((len(grid), 2), np.nan)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(times)
    t, a = times[valid], xy[valid]
    if len(t) < 2:
        return out
    order = np.argsort(t, kind="stable")
    t, a = t[order], a[order]
    t, ix = np.unique(t, return_index=True)
    a = a[ix]
    if len(t) < 2:
        return out
    right = np.searchsorted(t, grid, side="left").clip(0, len(t)-1)
    exact = np.isclose(grid, t[right], atol=1e-9, rtol=0)
    out[exact] = a[right[exact]]
    left = np.maximum(right-1, 0)
    good = (~exact) & (grid >= t[0]) & (grid <= t[-1]) & ((t[right]-t[left]) <= max_gap)
    for axis in range(2):
        out[good, axis] = np.interp(grid[good], t, a[:, axis])
    return out


def smooth_valid(xy, hz, smooth_ms):
    """Centered odd boxcar, confined to each contiguous valid run."""
    result = xy.copy()
    window = max(1, int(round(smooth_ms*hz/1000)))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return result
    valid = np.isfinite(xy).all(axis=1)
    cuts = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, end in cuts.reshape(-1, 2):
        result[start:end] = uniform_filter1d(xy[start:end], size=window, axis=0, mode="nearest")
    return result


def decode_mask(path, threshold, min_area, limit=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    xy, area = [], []
    try:
        while limit is None or len(xy) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > threshold).astype(np.uint8)
            m = cv2.moments(mask, binaryImage=True)
            a = m["m00"]
            xy.append((m["m10"]/a, m["m01"]/a) if a >= min_area else (np.nan, np.nan))
            area.append(a)
    finally:
        cap.release()
    return np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(area), width, height, fps


def source_files(record, args):
    cid = record["content_id"]
    base = args.data_root / cid.split("_")[0]
    files = [base/"csv"/f"{cid}_video_ts.csv", base/"npy"/f"{cid}_gaze.csv"]
    if args.gaze_source == "raw":
        files.append(base/"csv"/f"{cid}_gaze.csv")
    files.extend(base/"sam"/cid/"mask"/f"obj_psm_{s}.mp4" for s in ("l", "r"))
    return files


def fingerprint(record, args, cache_version=CACHE_VERSION):
    settings = {k: getattr(args, k) for k in ("hz", "max_gap", "threshold", "min_area", "max_frames", "gaze_source", "gaze_units", "gaze_x_column", "gaze_y_column", "gaze_valid_values", "flip_gaze_y", "smooth_ms", "jump_px", "area_ratio")}
    settings["files"] = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in source_files(record, args)]
    settings["record"] = {"content_id": record["content_id"], "task": record["Task Name"]}
    settings["version"] = cache_version
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def extract(record, args):
    cid = record["content_id"]
    base = args.data_root / cid.split("_")[0]
    key = fingerprint(record, args)
    cache = args.output/"cache"/f"{cid}.npz"
    mask_cache = None
    if cache.exists() and args.reuse_cache:
        with np.load(cache, allow_pickle=False) as z:
            if str(z["fingerprint"]) == key:
                data = {k: z[k] for k in z.files}
                return data, json.loads(str(data["qc_json"]))
            if str(z["fingerprint"]) == fingerprint(record, args, cache_version=3):
                # Version 3's input bugs affect gaze, not decoded mask pixels.
                # Only reuse independently verified native mask trajectories;
                # always regenerate gaze, stage alignment, smoothing and QC.
                mask_cache = {k: z[k] for k in z.files}
    if args.phase == "analyze":
        raise ValueError("Matching trajectory cache unavailable; run --phase extract or all first")
    frames, ns, n_deleted = timeline(base, cid)
    allocations, labels, ids = stage_frames(base, cid, frames)
    origin = int(ns[0])
    native_time = (ns - origin)/1e9
    n = len(frames) if args.max_frames is None else min(args.max_frames, len(frames))
    frames, native_time, labels, ids = frames[:n], native_time[:n], labels[:n], ids[:n]
    grid = np.arange(int(np.floor(native_time[-1]*args.hz))+1)/args.hz
    if args.gaze_source == "raw":
        raw = read_rows(base/"csv"/f"{cid}_gaze.csv")
        raw = [r for r in raw if str(r.get("Capture Unix timestamp", "")).strip()]
        gt = np.asarray([(timestamp_ns(r["Capture Unix timestamp"])-origin)/1e9 for r in raw])
        xcol, ycol = args.gaze_x_column or "norm_x", args.gaze_y_column or "norm_y"
        valid_col = "Status"
    else:
        raw = allocations[:n]
        gt = native_time
        xcol, ycol = args.gaze_x_column or "gaze_x", args.gaze_y_column or "gaze_y"
        valid_col = "valid"
    if not raw:
        raise ValueError("No gaze samples with timestamps")
    gaze = np.asarray([[number(r[xcol]), number(r[ycol])] for r in raw])
    accepted = {status_code(value) for value in args.gaze_valid_values} - {None}
    good = np.asarray([status_code(r.get(valid_col)) in accepted for r in raw])
    good &= np.isfinite(gaze).all(axis=1)
    if args.gaze_units == "normalized":
        good &= (gaze >= 0).all(axis=1) & (gaze <= 1).all(axis=1)
    gaze[~good] = np.nan
    if not good.any():
        raise ValueError("No valid gaze samples after status and coordinate filtering")
    index = np.searchsorted(native_time, grid, side="right")-1
    index = index.clip(0, len(native_time)-1)
    grid_labels, grid_ids = labels[index].copy(), ids[index].copy()
    stale = grid-native_time[index] > args.max_gap
    grid_labels[stale], grid_ids[stale] = "none", -1
    data = {"time_s": grid, "stage": grid_labels, "occurrence": grid_ids,
            "native_time_s": native_time, "native_stage": labels, "native_occurrence": ids,
            "frame_number": frames, "fingerprint": np.asarray(key)}
    qc = {"content_id": cid, "user": cid.split("_")[0], "task": record["Task Name"], "status": "ok",
          "frames": n, "deleted_rows": n_deleted, "hz": args.hz,
          "measured_video_hz": float(1/np.median(np.diff(native_time))),
          "video_interval_p95_s": float(np.percentile(np.diff(native_time), 95)),
          "partial_record": args.max_frames is not None, "duration_s": float(native_time[-1]),
          "stages": sorted(set(labels)-EMPTY_STAGES), "gaze_source": args.gaze_source,
          "cache_version": CACHE_VERSION, "accepted_gaze_samples": int(good.sum())}
    reusable_masks = (mask_cache is not None
                      and np.array_equal(mask_cache["frame_number"], frames)
                      and np.array_equal(mask_cache["native_time_s"], native_time))
    for side in ("l", "r"):
        path = base/"sam"/cid/"mask"/f"obj_psm_{side}.mp4"
        if reusable_masks:
            points, areas = mask_cache[f"native_psm_{side}"], mask_cache[f"area_{side}"]
            width, height = mask_cache[f"resolution_{side}"]
            fps = json.loads(str(mask_cache["qc_json"]))[f"{side}_mask_fps"]
        else:
            points, areas, width, height, fps = decode_mask(path, args.threshold, args.min_area, args.max_frames)
        if len(points) != n:
            raise ValueError(f"{side}: {len(points)} decoded mask frames vs {n} retained timestamps")
        gaze_px = gaze*np.array([width, height]) if args.gaze_units == "normalized" else gaze.copy()
        if args.flip_gaze_y:
            gaze_px[:, 1] = height-gaze_px[:, 1]
        inside = (gaze_px >= 0).all(axis=1) & (gaze_px <= [width, height]).all(axis=1)
        gaze_px[~inside] = np.nan
        aligned_gaze = interpolate_valid(gt, gaze_px, grid, args.max_gap)
        aligned_psm = interpolate_valid(native_time, points, grid, args.max_gap)
        if not np.isfinite(aligned_gaze).all(axis=1).any():
            raise ValueError(f"{side}: no valid gaze samples overlap the video timeline")
        data[f"resolution_{side}"] = np.asarray([width, height])
        data[f"gaze_raw_{side}"] = aligned_gaze
        data[f"psm_raw_{side}"] = aligned_psm
        data[f"gaze_{side}"] = smooth_valid(aligned_gaze, args.hz, args.smooth_ms)
        data[f"psm_{side}"] = smooth_valid(aligned_psm, args.hz, args.smooth_ms)
        data[f"native_psm_{side}"] = points
        data[f"area_{side}"] = areas
        qc[f"{side}_resolution"] = f"{width}x{height}"
        qc[f"{side}_mask_fps"] = float(fps)
        qc[f"{side}_native_mask_cache_reused"] = bool(reusable_masks)
        qc[f"{side}_gaze_valid_fraction"] = float(np.isfinite(aligned_gaze).all(axis=1).mean())
        qc[f"{side}_psm_valid_fraction"] = float(np.isfinite(aligned_psm).all(axis=1).mean())
        if len(points) > 1:
            qc[f"{side}_centroid_jumps"] = int(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1) > args.jump_px))
            ratio = np.maximum(areas[1:], areas[:-1])/np.maximum(np.minimum(areas[1:], areas[:-1]), 1)
            qc[f"{side}_large_area_changes"] = int(np.sum(ratio > args.area_ratio))
    data["qc_json"] = np.asarray(json.dumps(qc))
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".npz", dir=cache.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **data)
        os.replace(name, cache)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return data, qc


def curve_inputs(data, side, signal, hz, max_gap=.1):
    g, p = data[f"gaze_{side}"], data[f"psm_{side}"]
    labels, ids = data["stage"], data["occurrence"]
    if signal == "velocity":
        g, p = np.diff(g, axis=0)*hz, np.diff(p, axis=0)*hz
        # Stage labels are evaluated at true velocity-interval midpoints.
        midpoint = (data["time_s"][:-1]+data["time_s"][1:])/2
        index = np.searchsorted(data["native_time_s"], midpoint, side="right")-1
        index = index.clip(0, len(data["native_time_s"])-1)
        labels, ids = data["native_stage"][index], data["native_occurrence"][index]
        stale = midpoint-data["native_time_s"][index] > max_gap
        labels, ids = labels.copy(), ids.copy()
        labels[stale], ids[stale] = "none", -1
    return g, p, labels, ids


def calculate(record, data, args, donor=None):
    curves = []
    for side in ("l", "r"):
        for signal in args.signals:
            g, p, labels, ids = curve_inputs(data, side, signal, args.hz, args.max_gap)
            if donor is not None:
                source = donor[f"gaze_{side}"] * (data[f"resolution_{side}"]/donor[f"resolution_{side}"])
                if signal == "velocity":
                    source = np.diff(source, axis=0)*args.hz
                # Align elapsed time without warping, repeat, or extrapolation.
                g = np.full_like(p, np.nan)
                take = min(len(g), len(source))
                g[:take] = source[:take]
            selections = [("__whole__", "whole", None)]
            if args.stages:
                for stage in sorted(set(labels)-EMPTY_STAGES):
                    occurrence = np.where(labels == stage, ids, -1)
                    selections.append((stage, "psm_stage", occurrence))
                    if args.within_stage:
                        selections.append((stage, "within_stage", occurrence))
            for stage, mode, occurrence in selections:
                rms_g = args.min_gaze_rms if signal == "position" else args.min_velocity_rms
                rms_p = args.min_psm_rms if signal == "position" else args.min_velocity_rms
                result = analyze_curve(g, p, args.hz, max_lag=args.max_lag,
                    min_overlap=args.min_overlap if mode == "whole" else args.stage_min_overlap,
                    occurrence_ids=occurrence, within_stage=mode == "within_stage",
                    min_gaze_rms=rms_g, min_psm_rms=rms_p,
                    spatial_tolerance=args.spatial_tolerance, compute_spatial=signal == "position" and donor is None)
                result.update(content_id=record["content_id"], user=record["content_id"].split("_")[0],
                              task=record["Task Name"], side=side, signal=signal, stage=stage, mode=mode)
                curves.append(result)
    return curves


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, allow_nan=False))


def _extract_job(payload):
    record, args = payload
    cid = record["content_id"]
    cv2.setNumThreads(args.opencv_threads)
    try:
        data, quality = extract(record, args)
        return cid, data, quality
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        return cid, None, {"content_id": cid, "user": cid.split("_")[0],
                           "task": record["Task Name"], "status": "error", "error": str(exc)}


def _init_analysis_worker(records, datasets, args):
    global _ANALYSIS_RECORDS, _ANALYSIS_DATASETS, _ANALYSIS_ARGS
    _ANALYSIS_RECORDS, _ANALYSIS_DATASETS, _ANALYSIS_ARGS = records, datasets, args
    cv2.setNumThreads(args.opencv_threads)


def _observed_job(cid):
    return calculate(_ANALYSIS_RECORDS[cid], _ANALYSIS_DATASETS[cid], _ANALYSIS_ARGS)


def _null_job(pairings):
    shuffled = []
    for cid, donor_id in pairings:
        curves = calculate(_ANALYSIS_RECORDS[cid], _ANALYSIS_DATASETS[cid],
                           _ANALYSIS_ARGS, donor=_ANALYSIS_DATASETS[donor_id])
        # The unpaired aggregation only consumes metadata and r/lag. Avoid
        # transporting unused NaN spatial arrays between worker processes.
        shuffled.extend({k: v for k, v in c.items()
                         if not isinstance(v, np.ndarray) or k in {"lag_s", "r"}}
                        for c in curves)
    return shuffled


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Position xcorr of gaze versus PSM centroids; positive lag means gaze leads.")
    parser.add_argument("--index", type=Path, default=Path("Index/content_index.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/nas/Users/yxl3527/Dataset/dvss/Active_Data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gaze_psm_xcorr"))
    parser.add_argument("--phase", choices=["all", "extract", "analyze"], default="all")
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tasks", nargs="+", default=["A", "C"])
    parser.add_argument("--skill", default="novice")
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--hz", default="auto", help="Uniform Hz or auto: rounded median native video rate across selected records")
    parser.add_argument("--max-lag", type=float, default=2.)
    parser.add_argument("--max-gap", type=float, default=.1)
    parser.add_argument("--min-overlap", type=float, default=5.)
    parser.add_argument("--stage-min-overlap", type=float, default=2.)
    parser.add_argument("--stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--within-stage", action="store_true")
    parser.add_argument("--signals", nargs="+", choices=["position", "velocity"], default=["position", "velocity"])
    parser.add_argument("--smooth-ms", type=float, default=80.)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--spatial-tolerance", type=float, default=50., help="Exploratory pixel proximity radius; calibrate for your data")
    parser.add_argument("--min-gaze-rms", type=float, default=1., help="Minimum pooled position RMS in pixels")
    parser.add_argument("--min-psm-rms", type=float, default=1.)
    parser.add_argument("--min-velocity-rms", type=float, default=1., help="Minimum pooled velocity RMS in pixels/second")
    parser.add_argument("--gaze-source", choices=["raw", "allocation"], default="raw")
    parser.add_argument("--gaze-units", choices=["normalized", "pixels"], default="normalized")
    parser.add_argument("--gaze-x-column")
    parser.add_argument("--gaze-y-column")
    parser.add_argument("--gaze-valid-values", "--gaze-status", nargs="+", default=["2"])
    parser.add_argument("--flip-gaze-y", action="store_true")
    parser.add_argument("--jump-px", type=float, default=100., help="QC-only centroid jump threshold")
    parser.add_argument("--area-ratio", type=float, default=3., help="QC-only mask area change ratio")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-users", type=int, default=3, help="Minimum stable novices; sparse stages still exploratory")
    parser.add_argument("--max-frames", type=int, help="Partial-record smoke test only")
    parser.add_argument("--opencv-threads", type=int, default=1, help="OpenCV image-operation threads during mask extraction")
    parser.add_argument("--workers", type=int, default=1, help="Parallel extraction and xcorr worker processes; results retain index/replicate order")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    positive = (args.max_gap, args.min_overlap, args.stage_min_overlap, args.spatial_tolerance, args.min_area, args.jump_px)
    nonnegative = (args.max_lag, args.smooth_ms, args.min_gaze_rms, args.min_psm_rms, args.min_velocity_rms)
    if not all(np.isfinite(v) and v > 0 for v in positive) or not all(np.isfinite(v) and v >= 0 for v in nonnegative) or args.bootstrap < 0 or args.shuffles < 0 or args.min_users < 1 or not 0 <= args.threshold <= 255 or args.area_ratio <= 1 or args.opencv_threads < 1 or args.workers < 1 or (args.max_frames is not None and args.max_frames < 2):
        parser.error("Invalid sampling, support, smoothing, threshold or resampling settings")
    if args.hz != "auto":
        try:
            args.hz = float(args.hz)
            if not np.isfinite(args.hz) or args.hz <= 0:
                raise ValueError()
        except ValueError:
            parser.error("--hz must be auto or a positive number")
    return args


def main(argv=None):
    args = parse_args(argv)
    cv2.setNumThreads(args.opencv_threads)
    records = {r["content_id"]: r for r in read_rows(args.index) if r["src_skill"].strip().lower() == args.skill.lower() and r["Task Name"] in args.tasks and (not args.records or r["content_id"] in args.records)}
    if not records:
        raise SystemExit("No matching index records")
    if args.records and set(args.records)-records.keys():
        raise SystemExit(f"Requested records absent from selected index: {sorted(set(args.records)-records.keys())}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.hz == "auto":
        rates = []
        for record in records.values():
            try:
                cid = record["content_id"]
                _, ns, _ = timeline(args.data_root/cid.split("_")[0], cid)
                rates.append(1e9/np.median(np.diff(ns)))
            except (OSError, ValueError, KeyError) as exc:
                print(f"Rate preflight {record['content_id']}: {exc}", flush=True)
        if not rates:
            raise SystemExit("Cannot infer sample rate from selected video timestamps")
        args.hz = float(max(1, round(np.median(rates))))
    print(f"Uniform rate: {args.hz:g} Hz; {len(records)} records; {len({c.split('_')[0] for c in records})} novices", flush=True)
    save_json(args.output/"config.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cache_version": CACHE_VERSION, "primary_signal": "position", "stage_anchor": "psm_time", "partial_record": args.max_frames is not None})
    datasets, qc = {}, []
    payloads = [(record, args) for record in records.values()]
    with ExitStack() as stack:
        if args.workers > 1:
            extraction_pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context("spawn")))
            extracted = extraction_pool.map(_extract_job, payloads)
        else:
            extracted = map(_extract_job, payloads)
        for i, (cid, data, quality) in enumerate(extracted, 1):
            print(f"[{i}/{len(records)}] {cid}: {quality['status']}", flush=True)
            qc.append(quality)
            if data is not None:
                datasets[cid] = data
            else:
                print(f"  ERROR: {quality['error']}", flush=True)
    save_json(args.output/"quality.json", qc)
    if args.phase == "extract":
        print(f"Cached {len(datasets)} trajectories in {args.output/'cache'}", flush=True)
        return 1 if any(q["status"] == "error" for q in qc) else 0
    # Draw all controls in the parent, in index order, before parallel work.
    # Thus worker count cannot change the randomization or aggregation order.
    rng = np.random.default_rng(args.seed)
    donors = {}
    for cid in datasets:
        eligible = [other for other in datasets
                    if other.split("_")[0] != cid.split("_")[0]
                    and records[other]["Task Name"] == records[cid]["Task Name"]]
        by_user = {}
        for other in eligible:
            by_user.setdefault(other.split("_")[0], []).append(other)
        donors[cid] = by_user
    if args.shuffles and any(not v for v in donors.values()):
        print("Unpaired reference unavailable for records without a different novice in the same task; those groups have no complete null replicate.", flush=True)
    pairings = []
    for _ in range(args.shuffles):
        replicate = []
        for cid, by_user in donors.items():
            if by_user:
                donor_user = rng.choice(sorted(by_user))
                replicate.append((cid, rng.choice(by_user[donor_user])))
        pairings.append(replicate)
    with ExitStack() as stack:
        if args.workers > 1:
            # On Linux, read-only aligned trajectories are shared copy-on-write.
            method = "fork" if hasattr(os, "fork") else "spawn"
            analysis_pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context(method),
                initializer=_init_analysis_worker, initargs=(records, datasets, args)))
            observed = analysis_pool.map(_observed_job, datasets)
        else:
            _init_analysis_worker(records, datasets, args)
            observed = map(_observed_job, datasets)
        curves = []
        for i, result in enumerate(observed, 1):
            curves.extend(result)
            print(f"Observed xcorr [{i}/{len(datasets)}]", flush=True)
        shuffled_results = (analysis_pool.map(_null_job, pairings) if args.workers > 1
                            else map(_null_job, pairings))
        def null_replicates():
            for rep, shuffled in enumerate(shuffled_results, 1):
                print(f"Unpaired reference [{rep}/{args.shuffles}]", flush=True)
                yield shuffled
        users, groups, contrasts = aggregate(curves, bootstrap=args.bootstrap, seed=args.seed,
                                             min_users=args.min_users, baseline_replicates=null_replicates())
    save_json(args.output/"results.json", {"record_curves": curves, "user_curves": users, "group_curves": groups, "contrasts": contrasts})
    from tools.gaze_psm_report import generate_outputs
    generate_outputs(args.output, curves, users, groups, contrasts, qc, make_plots=not args.no_plots)
    print(f"Saved {len(curves)} recording curves and {len(groups)} groups to {args.output}", flush=True)
    return 1 if any(q["status"] == "error" for q in qc) else 0
