"""Quantify raw observation playback versus the required source-content clock.

Exact coordinate matches are descriptive provenance checks, not gaze filtering.
They do not replace the pre-existing NPY, its mapping or the requested time axis.
"""

from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = Path(r"D:\Dataset\dvss")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": len(a), "min": float(a.min()), "p05": float(np.quantile(a, .05)),
            "median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def key(x, y):
    return round(float(x), 5), round(float(y), 5)


def main():
    ledger = json.loads((OUT / "input_ledger.json").read_text(encoding="utf-8"))
    audits, input_hashes = [], {}
    for r in ledger:
        if not r["novice_pair"]:
            continue
        record_id = r["record_id"]
        raw_path = ROOT / "Passive_Data" / f"User{r['observer']}" / "csv" / f"{record_id}_gaze.csv"
        with raw_path.open(encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.DictReader(stream))
        input_hashes[str(raw_path)] = sha(raw_path)
        if not r["candidate"]:
            continue
        time_column = next(k for k in raw[0] if k.startswith("TIME("))
        tick_column = next(k for k in raw[0] if k.startswith("TIMETICK("))
        rt = np.asarray([float(z[time_column]) for z in raw])
        ticks = np.asarray([int(Decimal(z[tick_column])) for z in raw], dtype=np.int64)
        rv = np.asarray([int(Decimal(z["VID_FRAME"])) for z in raw], dtype=np.int64)
        raw_keys = [key(z["BPOGX"], z["BPOGY"]) for z in raw]
        raw_lookup = defaultdict(list)
        for i, pair in enumerate(raw_keys):
            raw_lookup[pair].append(i)
        a = np.load(r["gaze_npy_path"], allow_pickle=False)
        finite = np.isfinite(a["gaze_x"]) & np.isfinite(a["gaze_y"])
        npy_keys = [key(x, y) if good else None for x, y, good in zip(a["gaze_x"], a["gaze_y"], finite)]
        npy_counts = Counter(npy_keys)
        unique_matches = [(i, raw_lookup[pair][0]) for i, pair in enumerate(npy_keys)
                          if pair is not None and npy_counts[pair] == 1 and len(raw_lookup[pair]) == 1]
        source_ns = a["Timestamp(ns)"].astype(np.int64)
        source_t = (source_ns - source_ns[0]) / 1e9
        # Do not infer a clock from nominal FPS. This is only a comparison axis.
        frame_delta = [int(rv[j]) - i for i, j in unique_matches]
        raw_time_minus_content = [rt[j] - source_t[i] for i, j in unique_matches]
        raw_time_minus_container = [rt[j] - i / r["passive_video_fps"] for i, j in unique_matches]
        raw_pair_matches = sum(pair in raw_lookup for pair in npy_keys if pair is not None)
        matched_content = np.asarray([source_t[i] for i, _ in unique_matches])
        matched_raw = np.asarray([rt[j] for _, j in unique_matches])
        slope, intercept = np.polyfit(matched_content, matched_raw, 1) if len(unique_matches) > 1 else (np.nan, np.nan)
        fit_residuals = matched_raw - (slope * matched_content + intercept)
        video_slope, video_intercept = np.polyfit(rt, rv, 1)
        nominal_frame_residual = rv - rt * r["passive_video_fps"]
        fit_frame_residual = rv - (video_slope * rt + video_intercept)
        match_positions = np.asarray([i for i, _ in unique_matches])
        drift_shape = []
        for fraction in (0, .1, .25, .5, .75, .9, 1):
            selected = int(np.argmin(np.abs(match_positions - fraction * (len(a) - 1))))
            source_row, raw_row = unique_matches[selected]
            drift_shape.append({
                "fraction_of_source_video": fraction,
                "source_retained_row": source_row,
                "original_frame_number": int(a["Frame_Number"][source_row]),
                "source_content_elapsed_s": float(source_t[source_row]),
                "matched_raw_row": raw_row,
                "matched_raw_time_s": float(rt[raw_row]),
                "matched_raw_vid_frame": int(rv[raw_row]),
                "raw_vid_frame_minus_source_row": int(rv[raw_row]) - source_row,
            })
        audit = {
            "record_id": record_id, "content_id": r["content_id"], "task": r["task"],
            "raw_csv": str(raw_path), "raw_sha256": input_hashes[str(raw_path)],
            "raw_media_names": r["media_names"], "raw_rows": len(raw),
            "raw_time_column": time_column, "raw_tick_column": tick_column,
            "raw_media_time_first_s": float(rt[0]), "raw_media_time_last_s": float(rt[-1]),
            "raw_media_time_span_s": float(rt[-1] - rt[0]),
            "raw_tick_span_s": float((ticks[-1] - ticks[0]) / 10000000),
            "raw_vid_frame_first": int(rv[0]), "raw_vid_frame_last": int(rv[-1]),
            "source_content_precise_span_s": r["timeline_duration_s"],
            "passive_container_duration_s": r["passive_container_duration_s"],
            "container_minus_content_duration_s": r["passive_container_duration_s"] - r["timeline_duration_s"],
            "native_npy_finite_pairs": int(finite.sum()),
            "native_pairs_matching_raw_BPOG_at_5dec": raw_pair_matches,
            "native_pairs_matching_raw_BPOG_fraction": raw_pair_matches / int(finite.sum()),
            "unique_raw_npy_coordinate_match_count": len(unique_matches),
            "raw_vid_frame_minus_retained_source_row": summary(frame_delta),
            "raw_media_time_minus_source_content_time_s": summary(raw_time_minus_content),
            "raw_media_time_minus_nominal_container_row_time_s": summary(raw_time_minus_container),
            "matched_raw_time_fit_to_content_time": {
                "slope": float(slope), "intercept_s": float(intercept),
                "residual_s": summary(fit_residuals)},
            "raw_vid_frame_fit_to_raw_time": {
                "slope_frames_per_s": float(video_slope),
                "intercept_frames": float(video_intercept),
                "residual_frames": summary(fit_frame_residual),
                "vid_frame_minus_TIME_times_passive_fps": summary(nominal_frame_residual),
                "empirical_rule": "VID_FRAME agrees with floor(raw TIME * passive container FPS), allowing raw TIME decimal rounding."
            },
            "drift_shape_samples": drift_shape,
            "playback_alignment_status": "existing NPY Frame_Number and canonical timeline mapping verified; correspondence to actual raw playback frame is discrepant and remains unresolved",
            "construction_code_status": "Native passive NPY constructor not found in local repositories. Timing relation inferred independently from unique exact gaze-coordinate pairs.",
            "interpretation": "Analysis clock remains source precise video_ts by original Frame_Number. Raw TIME, VID_FRAME and nominal container time are separate acquisition/playback provenance. These comparisons do not authorize remapping or stretching gaze."
        }
        audits.append(audit)
        r["playback_alignment_status"] = audit["playback_alignment_status"]
        print(record_id, "coordinate matches", audit["native_pairs_matching_raw_BPOG_fraction"],
              "rawtime fit", slope, intercept, "frame delta", audit["raw_vid_frame_minus_retained_source_row"]["median"], flush=True)
    proof_paths = [
        Path(r"D:\GitHub_Project\DVSS_data_analysis\Selected\Ring_Rollercoaster_Novice_CoverSet.csv"),
        Path(r"D:\GitHub_Project\DVSS_data_analysis\Task Score - User2.csv"),
        Path(r"D:\GitHub_Project\DVSS_data_analysis\Passive_Observation_Assignment_Balanced.csv"),
        ROOT / "Passive_Data/User6/stimuli/rename_log_fix_taskid_20251220-185616.csv"
    ]
    input_hashes.update({str(p): sha(p) for p in proof_paths})
    b_video = ROOT / "Passive_Data/User6/stimuli/User6_4_User2_1_1.mp4"
    input_hashes[str(b_video)] = sha(b_video)
    (OUT / "raw_playback_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "raw_and_task_evidence_hashes.json").write_text(json.dumps(input_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "input_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(k for r in ledger for k in r))
    with (OUT / "input_ledger.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in ledger:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    confirmation = {
        "status": "confirmed", "evidence_type": "user_confirmation",
        "exact_answer": "视频画面归一化坐标 1280*1024",
        "mapping": "x_px = gaze_x * 1280; y_px = gaze_y * 1024",
        "crop_offset_px": [0, 0], "stereo_panel_offset_px": [0, 0],
        "scope": "native passive gaze NPY coordinates refer to the full single-camera media image"
    }
    s = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    s["coordinate_mapping_confirmation"] = confirmation
    s["geometry_scope"] = "Full-frame 1280x1024 video-media normalized gaze. Owner confirmed mapping; decoded source/passive samples retain the same image geometry."
    s["raw_playback_audit"] = str(OUT / "raw_playback_audit.json")
    s["raw_and_task_evidence_hashes"] = str(OUT / "raw_and_task_evidence_hashes.json")
    (OUT / "audit_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
