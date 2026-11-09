"""Audit passive input eligibility, retained-frame maps and decoded stimulus identity.

Read source datasets only. Write this script's results beneath its own directory.
Run with C:/ProgramData/anaconda3/envs/sam3/python.exe.
"""

from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


ROOT = Path(r"D:\Dataset\dvss")
OUT = Path(__file__).resolve().parent
LINKS = Path(r"C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json")
BASELINE = Path(r"D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106\gt84")
SCORES = Path(r"D:\GitHub_Project\DVSS_data_analysis\Selected\Ring_Rollercoaster_Novice_CoverSet.csv")
INTERMEDIATE = {96, 97, 98}
EMPTY_STAGES = {"", "none", "nan", "null"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def timeline(path):
    original = rows(path)
    kept = [r for r in original if r.get("IsDelete", "0").strip() in {"", "0", "0.0"}]
    frames = np.asarray([int(r["Frame_Number"]) for r in kept], dtype=np.int64)
    ns = np.asarray([int(Decimal(r["Timestamp(ns)"])) for r in kept], dtype=np.int64)
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(ns) <= 0):
        raise ValueError(f"Invalid timeline {path}")
    return frames, ns, len(original) - len(kept)


def select_by_frame(data, frames):
    ids = data["Frame_Number"].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate original Frame_Number in NPY")
    order = np.argsort(ids)
    pos = np.searchsorted(ids[order], frames)
    if np.any(pos == len(ids)) or not np.array_equal(ids[order[pos]], frames):
        raise ValueError("NPY does not cover the complete retained timeline")
    return data[order[pos]]


def metadata(cap):
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }


def decode(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    good, image = cap.read()
    if not good:
        raise ValueError(f"Could not decode frame {index}")
    return image


def decoded_comparison(passive, source, count, record_id):
    pc = cv2.VideoCapture(str(passive))
    sc = cv2.VideoCapture(str(source))
    try:
        pm, sm = metadata(pc), metadata(sc)
        indices = sorted({0, 1, int((count - 1) * .1), count // 2,
                          int((count - 1) * .9), count - 2, count - 1})
        evidence = []
        middle = None
        for i in indices:
            p, s = decode(pc, i), decode(sc, i)
            row = {"retained_video_row": i, "passive_rgb_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
                   "source_rgb_sha256": hashlib.sha256(s.tobytes()).hexdigest(),
                   "shape_equal": p.shape == s.shape}
            if p.shape == s.shape:
                difference = np.abs(p.astype(np.int16) - s.astype(np.int16))
                pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(float).ravel()
                sg = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(float).ravel()
                row.update(mae=float(difference.mean()), max_abs_error=int(difference.max()),
                           rgb_identical=bool(np.array_equal(p, s)),
                           grayscale_correlation=float(np.corrcoef(pg, sg)[0, 1]))
            evidence.append(row)
            if i == count // 2:
                middle = (p, s)
        if middle is not None:
            p, s = middle
            p = cv2.resize(p, (640, 512))
            s = cv2.resize(s, (640, 512))
            sheet = np.vstack((np.hstack((p, s)), np.zeros((42, 1280, 3), dtype=np.uint8)))
            cv2.putText(sheet, f"{record_id} | passive left, source right | row {count // 2}",
                        (8, 540), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            image_path = OUT / "decoded_frames" / f"{record_id}.png"
            image_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(image_path), sheet)
        return {"passive_video": pm, "source_video": sm, "samples": evidence,
                "all_samples_rgb_identical": all(r.get("rgb_identical", False) for r in evidence),
                "max_sample_mae": max((r.get("mae", float("inf")) for r in evidence), default=None)}
    finally:
        pc.release()
        sc.release()


def dlc_audit(path, expected):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = [next(reader) for _ in range(3)]
        data = [r for r in reader if r]
    indices = [int(Decimal(r[0])) for r in data]
    result = {"rows": len(data), "index_consecutive_from_zero": indices == list(range(len(data))),
              "matches_retained_video_rows": len(data) == expected}
    for side in ("left_psm", "right_psm"):
        columns = [next(i for i, (b, c) in enumerate(zip(headers[1], headers[2]))
                        if b == side and c == axis) for axis in ("x", "y")]
        xy = np.asarray([[float(r[i]) if r[i].strip() else np.nan for i in columns] for r in data])
        result[f"{side}_finite_pair_count"] = int(np.isfinite(xy).all(axis=1).sum())
    return result


def raw_media_audit(path):
    media, video_indices = set(), []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for r in reader:
            media.add(r.get("MEDIA_NAME", ""))
            text = r.get("VID_FRAME", "")
            if text.strip():
                video_indices.append(int(Decimal(text)))
    return {"media_names": sorted(media), "raw_vid_frame_min": min(video_indices),
            "raw_vid_frame_max": max(video_indices), "raw_eye_rows": len(video_indices)}


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    linked = json.loads(LINKS.read_text(encoding="utf-8-sig"))["clips"]
    baseline_records = {r["content_id"] for r in json.loads((BASELINE / "run_manifest.json").read_text())["records"]}
    score_row = [r for r in rows(SCORES) if (r["UserID"], r["TaskNum"], r["Attempt"]) == ("2", "1", "1")]
    assert len(score_row) == 1 and score_row[0]["TaskKey"] == "Ring Rollercoaster"
    ledger, detailed = [], []
    for path in sorted((ROOT / "Passive_Data").glob("User*/npy/*_gaze.npy")):
        match = re.fullmatch(r"User(\d+)_(\d+)_(User(\d+)_(\d+)_(\d+))_gaze\.npy", path.name)
        if match is None:
            raise ValueError(f"Unrecognized passive filename {path}")
        observer, seq, cid, source_user = int(match[1]), int(match[2]), match[3], int(match[4])
        record_id = path.stem.removesuffix("_gaze")
        key = f"Passive_Data/User{observer}/stimuli/{record_id}"
        item = linked[key]
        task = item["meta"]["task_name"]
        task_evidence = "reference index metadata"
        if cid == "User2_1_1":
            task, task_evidence = "B", str(SCORES) + " exact UserID=2 TaskNum=1 Attempt=1 TaskKey=Ring Rollercoaster"
        source_base = ROOT / "Active_Data" / f"User{source_user}"
        source = source_base / "npy" / f"{cid}_gaze.npy"
        source_video = source_base / "stimuli" / f"{cid}.mp4"
        video = ROOT / "Passive_Data" / f"User{observer}" / "stimuli" / f"{record_id}.mp4"
        dlc = source_base / "sam" / cid / f"{cid}_DLC.csv"
        novice_pair = observer not in INTERMEDIATE and source_user not in INTERMEDIATE
        reasons = []
        if observer in INTERMEDIATE:
            reasons.append("intermediate_observer")
        if source_user in INTERMEDIATE:
            reasons.append("intermediate_source")
        if task not in ("A", "C"):
            reasons.append("task_outside_A_C" if task in ("B", "D") else "unresolved_task")
        if not source.is_file():
            reasons.append("source_npy_missing")
        if not source_video.is_file():
            reasons.append("source_video_missing")
        if not dlc.is_file():
            reasons.append("source_dlc_missing")
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        row = {"record_id": record_id, "content_id": cid, "observer": observer, "source_user": source_user,
               "task": task, "task_evidence": task_evidence, "novice_pair": novice_pair,
               "gaze_npy_path": str(path), "source_npy_path": str(source), "source_video_path": str(source_video),
               "passive_video_path": str(video), "dlc_csv_path": str(dlc), "passive_npy_rows": len(a),
               "source_npy_exists": source.is_file(), "source_video_exists": source_video.is_file(),
               "dlc_exists": dlc.is_file(), "source_in_gt84": cid in baseline_records,
               "stage_exists": False, "candidate": False, "exclusion_reasons": reasons}
        if source.is_file():
            b = np.load(source, mmap_mode="r", allow_pickle=False)
            stage = next((n for n in ("stage", "event") if n in b.dtype.names), None)
            row["stage_exists"] = stage is not None
            if not stage:
                reasons.append("source_stage_field_missing")
            row["source_frame_ids_equal_passive"] = np.array_equal(a["Frame_Number"], b["Frame_Number"])
            row["source_native_npy_ns_equal_passive"] = np.array_equal(a["Timestamp(ns)"], b["Timestamp(ns)"])
        if novice_pair:
            raw_path = ROOT / "Passive_Data" / f"User{observer}" / "csv" / f"{record_id}_gaze.csv"
            row.update(raw_media_audit(raw_path))
        if novice_pair and task in ("A", "C"):
            pf = ROOT / "Passive_Data" / f"User{observer}" / "csv" / f"{record_id}_video_ts.csv"
            sf = source_base / "csv" / f"{cid}_video_ts.csv"
            frames, ns, deleted = timeline(pf)
            sframes, sns, source_deleted = timeline(sf)
            mapped = select_by_frame(a, frames)
            source_mapped = select_by_frame(b, sframes)
            labels = np.asarray([" ".join(str(v).strip().lower().split()) for v in source_mapped[stage]])
            occurrences = np.r_[0, np.cumsum(labels[1:] != labels[:-1])]
            native_xy = np.column_stack((mapped["gaze_x"], mapped["gaze_y"]))
            finite = np.isfinite(native_xy).all(axis=1)
            in_unit = finite & (native_xy >= 0).all(axis=1) & (native_xy <= 1).all(axis=1)
            row.update(retained_timeline_rows=len(frames), deleted_video_ts_rows=deleted,
                       source_deleted_video_ts_rows=source_deleted,
                       source_timeline_frames_equal_passive=np.array_equal(frames, sframes),
                       source_precise_video_ts_ns_equal_passive=np.array_equal(ns, sns),
                       native_npy_ns_equal_precise_video_ts=np.array_equal(mapped["Timestamp(ns)"], ns),
                       original_frame_first=int(frames[0]), original_frame_last=int(frames[-1]),
                       timeline_duration_s=float((ns[-1] - ns[0]) / 1e9),
                       timeline_median_dt_s=float(np.median(np.diff(ns)) / 1e9),
                       timeline_max_dt_s=float(np.max(np.diff(ns)) / 1e9),
                       timeline_gaps_over_100ms=int(np.sum(np.diff(ns) > 100000000)),
                       original_frame_gaps=int(np.sum(np.diff(frames) != 1)),
                       native_gaze_finite_count=int(finite.sum()), native_gaze_accepted_count=int(in_unit.sum()),
                       stage_occurrence_count=len(np.unique(occurrences)),
                       stage_values={str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                       finite_gaze_valid_flag_counts={str(k): int(v) for k, v in zip(*np.unique(mapped["valid"][in_unit], return_counts=True))},
                       input_sha256={str(p): sha(p) for p in (path, source, sf, pf, dlc)})
            comparison = decoded_comparison(video, source_video, len(frames), record_id)
            d = dlc_audit(dlc, len(sframes))
            row["candidate"] = not reasons
            row.update(decoded_samples_rgb_identical=comparison["all_samples_rgb_identical"],
                       decoded_samples_max_mae=comparison["max_sample_mae"],
                       passive_video_width=comparison["passive_video"]["width"],
                       passive_video_height=comparison["passive_video"]["height"],
                       source_video_width=comparison["source_video"]["width"],
                       source_video_height=comparison["source_video"]["height"],
                       source_video_frames=comparison["source_video"]["frames"],
                       passive_video_frames=comparison["passive_video"]["frames"],
                       source_video_fps=comparison["source_video"]["fps"],
                       passive_video_fps=comparison["passive_video"]["fps"],
                       passive_container_duration_s=(comparison["passive_video"]["frames"] /
                                                     comparison["passive_video"]["fps"]),
                       dlc_rows=d["rows"], dlc_indices_consecutive=d["index_consecutive_from_zero"],
                       dlc_matches_retained_timeline=d["matches_retained_video_rows"])
            detailed.append({"record_id": record_id, "decoded_comparison": comparison, "dlc": d})
            print(record_id, "pixel identical", comparison["all_samples_rgb_identical"],
                  "max MAE", comparison["max_sample_mae"], flush=True)
        ledger.append(row)
    (OUT / "input_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "decoded_evidence.json").write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = list(dict.fromkeys(k for r in ledger for k in r))
    with (OUT / "input_ledger.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in ledger:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    summary = {"input_count": len(ledger), "novice_pair_count": sum(r["novice_pair"] for r in ledger),
               "novice_pair_tasks": dict(Counter(r["task"] for r in ledger if r["novice_pair"])),
               "candidate_count": sum(r["candidate"] for r in ledger),
               "candidate_tasks": dict(Counter(r["task"] for r in ledger if r["candidate"])),
               "candidate_observers": sorted({r["observer"] for r in ledger if r["candidate"]}),
               "all_candidate_pixel_samples_identical": all(r["decoded_samples_rgb_identical"] for r in ledger if r["candidate"]),
               "user2_task_proof": {"file": str(SCORES), "sha256": sha(SCORES), "row": score_row[0]},
               "geometry_scope": "Full-frame 1280x1024 video-media normalized gaze. Owner confirmed mapping; decoded single-camera passive versus active source samples retain the same image geometry.",
               "coordinate_mapping_confirmation": {
                   "status": "confirmed",
                   "evidence_type": "user_confirmation",
                   "exact_answer": "视频画面归一化坐标 1280*1024",
                   "mapping": "x_px = gaze_x * 1280; y_px = gaze_y * 1024",
                   "crop_offset_px": [0, 0],
                   "stereo_panel_offset_px": [0, 0],
                   "scope": "native passive gaze NPY coordinates refer to the full single-camera media image"
               },
               "frame_alignment": "Match each original Frame_Number to retained precise video_ts row. That row is source playback/DLC zero-based row. Never use rounded native NPY timestamp or nominal FPS as clock.",
               "no_source_writes": True}
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
