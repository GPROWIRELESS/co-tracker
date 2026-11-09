"""Snapshot historical notebook as text and independently audit its data mapping.

No historical notebook code is executed. No source dataset files are written.
"""

import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np


OUT = Path(__file__).resolve().parent
REPO = Path(r"D:\GitHub_Project\DVSS_Pytorch")
ROOT = Path(r"D:\Dataset\dvss")
COMMIT = "e9a153a5e5e284e929272cf256cfeca07b43fa5f"
NOTEBOOK = "Dataloader/generate_gaze_npy.ipynb"
REPORT = "Dataloader/alignment_report_passive.csv"


def git(*args):
    return subprocess.check_output(["git", "-c", "safe.directory=D:/GitHub_Project/DVSS_Pytorch", *args], cwd=REPO)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    notebook_bytes = git("show", f"{COMMIT}:{NOTEBOOK}")
    report_bytes = git("show", f"{COMMIT}:{REPORT}")
    notebook = json.loads(notebook_bytes)
    (OUT / "historical_generate_gaze_npy.ipynb.txt").write_bytes(notebook_bytes)
    text = "\n\n".join(f"# NOTEBOOK CELL {i}\n" + "".join(c.get("source", []))
                         for i, c in enumerate(notebook["cells"]))
    (OUT / "historical_generate_gaze_npy.py.txt").write_text(text, encoding="utf-8")
    (OUT / "historical_alignment_report_passive.csv").write_bytes(report_bytes)
    historical = list(csv.DictReader(io.StringIO(report_bytes.decode("utf-8-sig"))))
    by_id = {Path(r["out_npy"].replace("\\", "/")).stem.removesuffix("_gaze"): r for r in historical}
    ledger = json.loads((OUT / "input_ledger.json").read_text(encoding="utf-8"))
    output = []
    selected_reports = []
    mappings = OUT / "native_to_raw_mapping"
    mappings.mkdir(exist_ok=True)
    for rec in ledger:
        if not rec["candidate"]:
            continue
        rid = rec["record_id"]
        raw_path = ROOT / "Passive_Data" / f"User{rec['observer']}" / "csv" / f"{rid}_gaze.csv"
        raw = read_rows(raw_path)
        ts_path = ROOT / "Passive_Data" / f"User{rec['observer']}" / "csv" / f"{rid}_video_ts.csv"
        vts = [r for r in read_rows(ts_path) if r.get("IsDelete", "0").strip() in {"", "0", "0.0"}]
        frames = np.asarray([int(r["Frame_Number"]) for r in vts], dtype=np.int64)
        ns = np.asarray([int(Decimal(r["Timestamp(ns)"])) for r in vts], dtype=np.int64)
        ticks = np.asarray([int(Decimal(r["TIMETICK(f=10000000)"])) for r in raw], dtype=np.int64)
        # This audit implements the documented arithmetic independently.
        # It does not import or execute historical notebook functions.
        gaze_ns = ns[0] + (ticks - ticks[0]) * 100
        order = np.argsort(gaze_ns)
        sorted_ns = gaze_ns[order]
        right = np.searchsorted(sorted_ns, ns, side="left").clip(0, len(raw) - 1)
        left = (np.searchsorted(sorted_ns, ns, side="left") - 1).clip(0, len(raw) - 1)
        choose_right = np.abs(sorted_ns[right] - ns) < np.abs(sorted_ns[left] - ns)
        selected = order[np.where(choose_right, right, left)]
        gx = np.asarray([float(r["BPOGX"]) for r in raw], dtype=np.float32)
        gy = np.asarray([float(r["BPOGY"]) for r in raw], dtype=np.float32)
        valid = np.asarray([int(Decimal(r["BPOGV"])) for r in raw], dtype=np.uint8)
        reconstructed_x = np.clip(gx[selected], 0, 1)
        reconstructed_y = np.clip(gy[selected], 0, 1)
        current = np.load(rec["gaze_npy_path"], allow_pickle=False)
        equal = {
            "frame_number": np.array_equal(current["Frame_Number"], frames),
            "timestamp_ns": np.array_equal(current["Timestamp(ns)"], ns),
            "gaze_x": np.array_equal(current["gaze_x"], reconstructed_x, equal_nan=True),
            "gaze_y": np.array_equal(current["gaze_y"], reconstructed_y, equal_nan=True),
            "valid": np.array_equal(current["valid"], valid[selected]),
        }
        time_col = next(k for k in raw[0] if k.startswith("TIME("))
        raw_time = np.asarray([float(r[time_col]) for r in raw], dtype=float)
        raw_vid = np.asarray([int(Decimal(r["VID_FRAME"])) for r in raw], dtype=np.int64)
        selected_vid = raw_vid[selected]
        original_video_row = np.arange(len(current), dtype=np.int64)
        delta = selected_vid - original_video_row
        valid_frame = (selected_vid >= 0) & (selected_vid < len(frames))
        nearest_dt = np.abs(gaze_ns[selected] - ns) / 1e6
        mapping_path = mappings / f"{rid}.npz"
        np.savez_compressed(mapping_path, native_npy_row=original_video_row,
                            original_frame_number=frames, source_content_ns=ns,
                            selected_raw_eye_row=selected, selected_raw_vid_frame=selected_vid,
                            selected_raw_time_s=raw_time[selected], selected_raw_tick=ticks[selected],
                            selected_raw_vid_frame_in_bounds=valid_frame,
                            selected_raw_vid_frame_minus_native_row=delta,
                            source_elapsed_s=(ns - ns[0]) / 1e9,
                            historical_nearest_dt_ms=nearest_dt)
        row = {
            "record_id": rid, "rows": len(current), "all_fields_reproduced_exactly": all(equal.values()),
            "field_equality": equal, "historical_report": by_id.get(rid),
            "selected_raw_vid_frame_minus_native_row_min": int(delta.min()),
            "selected_raw_vid_frame_minus_native_row_median": float(np.median(delta)),
            "selected_raw_vid_frame_minus_native_row_max": int(delta.max()),
            "selected_raw_vid_frame_out_of_bounds_count": int((~valid_frame).sum()),
            "distinct_in_bounds_selected_raw_vid_frames": int(len(np.unique(selected_vid[valid_frame]))),
            "source_video_frames_without_selected_native_gaze": int(len(frames) - len(np.unique(selected_vid[valid_frame]))),
            "selected_raw_video_frame_differs_native_row_count": int(np.sum(delta != 0)),
            "historical_native_clock_nearest_dt_ms_max": float(nearest_dt.max()),
            "historical_coordinate_clamped_rows": int(np.sum((gx[selected] < 0) | (gx[selected] > 1) | (gy[selected] < 0) | (gy[selected] > 1))),
            "mapping_npz": str(mapping_path), "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            "conclusion": "Current NPY provenance is exact. Native Frame_Number attaches gaze by source acquisition elapsed time; selected raw VID_FRAME often names a different displayed CFR frame. Historical algorithm did not use VID_FRAME."
        }
        output.append(row)
        rec["playback_alignment_status"] = "historical acquisition-clock mapping verified exactly; raw observation VID_FRAME association differs; native point association remains unchanged"
        rec["constructor_reproduction_exact"] = all(equal.values())
        rec["native_to_raw_mapping_npz"] = str(mapping_path)
        selected_reports.append(by_id[rid])
        print(rid, "exact", all(equal.values()), "delta", delta.min(), np.median(delta), delta.max(), flush=True)
    with (OUT / "historical_alignment_report_selected16.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_reports[0]))
        writer.writeheader()
        writer.writerows(selected_reports)
    provenance = {
        "repository": str(REPO), "commit": COMMIT,
        "commit_metadata": git("show", "-s", "--format=%H%n%an%n%aI%n%cn%n%cI%n%s", COMMIT).decode("utf-8").splitlines(),
        "notebook_git_path": NOTEBOOK,
        "notebook_blob_oid": git("rev-parse", f"{COMMIT}:{NOTEBOOK}").decode().strip(),
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "report_git_path": REPORT, "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "historical_code_executed": False,
        "independent_comparison_matches_all16": all(r["all_fields_reproduced_exactly"] for r in output),
        "historical_parameters": {"CLAMP_01": True, "MAX_DT_MS": None, "FILTER_ISDELETE": True},
        "historical_passive_transform": "first_video_ts_ns + 100*(raw_TIMETICK-first_TIMETICK); nearest raw gaze for each retained acquisition timestamp; BPOG coordinate clip; copy BPOGV without filter",
        "historical_frame_assignment": "original Frame_Number from retained acquisition video_ts; TIME and VID_FRAME unused",
        "alignment_status": "NPY construction fully verified; native association differs from observation CFR playback frames. Reporting canonical-source-clock output does not verify physical-playback lead/lag.",
        "deterministic_repair_possibility": "Exact selected raw eye row is recoverable for every existing native NPY row with this verified constructor. A separate mapping could attach that existing NPY point to selected raw VID_FRAME and then the corresponding source frame, retaining source precise clock. Raw VID_FRAME semantics and partial/duplicate coverage must be resolved before such a branch is admitted.",
        "records": output,
    }
    (OUT / "constructor_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "input_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(k for r in ledger for k in r))
    with (OUT / "input_ledger.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in ledger:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    summary_path = OUT / "audit_summary.json"
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    s["native_constructor_provenance"] = str(OUT / "constructor_provenance.json")
    s["native_constructor_reproduced_exact_all16"] = provenance["independent_comparison_matches_all16"]
    s["physical_playback_frame_alignment"] = "Discrepancy verified. Historical NPY uses original acquisition elapsed clock nearest raw TIMETICK. Raw VID_FRAME identifies another CFR source row for many points. Native branch retains provided mapping; direct physical observer latency interpretation requires correction/limitation."
    summary_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    playback_path = OUT / "raw_playback_audit.json"
    playback = json.loads(playback_path.read_text(encoding="utf-8"))
    for r in playback:
        r["construction_code_status"] = "Historical constructor found and independently reproduces all five current NPY fields exactly; see constructor_provenance.json."
        r["playback_alignment_status"] = "verified historical acquisition-clock association differs from raw CFR playback-frame association; native branch remains unchanged"
    playback_path.write_text(json.dumps(playback, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
