#!/usr/bin/env python3
"""Check source left/right mask videos against cached trajectories and outputs.

Reads source videos without modifying them. Samples each mask video, compares
its centroid and area to both cache sides, verifies the same-side interpolation
and smoothing, and checks whole-task position peak values and visual overlays.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/psm_mask_side_audit_matplotlib")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from tools.gaze_psm_pipeline import interpolate_valid, smooth_valid


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def error(actual, expected):
    if not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        return float("inf")
    valid = np.isfinite(actual)
    return float(np.max(abs(actual[valid]-expected[valid]))) if valid.any() else 0.


def position_r(gaze, psm, shift):
    n = len(gaze)
    if shift >= 0:
        g, p = gaze[:n-shift], psm[shift:]
    else:
        g, p = gaze[-shift:], psm[:n+shift]
    valid = np.isfinite(g).all(axis=1) & np.isfinite(p).all(axis=1)
    g, p = g[valid], p[valid]
    g, p = g-g.mean(axis=0), p-p.mean(axis=0)
    return float(np.sum(g*p)/np.sqrt(np.sum(g*g)*np.sum(p*p)))


def overlays(destination, selected, data_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 4.7*len(selected)), squeeze=False)
    for row, (cid, entry) in enumerate(selected.items()):
        index = entry["index"]
        video = data_root/cid.split("_")[0]/"stimuli"/f"{cid}.mp4"
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot read RGB frame: {video}, index {index}")
        finally:
            capture.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{cid} / source RGB\nNative video frame {index} (zero based)", fontsize=12)
        for col, (side, color) in enumerate((("l", np.array([0., .45, .7])), ("r", np.array([.85, .33, 0.]))), start=1):
            mask = entry[side]["mask"]
            assert mask.shape == rgb.shape[:2]
            ax = axes[row, col]
            ax.imshow(rgb)
            tinted = np.zeros((*mask.shape, 4))
            tinted[mask, :3], tinted[mask, 3] = color, .35
            ax.imshow(tinted)
            if mask.any() and not mask.all():
                ax.contour(mask, levels=[.5], colors=[color], linewidths=1.)
            xy = entry[side]["centroid"]
            if np.isfinite(xy).all():
                ax.plot(*xy, marker="x", color="white", ms=11, mew=2)
                ax.plot(*xy, marker="+", color=color, ms=11, mew=1.3)
            status = " / empty source mask" if not mask.any() else ""
            ax.set_title(f"obj_psm_{side}.mp4\nMatches native_psm_{side} / {'Left' if side == 'l' else 'Right'} PSM label{status}", fontsize=12)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Source mask identity check: RGB, left mask, right mask", fontsize=18, y=.995)
    fig.text(.5, .012, "Colored regions come directly from the source mask videos. Crosses mark their centroids, checked against the cache.\n"
             "Labels follow the source filenames; these examples do not validate segmentation quality in every frame or establish robot-arm IDs.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .97), h_pad=3.2, w_pad=1.4)
    fig.savefig(destination/"source_mask_overlays.png", dpi=160, bbox_inches="tight")
    fig.savefig(destination/"source_mask_overlays.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/gaze_psm_xcorr"))
    parser.add_argument("--output", type=Path, default=Path("outputs/score_corr/mask_side_audit"))
    parser.add_argument("--samples-per-video", type=int, default=9)
    opts = parser.parse_args()
    if opts.samples_per_video < 1:
        parser.error("--samples-per-video must be positive")
    config = json.loads((opts.input/"config.json").read_text())
    data_root = Path(config["data_root"])
    with (opts.input/"record_summary.csv").open(newline="") as stream:
        peaks = {(r["content_id"], r["side"]): r for r in csv.DictReader(stream)
                 if r["signal"] == "position" and r["mode"] == "whole"}
    cv2.setNumThreads(1)
    samples, records, selected = [], [], {}
    overlay_ids = {"User1_1_1", "User3_2_1", "User11_2_1", "User1_3_1"}
    paths = sorted((opts.input/"cache").glob("*.npz"))
    assert paths
    opts.output.mkdir(parents=True, exist_ok=True)
    for recording_index, path in enumerate(paths, start=1):
        cid = path.stem
        with np.load(path, allow_pickle=False) as cache:
            names = ["native_time_s", "time_s", "qc_json", "frame_number"]
            names += [f"{name}_{side}" for side in ("l", "r") for name in ("native_psm", "area", "psm_raw", "psm", "gaze", "resolution")]
            data = {name: cache[name] for name in names}
        qc = json.loads(str(data["qc_json"]))
        n = len(data["native_time_s"])
        indices = np.unique(np.rint(np.linspace(.05, .95, opts.samples_per_video)*(n-1)).astype(int))
        middle = int(indices[len(indices)//2])
        if cid in overlay_ids:
            # Representative RGB videos and masks have identical retained frame counts.
            assert qc["deleted_rows"] == 0
            selected[cid] = {"index": middle}
        for side in ("l", "r"):
            opposite = "r" if side == "l" else "l"
            native = data[f"native_psm_{side}"]
            aligned = interpolate_valid(data["native_time_s"], native, data["time_s"], config["max_gap"])
            smoothed = smooth_valid(aligned, config["hz"], config["smooth_ms"])
            raw_error = error(aligned, data[f"psm_raw_{side}"])
            smooth_error = error(smoothed, data[f"psm_{side}"])
            peak = peaks[(cid, side)]
            lag = float(peak["peak_lag_s"])
            peak_error = abs(position_r(data[f"gaze_{side}"], smoothed, round(lag*config["hz"]))-float(peak["peak_r"]))
            mask_path = data_root/cid.split("_")[0]/"sam"/cid/"mask"/f"obj_psm_{side}.mp4"
            before = mask_path.stat()
            capture = cv2.VideoCapture(str(mask_path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open source mask: {mask_path}")
            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                side_samples = []
                for index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Cannot read {mask_path} at frame {index}")
                    assert int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == index+1
                    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > config["threshold"]
                    assert list(frame.shape[1::-1]) == list(data[f"resolution_{side}"])
                    yy, xx = np.nonzero(mask)
                    area = len(xx)
                    xy = np.array([xx.mean(), yy.mean()]) if area >= config["min_area"] else np.array([np.nan, np.nan])
                    own = native[index]
                    other = data[f"native_psm_{opposite}"][index]
                    matches = bool(np.allclose(xy, own, atol=1e-9, rtol=0, equal_nan=True))
                    other_matches = bool(np.allclose(xy, other, atol=1e-9, rtol=0, equal_nan=True))
                    row = {"content_id": cid, "side": side, "frame_index": int(index), "source_mask": str(mask_path),
                           "decoded_x": xy[0], "decoded_y": xy[1], "cached_same_x": own[0], "cached_same_y": own[1],
                           "cached_opposite_x": other[0], "cached_opposite_y": other[1], "matches_same_side": matches,
                           "matches_opposite_side": other_matches, "discriminates_sides": bool(np.isfinite(xy).all() and not other_matches),
                           "area_matches": bool(area == data[f"area_{side}"][index]), "area": area}
                    samples.append(row)
                    side_samples.append(row)
                    if cid in overlay_ids and index == middle:
                        selected[cid][side] = {"mask": mask, "centroid": xy}
            finally:
                capture.release()
            after = mask_path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            records.append({"content_id": cid, "task": qc["task"], "side": side, "source_mask": str(mask_path),
                            "source_frames": frame_count, "cached_frames": n, "frame_count_matches": frame_count == n,
                            "samples": len(side_samples), "matching_samples": sum(r["matches_same_side"] for r in side_samples),
                            "discriminating_samples": sum(r["discriminates_sides"] for r in side_samples),
                            "area_mismatches": sum(not r["area_matches"] for r in side_samples),
                            "interpolated_max_error_px": raw_error, "smoothed_max_error_px": smooth_error,
                            "reported_whole_task_peak_error": peak_error, "source_size": before.st_size,
                            "source_mtime_ns": before.st_mtime_ns})
        if recording_index % 10 == 0 or recording_index == len(paths):
            print(f"Checked {recording_index}/{len(paths)} recordings", flush=True)
    write_csv(opts.output/"sampled_frames.csv", samples)
    write_csv(opts.output/"record_checks.csv", records)
    good = all(r["frame_count_matches"] and r["samples"] == r["matching_samples"] and not r["area_mismatches"]
               and r["interpolated_max_error_px"] < 1e-8 and r["smoothed_max_error_px"] < 1e-8
               and r["reported_whole_task_peak_error"] < 1e-8 for r in records)
    overlays(opts.output, selected, data_root)
    validation = {
        "all_checks_passed": good, "recordings": len(paths), "mask_videos": len(records), "sampled_frames": len(samples),
        "same_side_matches": sum(r["matches_same_side"] for r in samples),
        "opposite_side_discriminating_frames": sum(r["discriminates_sides"] for r in samples),
        "sampled_centroid_mismatches": sum(not r["matches_same_side"] for r in samples),
        "sampled_area_mismatches": sum(not r["area_matches"] for r in samples),
        "sampled_empty_or_small_masks": sum(r["area"] < config["min_area"] for r in samples),
        "representative_empty_masks": [{"content_id": cid, "side": side, "frame_index": entry["index"]}
                                       for cid, entry in selected.items() for side in ("l", "r")
                                       if not entry[side]["mask"].any()],
        "max_reported_peak_error": max(r["reported_whole_task_peak_error"] for r in records),
        "mapping": {"obj_psm_l.mp4": "native_psm_l -> psm_raw_l -> psm_l -> side=l -> Left PSM",
                    "obj_psm_r.mp4": "native_psm_r -> psm_raw_r -> psm_r -> side=r -> Right PSM"},
        "scope": "Samples from every source mask video plus full cached interpolation/smoothing and whole-task peak checks; not every video frame",
        "semantic_limit": "Filename identity and representative overlays are checked; robot-arm IDs and every-frame segmentation quality are not established",
    }
    (opts.output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps(validation, indent=2), flush=True)
    if not good:
        raise RuntimeError("Mask-side audit found mismatches; inspect record_checks.csv and sampled_frames.csv")


if __name__ == "__main__":
    main()
