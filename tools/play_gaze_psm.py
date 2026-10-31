#!/usr/bin/env python3
"""Play source videos with frame-aligned gaze and left/right PSM mask centroids."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from tools.gaze_psm_pipeline import (
    interpolate_valid, number, read_rows, stage_frames, status_code, timeline,
    timestamp_ns,
)

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"gaze": "#ffe04b", "l": "#21d4fd", "r": "#ff8c42"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT/"Index/content_index.csv")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/mnt/nas/Users/yxl3527/Dataset/dvss/Active_Data"),
                        help="Active_Data directory, or its parent dvss directory")
    parser.add_argument("--tasks", nargs="+", default=["A", "C"])
    parser.add_argument("--skill", default="novice", help="Skill filter, or all")
    parser.add_argument("--records", nargs="+", help="Recording IDs; overrides task/skill filters")
    parser.add_argument("--gaze-source", choices=["raw", "allocation"], default="raw")
    parser.add_argument("--gaze-units", choices=["normalized", "pixels"], default="normalized")
    parser.add_argument("--gaze-x-column")
    parser.add_argument("--gaze-y-column")
    parser.add_argument("--gaze-valid-values", nargs="+", default=["2"])
    parser.add_argument("--flip-gaze-y", action="store_true")
    parser.add_argument("--max-gap", type=float, default=.1,
                        help="Maximum gap in seconds for raw-gaze interpolation")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--list", action="store_true", help="List selected recordings without opening a window")
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_gap) or args.max_gap <= 0:
        parser.error("--max-gap must be positive and finite")
    if not 0 <= args.threshold <= 255 or args.min_area < 1:
        parser.error("--threshold must be 0–255; --min-area must be positive")
    if (args.data_root/"Active_Data").is_dir():
        args.data_root = args.data_root/"Active_Data"
    return args


def select_records(args):
    rows = read_rows(args.index)
    selected = {}
    for row in rows:
        if args.records:
            keep = row["content_id"] in args.records
        else:
            keep = (row["Task Name"] in args.tasks
                    and (args.skill == "all" or row["src_skill"] == args.skill))
        if keep:
            selected[row["content_id"]] = row
    if args.records and set(args.records)-selected.keys():
        raise ValueError(f"Recording IDs missing from index: {', '.join(sorted(set(args.records)-selected.keys()))}")
    if not selected:
        raise ValueError("No recordings match the selection")
    return sorted(selected.values(), key=lambda row: tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", row["content_id"])))


def align_gaze(base, cid, allocations, ns, width, height, args):
    """Use exact capture timestamps for raw gaze; allocation gaze maps by frame ID."""
    origin = int(ns[0])
    native_time = (ns-origin)/1e9
    if args.gaze_source == "raw":
        rows = [r for r in read_rows(base/"csv"/f"{cid}_gaze.csv")
                if str(r.get("Capture Unix timestamp", "")).strip()]
        times = np.asarray([(timestamp_ns(r["Capture Unix timestamp"])-origin)/1e9
                            for r in rows])
        xcol, ycol = args.gaze_x_column or "norm_x", args.gaze_y_column or "norm_y"
        valid_col = "Status"
    else:
        rows, times = allocations, native_time
        xcol, ycol = args.gaze_x_column or "gaze_x", args.gaze_y_column or "gaze_y"
        valid_col = "valid"
    accepted = {status_code(value) for value in args.gaze_valid_values}-{None}
    xy = np.full((len(rows), 2), np.nan)
    for i, row in enumerate(rows):
        if status_code(row.get(valid_col)) not in accepted:
            continue
        point = np.asarray([number(row[xcol]), number(row[ycol])])
        if not np.isfinite(point).all():
            continue
        if args.gaze_units == "normalized":
            if np.any(point < 0) or np.any(point > 1):
                continue
            point *= [width, height]
        if args.flip_gaze_y:
            point[1] = height-point[1]
        if np.all(point >= 0) and np.all(point <= [width, height]):
            xy[i] = point
    if args.gaze_source == "allocation":
        return xy  # Do not interpolate through invalid allocation rows.
    aligned = interpolate_valid(times, xy, native_time, args.max_gap)
    valid = np.flatnonzero(np.isfinite(xy).all(axis=1))
    if len(valid) == 1:
        exact = np.isclose(native_time, times[valid[0]], atol=1e-9, rtol=0)
        aligned[exact] = xy[valid[0]]
    return aligned


def stage_segments(labels):
    """Keep each consecutive occurrence, including none and separated repeats."""
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1]
    return [(int(start), int(end), str(labels[start]) or "none")
            for start, end in zip(starts, np.r_[starts[1:], len(labels)])]


def stage_target(starts, frame, direction):
    target = bisect_right(starts, frame)-1+direction
    return starts[target] if 0 <= target < len(starts) else frame


def centroid(frame, threshold=127, min_area=10):
    mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > threshold).astype(np.uint8)
    moments = cv2.moments(mask, binaryImage=True)
    area = moments["m00"]
    point = (moments["m10"]/area, moments["m01"]/area) if area >= min_area else None
    return point, mask


class VideoReader:
    """Decode consecutive frames without seeking; seek exactly on navigation."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Cannot open {path}")
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.index, self.frame = -1, None

    def read(self, index):
        if index == self.index:
            return self.frame
        if index != self.index+1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index} of {self.path}")
        self.index, self.frame = index, frame
        return frame

    def close(self):
        self.cap.release()


class Recording:
    def __init__(self, record, args):
        self.readers = []
        self.cid = record["content_id"]
        base = args.data_root/self.cid.split("_")[0]
        self.frames, ns, deleted = timeline(base, self.cid)
        self.times = (ns-int(ns[0]))/1e9
        allocations, labels, _ = stage_frames(base, self.cid, self.frames)
        self.segments = stage_segments(labels)
        self.starts = [s[0] for s in self.segments]
        self.count = len(self.frames)
        try:
            self.video = self.open(base/"stimuli"/f"{self.cid}.mp4")
            self.masks = {side: self.open(base/"sam"/self.cid/"mask"/f"obj_psm_{side}.mp4")
                          for side in ("l", "r")}
            # Masks follow retained timestamps. RGB may still contain deleted frames.
            for side, reader in self.masks.items():
                if reader.count != self.count:
                    raise ValueError(f"{side} mask has {reader.count} frames, expected {self.count} retained timestamps")
                if (reader.width, reader.height) != (self.video.width, self.video.height):
                    raise ValueError(f"{side} mask resolution differs from the source video")
            if self.video.count == self.count:
                self.video_indices = np.arange(self.count)
            elif deleted and self.video.count == self.count+deleted:
                rows = read_rows(base/"csv"/f"{self.cid}_video_ts.csv")
                self.video_indices = np.asarray([i for i, r in enumerate(rows)
                    if str(r.get("IsDelete", "0")).strip() in {"", "0", "0.0"}])
            else:
                raise ValueError(f"RGB video has {self.video.count} frames; cannot align with {self.count} retained timestamps")
            self.gaze = align_gaze(base, self.cid, allocations, ns,
                                   self.video.width, self.video.height, args)
        except Exception:
            self.close()
            raise

    def open(self, path):
        reader = VideoReader(path)
        self.readers.append(reader)
        return reader

    def read(self, index, args):
        image = cv2.cvtColor(self.video.read(int(self.video_indices[index])), cv2.COLOR_BGR2RGB)
        overlays = {side: centroid(reader.read(index), args.threshold, args.min_area)
                    for side, reader in self.masks.items()}
        return image, overlays

    def close(self):
        for reader in self.readers:
            reader.close()


class Player:
    def __init__(self, args, records):
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import Image, ImageTk

        self.tk, self.messagebox = tk, messagebox
        self.Image, self.ImageTk = Image, ImageTk
        self.args, self.records = args, records
        self.recording, self.record_index = None, -1
        self.frame, self.playing, self.timer = 0, False, None
        self.render_timer, self.updating_slider = None, False
        self.window = tk.Tk()
        self.window.title("Gaze / PSM video player")
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry(f"{min(1400, sw-80)}x{min(1000, sh-100)}+30+30")
        self.window.minsize(760, 560)

        header = ttk.Frame(self.window, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Recording").pack(side="left")
        self.record_box = ttk.Combobox(header, state="readonly", width=38,
            values=[f"{r['content_id']}  |  Task {r['Task Name']}  |  {r['src_skill']}" for r in records])
        self.record_box.pack(side="left", padx=8)
        self.record_box.bind("<<ComboboxSelected>>", lambda e: self.open_record(self.record_box.current()))
        ttk.Button(header, text="Previous video", command=lambda: self.open_record(self.record_index-1)).pack(side="left", padx=3)
        ttk.Button(header, text="Next video", command=lambda: self.open_record(self.record_index+1)).pack(side="left", padx=3)

        stages = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        stages.pack(fill="x")
        ttk.Button(stages, text="← Previous stage", command=lambda: self.move_stage(-1)).pack(side="left")
        self.stage_box = ttk.Combobox(stages, state="readonly", width=65)
        self.stage_box.pack(side="left", fill="x", expand=True, padx=8)
        self.stage_box.bind("<<ComboboxSelected>>", lambda e: self.seek(self.recording.starts[self.stage_box.current()]))
        ttk.Button(stages, text="Next stage →", command=lambda: self.move_stage(1)).pack(side="left")

        legend = ttk.Frame(self.window, padding=(8, 0, 8, 5))
        legend.pack(fill="x")
        for label, color in (("● Gaze", COLORS["gaze"]), ("⊕ Left PSM centroid", COLORS["l"]), ("⊕ Right PSM centroid", COLORS["r"])):
            tk.Label(legend, text=label, bg="#222831", fg=color, padx=10, pady=3).pack(side="left", padx=(0, 6))
        self.show_masks = tk.BooleanVar(value=False)
        ttk.Checkbutton(legend, text="Show mask regions", variable=self.show_masks, command=self.render).pack(side="right")

        self.canvas = tk.Canvas(self.window, bg="#101318", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())

        self.status = tk.StringVar()
        ttk.Label(self.window, textvariable=self.status, padding=(8, 5)).pack(fill="x")
        self.slider = tk.Scale(self.window, from_=0, to=1, orient="horizontal",
                               showvalue=False, resolution=1, command=self.scrub)
        self.slider.pack(fill="x", padx=8)
        controls = ttk.Frame(self.window, padding=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="< Frame", command=lambda: self.move(-1)).pack(side="left", padx=3)
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=3)
        ttk.Button(controls, text="Frame >", command=lambda: self.move(1)).pack(side="left", padx=3)
        ttk.Label(controls, text="Speed").pack(side="left", padx=(15, 4))
        self.speed = tk.StringVar(value="1")
        self.speed_box = ttk.Combobox(controls, textvariable=self.speed, state="readonly",
                                     values=["0.25", "0.5", "1", "2"], width=5)
        self.speed_box.pack(side="left")
        self.speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        ttk.Label(controls, text="Space: play/pause   < / > or , / .: frame   ← / →: stage   Esc: quit").pack(side="right")

        # Bind before widget class handlers so arrows never alter the timeline,
        # a focused combobox, or a button instead of navigating stages.
        tag = f"PlayerKeys{id(self)}"
        self.window.bind_class(tag, "<KeyPress>", self.keypress)
        self.bind_keys(self.window, tag)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.open_record(0)
        self.canvas.focus_set()

    def bind_keys(self, widget, tag):
        widget.bindtags((tag, *widget.bindtags()))
        for child in widget.winfo_children():
            self.bind_keys(child, tag)

    def keypress(self, event):
        if event.keysym in {"comma", "less", "period", "greater"}:
            return self.move(-1 if event.keysym in {"comma", "less"} else 1)
        if event.keysym in {"Left", "Right"}:
            return self.move_stage(-1 if event.keysym == "Left" else 1)
        if event.keysym == "space":
            return self.toggle_play()
        if event.keysym == "Escape":
            self.close()
            return "break"

    def open_record(self, index):
        if not 0 <= index < len(self.records):
            return
        self.pause()
        self.status.set(f"Loading {self.records[index]['content_id']}…")
        self.window.update_idletasks()
        try:
            candidate = Recording(self.records[index], self.args)
            image, overlays = candidate.read(0, self.args)
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            if "candidate" in locals():
                candidate.close()
            self.record_box.current(max(0, self.record_index))
            self.status.set(str(exc))
            self.messagebox.showerror("Cannot load recording", str(exc), parent=self.window)
            return
        if self.recording is not None:
            self.recording.close()
        self.recording, self.record_index, self.frame = candidate, index, 0
        self.image, self.overlays = image, overlays
        self.record_box.current(index)
        self.stage_box.configure(values=[
            f"{i+1}. {name}  |  frames {start}–{end-1}  |  {candidate.times[start]:.2f}s"
            for i, (start, end, name) in enumerate(candidate.segments)])
        self.slider.configure(to=candidate.count-1)
        self.render()

    def pause(self):
        self.playing = False
        if self.timer is not None:
            self.window.after_cancel(self.timer)
            self.timer = None
        self.play_button.configure(text="Play")

    def seek(self, frame, pause=True):
        if pause:
            self.pause()
        if self.recording is None:
            return "break"
        frame = max(0, min(self.recording.count-1, int(frame)))
        try:
            image, overlays = self.recording.read(frame, self.args)
        except (ValueError, cv2.error) as exc:
            self.pause()
            self.status.set(str(exc))
            self.messagebox.showerror("Cannot read frame", str(exc), parent=self.window)
            return "break"
        self.frame, self.image, self.overlays = frame, image, overlays
        self.render()
        return "break"

    def move(self, delta):
        return self.seek(self.frame+delta)

    def move_stage(self, direction):
        if self.recording is not None:
            return self.seek(stage_target(self.recording.starts, self.frame, direction))
        return "break"

    def scrub(self, value):
        if not self.updating_slider and int(float(value)) != self.frame:
            self.seek(int(float(value)))

    def toggle_play(self):
        if self.recording is None:
            return "break"
        if self.playing:
            self.pause()
        else:
            if self.frame == self.recording.count-1:
                self.seek(0)
            self.playing = True
            self.play_button.configure(text="Pause")
            self.reset_clock()
            self.schedule_tick()
        return "break"

    def reset_clock(self):
        self.clock_start = time.monotonic()
        self.video_start = self.recording.times[self.frame]

    def change_speed(self, event=None):
        if self.playing:
            if self.timer is not None:
                self.window.after_cancel(self.timer)
                self.timer = None
            self.reset_clock()
            self.schedule_tick()

    def schedule_tick(self):
        if self.frame >= self.recording.count-1:
            self.pause()
            return
        due = (self.recording.times[self.frame+1]-self.video_start)/float(self.speed.get())
        delay = max(1, round(1000*(due-(time.monotonic()-self.clock_start))))
        self.timer = self.window.after(delay, self.tick)

    def tick(self):
        self.timer = None
        if self.playing:
            self.seek(self.frame+1, pause=False)
            if self.playing:
                self.schedule_tick()

    def resize(self, event=None):
        if self.render_timer is not None:
            self.window.after_cancel(self.render_timer)
        self.render_timer = self.window.after_idle(self.render)

    def render(self):
        self.render_timer = None
        if self.recording is None:
            return
        record = self.recording
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(width/record.video.width, height/record.video.height)
        dw, dh = max(1, round(record.video.width*scale)), max(1, round(record.video.height*scale))
        ox, oy = (width-dw)/2, (height-dh)/2
        image = self.image
        if self.show_masks.get():
            image = image.copy()
            for side, (_, mask) in self.overlays.items():
                color = tuple(int(COLORS[side][i:i+2], 16) for i in (1, 3, 5))
                selected = mask.astype(bool)
                image[selected] = .65*image[selected]+.35*np.asarray(color)
        image = self.Image.fromarray(image).resize((dw, dh), self.Image.Resampling.BILINEAR)
        self.photo = self.ImageTk.PhotoImage(image, master=self.window)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        points = {"gaze": record.gaze[self.frame],
                  **{side: point for side, (point, _) in self.overlays.items()}}
        for name, point in points.items():
            if point is None or not np.isfinite(point).all():
                continue
            x, y = ox+point[0]*dw/record.video.width, oy+point[1]*dh/record.video.height
            r = 7 if name == "gaze" else 9
            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="black", width=4,
                                    fill=COLORS[name] if name == "gaze" else "")
            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=COLORS[name], width=2)
            if name != "gaze":
                self.canvas.create_line(x-r-4, y, x+r+4, y, fill=COLORS[name], width=2)
                self.canvas.create_line(x, y-r-4, x, y+r+4, fill=COLORS[name], width=2)
        segment = bisect_right(record.starts, self.frame)-1
        self.stage_box.current(segment)
        self.updating_slider = True
        self.slider.set(self.frame)
        self.updating_slider = False
        detail = []
        for name, point in points.items():
            label = {"gaze": "Gaze", "l": "Left PSM", "r": "Right PSM"}[name]
            valid = point is not None and np.isfinite(point).all()
            detail.append(f"{label}: ({point[0]:.1f}, {point[1]:.1f})" if valid else f"{label}: unavailable")
        self.status.set(f"Frame {self.frame} / {record.count-1}  |  CSV {record.frames[self.frame]}  |  "
                        f"{record.times[self.frame]:.3f}s  |  Stage {segment+1}/{len(record.segments)}: "
                        f"{record.segments[segment][2]}\n"+"  |  ".join(detail))

    def close(self):
        self.pause()
        if self.render_timer is not None:
            self.window.after_cancel(self.render_timer)
        if self.recording is not None:
            self.recording.close()
        self.window.destroy()


def main(argv=None):
    args = parse_args(argv)
    try:
        records = select_records(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.list:
        for record in records:
            print(f"{record['content_id']}\tTask {record['Task Name']}\t{record['src_skill']}")
        return
    cv2.setNumThreads(1)
    app = Player(args, records)
    app.window.mainloop()


if __name__ == "__main__":
    main()
