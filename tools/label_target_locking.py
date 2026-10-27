#!/usr/bin/env python3
"""Review and edit target-locking points on every video frame, or annotate stage seeds."""
import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_items(inputs):
    records = {}
    for source in inputs:
        source = Path(source)
        paths = sorted(source.glob('*.json')) if source.is_dir() else [source]
        for path in paths:
            data = json.loads(path.read_text())
            if not isinstance(data, list):  # summary.json
                continue
            for item in data:
                if item.get('domain', 'active') == 'active':
                    records[item['video_path']] = item
    return list(records.values())


def stage_name(row):
    return row['stage'] if 'stage' in row else row['event']


def stages(rows, include_none=False):
    if any('event' not in row and 'stage' not in row for row in rows):
        raise ValueError('CSV has no stage or event column; stage annotations are required')
    result = []
    start = 0
    for end in range(1, len(rows) + 1):
        if end == len(rows) or stage_name(rows[end]) != stage_name(rows[start]):
            if include_none or stage_name(rows[start]).strip().lower() not in {'', 'none', 'nan'}:
                result.append((start, end))
            start = end
    return result


def save_csv(path, fields, rows):
    fields = [field for field in fields if not re.fullmatch(r'point_\d+_visible', field)]
    rows = [{key: value for key, value in row.items()
             if not re.fullmatch(r'point_\d+_visible', key)} for row in rows]
    backup = path.with_suffix(path.suffix + '.before_point_labels.bak')
    if not backup.exists():
        shutil.copy2(path, backup)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def record_paths(item, data_root):
    """Resolve /dvss/..., dvss/..., Active_Data/..., or legacy absolute paths."""
    parts = Path(item['video_path'].replace('\\', '/')).parts
    if 'Active_Data' not in parts:
        raise ValueError(f"Expected an Active_Data video path: {item['video_path']}")
    relative = Path(*parts[parts.index('Active_Data'):])
    if '..' in relative.parts:
        raise ValueError('Dataset paths cannot contain parent traversal')
    video = Path(data_root) / relative
    allocation = video.parent.parent / 'npy' / (video.stem + '_gaze.csv')
    return video, allocation


def frame_indices(rows, count, offset=None):
    if not rows or not all(('stage' in r or 'event' in r) and 'Frame_Number' in r for r in rows):
        raise ValueError('CSV requires rows with stage (or event) and Frame_Number columns')
    numbers = [int(r['Frame_Number']) for r in rows]
    offset = numbers[0] if offset is None else offset
    frames = [n - offset for n in numbers]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError('Frame_Number must increase strictly')
    if min(frames) < 0 or max(frames) >= count:
        raise ValueError(f'Allocation frames outside video (length {count}); set --frame-offset')
    return frames


def ensure_columns(fields, count):
    for column in ['point_label_mode'] + [f'point_{i}_{a}' for i in range(1, count+1)
                                         for a in ('x', 'y')]:
        if column not in fields:
            fields.append(column)


def seed_points(row, maximum=5):
    """Read a contiguous set of 1..maximum coordinate pairs; reject partial pairs/gaps."""
    points = []
    gap = False
    for i in range(1, maximum+1):
        x, y = row.get(f'point_{i}_x', ''), row.get(f'point_{i}_y', '')
        if x == '' and y == '':
            gap = True
            continue
        if gap or x == '' or y == '':
            raise ValueError('Seed points must be complete consecutive x/y pairs')
        points.append((float(x), float(y)))
    if not points:
        raise ValueError('Select at least one point')
    return points


def has_seed(row, count):
    if row.get('point_label_mode') != 'manual':
        return False
    try:
        seed_points(row, count)
        return True
    except (ValueError, TypeError):
        return False


def seed_index(rows, start, end, count):
    seeds = [i for i in dict.fromkeys((start, end-1)) if has_seed(rows[i], count)]
    if len(seeds) > 1:
        raise ValueError('Stage has manual seeds at both endpoints; select one in the UI and save again')
    return seeds[0] if seeds else None


def is_skipped(rows, start, end):
    return all(row.get('point_label_mode') == 'skipped' for row in rows[start:end])


def set_stage_skipped(rows, start, end):
    for row in rows[start:end]:
        row['point_label_mode'] = 'skipped'
        for key in list(row):
            if re.fullmatch(r'point_\d+_(x|y)', key):
                row[key] = ''


def read_allocation(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = [] if 'Frame_Number' in fields else ['Frame_Number']
        if 'stage' not in fields and 'event' not in fields:
            missing.append('stage (or event)')
        if missing:
            raise ValueError(f'{path}: missing required columns: {", ".join(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: allocation CSV has no rows')
    return fields, rows


def labelable_items(items, data_root):
    ready, unavailable = [], []
    for item in items:
        try:
            _, allocation = record_paths(item, data_root)
            read_allocation(allocation)
            ready.append(item)
        except (OSError, ValueError, csv.Error) as exc:
            unavailable.append(str(exc))
    return ready, unavailable


def stage_types(items, data_root, include_none=False):
    names = set()
    for item in items:
        _, allocation = record_paths(item, data_root)
        _, rows = read_allocation(allocation)
        names.update(stage_name(rows[start]) for start, _ in stages(rows, include_none))
    return sorted(names)


def skip_selected_stages(rows, segments, selected):
    candidate = [dict(row) for row in rows]
    for start, end in segments:
        if stage_name(rows[start]) in selected:
            set_stage_skipped(candidate, start, end)
    return candidate


def set_manual_seed(rows, start, end, points, seed=None):
    if not 1 <= len(points) <= 5:
        raise ValueError('Select between 1 and 5 points')
    seed = start if seed is None else seed
    if seed not in (start, end-1):
        raise ValueError('Manual seed must be at the first or last frame of the stage')
    # Invalidate prior predictions and the other endpoint when the seed changes.
    for index in range(start, end):
        row = rows[index]
        row['point_label_mode'] = 'manual' if index == seed else ''
        for key in list(row):
            if re.fullmatch(r'point_\d+_(x|y)', key):
                row[key] = ''
        for i, (x, y) in enumerate(points, 1):
            row[f'point_{i}_x'] = x if index == seed else ''
            row[f'point_{i}_y'] = y if index == seed else ''


def add_data_arguments(parser):
    parser.add_argument('--items', nargs='+', type=Path, default=[
        ROOT/'Index/items/novice_active_taskA'],
        help='Index JSON files or directories; selects active recordings of any task or skill level')
    parser.add_argument('--data-root', type=Path,
                        default=Path('/mnt/nas/Users/yxl3527/Dataset/dvss'),
                        help='Local dvss directory containing Active_Data')
    parser.add_argument('--points', type=int, default=5, help='Maximum points per stage (1–5); at least one is required')
    parser.add_argument('--frame-offset', type=int, help='Subtract from Frame_Number; default: first CSV frame')
    parser.add_argument('--include-none', action='store_true')


class SeedLabeler:
    def __init__(self, args, items):
        import tkinter as tk
        from tkinter import messagebox, font as tkfont
        import cv2
        from PIL import Image, ImageTk
        self.tk, self.messagebox, self.cv2 = tk, messagebox, cv2
        self.Image, self.ImageTk = Image, ImageTk
        self.args, self.items = args, items
        self.record = -1
        self.cap = None
        self.window = tk.Tk()
        self.window.title('DVSS target-locking annotation')
        # Scale Tk's named fonts so labels, controls, and dialogs use twice the text size.
        for name in tkfont.names(self.window):
            font = tkfont.nametofont(name, root=self.window)
            font.configure(size=2 * int(font.cget('size')))

        self.status = tk.Label(self.window, justify='left')
        self.status.pack()
        self.seed_side = tk.StringVar(value='first')
        selector = tk.Frame(self.window)
        selector.pack()
        tk.Label(selector, text='Label on:').pack(side='left')
        for text, value in [('First frame (left)', 'first'), ('Last frame (right)', 'last')]:
            tk.Radiobutton(selector, text=text, variable=self.seed_side, value=value,
                           command=self.select_seed).pack(side='left')
        self.skip_stage = tk.BooleanVar(value=False)
        skip_panel = tk.LabelFrame(self.window, text='Skip stage types throughout the labeling queue (check to skip)')
        skip_panel.pack(fill='x', padx=12)
        self.skip_types = {}
        columns = max(1, min(4, self.window.winfo_screenwidth() // 420))
        for index, name in enumerate(stage_types(items, args.data_root, args.include_none)):
            variable = tk.BooleanVar(value=False)
            self.skip_types[name] = variable
            tk.Checkbutton(skip_panel, text=name, variable=variable,
                           command=self.update_skip_selection).grid(
                               row=index // columns, column=index % columns, sticky='w', padx=8)
        tk.Label(skip_panel, text='Selections apply to every occurrence in this and following recordings. '
                 'Right / Enter saves and skips automatically.',
                 wraplength=max(300, self.window.winfo_screenwidth()-100)).grid(
                     row=(len(self.skip_types)+columns-1)//columns, column=0,
                     columnspan=columns, sticky='w')
        frames_panel = tk.Frame(self.window)
        frames_panel.pack()
        first_panel = tk.Frame(frames_panel)
        first_panel.pack(side='left', padx=(0, 12))
        last_panel = tk.Frame(frames_panel)
        last_panel.pack(side='left')
        self.first_caption = tk.Label(first_panel, fg='#146b26')
        self.first_caption.pack()
        self.last_caption = tk.Label(last_panel, fg='white', bg='#b00020',
                                     font='TkHeadingFont', padx=12, pady=10)
        self.last_caption.pack()
        self.canvas = tk.Canvas(first_panel, highlightthickness=0)
        self.canvas.pack()
        self.last_canvas = tk.Canvas(last_panel, highlightthickness=0)
        self.last_canvas.pack()
        tk.Label(self.window, text='Choose FIRST or LAST above, then click the selected view | Right click / Backspace: undo | R: clear\n'
                 'Right / Enter: save + next stage | Left: previous stage | N / P: next / previous video\n'
                 'Esc: quit | Video switching / quit discards unsaved clicks',
                 wraplength=max(300, self.window.winfo_screenwidth()-100)).pack()
        self.canvas.bind('<Button-1>', lambda event: self.add(event, 'first'))
        self.last_canvas.bind('<Button-1>', lambda event: self.add(event, 'last'))
        self.last_canvas.bind('<Button-3>', self.undo)
        self.canvas.bind('<Button-3>', self.undo)
        self.window.bind('<BackSpace>', self.undo)
        self.window.bind('<r>', self.clear)
        self.window.bind('<Right>', self.advance)
        self.window.bind('<Return>', self.advance)
        self.window.bind('<Left>', self.previous)
        for key in ('<n>', '<N>'):
            self.window.bind(key, self.next_video)
        for key in ('<p>', '<P>'):
            self.window.bind(key, self.previous_video)
        self.window.bind('<Escape>', lambda e: self.window.destroy())
        self.next_record()

    def paths(self, item):
        return record_paths(item, self.args.data_root)

    def next_video(self, event=None):
        if self.record + 1 < len(self.items):
            self.next_record()

    def previous_video(self, event=None):
        if self.record > 0:
            self.next_record(direction=-1)

    def next_record(self, direction=1):
        target = self.record + direction
        if target < 0:
            return
        if self.cap is not None:
            self.cap.release()
        self.record = target
        if self.record == len(self.items):
            self.messagebox.showinfo('Finished', 'All selected recordings are labeled.')
            self.window.destroy()
            return
        video, self.csv_path = self.paths(self.items[self.record])
        self.fields, self.rows = read_allocation(self.csv_path)
        self.cap = self.cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        count = int(self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        self.frames = frame_indices(self.rows, count, self.args.frame_offset)
        self.segments = stages(self.rows, self.args.include_none)
        ensure_columns(self.fields, self.args.points)
        candidate = skip_selected_stages(self.rows, self.segments, self.selected_skip_types())
        if candidate != self.rows:
            save_csv(self.csv_path, self.fields, candidate)
            self.rows = candidate
        self.stage = 0
        if self.stage == len(self.segments):
            self.next_record(direction=direction)
        else:
            self.show()

    def selected_skip_types(self):
        return {name for name, variable in self.skip_types.items() if variable.get()}

    def update_skip_selection(self):
        start, _ = self.segments[self.stage]
        self.skip_stage.set(stage_name(self.rows[start]) in self.selected_skip_types())
        self.redraw()

    def show(self):
        start, end = self.segments[self.stage]
        self.skip_stage.set(stage_name(self.rows[start]) in self.selected_skip_types())
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.frames[start])
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {self.frames[start]}')
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.frames[end-1])
        ok, last_frame = self.cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {self.frames[end-1]}')
        self.height, self.width = frame.shape[:2]
        scale = min(1, (self.window.winfo_screenwidth()-120)/(2*self.width),
                    (self.window.winfo_screenheight()-600)/self.height)
        self.dw, self.dh = max(1, int(self.width*scale)), max(1, int(self.height*scale))
        self.picture = self.ImageTk.PhotoImage(self.Image.fromarray(
            self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)).resize((self.dw, self.dh)))
        self.canvas.config(width=self.dw, height=self.dh)
        self.last_picture = self.ImageTk.PhotoImage(self.Image.fromarray(
            self.cv2.cvtColor(last_frame, self.cv2.COLOR_BGR2RGB)).resize((self.dw, self.dh)))
        self.last_canvas.config(width=self.dw, height=self.dh)
        try:
            saved_seed = seed_index(self.rows, start, end, self.args.points)
        except ValueError:
            saved_seed = None  # Let the user replace conflicting endpoint annotations.
        self.seed_side.set('last' if saved_seed == end-1 and saved_seed != start else 'first')
        self.load_seed_points()
        self.redraw()

    def load_seed_points(self):
        start, end = self.segments[self.stage]
        row = self.rows[start if self.seed_side.get() == 'first' else end-1]
        self.points = [(float(row[f'point_{i}_x']), float(row[f'point_{i}_y']))
                       for i in range(1, self.args.points+1)
                       if row.get('point_label_mode') == 'manual'
                       and row.get(f'point_{i}_x', '') != '' and row.get(f'point_{i}_y', '') != '']

    def select_seed(self):
        # Switching views discards pending clicks; points from different frames never mix.
        self.load_seed_points()
        self.redraw()

    def redraw(self):
        active_first = self.seed_side.get() == 'first'
        for canvas, picture in ((self.canvas, self.picture), (self.last_canvas, self.last_picture)):
            canvas.delete('all')
            canvas.create_image(0, 0, image=picture, anchor='nw')
        active = self.canvas if active_first else self.last_canvas
        for i, (x, y) in enumerate([] if self.skip_stage.get() else self.points, 1):
            x, y = x*self.dw/self.width, y*self.dh/self.height
            active.create_oval(x-4, y-4, x+4, y+4, fill='red', outline='white')
            active.create_text(x+8, y-8, text=str(i), fill='yellow', font='TkDefaultFont')
        start, end = self.segments[self.stage]
        for caption, name, frame, selected in (
                (self.first_caption, 'First', self.frames[start], active_first),
                (self.last_caption, 'Last', self.frames[end-1], not active_first)):
            caption.config(text=f'{name} frame: {frame}\n' + (
                'SKIP SELECTED — NO LABELS NEEDED' if self.skip_stage.get() else (
                    'LABEL POINTS HERE' if selected else 'REFERENCE ONLY — SELECT ABOVE TO LABEL')),
                bg='#775500' if self.skip_stage.get() else ('#146b26' if selected else '#b00020'), fg='white',
                wraplength=self.dw)
        self.status.config(text=f'{self.record+1}/{len(self.items)}: {self.csv_path.stem}\n'
                           f'Stage {self.stage+1}/{len(self.segments)}: {stage_name(self.rows[start])} | ' +
                           ('STAGE TYPE SKIPPED — press Right / Enter to continue' if self.skip_stage.get() else
                            f'Labeling {self.seed_side.get()} frame | {len(self.points)}/{self.args.points} points (minimum 1)'))

    def add(self, event, side='first'):
        if self.skip_stage.get() or side != self.seed_side.get():
            return
        if len(self.points) < self.args.points:
            self.points.append((min(self.width-1, max(0, round(event.x*self.width/self.dw))),
                                min(self.height-1, max(0, round(event.y*self.height/self.dh)))))
            self.redraw()

    def undo(self, event=None):
        if self.points:
            self.points.pop()
            self.redraw()

    def clear(self, event=None):
        self.points = []
        self.redraw()

    def previous(self, event=None):
        if self.stage > 0:
            self.stage -= 1
            self.show()

    def advance(self, event=None):
        if not self.skip_stage.get() and not 1 <= len(self.points) <= self.args.points:
            self.messagebox.showwarning('Select points', f'Select 1 to {self.args.points} points first.')
            return
        start, end = self.segments[self.stage]
        self.status.config(text='Saving manual target points…')
        self.window.update_idletasks()
        try:
            candidate = skip_selected_stages(self.rows, self.segments, self.selected_skip_types())
            if self.skip_stage.get():
                set_stage_skipped(candidate, start, end)
            else:
                set_manual_seed(candidate, start, end, self.points,
                                start if self.seed_side.get() == 'first' else end-1)
            save_csv(self.csv_path, self.fields, candidate)
            self.rows = candidate
        except Exception as exc:
            self.messagebox.showerror('Stage not saved', str(exc))
            self.redraw()
            return
        self.stage += 1
        while self.stage < len(self.segments):
            start, end = self.segments[self.stage]
            if not is_skipped(self.rows, start, end):
                break
            self.stage += 1
        if self.stage == len(self.segments):
            self.next_record()
        else:
            self.show()


class Labeler:
    """Frame-by-frame review of saved predictions; edits remain pending until Save."""

    def __init__(self, args, items):
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageTk
        import cv2
        self.tk, self.messagebox = tk, messagebox
        self.Image, self.ImageTk, self.cv2 = Image, ImageTk, cv2
        self.args, self.items = args, items
        self.cap = None
        self.record = -1
        self.dirty = False
        self.playing = False
        self.timer = None
        self.dragging = False
        self.window = tk.Tk()
        self.window.title('DVSS tracking review')
        self.status = tk.Label(self.window, justify='left')
        self.status.pack(fill='x')
        self.canvas = tk.Canvas(self.window, highlightthickness=0, bg='black')
        self.canvas.pack()
        self.slider = tk.Scale(self.window, orient='horizontal', showvalue=True,
                               command=self.scrub)
        self.slider.pack(fill='x')
        controls = tk.Frame(self.window)
        controls.pack()
        for text, command in [('<', lambda: self.move(-1)), ('Play / Pause', self.toggle_play),
                              ('>', lambda: self.move(1)), ('J: -60', lambda: self.move(-60)),
                              ('L: +60', lambda: self.move(60)), ('Save', self.save),
                              ('Previous video', lambda: self.open_record(self.record-1)),
                              ('Next video', lambda: self.open_record(self.record+1))]:
            tk.Button(controls, text=text, command=command).pack(side='left')
        points = tk.Frame(self.window)
        points.pack()
        self.selected = tk.IntVar(value=1)
        tk.Label(points, text='Edit point:').pack(side='left')
        for i in range(1, args.points+1):
            tk.Radiobutton(points, text=str(i), variable=self.selected, value=i).pack(side='left')
        tk.Button(points, text='Delete selected point', command=self.delete_point).pack(side='left')
        tk.Button(points, text='Clear frame', command=self.clear_frame).pack(side='left')
        ranges = tk.Frame(self.window)
        ranges.pack()
        tk.Button(ranges, text='[ Set start', command=self.mark_start).pack(side='left')
        tk.Button(ranges, text='] Set end', command=self.mark_end).pack(side='left')
        tk.Button(ranges, text='Clear ALL points in range', command=self.clear_range).pack(side='left')
        tk.Label(self.window, text='Space: play/pause | < / > (or , / .): one frame | J / L: 60 frames\n'
                 '[ / ]: inclusive range | Click: place selected point | Drag a point: move | Delete: remove selected\n'
                 'Ctrl+S: save | Edits affect the current frame only; Save writes pending changes.').pack()
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'dragging', False))
        self.window.bind('<space>', self.toggle_play)
        for key, delta in [('comma', -1), ('less', -1), ('period', 1), ('greater', 1),
                           ('j', -60), ('J', -60), ('l', 60), ('L', 60)]:
            self.window.bind(f'<{key}>', lambda e, d=delta: self.move(d))
        self.window.bind('<bracketleft>', self.mark_start)
        self.window.bind('<bracketright>', self.mark_end)
        self.window.bind('<Delete>', self.delete_point)
        self.window.bind('<Control-s>', self.save)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.open_record(0)

    def pause(self):
        self.playing = False
        if self.timer is not None:
            self.window.after_cancel(self.timer)
            self.timer = None

    def confirm_leave(self):
        self.pause()
        if not self.dirty:
            return True
        answer = self.messagebox.askyesnocancel('Unsaved edits', 'Save changes before leaving this video?')
        if answer is None:
            return False
        return self.save() if answer else True

    def open_record(self, index):
        if not 0 <= index < len(self.items) or not self.confirm_leave():
            return
        if self.cap is not None:
            self.cap.release()
        video, self.csv_path = record_paths(self.items[index], self.args.data_root)
        self.fields, self.rows = read_allocation(self.csv_path)
        ensure_columns(self.fields, self.args.points)
        self.cap = self.cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        self.count = int(self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        frames = frame_indices(self.rows, self.count, self.args.frame_offset)
        self.offset = int(self.rows[0]['Frame_Number']) if self.args.frame_offset is None else self.args.frame_offset
        self.by_frame = dict(zip(frames, self.rows))
        fps = self.cap.get(self.cv2.CAP_PROP_FPS)
        self.delay = max(1, round(1000/fps)) if math.isfinite(fps) and fps > 0 else 33
        self.record, self.dirty = index, False
        self.range_start = self.range_end = None
        self.slider.config(to=self.count-1)
        self.frame = -1
        self.show_frame(0)

    def show_frame(self, frame):
        frame = min(self.count-1, max(0, int(frame)))
        if frame != self.frame:
            # Sequential playback avoids an expensive decoder seek on every frame.
            if frame != self.frame+1:
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = self.cap.read()
            if not ok:
                self.pause()
                self.messagebox.showerror('Video error', f'Cannot decode frame {frame}')
                return
            self.frame = frame
            self.height, self.width = image.shape[:2]
            scale = min(1, (self.window.winfo_screenwidth()-100)/self.width,
                        max(120, self.window.winfo_screenheight()-360)/self.height)
            self.dw, self.dh = max(1, int(self.width*scale)), max(1, int(self.height*scale))
            rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            self.picture = self.ImageTk.PhotoImage(self.Image.fromarray(rgb).resize((self.dw, self.dh)))
            self.canvas.config(width=self.dw, height=self.dh)
            self.slider.set(frame)
        self.redraw()

    def redraw(self):
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.picture, anchor='nw')
        row = self.by_frame.get(self.frame, {})
        for i in range(1, self.args.points+1):
            try:
                x, y = float(row.get(f'point_{i}_x', '')), float(row.get(f'point_{i}_y', ''))
            except (ValueError, TypeError):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            x, y = x*self.dw/self.width, y*self.dh/self.height
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', outline='white', width=2)
            self.canvas.create_text(x+10, y-10, text=str(i), fill='yellow', anchor='sw')
        self.status.config(text=f'{self.record+1}/{len(self.items)}: {self.csv_path.stem} | '
                           f'Frame {self.frame}/{self.count-1} | {row.get("stage", row.get("event", "No allocation row"))} | '
                           f'{row.get("point_label_mode", "")} | {"Playing" if self.playing else "Paused"}\n'
                           f'Selected range (inclusive): {self.range_start} to {self.range_end}'
                           f' | {"UNSAVED EDITS" if self.dirty else "Saved"}')

    def scrub(self, value):
        if hasattr(self, 'frame') and int(float(value)) != self.frame:
            self.pause()
            self.show_frame(int(float(value)))

    def move(self, delta):
        self.pause()
        self.show_frame(self.frame+delta)
        return 'break'

    def toggle_play(self, event=None):
        if self.playing:
            self.pause()
            self.redraw()
        else:
            self.playing = True
            self.tick()
        return 'break'

    def tick(self):
        self.timer = None
        if not self.playing:
            return
        if self.frame >= self.count-1:
            self.pause()
            self.redraw()
            return
        self.show_frame(self.frame+1)
        if self.playing:
            self.timer = self.window.after(self.delay, self.tick)

    def editable_row(self):
        if self.frame not in self.by_frame:
            row = {field: '' for field in self.fields}
            row['Frame_Number'] = str(self.frame+self.offset)
            self.by_frame[self.frame] = row
        return self.by_frame[self.frame]

    def place(self, event):
        row = self.editable_row()
        i = self.selected.get()
        row[f'point_{i}_x'] = float(min(self.width-1, max(0, event.x*self.width/self.dw)))
        row[f'point_{i}_y'] = float(min(self.height-1, max(0, event.y*self.height/self.dh)))
        row['point_label_mode'] = 'edited'
        self.dirty = True
        self.redraw()

    def click(self, event):
        self.pause()
        row = self.by_frame.get(self.frame, {})
        nearest, distance = None, 15**2
        for i in range(1, self.args.points+1):
            try:
                x, y = float(row.get(f'point_{i}_x', '')), float(row.get(f'point_{i}_y', ''))
            except (ValueError, TypeError):
                continue
            d = (event.x-x*self.dw/self.width)**2+(event.y-y*self.dh/self.height)**2
            if d < distance:
                nearest, distance = i, d
        if nearest is not None:
            self.selected.set(nearest)
        self.dragging = True
        self.place(event)

    def drag(self, event):
        if self.dragging:
            self.place(event)

    def delete_point(self, event=None):
        self.pause()
        row = self.editable_row()
        for axis in ('x', 'y'):
            row[f'point_{self.selected.get()}_{axis}'] = ''
        row['point_label_mode'] = 'edited'
        self.dirty = True
        self.redraw()

    def clear_frame(self):
        self.pause()
        self.clear_rows([self.editable_row()])

    def clear_rows(self, rows):
        for row in rows:
            # Clear all point columns, even when --points displays fewer slots.
            for key in row:
                if re.fullmatch(r'point_\d+_(x|y|visible)', key):
                    row[key] = ''
            row['point_label_mode'] = 'cleared'
        self.dirty = True
        self.redraw()

    def mark_start(self, event=None):
        self.pause()
        self.range_start = self.frame
        self.redraw()

    def mark_end(self, event=None):
        self.pause()
        self.range_end = self.frame
        self.redraw()

    def clear_range(self):
        self.pause()
        if self.range_start is None or self.range_end is None:
            self.messagebox.showwarning('Select a range', 'Use [ and ] to set both range endpoints first.')
            return
        start, end = sorted((self.range_start, self.range_end))
        self.clear_rows([row for frame, row in self.by_frame.items() if start <= frame <= end])

    def save(self, event=None):
        self.pause()
        try:
            rows = [row for frame, row in sorted(self.by_frame.items())]
            save_csv(self.csv_path, self.fields, rows)
        except Exception as exc:
            self.messagebox.showerror('Save failed', str(exc))
            return False
        self.rows, self.dirty = rows, False
        self.redraw()
        return True

    def close(self):
        if self.confirm_leave():
            if self.cap is not None:
                self.cap.release()
            self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--seed-mode', action='store_true', help='Use the original first/last-frame stage seed editor')
    parser.add_argument('--list', action='store_true', help='Print local video paths without opening UI')
    args = parser.parse_args()
    if not 1 <= args.points <= 5:
        parser.error('--points must be between 1 and 5')
    items = load_items(args.items)
    if not items:
        parser.error('No active recordings found in the supplied JSON lists')
    if args.list:
        for item in items:
            print(record_paths(item, args.data_root)[0])
        return
    items, unavailable = labelable_items(items, args.data_root)
    if unavailable:
        print(f'Excluded {len(unavailable)} recordings with missing or invalid stage allocations:', file=sys.stderr)
        for reason in unavailable:
            print(f'  {reason}', file=sys.stderr)
        print('These recordings require event stage annotations before labeling.', file=sys.stderr)
    if not items:
        parser.exit(1, 'No recordings with valid stage allocations are available. No files changed.\n')
    print(f'Opening labeling queue: {len(items)} recordings ready; {len(unavailable)} excluded.', flush=True)
    app = (SeedLabeler if args.seed_mode else Labeler)(args, items)
    try:
        app.window.mainloop()
    finally:
        if app.cap is not None:
            app.cap.release()


if __name__ == '__main__':
    main()
