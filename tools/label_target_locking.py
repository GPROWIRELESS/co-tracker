#!/usr/bin/env python3
"""Manually annotate target-locking points on the first or last frame of each event stage."""
import argparse
import csv
import json
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
                meta = item.get('meta', {})
                if (meta.get('src_skill', '').lower() == 'novice'
                        and meta.get('task_name', '').upper() == 'A'
                        and item.get('domain', 'active') == 'active'):
                    records[item['video_path']] = item
    return list(records.values())


def stages(rows, include_none=False):
    if any('event' not in row for row in rows):
        raise ValueError('Allocation CSV has no event column; stage annotations are required')
    result = []
    start = 0
    for end in range(1, len(rows) + 1):
        if end == len(rows) or rows[end]['event'] != rows[start]['event']:
            if include_none or rows[start]['event'].strip().lower() not in {'', 'none', 'nan'}:
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
    allocation = video.parent.parent / 'sam' / video.stem / (video.stem + '_allocation.csv')
    return video, allocation


def frame_indices(rows, count, offset=None):
    if not rows or not all('event' in r and 'Frame_Number' in r for r in rows):
        raise ValueError('Allocation requires rows with event and Frame_Number columns')
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
        if not {'event', 'Frame_Number'} <= set(fields):
            missing = sorted({'event', 'Frame_Number'} - set(fields))
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
        names.update(rows[start]['event'] for start, _ in stages(rows, include_none))
    return sorted(names)


def skip_selected_stages(rows, segments, selected):
    candidate = [dict(row) for row in rows]
    for start, end in segments:
        if rows[start]['event'] in selected:
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
        ROOT/'Index/items/novice_active_taskA'])
    parser.add_argument('--data-root', type=Path,
                        default=Path('/mnt/nas/Users/yxl3527/Dataset/dvss'),
                        help='Local dvss directory containing Active_Data')
    parser.add_argument('--points', type=int, default=5, help='Maximum points per stage (1–5); at least one is required')
    parser.add_argument('--frame-offset', type=int, help='Subtract from Frame_Number; default: first CSV frame')
    parser.add_argument('--include-none', action='store_true')


class Labeler:
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
                 'Right / Enter: save + next | Left: previous | Esc: quit (unsaved clicks discarded)',
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
        self.window.bind('<Escape>', lambda e: self.window.destroy())
        self.next_record()

    def paths(self, item):
        return record_paths(item, self.args.data_root)

    def next_record(self):
        if self.cap is not None:
            self.cap.release()
        self.record += 1
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
        while self.stage < len(self.segments) and self.complete(self.stage):
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

    def selected_skip_types(self):
        return {name for name, variable in self.skip_types.items() if variable.get()}

    def update_skip_selection(self):
        start, _ = self.segments[self.stage]
        self.skip_stage.set(self.rows[start]['event'] in self.selected_skip_types())
        self.redraw()

    def complete(self, index):
        start, end = self.segments[index]
        if is_skipped(self.rows, start, end):
            return True
        try:
            return seed_index(self.rows, start, end, self.args.points) is not None
        except ValueError:
            return False

    def show(self):
        start, end = self.segments[self.stage]
        self.skip_stage.set(self.rows[start]['event'] in self.selected_skip_types())
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
                           f'Stage {self.stage+1}/{len(self.segments)}: {self.rows[start]["event"]} | ' +
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--list', action='store_true', help='Print local video paths without opening UI')
    args = parser.parse_args()
    if not 1 <= args.points <= 5:
        parser.error('--points must be between 1 and 5')
    items = load_items(args.items)
    if not items:
        parser.error('No novice active task A recordings found in the supplied JSON lists')
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
    app = Labeler(args, items)
    try:
        app.window.mainloop()
    finally:
        if app.cap is not None:
            app.cap.release()


if __name__ == '__main__':
    main()
