#!/usr/bin/env python3
"""Propagate manually annotated target-locking seeds within each event stage."""
import argparse
import csv
import math
from pathlib import Path

from label_target_locking import (
    ROOT, add_data_arguments, ensure_columns, frame_indices, stage_name,
    load_items, record_paths, save_csv, stages, seed_index, seed_points, is_skipped, read_allocation,
)


def track_stage(model, cap, first, last, points, device, reverse=False):
    import cv2
    import numpy as np
    import torch
    queries = torch.tensor([[[0, x, y] for x, y in points]], dtype=torch.float32, device=device)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    buffer = []
    cursor = last if reverse else first
    tracks = visibility = None
    with torch.inference_mode():
        while True:
            if reverse:
                length = min(2 * model.step - len(buffer), cursor-first+1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cursor-length+1)
                block = []
                for index in range(length):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f'Cannot decode frame {cursor-length+1+index}')
                    block.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buffer.extend(reversed(block))
                cursor -= length
            else:
                while len(buffer) < 2 * model.step and cursor <= last:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f'Cannot decode frame {cursor}')
                    buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cursor += 1
            chunk = torch.from_numpy(np.stack(buffer)).permute(0, 3, 1, 2)[None].to(
                device=device, dtype=torch.float32)
            if tracks is None:
                model(chunk[:, :1], is_first_step=True, queries=queries, add_support_grid=True)
            tracks, visibility = model(chunk, add_support_grid=True)
            if (reverse and cursor < first) or (not reverse and cursor > last):
                break
            buffer = buffer[model.step:]
    tracks = tracks[0, :last-first+1].cpu().numpy()
    visibility = visibility[0, :last-first+1].cpu().numpy()
    height, width = buffer[0].shape[:2]
    visibility &= (np.isfinite(tracks).all(axis=-1)
                   & (tracks[..., 0] >= 0) & (tracks[..., 0] < width)
                   & (tracks[..., 1] >= 0) & (tracks[..., 1] < height))
    return tracks, visibility


def is_tracked(rows, start, end, count):
    seed = seed_index(rows, start, end, count)
    if seed is None:
        return False
    return all(
        row.get('point_label_mode') == 'tracked'
        for index, row in enumerate(rows[start:end], start)
        if index != seed)


def propagate_stage(rows, frames, start, end, count, model, cap, device):
    """Return updated rows, preserving the exact human seed and other stages."""
    if is_skipped(rows, start, end):
        return [dict(row) for row in rows]
    seed = seed_index(rows, start, end, count)
    if seed is None:
        raise ValueError('Missing manual endpoint seed')
    points = seed_points(rows[seed], count)
    candidate = [dict(row) for row in rows]
    if end == start+1:
        return candidate
    reverse = seed != start
    tracks, visibility = track_stage(model, cap, frames[start], frames[end-1], points, device, reverse=reverse)
    for index in range(start, end):
        if index == seed:
            continue
        offset = frames[end-1] - frames[index] if reverse else frames[index] - frames[start]
        candidate[index]['point_label_mode'] = 'tracked'
        for i in range(1, len(points)+1):
            if not visibility[offset][i-1]:
                candidate[index].update({f'point_{i}_x': '', f'point_{i}_y': ''})
                continue
            x, y = tracks[offset][i-1]
            if not math.isfinite(float(x)) or not math.isfinite(float(y)):
                raise ValueError(f'Nonfinite prediction at allocation row {index}')
            candidate[index].update({f'point_{i}_x': float(x), f'point_{i}_y': float(y)})
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/scaled_online.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overwrite', action='store_true', help='Recompute already tracked stages')
    parser.add_argument('--check-only', action='store_true', help='Validate all inputs and seeds without inference or writes')
    args = parser.parse_args()
    if not 1 <= args.points <= 5:
        parser.error('--points must be between 1 and 5')
    items = load_items(args.items)
    if not items:
        parser.error('No active recordings found in the supplied JSON lists')
    import cv2
    jobs = []
    errors = []
    # Verify the whole annotation queue before loading the model or writing data.
    for item in items:
        try:
            video, allocation = record_paths(item, args.data_root)
            fields, rows = read_allocation(allocation)
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():
                    raise ValueError(f'Cannot open {video}')
                frames = frame_indices(rows, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), args.frame_offset)
                width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            finally:
                cap.release()
            segments = stages(rows, args.include_none)
            for start, end in segments:
                if is_skipped(rows, start, end):
                    continue
                seed = seed_index(rows, start, end, args.points)
                if seed is None:
                    errors.append(f'{allocation}: missing manual endpoint seed for stage at frame {frames[start]}')
                    continue
                for i, (x, y) in enumerate(seed_points(rows[seed], args.points), 1):
                    if not (0 <= x < width and 0 <= y < height):
                        errors.append(f'{allocation}: point {i} outside image at frame {frames[start]}')
            jobs.append((video, allocation, fields, rows, frames, segments))
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        parser.exit(1, '\n'.join(errors) + '\nFinish annotation or correct inputs before tracking. No files changed.\n')
    print(f'Validated {len(jobs)} recordings, {sum(len(j[-1]) for j in jobs)} stages.', flush=True)
    if args.check_only:
        return
    import torch
    from cotracker.predictor import CoTrackerOnlinePredictor
    model = None
    for video, allocation, fields, rows, frames, segments in jobs:
        ensure_columns(fields, args.points)
        cap = cv2.VideoCapture(str(video))
        try:
            for start, end in segments:
                if is_skipped(rows, start, end):
                    continue
                if not args.overwrite and is_tracked(rows, start, end, args.points):
                    continue
                if model is None and end > start+1:
                    model = CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).to(args.device).eval()
                print(f'{video.stem}: {stage_name(rows[start])}, frames {frames[start]}–{frames[end-1]}', flush=True)
                candidate = propagate_stage(rows, frames, start, end, args.points, model, cap, args.device)
                save_csv(allocation, fields, candidate)
                rows = candidate
            if any(field.startswith('point_') and field.endswith('_visible') for field in fields):
                save_csv(allocation, fields, rows)
        finally:
            cap.release()
    print('Tracking complete.')


if __name__ == '__main__':
    main()
