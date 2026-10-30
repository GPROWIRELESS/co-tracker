#!/usr/bin/env python3
"""Track manual target-locking seeds through their stage and following skipped stages."""
import argparse
import csv
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
    return tracks, visibility


def is_tracked(rows, start, end, count):
    seed = seed_index(rows, start, end, count)
    if seed is None:
        return False
    return all(
        row.get('point_label_mode') == 'tracked'
        for index, row in enumerate(rows[start:end], start)
        if index != seed)


def propagate_stage(rows, frames, start, end, count, model, cap, device, tracking_end=None):
    """Return updated rows, preserving the exact human seed and other stages."""
    if is_skipped(rows, start, end):
        return [dict(row) for row in rows]
    seed = seed_index(rows, start, end, count)
    if seed is None:
        raise ValueError('Missing manual endpoint seed')
    points = seed_points(rows[seed], count)
    candidate = [dict(row) for row in rows]
    seed_end = end
    end = tracking_end if tracking_end is not None else end
    if end == start+1:
        return candidate
    reverse = seed != start
    if reverse and end > seed_end:
        candidate = propagate_stage(rows, frames, start, seed_end, count, model, cap, device)
        # Continue forward from the same manual endpoint, preserving backward stage results.
        return propagate_stage(candidate, frames, seed, seed+1, count, model, cap, device,
                               tracking_end=end)
    tracks, _visibility = track_stage(model, cap, frames[start], frames[end-1], points, device, reverse=reverse)
    for index in range(start, end):
        if index == seed:
            continue
        offset = frames[end-1] - frames[index] if reverse else frames[index] - frames[start]
        candidate[index]['point_label_mode'] = 'tracked' if index < seed_end else 'tracked_inherited'
        for i in range(1, len(points)+1):
            # Save raw coordinates for every prediction, regardless of visibility or bounds.
            x, y = tracks[offset][i-1]
            candidate[index].update({f'point_{i}_x': float(x), f'point_{i}_y': float(y)})
    return candidate


def is_continuation(rows, start, end):
    return all(row.get('point_label_mode') in {'skipped', 'tracked_inherited'}
               for row in rows[start:end])


def continuation_groups(rows, segments):
    """Attach unseeded skipped stages to the preceding manually seeded stage."""
    groups = []
    for start, end in segments:
        if is_continuation(rows, start, end):
            if groups:
                groups[-1][2] = end
            else:
                print(f'No preceding seed for initial skipped stage at row {start}; leaving unchanged.', flush=True)
        else:
            groups.append([start, end, end])
    return groups


def load_tracking_model(checkpoint, device, window_len):
    from cotracker.predictor import CoTrackerOnlinePredictor
    # Load the native 16-frame checkpoint before changing the inference window.
    # Its saved time embedding is interpolated by the model during forward().
    model = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).to(device).eval()
    model.model.window_len = window_len
    model.step = window_len // 2
    return model


def run_jobs(jobs, args):
    """Process whole recordings sequentially on one device."""
    import cv2
    model = None
    for video, allocation, fields, rows, frames, segments in jobs:
        ensure_columns(fields, args.points)
        cap = cv2.VideoCapture(str(video))
        try:
            groups = continuation_groups(rows, segments)
            for start, end, tracking_end in groups:
                inherited_done = all(row.get('point_label_mode') == 'tracked_inherited'
                                     for row in rows[end:tracking_end])
                if not args.overwrite and is_tracked(rows, start, end, args.points) and inherited_done:
                    continue
                if model is None and tracking_end > start+1:
                    model = load_tracking_model(args.checkpoint, args.device, args.window_len)
                print(f'[{args.device}] {video.stem}: {stage_name(rows[start])}, frames {frames[start]}–{frames[tracking_end-1]}', flush=True)
                candidate = propagate_stage(rows, frames, start, end, args.points, model, cap, args.device, tracking_end=tracking_end)
                save_csv(allocation, fields, candidate)
                rows = candidate
            if any(field.startswith('point_') and field.endswith('_visible') for field in fields):
                save_csv(allocation, fields, rows)
        finally:
            cap.release()


def gpu_worker(rank, partitions, devices, args):
    import torch
    args.device = devices[rank]
    torch.cuda.set_device(args.device)
    run_jobs(partitions[rank], args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/scaled_online.pth')
    parser.add_argument('--device', default='cuda', help='Device for single-worker execution')
    parser.add_argument('--devices', nargs='+', help='One worker per GPU, e.g. cuda:0 cuda:1; overrides --device')
    parser.add_argument('--window-len', type=int, default=24,
                        help='Even inference window size (default: 24; original: 16); overlap is half')
    parser.add_argument('--overwrite', action='store_true', help='Recompute already tracked stages')
    parser.add_argument('--check-only', action='store_true', help='Validate all inputs and seeds without inference or writes')
    args = parser.parse_args()
    if not 1 <= args.points <= 5:
        parser.error('--points must be between 1 and 5')
    if args.window_len < 2 or args.window_len % 2:
        parser.error('--window-len must be an even integer of at least 2')
    print(f'Window: {args.window_len} frames, step: {args.window_len // 2}', flush=True)
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
                if is_continuation(rows, start, end):
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
    if args.devices:
        import torch
        devices = []
        for value in args.devices:
            try:
                device = torch.device(value)
            except (RuntimeError, ValueError) as exc:
                parser.error(str(exc))
            if device.type != 'cuda' or device.index is None:
                parser.error('--devices requires explicit CUDA devices, e.g. cuda:0 cuda:1')
            if device.index >= torch.cuda.device_count():
                parser.error(f'Unavailable GPU: {device}')
            devices.append(str(device))
        if len(set(devices)) != len(devices):
            parser.error('--devices must not contain duplicates')
        # Keep each CSV in exactly one worker, even if multiple input items alias it.
        allocations = [job[1].resolve() for job in jobs]
        if len(set(allocations)) != len(allocations):
            parser.error('Multiple recordings reference the same allocation CSV; remove duplicates')
        devices = devices[:len(jobs)]
        partitions = [jobs[index::len(devices)] for index in range(len(devices))]
        print('Recordings per GPU: ' + ', '.join(
            f'{device}: {len(part)}' for device, part in zip(devices, partitions)), flush=True)
        # Spawn independent processes: CUDA state must never be inherited with fork.
        torch.multiprocessing.spawn(
            gpu_worker, args=(partitions, devices, args), nprocs=len(devices), join=True)
    else:
        run_jobs(jobs, args)
    print('Tracking complete.')


if __name__ == '__main__':
    main()
