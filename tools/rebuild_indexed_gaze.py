#!/usr/bin/env python3
"""Rebuild only indexed gaze NPY files from sibling CSVs; requires NumPy."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INDEXES = [REPO / 'Index/items' / name for name in
                   ('novice_active_taskA', 'novice_active_taskC')]


def references(value):
    if isinstance(value, dict):
        if 'gaze_npy_path' in value:
            yield value['gaze_npy_path']
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def selected_paths(index_dirs, root):
    selected = set()
    for directory in index_dirs:
        files = sorted(directory.rglob('*.json'))
        if not files:
            raise ValueError(f'No JSON files in {directory}')
        for file in files:
            for reference in references(json.loads(file.read_text(encoding='utf-8-sig'))):
                if not isinstance(reference, str) or not reference.startswith('/dvss/'):
                    raise ValueError(f'{file}: unsupported gaze path {reference!r}')
                relative = PurePosixPath(reference[len('/dvss/'):])
                if '..' in relative.parts or relative.suffix != '.npy':
                    raise ValueError(f'{file}: unsafe gaze path {reference!r}')
                path = (root / Path(*relative.parts)).resolve()
                path.relative_to(root)
                selected.add(path)
    if not selected:
        raise ValueError('No gaze_npy_path entries found')
    return sorted(selected)


def read_csv(csv_path, dtype):
    """Read by field name, retaining NPY schema and integer timestamp precision."""
    if not dtype.names or dtype.hasobject:
        raise ValueError(f'{csv_path}: expected a numeric structured NPY dtype')
    for name in dtype.names:
        if dtype[name].kind not in 'iuf':
            raise ValueError(f'{csv_path}: unsupported field type {name}: {dtype[name]}')
    rows = []
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f'{csv_path}: missing or duplicate CSV headers')
        missing = set(dtype.names) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'{csv_path}: missing columns {sorted(missing)}')
        for line, row in enumerate(reader, 2):
            values = []
            for name in dtype.names:
                field = dtype[name]
                try:
                    if field.kind in 'iu':
                        decimal = Decimal(row[name])
                        if not decimal.is_finite() or decimal != decimal.to_integral_value():
                            raise ValueError('expected an integer')
                        value = int(decimal)
                        limits = np.iinfo(field)
                        if not limits.min <= value <= limits.max:
                            raise ValueError('integer out of range')
                    else:
                        # Empty gaze coordinates represent missing measurements.
                        value = float(row[name]) if row[name] and row[name].strip() else float('nan')
                        if np.isinf(value) or (np.isfinite(value) and abs(value) > np.finfo(field).max):
                            raise ValueError('floating point value out of range')
                    values.append(value)
                except (ValueError, TypeError, InvalidOperation) as error:
                    raise ValueError(f'{csv_path}:{line}: invalid {name}={row.get(name)!r}') from error
            rows.append(tuple(values))
    if not rows:
        raise ValueError(f'{csv_path}: no data rows')
    return np.array(rows, dtype=dtype)


def replace_npy(path, array):
    # Replace atomically: no backup, and no half-written NPY if saving fails.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Local directory corresponding to /dvss, e.g. D:\\Dataset\\dvss')
    parser.add_argument('--index-dirs', type=Path, nargs='+', default=DEFAULT_INDEXES)
    parser.add_argument('--manifest', type=Path, default=REPO / 'gaze_upload_manifest.json')
    parser.add_argument('--dry-run', action='store_true', help='Validate every CSV, write nothing')
    args = parser.parse_args()
    root = args.data_root.resolve()
    paths = selected_paths(args.index_dirs, root)
    schemas = {}
    # Fully validate the batch before overwriting anything. Keep only schemas in memory.
    for path in paths:
        existing = np.load(path, mmap_mode='r', allow_pickle=False)
        if existing.ndim != 1:
            raise ValueError(f'{path}: expected a one-dimensional structured array')
        dtype = existing.dtype
        del existing
        array = read_csv(path.with_suffix('.csv'), dtype)
        schemas[path] = dtype
        print(f'{"WOULD REBUILD" if args.dry_run else "VALIDATED"} {path} rows={len(array)} dtype={dtype}')
    if args.dry_run:
        print(f'Dry run: {len(paths)} NPY + {len(paths)} CSV selected; no files written.')
        return
    # Remove an old success manifest before starting a new conversion batch.
    manifest = args.manifest.resolve()
    if manifest in paths or manifest in [p.with_suffix('.csv') for p in paths]:
        raise ValueError('Manifest must not overwrite a selected dataset file')
    manifest.unlink(missing_ok=True)
    for path in paths:
        replace_npy(path, read_csv(path.with_suffix('.csv'), schemas[path]))
        print(f'REBUILT {path}')
    files = [p.relative_to(root).as_posix() for path in paths
             for p in (path, path.with_suffix('.csv'))]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'data_root': str(root), 'files': files}, indent=2) + '\n',
                        encoding='utf-8')
    print(f'Rebuilt {len(paths)} NPY files. Upload manifest ({len(files)} files): {manifest}')


if __name__ == '__main__':
    main()
