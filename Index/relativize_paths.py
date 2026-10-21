#!/usr/bin/env python3
"""Normalize dataset paths in Index JSON/CSV files. Run with --dry-run to preview.

Dataset references use the requested /dvss/... convention (a logical dataset
root, despite the leading slash). Index references are repository-relative.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import re


def normalize(value):
    if not isinstance(value, str):
        return value
    posix = value.replace('\\', '/')
    # Only rewrite path strings, leaving ordinary metadata untouched.
    if not (posix.startswith('/') or re.match(r'^[A-Za-z]:/', posix)):
        return value
    for directory in ('Active_Data', 'Passive_Data'):
        marker = '/' + directory + '/'
        if marker in posix:
            return '/dvss/' + directory + '/' + posix.split(marker, 1)[1]
    if '/dvss/' in posix:
        return '/dvss/' + posix.split('/dvss/', 1)[1]
    if '/Index/' in posix:
        return 'Index/' + posix.split('/Index/', 1)[1]
    if '/DVSS_Pytorch/' in posix:
        return posix.split('/DVSS_Pytorch/', 1)[1]
    return value


def transform(value):
    if isinstance(value, dict):
        return {key: transform(item) for key, item in value.items()}
    if isinstance(value, list):
        return [transform(item) for item in value]
    return normalize(value)


def rewrite(path, dry_run=False):
    original = path.read_bytes()
    bom = original.startswith(b'\xef\xbb\xbf')
    text = original.decode('utf-8-sig')
    if path.suffix.lower() == '.json':
        data = json.loads(text)
        updated = transform(data)
        if data == updated:
            return False
        output = json.dumps(updated, ensure_ascii=False, indent=2) + '\n'
    else:
        rows = list(csv.reader(io.StringIO(text, newline='')))
        updated = [[normalize(cell) for cell in row] for row in rows]
        if rows == updated:
            return False
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\r\n' if '\r\n' in text else '\n')
        writer.writerows(updated)
        output = stream.getvalue()
    if not dry_run:
        path.write_bytes(output.encode('utf-8-sig' if bom else 'utf-8'))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    paths = sorted(p for p in args.root.rglob('*') if p.suffix.lower() in {'.json', '.csv'})
    # Validate every input before changing any files.
    for path in paths:
        if path.suffix.lower() == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
        else:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                list(csv.reader(stream))
    changed = 0
    for path in paths:
        if rewrite(path, args.dry_run):
            changed += 1
            print(path.relative_to(args.root))
    print(f'{"Would update" if args.dry_run else "Updated"} {changed} of {len(paths)} JSON/CSV files.')


if __name__ == '__main__':
    main()
