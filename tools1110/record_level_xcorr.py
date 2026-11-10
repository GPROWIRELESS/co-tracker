"""Compute record peaks from an already-matched NPY/DLC JSON path list.

Run prepare_xcorr_path_list.py separately first. No index/dataset discovery,
no plots. Output: two rows per record (left/right PSM), including failures.
Source paths in the CSV start at dvss (backslash-separated); the path list
keeps loadable absolute paths. All index meta fields become CSV columns.
Example: python record_level_xcorr.py --pairs matched_npy_dlc_paths.json
"""
import argparse
import json
from pathlib import Path, PureWindowsPath

from xcorr_common import DEFAULT_PAIRS, RECORD_FIELDS, compute_records, read_path_list, write_csv


def dataset_relative_path(path):
    """D:\\Dataset\\dvss\\Active_Data\\... -> dvss\\Active_Data\\..."""
    parts = PureWindowsPath(str(path)).parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "dvss":
            return str(PureWindowsPath(*parts[index:]))
    raise ValueError(f"Source path has no dvss dataset directory: {path}")


def format_record_output(entries, rows):
    """Format CSV only; do not modify computation rows or loadable paths.

    Metadata uses original field names, with task_name sharing the existing
    column. Any other name colliding with a result column uses meta_<name>.
    Nulls become blank CSV cells; any nested metadata is serialized as JSON.
    """
    by_id = {entry["record_id"]: entry for entry in entries}
    meta_columns = {}
    for entry in entries:
        meta = entry.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be an object: {entry['record_id']}")
        if "task_name" in meta and meta["task_name"] != entry["task_name"]:
            raise ValueError(f"Conflicting meta task_name: {entry['record_id']}")
        for key in meta:
            column = "meta_" + key if key in RECORD_FIELDS and key != "task_name" else key
            meta_columns[key] = column
    if len(set(meta_columns.values())) != len(meta_columns):
        raise ValueError("Metadata field names collide after result-column disambiguation")
    fields = RECORD_FIELDS + [column for column in meta_columns.values() if column not in RECORD_FIELDS]
    output = []
    for row in rows:
        formatted = row.copy()
        for key in ("gaze_npy_path", "dlc_csv_path"):
            formatted[key] = dataset_relative_path(row[key])
        for key, value in by_id[row["record_id"]].get("meta", {}).items():
            formatted[meta_columns[key]] = (json.dumps(value, ensure_ascii=False)
                                            if isinstance(value, (dict, list)) else value)
        output.append(formatted)
    return output, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "record_level_xcorr.csv")
    args = parser.parse_args()
    entries = read_path_list(args.pairs)
    rows, fields = format_record_output(entries, compute_records(entries))
    write_csv(args.output, rows, fields)
    print(f"Saved {len(rows)} side results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
