"""Separate index discovery and completeness audit, without correlation.

Writes a matched JSON path list and a CSV audit of ALL indexed records.
The path list preserves each record's complete index meta object.
Only existing, parseable, nonempty, equal-length NPY/DLC pairs enter the list.
Missing/unmatched records remain in the audit and are never silently truncated.
The computation scripts do not import or invoke this discovery script.
"""
import argparse
import json
from pathlib import Path

from xcorr_common import load_dlc, load_gaze, write_csv

DEFAULT_INDEX = Path(r"C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\dvss_pytorch\Index\index_novice_active.json")
DEFAULT_ROOT = Path(r"D:\Dataset\dvss")
HERE = Path(__file__).resolve().parent
AUDIT_FIELDS = ["record_id", "task_name", "gaze_npy_path", "dlc_csv_path",
                "gaze_rows", "dlc_rows", "status", "error"]


def prepare(index, root):
    root = Path(root).resolve()
    clips = json.loads(Path(index).read_text(encoding="utf-8-sig"))["clips"]
    matched, audit = [], []
    seen = set()
    for key, clip in clips.items():
        meta = clip["meta"]
        cid = meta["content_id"]
        if cid in seen:
            raise ValueError(f"Duplicate record_id in index: {cid}")
        seen.add(cid)
        gaze_path = (root / clip["gaze_npy_path"]).resolve()
        dlc_path = gaze_path.parent.parent / "sam" / cid / f"{cid}_DLC.csv"
        row = dict(record_id=cid, task_name=meta["task_name"], gaze_npy_path=str(gaze_path),
                   dlc_csv_path=str(dlc_path), gaze_rows="", dlc_rows="", status="error", error="")
        errors = []
        for name, path, loader in (("gaze", gaze_path, load_gaze), ("dlc", dlc_path, load_dlc)):
            try:
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                loaded = loader(path)
                row[f"{name}_rows"] = len(loaded if name == "gaze" else loaded[0])
            except (OSError, ValueError, TypeError, EOFError) as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if not errors:
            if row["gaze_rows"] != row["dlc_rows"]:
                errors.append(f"Row mismatch: gaze={row['gaze_rows']}, DLC={row['dlc_rows']}")
            elif row["gaze_rows"] == 0:
                errors.append("Empty record")
        if errors:
            row["error"] = " | ".join(errors)
        else:
            row["status"] = "matched"
            matched.append({k: row[k] for k in ("record_id", "task_name", "gaze_npy_path", "dlc_csv_path")})
            matched[-1]["n_rows"] = row["gaze_rows"]
            matched[-1]["meta"] = meta.copy()
        audit.append(row)
    return matched, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=HERE / "matched_npy_dlc_paths.json")
    parser.add_argument("--audit-output", type=Path, default=HERE / "xcorr_path_audit.csv")
    args = parser.parse_args()
    matched, audit = prepare(args.index, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.audit_output, audit, AUDIT_FIELDS)
    print(f"Indexed={len(audit)}, matched={len(matched)}, excluded={len(audit)-len(matched)}")
    print(f"Path list: {args.output.resolve()}\nAudit: {args.audit_output.resolve()}")


if __name__ == "__main__":
    main()
