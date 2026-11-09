"""Record the user's primary row mapping and reverify source evidence hashes."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
EXACT_INSTRUCTION = "直接按行数对应就可以"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ledger = read_json("input_ledger.json")
    candidates = [r for r in ledger if r["candidate"]]
    checks = []
    for r in candidates:
        lengths = {k: int(r[k]) for k in (
            "passive_npy_rows", "retained_timeline_rows", "source_video_frames",
            "passive_video_frames", "dlc_rows")}
        equal_counts = len(set(lengths.values())) == 1
        equal_keys = all(r[k] for k in (
            "source_frame_ids_equal_passive", "source_timeline_frames_equal_passive",
            "source_precise_video_ts_ns_equal_passive", "dlc_indices_consecutive",
            "dlc_matches_retained_timeline"))
        if not (equal_counts and equal_keys):
            raise ValueError(f"Primary row mapping audit failed for {r['record_id']}")
        checks.append({"record_id": r["record_id"], "row_counts": lengths,
                       "equal_counts": equal_counts, "frame_and_timeline_key_checks": equal_keys})
        r["explicit_user_authorised_row_mapping"] = True
        r["primary_association"] = "native passive NPY row i -> passive video frame i -> source video frame i -> source DLC row i"
        r["primary_clock"] = "unchanged precise source video_ts matched by original Frame_Number"
        r["raw_vid_frame_role"] = "diagnostic only under explicit user instruction; no exclusion or remapping"
        r["playback_alignment_status"] = "primary row association explicitly authorised by user after equal counts and original key checks; raw VID_FRAME differences retained as diagnostics"
    write_json("input_ledger.json", ledger)
    fields = list(dict.fromkeys(k for r in ledger for k in r))
    with (OUT / "input_ledger.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in ledger:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in r.items()})
    authorisation = {
        "explicit_user_authorised_row_mapping": True,
        "evidence_type": "user_instruction",
        "exact_instruction": EXACT_INSTRUCTION,
        "primary_association": "native passive NPY row i <-> passive video frame i <-> source video frame i <-> source DLC row i",
        "clock": "precise source video_ts remains unchanged and is matched using original Frame_Number",
        "raw_vid_frame_role": "diagnostic only, with historical constructor provenance preserved",
        "exclude_for_raw_vid_frame_discrepancy": False,
        "remap_gaze_using_raw_vid_frame": False,
        "candidate_count": len(candidates),
        "row_and_key_checks": checks,
    }
    write_json("row_mapping_authorisation.json", authorisation)
    summary = read_json("audit_summary.json")
    summary["explicit_user_authorised_row_mapping"] = True
    summary["row_mapping_exact_instruction"] = EXACT_INSTRUCTION
    summary["primary_association"] = authorisation["primary_association"]
    summary["primary_clock"] = authorisation["clock"]
    summary["row_mapping_authorisation"] = str(OUT / "row_mapping_authorisation.json")
    summary["physical_playback_frame_alignment"] = "Primary analysis uses the user-authorised row association. Historical raw VID_FRAME discrepancies are retained as diagnostics and cause no exclusion or gaze remapping."
    summary["frame_alignment"] = "After verified equal counts and original key checks, NPY row i corresponds to passive/source video frame i and source DLC row i. Precise video_ts lookup by original Frame_Number is unchanged."
    write_json("audit_summary.json", summary)
    constructor = read_json("constructor_provenance.json")
    constructor.pop("deterministic_repair_possibility", None)
    constructor["primary_user_instruction"] = EXACT_INSTRUCTION
    constructor["primary_association_policy"] = authorisation["primary_association"]
    constructor["raw_mapping_artifacts_role"] = "historical diagnostics only, no exclusion or remapping in primary analysis"
    write_json("constructor_provenance.json", constructor)
    playback = read_json("raw_playback_audit.json")
    for r in playback:
        r["primary_user_instruction"] = EXACT_INSTRUCTION
        r["raw_vid_frame_role"] = "diagnostic only; primary row correspondence authorised by user"
    write_json("raw_playback_audit.json", playback)
    before = read_json("raw_and_task_evidence_hashes.json")
    after = {name: sha(name) for name in before}
    changed = [name for name in before if before[name] != after[name]]
    verification = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all raw passive gaze CSV and task/rename evidence referenced by input audit",
        "before_manifest_path": str(OUT / "raw_and_task_evidence_hashes.json"),
        "before_combined_sha256": canonical_digest(before),
        "after_combined_sha256": canonical_digest(after),
        "source_file_count": len(before),
        "all_source_bytes_unchanged": not changed,
        "changed_paths": changed,
        "before_sha256": before,
        "after_sha256": after,
    }
    write_json("source_unchanged_verification.json", verification)
    if changed:
        raise ValueError(f"Source evidence changed: {changed}")
    summary["raw_and_task_source_unchanged_verification"] = str(OUT / "source_unchanged_verification.json")
    summary["raw_and_task_source_bytes_unchanged"] = True
    summary["raw_and_task_evidence_combined_sha256"] = verification["after_combined_sha256"]
    write_json("audit_summary.json", summary)
    print(json.dumps({"candidate_count": len(candidates),
                      "explicit_user_authorised_row_mapping": True,
                      "source_file_count": len(before),
                      "all_source_bytes_unchanged": True,
                      "combined_evidence_sha256": verification["after_combined_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
