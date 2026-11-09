"""Auditable, exploratory distal tool candidates from separate PSM masks.

The mask-derived point has NOT been validated as an anatomical jaw-tip. A
heuristic rejection is explicit and never falls back to a wrist in the same
track. Target coordinates are never changed. The main analysis may keep its
DLC wrist definition until independent distal-tip accuracy is established.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

VERSION = "component_pca_distal_v1"
THRESHOLD = 127
VALIDATION_STATUS = "failed_sample_visual_acceptance_not_a_validated_jawtip"


def _dump_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _root(data_root):
    p = Path(data_root)
    return p if p.name.lower() == "active_data" else p / "Active_Data"


def mask_paths(cid, data_root):
    p = _root(data_root) / cid.split("_")[0] / "sam" / cid / "mask"
    return {side: p / f"obj_psm_{side}.mp4" for side in ("l", "r", "tip")}


def video_info(path):
    info = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return info
    stat = path.stat()
    cap = cv2.VideoCapture(str(path))
    try:
        info.update(opened=cap.isOpened(), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    finally:
        cap.release()
    return info


def estimate_distal(mask, wrist):
    """Return one proposal and auditable geometry; abstain when direction is weak.

    Select the connected mask component nearest the DLC wrist (<=15 px).
    Principal-axis sign is supported by wrist-minus-component-centroid; the
    expected proximal metal collar is broader than the jaw. Require a >=10 px
    axial wrist offset and elongation >=1.5. The distal 3 px contour cap is
    averaged to suppress a single outlying pixel. A broad cap, separated cap
    branches or a wrist-to-candidate length outside 20..260 px causes rejection.
    These are transparent engineering QC rules, not calibrated probabilities.
    A geometrically accepted estimate still requires visual/annotated validation.
    """
    result = dict(x=np.nan, y=np.nan, accepted=False, reason="invalid_wrist",
                  quality_score=0.0, component_area=0, wrist_component_distance_px=np.nan,
                  elongation=np.nan, axial_wrist_offset_px=np.nan, tip_distance_px=np.nan,
                  cap_width_px=np.nan, direction_x=np.nan, direction_y=np.nan)
    wrist = np.asarray(wrist, float)
    if wrist.shape != (2,) or not np.isfinite(wrist).all():
        return result
    binary = np.asarray(mask, bool).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2D binary array")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    choices = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 100:
            continue
        yy, xx = np.where(labels == k)
        distance = np.sqrt(np.min((xx-wrist[0])**2 + (yy-wrist[1])**2))
        choices.append((float(distance), k))
    if not choices or min(choices)[0] > 15:
        result["reason"] = "no_component_within_15px_of_wrist"
        return result
    distance, k = min(choices)
    yy, xx = np.where(labels == k)
    coords = np.column_stack((xx, yy)).astype(float)
    center = coords.mean(axis=0)
    evals, vecs = np.linalg.eigh(np.cov(coords.T))
    axis = vecs[:, -1]
    shift = float(np.dot(wrist-center, axis))
    if shift < 0:
        axis = -axis
        shift = -shift
    result.update(component_area=len(coords), wrist_component_distance_px=distance,
                  elongation=float(evals[-1]/max(evals[0], 1e-9)),
                  axial_wrist_offset_px=shift, direction_x=axis[0], direction_y=axis[1])
    project = (coords-center) @ axis
    axial = float(np.percentile(project, 99.8))
    cap = coords[project >= axial-3]
    xy = cap.mean(axis=0)
    perp = np.array([-axis[1], axis[0]])
    width = float(np.ptp(cap @ perp))
    tip_distance = float(np.linalg.norm(xy-wrist))
    cap_img = np.zeros(binary.shape, np.uint8)
    cap_img[cap[:, 1].astype(int), cap[:, 0].astype(int)] = 1
    _, _, cs, _ = cv2.connectedComponentsWithStats(cap_img, 8)
    branches = int(np.sum(cs[1:, cv2.CC_STAT_AREA] >= max(3, .15*len(cap))))
    result.update(x=float(xy[0]), y=float(xy[1]), cap_width_px=width,
                  tip_distance_px=tip_distance, cap_branches=branches)
    if result["elongation"] < 1.5:
        result["reason"] = "shape_not_elongated"
    elif shift < 10:
        result["reason"] = "proximal_distal_direction_ambiguous"
    elif not 20 <= tip_distance <= 260:
        result["reason"] = "implausible_wrist_to_candidate_length"
    elif width > 45:
        result["reason"] = "distal_cap_too_broad"
    elif branches > 1:
        result["reason"] = "multiple_distal_caps"
    else:
        result["accepted"] = True
        result["reason"] = "heuristic_only_unvalidated"
        result["quality_score"] = float(min(1, shift/30) * min(1, result["elongation"]/4))
    return result


def _read_mask(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode mask frame {index}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > THRESHOLD


def propose_regions(d, data_root, repo=None, audit_dir=None):
    """Generate all-frame exploratory tracks; both sides share one strict support.

    Returns left_xy/right_xy [N,2], valid [N], confidence [N,2], reason [N,2],
    source (str), raw_left_xy/raw_right_xy. Rejected proposals remain NaN.
    confidence is an uncalibrated engineering geometry score. No training or
    testing decisions enter the estimator. This function decodes every native
    mask frame and can be expensive; the audit-only CLI uses sampled frames.
    d must contain the entire native sequence in the same order as the masks,
    not a stage-filtered subset. Metadata length is necessary but does not prove
    synchronization. The sample visual audit found false proximal selections;
    valid is only geometry support, NOT independently validated tip accuracy.
    """
    cid, n = d["content_id"], len(d["frames"])
    paths = mask_paths(cid, data_root)
    result = {"left_xy": np.full((n,2), np.nan), "right_xy": np.full((n,2), np.nan),
              "valid": np.zeros(n, bool), "confidence": np.zeros((n,2)),
              "reason": np.full((n,2), "missing_mask", dtype="U64"),
              "source": VERSION, "validation_status": VALIDATION_STATUS,
              "eligible_as_primary": False,
              "raw_left_xy": np.asarray(d["left"]).copy(),
              "raw_right_xy": np.asarray(d["right"]).copy()}
    for j, (side, name) in enumerate((("l", "left"),("r", "right"))):
        info = video_info(paths[side])
        if not (info.get("opened") and info.get("frames") == n and
                info.get("width") == d["width"] and info.get("height") == d["height"]):
            result["reason"][:, j] = "missing_or_misaligned_mask"
            continue
        cap = cv2.VideoCapture(str(paths[side]))
        try:
            for i in range(n):
                ok, frame = cap.read()
                if not ok:
                    result["reason"][i:, j] = "mask_decode_failed"
                    break
                a = estimate_distal(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)>THRESHOLD, d[name][i])
                result["reason"][i,j] = a["reason"]
                result["confidence"][i,j] = a["quality_score"]
                if a["accepted"]:
                    result[name+"_xy"][i] = (a["x"],a["y"])
        finally:
            cap.release()
    result["valid"] = np.isfinite(result["left_xy"]).all(1) & np.isfinite(result["right_xy"]).all(1)
    if audit_dir:
        out = Path(audit_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/(cid+"_distal_candidates.npz"), **result)
    return result


def self_test():
    """Focused software checks; these do not establish anatomical accuracy."""
    blank = np.zeros((200, 220), bool)
    assert estimate_distal(blank, [60, 80])["reason"] == "no_component_within_15px_of_wrist"
    assert estimate_distal(blank, [np.nan, 80])["reason"] == "invalid_wrist"
    # A symmetric silhouette cannot establish the distal direction.
    symmetric = blank.copy()
    symmetric[75:86, 20:181] = True
    assert not estimate_distal(symmetric, [100, 80])["accepted"]
    # A synthetic proximal collar plus narrower straight distal shaft.
    toy = blank.copy()
    toy[60:101, 20:81] = True
    toy[75:86, 80:161] = True
    estimate = estimate_distal(toy, [85, 80])
    assert estimate["accepted"] and estimate["x"] > 150
    mirrored = estimate_distal(toy[:, ::-1], [219-85, 80])
    assert mirrored["accepted"] and abs(mirrored["x"]-(219-estimate["x"])) < 1e-8
    # A missing mask never substitutes raw wrist coordinates.
    d = dict(content_id="NonexistentRegionAudit_0_0", frames=np.arange(2),
             width=220, height=200, left=np.zeros((2,2)), right=np.ones((2,2)))
    missing = propose_regions(d, Path(__file__).parent / "nonexistent_audit_data")
    assert not missing["valid"].any() and np.isnan(missing["left_xy"]).all()
    assert np.array_equal(missing["raw_right_xy"], d["right"])
    return dict(passed=True, checks=6, establishes_anatomical_accuracy=False)


def inventory(frame_dir, data_root, out):
    """Check all stored records by metadata. This does not decode every mask frame."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, flags = [], []
    for saved in sorted(Path(frame_dir).glob("*.npz")):
        cid = saved.stem
        with np.load(saved, allow_pickle=False) as d:
            n = len(d["frame_number"])
            W, H = d["resolution"]
            slot_key = "slotmask" if "slotmask" in d else "target_slotmask" if "target_slotmask" in d else None
            mode_key = "label_mode" if "label_mode" in d else "target_label_mode" if "target_label_mode" in d else None
            row = dict(content_id=cid, expected_frames=n, width=int(W), height=int(H))
            paths = mask_paths(cid, data_root)
            for side, p in paths.items():
                info = video_info(p)
                row.update({side+"_"+k:v for k,v in info.items()})
                row[side+"_metadata_match"] = bool(info.get("opened") and info.get("frames")==n
                    and info.get("width")==W and info.get("height")==H)
            rows.append(row)
            for occurrence in np.unique(d["occurrence"]):
                ii = np.flatnonzero(d["occurrence"] == occurrence)
                stage = str(d["stage"][ii[0]])
                if stage in ("", "none", "drop ring"):
                    continue
                slots = d["target_slots"][ii]
                center = d["target_xy"][ii]
                good = np.isfinite(slots).all(2) & (slots>=0).all(2) & (slots<[W,H]).all(2)
                radius = np.linalg.norm(slots-center[:,None,:],axis=2)
                radius[~good] = np.nan
                usable = np.isfinite(center).all(1)
                spread = np.array([np.nanmax(r) if np.isfinite(r).any() else np.nan for r in radius])
                near = np.minimum(np.linalg.norm(center-d["left_xy"][ii],axis=1),
                                  np.linalg.norm(center-d["right_xy"][ii],axis=1))
                nf = int(usable.sum())
                modes = dict(zip(*np.unique(d[mode_key][ii],return_counts=True))) if mode_key else {"unavailable":len(ii)}
                slotmask = d[slot_key][ii] if slot_key else np.sum(good.astype(np.uint8) * (2**np.arange(good.shape[1])), axis=1)
                flags.append(dict(content_id=cid, occurrence=int(occurrence), stage=stage,
                    first_frame=int(d["frame_number"][ii[0]]), last_frame=int(d["frame_number"][ii[-1]]),
                    frames=len(ii), target_frames=nf, target_role="unverified_from_annotation_protocol",
                    target_role_basis="no_object_identity_in_coordinate_fields",
                    semantic_flags="target_role_requires_protocol_confirmation",
                    count_changes=int(np.sum(slotmask[1:]!=slotmask[:-1])),
                    count_changes_definition="changes_of_valid_slot_bitmask_not_only_number_of_points",
                    target_slotmask_key=slot_key or "reconstructed_from_valid_slots", target_mode_key=mode_key or "unavailable",
                    max_point_radius_px=float(np.nanmax(spread)) if nf else np.nan,
                    frames_point_radius_gt50=int(np.sum(spread>50)),
                    frac_center_within50_of_wrist=float(np.mean(near[usable]<50)) if nf else np.nan,
                    mode_counts=json.dumps({str(k):int(v) for k,v in modes.items()}),
                    coordinate_override="none", flags_exclude_from_primary=False))
    _dump_csv(Path(out)/"mask_inventory.csv", rows)
    _dump_csv(Path(out)/"occurrence_geometry_flags.csv", flags)
    summary = dict(records=len(rows), metadata_both_sides_match=sum(r["l_metadata_match"] and r["r_metadata_match"] for r in rows),
                   tip_mask_metadata_match=sum(r["tip_metadata_match"] for r in rows), occurrences=len(flags),
                   coordinates_modified=False, all_frames_decoded=False,
                   flags_are="geometric descriptors, not diagnoses of drift or target object identity")
    (Path(out)/"inventory_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return rows, flags


def review_samples(frame_dir, data_root, out):
    """Replicate eight existing overlays and add another recording per task/stage."""
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    tasks={"A": ["reach ring","grab ring","transfer ring","place ring"],
           "C": ["reach string","wrap around psm","pull through and cinch","threading string through tower"]}
    saved = {p.stem:p for p in Path(frame_dir).glob("*.npz")}
    requests=[]
    for task, names in tasks.items():
        preferred = "User1_1_1" if task=="A" else "User1_5_2"
        for stage in names:
            chosen=0
            for cid in [preferred,*sorted(k for k in saved if k!=preferred)]:
                if chosen>=2:
                    break
                with np.load(saved[cid],allow_pickle=False) as d:
                    ii=np.flatnonzero((d["stage"]==stage)&d["valid_real_main"])
                    if not len(ii):continue
                    i=int(ii[len(ii)//2]);requests.append((cid,task,stage,i,chosen==0));chosen+=1
    samples=[]
    for j,(cid,task,stage,i,original) in enumerate(requests):
        with np.load(saved[cid],allow_pickle=False) as d:
            frame_no=int(d["frame_number"][i])
            video=_root(data_root)/cid.split("_")[0]/"stimuli"/(cid+".mp4")
            cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();cap.release()
            if not ok:raise ValueError(f"Cannot decode source frame {cid}/{i}")
            paths=mask_paths(cid,data_root)
            image=img.copy()
            local=[]
            for side,name,col in (("l","left",(255,150,20)),("r","right",(20,180,255))):
                cap=cv2.VideoCapture(str(paths[side]))
                try:mask=_read_mask(cap,i)
                finally:cap.release()
                wrist=d[name+"_xy"][i]
                a=estimate_distal(mask,wrist)
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image,contours,-1,col,1)
                w=tuple(np.round(wrist).astype(int));cv2.drawMarker(image,w,(0,255,0),cv2.MARKER_CROSS,15,2)
                cv2.circle(image,w,50,(0,150,0),1)
                if np.isfinite([a["x"],a["y"]]).all():
                    tip=(int(round(a["x"])),int(round(a["y"])))
                    color=(255,0,255) if a["accepted"] else (0,0,255)
                    cv2.drawMarker(image,tip,color,cv2.MARKER_DIAMOND,18,2)
                    cv2.line(image,w,tip,color,1)
                    cv2.circle(image,tip,50,color,1)
                samples.append(dict(sample_id=j, content_id=cid,task=task,stage=stage,
                    frame_index=i,frame_number=frame_no,occurrence=int(d["occurrence"][i]),
                    replicates_original_overlay=original,side=side,wrist_x=wrist[0],wrist_y=wrist[1],
                    target_x=d["target_xy"][i,0],target_y=d["target_xy"][i,1],
                    target_mode=str(d["label_mode"][i]),**a))
                local.append(side+": "+a["reason"])
            xy=d["target_xy"][i];cv2.circle(image,tuple(np.round(xy).astype(int)),50,(0,255,255),2)
            cv2.rectangle(image,(0,0),(image.shape[1],65),(0,0,0),-1)
            for k,line in enumerate([f"{j:02d} {cid} {stage} frame {frame_no}","Green = wrist; magenta = candidate; red = rejected proposal; yellow = target", " / ".join(local)]):
                cv2.putText(image,line,(8,18+20*k),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
            name=f"sample_{j:02d}_{cid}_{stage.replace(' ','_')}.jpg"
            cv2.imwrite(str(out/name),image)
    _dump_csv(out/"candidate_samples.csv",samples)
    return samples


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames",type=Path,required=True)
    p.add_argument("--data-root",type=Path,default=Path("D:/Dataset/dvss"))
    p.add_argument("--repo",type=Path,default=Path("D:/GitHub_Project/co-tracker"))
    p.add_argument("--out",type=Path,default=None)
    args=p.parse_args()
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.out = args.repo / "outputs" / "target_locking_region_review" / stamp
    args.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    rows,_=inventory(args.frames,args.data_root,args.out)
    print(f"Metadata inventory: {len(rows)} recordings",flush=True)
    samples=review_samples(args.frames,args.data_root,args.out)
    print(f"Sample review: {len(samples)} side/frame proposals",flush=True)


if __name__=="__main__":
    main()
