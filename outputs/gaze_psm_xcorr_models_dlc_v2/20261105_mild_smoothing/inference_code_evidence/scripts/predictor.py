"""Unified prior-only spatial inference, per-video metrics and AVC1 overlays."""

import csv
import json
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from dvss_pytorch.data import DVSSDataModule, load_gaze
from dvss_pytorch.data.cache import prepare_temporal_cache
from dvss_pytorch.data.io import resolve_items
from dvss_pytorch.model import SpatialCVAE
from dvss_pytorch.utils.io import args_parser, load_model, checkpoint_path
from dvss_pytorch.utils.metrics import score_predictions
from dvss_pytorch.utils.visualization import probe_video, render_heatmap
from dvss_pytorch.model.encoders import file_sha256


@torch.inference_mode()
def main(args, data=None):
    model = load_model(args["checkpoint"] or args["save_dir"]).eval()
    if not isinstance(model, SpatialCVAE):
        # Preserve the original Lightning test path for coordinate-only baselines.
        from lightning.pytorch import Trainer

        data = data or DVSSDataModule.from_model(
            args["task"],
            model,
            skill=args.get("skill"),
            domain=args.get("domain"),
            batch_size=args["batch_size"],
            num_workers=args["num_workers"],
            data_root=args.get("data_root"),
            manifest_dir=args.get("manifest_dir"),
        )
        Trainer(
            accelerator=args["accelerator"], devices=1,
            logger=False, enable_progress_bar=True,
        ).test(
            model, datamodule=data
        )
        return
    device = (
        "cuda" if args["accelerator"] != "cpu" and torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    split = args.get("split", "test")
    data = data or DVSSDataModule.from_model(
        args["task"],
        model,
        skill=args.get("skill"),
        domain=args.get("domain"),
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        cache_dir=args["cache_dir"],
        segment_stride=args["test_stride"],
        test_stride=args["test_stride"],
        filter_mode="none",
        predict_split=split,
        data_root=args.get("data_root"),
        manifest_dir=args.get("manifest_dir"),
    )
    prepare_temporal_cache(data, model, (split,), accelerator=args["accelerator"],
                           enabled=not args.get("no_cache", False))
    data.setup("predict")
    # Cached inference also reads source labels/videos for metrics and rendering.
    data.items[split] = resolve_items(data.items[split], data.data_root)
    out = Path(args["save_dir"]) / f'inference_{split}_k{args["num_samples"]}'
    checkpoint = checkpoint_path(args["checkpoint"] or args["save_dir"])
    provenance = dict(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=file_sha256(checkpoint),
        manifest_sha256=file_sha256(data.manifest_path(split)),
        skill=data.skill,
        domain=data.domain,
        task=data.task,
        data_root=str(data.data_root) if data.data_root else None,
        split=split,
        num_samples=args["num_samples"],
        seed=args["seed"],
        batch_size=args["batch_size"],
        decode_chunk=args["decode_chunk"],
        max_frames=0,
        model_class=type(model).__name__,
        test_stride=args["test_stride"],
        cached_inputs=bool(data.cache_dir),
        cache_path=str(data.cache_paths.get(split, data.cache_dir / split)) if data.cache_dir else None,
    )
    existing = out / "provenance.json"
    if existing.exists() and json.loads(existing.read_text()) != provenance:
        raise ValueError(
            "Prediction directory belongs to a different checkpoint or configuration"
        )
    out.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(provenance, indent=2))
    generator = torch.Generator(device=device).manual_seed(args["seed"])
    by_clip = {}
    for batch in tqdm(
        data.predict_dataloader(), desc=f"Inference {data.domain or 'data'}/{split}",
        unit="batch", dynamic_ncols=True, mininterval=1,
    ):
        inputs = {
            key: batch[key].to(device) for key in ("images", "features") if key in batch
        }
        pred = model.sample(
            **inputs,
            num_samples=args["num_samples"],
            generator=generator,
            decode_chunk=args["decode_chunk"],
        )
        for j, ci in enumerate(batch["clip_id"].tolist()):
            item = by_clip.setdefault(
                ci, dict(samples=[], probability=[], frame_index=[])
            )
            item["samples"].append(pred["samples"][j].cpu().numpy())
            item["probability"].append(pred["probability"][j].cpu().numpy())
            item["frame_index"].append(int(batch["anchor"][j]))
    rows = []
    verified = {}
    for ci, parts in tqdm(
        by_clip.items(), desc="Metrics / rendering", unit="video",
        dynamic_ncols=True, mininterval=1,
    ):
        item = data.items[split][ci]
        name = Path(item["video_path"]).stem
        ids = np.array(parts["frame_index"])
        gaze = load_gaze(item["gaze_npy_path"])
        xy = np.stack([gaze["gaze_x"][ids], gaze["gaze_y"][ids]], -1)
        result = {k: np.asarray(v) for k, v in parts.items()}
        result.update(
            target=np.clip(np.nan_to_num(xy), 0, 1).astype(np.float32),
            valid=np.isfinite(xy).all(-1),
            frame_number=gaze["Frame_Number"][ids],
            timestamp_ns=gaze["Timestamp(ns)"][ids],
            valid_code=gaze["valid"][ids],
        )
        np.savez_compressed(out / f"{name}_predictions.npz", **result)
        if args.get("export_csv", False):
            with (out / f"{name}_predictions.csv").open("w", newline="") as stream:
                csv_writer = csv.writer(stream)
                csv_writer.writerow(
                    [
                        "frame_index",
                        "frame_number",
                        "timestamp_ns",
                        "is_valid",
                        "gt_x",
                        "gt_y",
                    ]
                    + [
                        f"sample_{k}_{axis}"
                        for k in range(args["num_samples"])
                        for axis in ("x", "y")
                    ]
                )
                for i, frame in enumerate(ids):
                    csv_writer.writerow(
                        [
                            int(frame),
                            int(result["frame_number"][i]),
                            int(result["timestamp_ns"][i]),
                            int(result["valid"][i]),
                            *result["target"][i],
                            *result["samples"][i].flatten(),
                        ]
                    )
        info = probe_video(item["video_path"])
        rows.append(
            dict(
                clip=name,
                scope=(
                    "all_frames"
                    if args["test_stride"] == 1
                    else f"stride{args['test_stride']}"
                ),
                **score_predictions(
                    result, int(info["width"]), int(info["height"]), model.label_sigma
                ),
            )
        )
        if args["render"]:
            verified[name] = render_heatmap(
                item["video_path"],
                result,
                out / "heatmap_filtered" / f"{name}_avc1.mp4",
                gaze=gaze,
            )
    n = sum(r["n"] for r in rows)
    aggregate = dict(
        clip="ALL", scope=rows[0]["scope"], n=n, num_samples=args["num_samples"]
    )
    for key in rows[0]:
        if key in aggregate:
            continue
        power = 2 if "rmse" in key or key == "sample_std_norm" else 1
        aggregate[key] = (sum(r[key] ** power * r["n"] for r in rows) / n) ** (
            1 / power
        )
    rows.append(aggregate)
    (out / "metrics.json").write_text(
        json.dumps(
            dict(
                provenance=provenance,
                checkpoint=str(checkpoint.resolve()),
                model_class=type(model).__name__,
                cached_inputs=bool(data.cache_dir),
                test_stride=args["test_stride"],
                metrics=rows,
            ),
            indent=2,
        )
    )
    if args.get("export_csv", False):
        with (out / "rmse.csv").open("w", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
    if verified:
        (out / "heatmap_filtered/video_verification.json").write_text(
            json.dumps(verified, indent=2)
        )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main(args_parser())
