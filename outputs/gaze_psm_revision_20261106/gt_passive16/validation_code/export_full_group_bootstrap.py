"""Archive every existing group's frozen participant bootstrap peak draws.

This exports the draws already specified by GT84's group estimator and seeds.
It adds provenance files and leaves the primary plotting archives untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from validate_passive_results import (
    GROUP_FIELDS, as_number, compare_arrays, compare_scalar, fisher,
    group_key, read_csv, read_curves, selected_curves, stable_seed,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(output):
    protected = {name: sha(output / name) for name in
                 ("group_bootstrap_index.json", "group_bootstrap_peak_lags.npz")}
    summaries = read_csv(output / "group_summary.csv") + read_csv(output / "gt_selected_group_summary.csv")
    summary_lookup = {group_key(row): row for row in summaries}
    curves = read_curves(output / "record_curves.jsonl")
    grouped = defaultdict(lambda: defaultdict(list))
    for curve in curves + selected_curves(curves):
        if np.isfinite(curve["r"]).all():
            grouped[group_key(curve)][str(curve["user"])].append(curve)
    long = defaultdict(list)
    for row in read_csv(output / "group_curves.csv") + read_csv(output / "gt_selected_group_curves.csv"):
        long[group_key(row)].append(row)
    samples, index = {}, []
    errors = []
    for key, row in sorted(summary_lookup.items()):
        users = sorted(grouped[key])
        lag = np.array([float(point["lag_s"]) for point in long[key]])
        count = int(row["n_bootstrap"])
        assert count == 5000, (key, count)
        peak_lags = np.full(count, np.nan)
        if len(users) >= 3:
            people = np.stack([fisher([curve["r"] for curve in grouped[key][user]]) for user in users])
            rng = np.random.default_rng(stable_seed(42, (key[:-1], tuple(users))))
            draws = rng.integers(0, len(users), (count, len(users)))
            z = np.arctanh(np.clip(people, -1 + 1e-7, 1 - 1e-7))
            resampled_curves = np.tanh(z[draws].mean(1))
            peak_lags = lag[np.argmax(resampled_curves, axis=1)]
            low, high = np.percentile(resampled_curves, [2.5, 97.5], axis=0)
            compare_arrays([as_number(point["r_resample_low"]) for point in long[key]], low,
                           f"full_bootstrap_curve_low/{key}", errors, 1e-11)
            compare_arrays([as_number(point["r_resample_high"]) for point in long[key]], high,
                           f"full_bootstrap_curve_high/{key}", errors, 1e-11)
        token = f"g{len(index):04d}"
        samples[token] = peak_lags
        index.append(dict(zip(GROUP_FIELDS, key), token=token, users=users,
                          n_stable_users=len(users), n_complete_records=int(row["n_complete_records"]),
                          n_requested=count, n_finite_peak_draws=int(np.isfinite(peak_lags).sum()),
                          status=row["status"], lag_min_s=float(lag[0]), lag_max_s=float(lag[-1]),
                          seed_base=42, seed_derived=stable_seed(42, (key[:-1], tuple(users))),
                          seed_rule="SHA256(repr((42, (group_key_without_source, tuple(sorted_observers))))), first8 bytes little-endian",
                          unsupported_array="5000 NaNs when fewer than 3 complete-support observers"))
    archive_path = output / "group_bootstrap_full_peak_lags.npz"
    index_path = output / "group_bootstrap_full_index.json"
    assert not archive_path.exists() and not index_path.exists(), "Refusing to overwrite previous export"
    np.savez_compressed(archive_path, **samples)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    supported = 0
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {row["token"] for row in index}
        for item in index:
            key = group_key(item)
            row = summary_lookup[key]
            values = archive[item["token"]]
            assert values.shape == (5000,), (key, values.shape)
            if item["n_stable_users"] < 3:
                assert np.isnan(values).all(), key
                continue
            assert np.isfinite(values).all(), key
            low, high = np.percentile(values, [2.5, 97.5])
            compare_scalar(row["peak_lag_resample_low"], low, f"full_bootstrap_peak_low/{key}", errors)
            compare_scalar(row["peak_lag_resample_high"], high, f"full_bootstrap_peak_high/{key}", errors)
            boundary = np.mean((values == item["lag_min_s"]) | (values == item["lag_max_s"]))
            compare_scalar(row["bootstrap_boundary_frequency"], boundary, f"full_bootstrap_boundary/{key}", errors)
            supported += 1
    unchanged = all(sha(output / name) == fingerprint for name, fingerprint in protected.items())
    result = dict(passed=not errors and unchanged, groups_exported=len(index),
                  supported_groups_checked=supported, unsupported_groups_with_nan_arrays=len(index) - supported,
                  draws_per_group=5000, variants=sorted({row["variant"] for row in index}),
                  original_primary_bootstrap_archives_unchanged=unchanged,
                  original_primary_archive_sha256=protected, errors=errors)
    (output / "group_bootstrap_full_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    if not result["passed"]:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(export(arguments.output), ensure_ascii=False))
