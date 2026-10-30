"""Participant-level aggregation for gaze/PSM cross-correlation curves.

Recordings, not sample counts, receive equal weight within each participant.
Participants receive equal weight in group estimates. Bootstrap uncertainty is
computed by resampling participant identities, never individual time samples.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np


GROUP_FIELDS = ("task", "side", "signal", "stage", "mode")
_EPS = 1e-7


def _key(curve):
    return tuple(curve[field] for field in GROUP_FIELDS)


def _mean(values):
    """NaN-aware average without empty-slice warnings."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    total = np.where(finite, array, 0).sum(axis=0)
    count = finite.sum(axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def fisher_mean(values):
    """Equal-weight Fisher-z mean; NaNs contribute no weight."""
    array = np.asarray(values, dtype=float)
    array = np.where(np.isfinite(array), array, np.nan)
    z = np.arctanh(np.clip(array, -1 + _EPS, 1 - _EPS))
    return np.tanh(_mean(z))


def _quantiles(values, axis=0):
    array = np.asarray(values, dtype=float)
    if not array.size:
        return np.nan, np.nan
    return np.nanpercentile(array, [2.5, 97.5], axis=axis)


def _peak(lags, r):
    finite = np.isfinite(r)
    if not finite.any():
        return {"peak_lag_s": np.nan, "peak_r": np.nan,
                "peak_at_boundary": False, "status": "no_valid_curve"}
    index = int(np.nanargmax(r))
    boundary = index in (0, len(r) - 1)
    return {"peak_lag_s": float(lags[index]), "peak_r": float(r[index]),
            "peak_at_boundary": boundary,
            "status": "boundary_peak" if boundary else "ok"}


def _validate_lags(curves):
    first = np.asarray(curves[0]["lag_s"], dtype=float)
    if first.ndim != 1 or not len(first) or np.any(np.diff(first) <= 0):
        raise ValueError("lag_s must be a nonempty, strictly increasing vector")
    for curve in curves:
        if not np.array_equal(np.asarray(curve["lag_s"], dtype=float), first):
            raise ValueError("Cannot aggregate curves with different lag grids")
        if np.asarray(curve["r"]).shape != first.shape:
            raise ValueError("Correlation array does not match lag_s")
    return first


def _field(curve, name, n, default=np.nan):
    value = np.asarray(curve.get(name, np.full(n, default)), dtype=float)
    if value.shape != (n,):
        raise ValueError(f"{name} array does not match lag_s")
    return value


def _participant_curves(curves):
    groups = defaultdict(list)
    for curve in curves:
        groups[(str(curve["user"]), *_key(curve))].append(curve)
    output = []
    for key, records in sorted(groups.items()):
        lags = _validate_lags(records)
        valid = np.isfinite(np.stack([r["r"] for r in records]))
        row = dict(zip(("user", *GROUP_FIELDS), key))
        row.update(lag_s=lags.copy(), r=fisher_mean([r["r"] for r in records]),
                   n_records=valid.sum(axis=0), record_ids=[r["content_id"] for r in records])
        # Equal counts can still hide replacement of one attempt by another.
        # Keep a full-record-support sensitivity curve without changing the
        # primary available-attempt estimate.
        stable_records = valid.all(axis=1)
        row.update(record_support_changes=bool(np.any(valid != valid[:, :1])),
                   stable_record_ids=[r["content_id"] for r, keep in zip(records, stable_records) if keep],
                   n_stable_records=int(stable_records.sum()),
                   r_fixed_records=(fisher_mean(np.stack([r["r"] for r in records])[stable_records])
                                    if stable_records.any() else np.full(len(lags), np.nan)))
        # Counts describe samples supporting the curve, not weighting factors.
        for name in ("pairs", "occurrences"):
            arrays = np.stack([_field(r, name, len(lags), 0) for r in records])
            row[name] = np.where(valid, arrays, 0).sum(axis=0).astype(int)
        for name in ("distance_median_px", "proximity", "gaze_rms", "psm_rms"):
            arrays = np.stack([_field(r, name, len(lags)) for r in records])
            row[name] = _mean(np.where(valid, arrays, np.nan))
        row.update(_peak(lags, row["r"]))
        output.append(row)
    return output


def _bootstrap_curves(r, lags, count, rng):
    """Return group-curve bands and peak locations, with participant resampling."""
    k = len(lags)
    if count == 0 or len(r) == 0:
        return np.full(k, np.nan), np.full(k, np.nan), np.array([]), np.nan
    indices = rng.integers(0, len(r), size=(count, len(r)))
    z = np.arctanh(np.clip(r, -1 + _EPS, 1 - _EPS))
    # Every participant here is finite at every lag, so ordinary means apply.
    means = np.tanh(z[indices].mean(axis=1))
    peaks = np.argmax(means, axis=1)
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    boundary = np.mean((peaks == 0) | (peaks == k - 1))
    return low, high, lags[peaks], float(boundary)


def _group_curve(key, people, bootstrap, rng, min_users):
    lags = _validate_lags(people)
    arrays = np.stack([p["r"] for p in people])
    stable = np.isfinite(arrays).all(axis=1)
    selected = [p for p, keep in zip(people, stable) if keep]
    n = len(selected)
    row = dict(zip(GROUP_FIELDS, key))
    row.update(lag_s=lags.copy(), r_available=fisher_mean(arrays),
               n_users=np.isfinite(arrays).sum(axis=0),
               stable_users=[p["user"] for p in selected], n_stable_users=n,
               excluded_partial_users=[p["user"] for p, keep in zip(people, stable) if not keep],
               n_bootstrap=int(bootstrap))
    fixed_people = [p for p in selected if np.isfinite(p["r_fixed_records"]).all()]
    fixed_supported = len(fixed_people) >= min_users
    row.update(record_support_changes=any(p["record_support_changes"] for p in selected),
               fixed_record_users=[p["user"] for p in fixed_people],
               n_fixed_record_users=len(fixed_people),
               r_fixed_records=(fisher_mean([p["r_fixed_records"] for p in fixed_people])
                                if fixed_supported else np.full(len(lags), np.nan)),
               # This comparison holds the sensitivity participant set fixed,
               # isolating attempt composition from participant composition.
               r_primary_fixed_record_users=(fisher_mean([p["r"] for p in fixed_people])
                                             if fixed_supported else np.full(len(lags), np.nan)))
    supported = n >= min_users
    row["r"] = fisher_mean(arrays[stable]) if supported else np.full(len(lags), np.nan)
    row.update(_peak(lags, row["r"]))
    if not supported:
        row["status"] = "insufficient_users" if np.isfinite(arrays).any() else "no_valid_curve"
    for name in ("pairs", "occurrences", "n_records"):
        row[name] = (np.stack([p[name] for p in selected]).sum(axis=0).astype(int)
                     if selected else np.zeros(len(lags), dtype=int))
    for name in ("distance_median_px", "proximity", "gaze_rms", "psm_rms"):
        row[name] = (_mean([p[name] for p in selected]) if selected else np.full(len(lags), np.nan))
    low, high, peaks, boundary = _bootstrap_curves(arrays[stable], lags,
                                                  bootstrap if supported else 0, rng)
    row.update(r_ci_low=low, r_ci_high=high, bootstrap_peak_lag_s=peaks,
               boundary_frequency=boundary)
    row["peak_lag_ci_low"], row["peak_lag_ci_high"] = _quantiles(peaks)
    zero = int(np.argmin(np.abs(lags)))
    peak = int(np.nanargmax(row["r"])) if supported else None
    row.update(spatial_at_zero_px=float(row["distance_median_px"][zero]),
               proximity_at_zero=float(row["proximity"][zero]),
               spatial_at_peak_px=float(row["distance_median_px"][peak]) if peak is not None else np.nan,
               proximity_at_peak=float(row["proximity"][peak]) if peak is not None else np.nan)
    # Null fields are always present so CSV/figure schemas do not depend on flags.
    row.update(null_r_low=np.full(len(lags), np.nan), null_r_high=np.full(len(lags), np.nan),
               null_peak_r=np.array([]), null_peak_r_95=np.nan, null_peak_p=np.nan,
               baseline_label="unpaired_reference", n_null_replicates=0)
    return row


def _contrasts(groups, people_lookup, bootstrap, rng, min_users):
    """A/C task contrasts and same-task stage contrasts, both paired by person."""
    output = []
    for a, b in combinations(groups, 2):
        # Instruments, signal types, and conditioning modes must agree.
        if any(a[f] != b[f] for f in ("side", "signal", "mode")):
            continue
        if a["task"] != b["task"] and a["stage"] == b["stage"] == "__whole__":
            dimension, condition_a, condition_b = "task", a["task"], b["task"]
        elif a["task"] == b["task"] and a["stage"] != b["stage"] and "__whole__" not in (a["stage"], b["stage"]):
            dimension, condition_a, condition_b = "stage", a["stage"], b["stage"]
        else:
            continue
        users = sorted(set(a["stable_users"]) & set(b["stable_users"]))
        row = {f: a[f] for f in ("side", "signal", "mode")}
        row.update(dimension=dimension, condition_a=condition_a, condition_b=condition_b,
                   task=a["task"] if dimension == "stage" else "paired",
                   stage=a["stage"] if dimension == "task" else "paired",
                   stable_users=users, n_users=len(users),
                   difference_s=np.nan, ci_low=np.nan, ci_high=np.nan,
                   bootstrap_difference_s=np.array([]),
                   peak_lag_a_s=np.nan, peak_lag_b_s=np.nan,
                   status="insufficient_users", difference_definition="condition_b_minus_condition_a")
        if len(users) >= min_users:
            lags = _validate_lags([a, b])
            aa = np.stack([people_lookup[(_key(a), u)]["r"] for u in users])
            bb = np.stack([people_lookup[(_key(b), u)]["r"] for u in users])
            pa = _peak(lags, fisher_mean(aa))
            pb = _peak(lags, fisher_mean(bb))
            row.update(peak_lag_a_s=pa["peak_lag_s"], peak_lag_b_s=pb["peak_lag_s"],
                       difference_s=pb["peak_lag_s"] - pa["peak_lag_s"],
                       status="boundary_peak" if pa["peak_at_boundary"] or pb["peak_at_boundary"] else "ok")
            if bootstrap:
                # One set of sampled identities is applied to both conditions.
                indices = rng.integers(0, len(users), size=(bootstrap, len(users)))
                za = np.arctanh(np.clip(aa, -1 + _EPS, 1 - _EPS))
                zb = np.arctanh(np.clip(bb, -1 + _EPS, 1 - _EPS))
                ia = np.argmax(za[indices].mean(axis=1), axis=1)
                ib = np.argmax(zb[indices].mean(axis=1), axis=1)
                differences = lags[ib] - lags[ia]
                row["bootstrap_difference_s"] = differences
                row["ci_low"], row["ci_high"] = _quantiles(differences)
        output.append(row)
    return output


def aggregate(curves, bootstrap=1000, seed=42, min_users=3, baseline_replicates=None):
    """Return ``(participant_curves, group_curves, paired_contrasts)``.

    The primary group curve uses participants with finite estimates at *every*
    requested lag. ``r_available`` and ``n_users`` retain partial-data diagnostics.
    ``r_fixed_records`` is a sensitivity curve using only complete-lag attempts;
    its participant set is disclosed separately. ``r_primary_fixed_record_users``
    holds that participant set fixed when inspecting attempt-composition effects.
    Distances are means of recording medians within person, then person means;
    ``proximity`` uses the same hierarchy. They are never evaluated at individual
    records' optimized lags for a group spatial-correspondence claim.

    Baselines are an iterable of shuffled recording-curve lists, allowing the
    caller to stream one replicate at a time. Each replicate is
    aggregated identically, using the observed group's fixed participant set.
    The tail fraction ``null_peak_p`` includes the finite-simulation correction
    and compares maxima over the complete lag search; it is an empirical
    unpaired-reference statistic, not a universal independence or causal test.
    """
    if bootstrap < 0 or int(bootstrap) != bootstrap or min_users < 1 or int(min_users) != min_users:
        raise ValueError("bootstrap must be a nonnegative integer and min_users >= 1")
    if not curves:
        return [], [], []
    people = _participant_curves(curves)
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for person in people:
        by_group[_key(person)].append(person)
    groups = [_group_curve(key, values, int(bootstrap), rng, min_users)
              for key, values in sorted(by_group.items())]
    people_lookup = {(_key(p), p["user"]): p for p in people}
    contrasts = _contrasts(groups, people_lookup, int(bootstrap), rng, min_users)
    if baseline_replicates is not None:
        references = defaultdict(list)
        for replicate in baseline_replicates:
            null_people = _participant_curves(replicate)
            lookup = {(_key(p), p["user"]): p for p in null_people}
            for group in groups:
                users = group["stable_users"]
                if len(users) < min_users:
                    continue
                selected = [lookup.get((_key(group), user)) for user in users]
                if any(p is None for p in selected):
                    continue
                _validate_lags([group, *selected])
                r = np.stack([p["r"] for p in selected])
                # A baseline replicate must preserve the observed complete user
                # set and full lag support; partial replicates are not substituted.
                if np.isfinite(r).all():
                    references[_key(group)].append(fisher_mean(r))
        for group in groups:
            values = references[_key(group)]
            if values:
                array = np.stack(values)
                group["null_r_low"], group["null_r_high"] = np.percentile(array, [2.5, 97.5], axis=0)
                maxima = array.max(axis=1)
                group.update(null_peak_r=maxima, null_peak_r_95=float(np.percentile(maxima, 95)),
                             null_peak_p=float((1 + np.count_nonzero(maxima >= group["peak_r"])) / (len(maxima) + 1)),
                             n_null_replicates=len(maxima))
    return people, groups, contrasts
