"""CSV and figure exports for the gaze/PSM cross-correlation pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


META = (
    "content_id", "user", "task", "side", "signal", "stage", "mode",
    "n_stable_users", "n_fixed_record_users", "n_stable_records", "record_support_changes",
)
LAG_FIELDS = (
    "lag_s", "r", "r_available", "r_ci_low", "r_ci_high", "null_r_low",
    "null_r_high", "pairs", "occurrences", "gaze_rms", "psm_rms",
    "distance_median_px", "proximity", "n_records", "n_users",
    "r_fixed_records", "r_primary_fixed_record_users",
)


def _value(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), allow_nan=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, allow_nan=True)
    return value


def _write(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _value(value) for key, value in row.items()} for row in rows)


def _long_rows(curves):
    for curve in curves:
        lags = np.asarray(curve.get("lag_s", []))
        arrays = {key: np.asarray(curve[key]) for key in LAG_FIELDS
                  if key in curve and np.ndim(curve[key]) == 1
                  and len(curve[key]) == len(lags)}
        metadata = {key: curve[key] for key in META if key in curve}
        for index in range(len(lags)):
            yield {**metadata, **{key: value[index] for key, value in arrays.items()}}


def _summary(curve):
    # Preserve explicit group/user estimates; derive peaks for record curves.
    result = {key: value for key, value in curve.items()
              if not isinstance(value, np.ndarray) and key not in LAG_FIELDS}
    if "peak_lag_s" not in result:
        r = np.asarray(curve.get("r", []), dtype=float)
        lags = np.asarray(curve.get("lag_s", []), dtype=float)
        if np.isfinite(r).any():
            index = int(np.nanargmax(r))
            result.update(peak_lag_s=lags[index], peak_r=r[index],
                          peak_at_boundary=index in (0, len(r)-1),
                          status="boundary_peak" if index in (0, len(r)-1) else "ok")
            for source, destination in (("pairs", "pairs_at_peak"),
                                        ("occurrences", "occurrences_at_peak")):
                if source in curve:
                    result[destination] = curve[source][index]
        else:
            result.update(peak_lag_s=np.nan, peak_r=np.nan,
                          peak_at_boundary=False, status="insufficient_support_or_variation")
    return result


def _matches(curve, **conditions):
    return all(curve.get(key) == value for key, value in conditions.items())


def _finite(curve, field="r"):
    return np.isfinite(np.asarray(curve.get(field, []), dtype=float)).any()


def _task_order(curves):
    return sorted({str(c["task"]) for c in curves if "task" in c})


def _decorate(ax, ylabel="Position correlation r"):
    ax.axvline(0, color="0.35", linewidth=.8, linestyle=":")
    ax.set_xlabel("Lag (s); positive = gaze leads")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)


def _curve_panel(ax, groups, users, conditions):
    selected = [g for g in groups if _matches(g, **conditions)]
    for user in users:
        if _matches(user, **conditions) and _finite(user):
            ax.plot(user["lag_s"], user["r"], color="0.65", linewidth=.65, alpha=.55)
    if not selected:
        ax.text(.5, .5, "No supported curve", ha="center", va="center", transform=ax.transAxes)
        return
    group = selected[0]
    lag = np.asarray(group["lag_s"])
    if "null_r_low" in group and "null_r_high" in group:
        low, high = np.asarray(group["null_r_low"]), np.asarray(group["null_r_high"])
        if np.isfinite(low).any():
            ax.fill_between(lag, low, high, color="0.5", alpha=.18,
                            label="Unpaired reference (95%)")
    if "r_ci_low" in group and "r_ci_high" in group:
        ax.fill_between(lag, group["r_ci_low"], group["r_ci_high"],
                        color="tab:blue", alpha=.20, label="Novice bootstrap (95%)")
    if _finite(group):
        ax.plot(lag, group["r"], color="tab:blue", linewidth=1.9, label="Equal-novice mean")
        if np.isfinite(group.get("peak_lag_s", np.nan)):
            ax.plot(group["peak_lag_s"], group["peak_r"], "o", color="tab:blue", markersize=4)
    if group.get("record_support_changes") and _finite(group, "r_fixed_records"):
        ax.plot(lag, group["r_fixed_records"], color="tab:orange", linestyle="--", linewidth=1.1,
                label=f"Fixed-record sensitivity (n={group.get('n_fixed_record_users', 0)})")
    ax.text(.03, .97, f"Stable novices: {group.get('n_stable_users', 0)}\n{group.get('status', '')}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    ax.set_ylim(-1, 1)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper right")


def _save(fig, path):
    """Watermark all figures from an explicitly partial-record smoke test."""
    path = Path(path)
    config_path = path.parent.parent/"config.json"
    try:
        partial = bool(json.loads(config_path.read_text()).get("partial_record", False))
    except (OSError, ValueError, TypeError):
        partial = False
    if partial:
        fig.text(.5, .51, "PARTIAL RECORD / SMOKE TEST", ha="center", va="center",
                 fontsize=25, rotation=24, color="firebrick", alpha=.16, zorder=1000)
    fig.savefig(path, dpi=170)


def _whole_figures(plt, directory, groups, users):
    whole = [g for g in groups if g.get("mode") == "whole"]
    tasks = _task_order(whole)
    if not tasks:
        return
    for signal in sorted({g["signal"] for g in whole}):
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False,
                                 figsize=(11, max(3.4, 3.4*len(tasks))))
        for row, task in enumerate(tasks):
            for col, side in enumerate(("l", "r")):
                ax = axes[row, col]
                conditions = dict(task=task, side=side, signal=signal, stage="__whole__", mode="whole")
                _curve_panel(ax, groups, users, conditions)
                _decorate(ax, f"{signal.capitalize()} correlation r")
                ax.set_title(f"Task {task} / PSM {side.upper()}")
        fig.suptitle(f"Whole-task {signal} xcorr; faint lines are novices")
        fig.tight_layout(rect=(0, 0, 1, .96))
        _save(fig, directory/f"whole_task_{signal}.png")
        plt.close(fig)


def _spatial_figures(plt, directory, groups):
    groups = [g for g in groups if g.get("mode") == "whole" and g.get("signal") == "position"]
    tasks = _task_order(groups)
    if not tasks:
        return
    for field, ylabel, filename in (("distance_median_px", "Mean novice median distance (px)", "spatial_distance.png"),
                                     ("proximity", "Mean novice proximity fraction", "spatial_proximity.png")):
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(11, 3.3*len(tasks)))
        for row, task in enumerate(tasks):
            for col, side in enumerate(("l", "r")):
                ax = axes[row, col]
                selected = [g for g in groups if _matches(g, task=task, side=side)]
                if selected:
                    g = selected[0]
                    if _finite(g, field):
                        ax.plot(g["lag_s"], g[field], color="tab:green")
                    if np.isfinite(g.get("peak_lag_s", np.nan)):
                        ax.axvline(g["peak_lag_s"], color="tab:blue", linestyle="--", label="Group xcorr peak")
                        ax.legend(fontsize=8)
                    ax.text(.03, .03, f"Stable novices: {g.get('n_stable_users', 0)}", transform=ax.transAxes, fontsize=8)
                _decorate(ax, ylabel)
                if field == "proximity":
                    ax.set_ylim(0, 1)
                ax.set_title(f"Task {task} / PSM {side.upper()}")
        fig.tight_layout()
        _save(fig, directory/filename)
        plt.close(fig)


def _stage_figures(plt, directory, groups, users):
    staged = [g for g in groups if g.get("mode") == "psm_stage"]
    for task in _task_order(staged):
        for signal in sorted({g["signal"] for g in staged if g["task"] == task}):
            stages = sorted({g["stage"] for g in staged if _matches(g, task=task, signal=signal)})
            if not stages:
                continue
            fig, axes = plt.subplots(len(stages), 2, squeeze=False, figsize=(11, max(3.3, 3.1*len(stages))))
            for row, stage in enumerate(stages):
                for col, side in enumerate(("l", "r")):
                    ax = axes[row, col]
                    conditions = dict(task=task, side=side, signal=signal, stage=stage, mode="psm_stage")
                    _curve_panel(ax, groups, users, conditions)
                    within = [g for g in groups if _matches(g, **{**conditions, "mode": "within_stage"})]
                    if within and _finite(within[0]):
                        ax.plot(within[0]["lag_s"], within[0]["r"], color="tab:orange", linestyle="--", label="Within occurrence")
                        ax.legend(fontsize=7, loc="upper right")
                    _decorate(ax, f"{signal.capitalize()} correlation r")
                    ax.set_title(f"{stage} / PSM {side.upper()}")
            fig.suptitle(f"Task {task}: stages anchored to PSM time")
            fig.tight_layout(rect=(0, 0, 1, .98))
            _save(fig, directory/f"stages_{task}_{signal}.png")
            plt.close(fig)


def _record_figures(plt, directory, records):
    selected = [c for c in records if c.get("mode") == "whole"]
    tasks = _task_order(selected)
    if not tasks:
        return
    users = sorted({c["user"] for c in selected})
    colors = {user: plt.get_cmap("tab20")(i % 20) for i, user in enumerate(users)}
    for signal in sorted({c["signal"] for c in selected}):
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(10, 3.4*len(tasks)))
        for row, task in enumerate(tasks):
            for col, side in enumerate(("l", "r")):
                ax = axes[row, col]
                for curve in selected:
                    if not _matches(curve, task=task, side=side, signal=signal):
                        continue
                    s = _summary(curve)
                    if not np.isfinite(s["peak_lag_s"]):
                        continue
                    color = colors[curve["user"]]
                    ax.scatter(s["peak_lag_s"], s["peak_r"], s=25,
                               facecolors="none" if s["peak_at_boundary"] else color,
                               edgecolors=color, alpha=.8)
                _decorate(ax, "Record peak correlation r")
                ax.set_xlabel("Record peak lag (s); positive = gaze leads")
                ax.set_ylim(-1, 1)
                ax.set_title(f"Task {task} / PSM {side.upper()}")
        fig.suptitle(f"{signal.capitalize()} record peaks; colors = novices, hollow = boundary")
        fig.tight_layout(rect=(0, 0, 1, .96))
        _save(fig, directory/f"record_peaks_{signal}.png")
        plt.close(fig)


def _paired_figures(plt, directory, users):
    selected = [u for u in users if u.get("mode") == "whole" and u.get("task") in ("A", "C")]
    for signal in sorted({u["signal"] for u in selected}):
        fig, axes = plt.subplots(1, 2, squeeze=False, figsize=(9, 4))
        for col, side in enumerate(("l", "r")):
            ax = axes[0, col]
            n = 0
            people = sorted({u["user"] for u in selected})
            for person in people:
                pair = {u["task"]: u for u in selected if _matches(u, user=person, side=side, signal=signal)}
                if set(pair) != {"A", "C"}:
                    continue
                estimates = [_summary(pair[task]) for task in ("A", "C")]
                if any(not np.isfinite(e.get("peak_lag_s", np.nan)) for e in estimates):
                    continue
                n += 1
                color = "tab:orange" if any(e.get("peak_at_boundary") for e in estimates) else "0.45"
                ax.plot([0, 1], [e["peak_lag_s"] for e in estimates], "o-", color=color, alpha=.7, linewidth=.8, markersize=4)
            ax.axhline(0, color="0.4", linestyle=":", linewidth=.8)
            ax.set_xticks([0, 1], ["Task A", "Task C"])
            ax.set_xlim(-.2, 1.2)
            ax.set_ylabel("Novice peak lag (s); positive = gaze leads")
            ax.set_title(f"PSM {side.upper()}: {n} paired novices")
            ax.grid(alpha=.15)
        fig.suptitle(f"{signal.capitalize()} paired novice peaks; orange includes boundary estimates")
        fig.tight_layout(rect=(0, 0, 1, .94))
        _save(fig, directory/f"paired_novices_{signal}.png")
        plt.close(fig)


def _bootstrap_figures(plt, directory, groups):
    selected = [g for g in groups if g.get("mode") == "whole"]
    tasks = _task_order(selected)
    for signal in sorted({g["signal"] for g in selected}):
        selected_signal = [g for g in selected if g["signal"] == signal]
        if not any(np.isfinite(np.asarray(g.get("bootstrap_peak_lag_s", []), dtype=float)).any()
                   for g in selected_signal):
            continue
        fig, axes = plt.subplots(len(tasks), 2, squeeze=False, figsize=(11, 3.2*len(tasks)))
        for row, task in enumerate(tasks):
            for col, side in enumerate(("l", "r")):
                ax = axes[row, col]
                matching = [g for g in selected_signal if _matches(g, task=task, side=side)]
                if matching:
                    group = matching[0]
                    peaks = np.asarray(group.get("bootstrap_peak_lag_s", []), dtype=float)
                    peaks = peaks[np.isfinite(peaks)]
                    if len(peaks):
                        lag = np.asarray(group["lag_s"], dtype=float)
                        step = np.median(np.diff(lag)) if len(lag) > 1 else .1
                        edges = np.linspace(lag.min()-step/2, lag.max()+step/2, min(41, len(lag))+1)
                        ax.hist(peaks, bins=edges, color="tab:blue", alpha=.65)
                        if np.isfinite(group.get("peak_lag_s", np.nan)):
                            ax.axvline(group["peak_lag_s"], color="tab:blue", linestyle="--", label="Observed group peak")
                            ax.legend(fontsize=7, loc="upper right")
                        frequency = group.get("boundary_frequency", np.nan)
                        ax.text(.03, .97, f"Replicates: {len(peaks)}\nBoundary frequency: {frequency:.1%}",
                                transform=ax.transAxes, va="top", fontsize=7,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
                    else:
                        ax.text(.5, .5, "No supported bootstrap estimate", ha="center", va="center", transform=ax.transAxes)
                _decorate(ax, "Bootstrap replicate count")
                ax.set_title(f"Task {task} / PSM {side.upper()}")
        fig.suptitle(f"{signal.capitalize()} group peak-lag bootstrap distribution")
        fig.tight_layout(rect=(0, 0, 1, .96))
        _save(fig, directory/f"bootstrap_peak_lags_{signal}.png")
        plt.close(fig)


def generate_outputs(output, record_curves, user_curves, group_curves, contrasts, qc, make_plots=True):
    """Write auditable CSVs and optional figures, including unsupported results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for label, curves in (("record", record_curves), ("novice", user_curves), ("group", group_curves)):
        _write(output/f"{label}_curves.csv", _long_rows(curves),
               [key for key in META if any(key in c for c in curves)] +
               [key for key in LAG_FIELDS if any(key in c for c in curves)])
        _write(output/f"{label}_summary.csv", (_summary(c) for c in curves))
    _write(output/"contrasts.csv", ({k: v for k, v in c.items() if k != "bootstrap_difference_s"} for c in contrasts))
    _write(output/"quality.csv", qc)
    bootstrap_rows = []
    for group in group_curves:
        meta = {key: group[key] for key in META if key in group}
        for index, value in enumerate(group.get("bootstrap_peak_lag_s", [])):
            bootstrap_rows.append({**meta, "replicate": index, "peak_lag_s": value})
    _write(output/"bootstrap_peaks.csv", bootstrap_rows)
    null_rows = []
    for group in group_curves:
        meta = {key: group[key] for key in META if key in group}
        for index, value in enumerate(group.get("null_peak_r", [])):
            null_rows.append({**meta, "replicate": index, "peak_r": value})
    _write(output/"unpaired_peaks.csv", null_rows)
    contrast_rows = []
    for contrast in contrasts:
        meta = {key: value for key, value in contrast.items()
                if key not in ("bootstrap_difference_s", "stable_users") and not isinstance(value, np.ndarray)}
        for index, value in enumerate(contrast.get("bootstrap_difference_s", [])):
            contrast_rows.append({**meta, "replicate": index, "difference_s": value})
    _write(output/"bootstrap_contrasts.csv", contrast_rows)
    if make_plots:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        directory = output/"figures"
        directory.mkdir(exist_ok=True)
        _whole_figures(plt, directory, group_curves, user_curves)
        _spatial_figures(plt, directory, group_curves)
        _stage_figures(plt, directory, group_curves, user_curves)
        _record_figures(plt, directory, record_curves)
        _paired_figures(plt, directory, user_curves)
        _bootstrap_figures(plt, directory, group_curves)
