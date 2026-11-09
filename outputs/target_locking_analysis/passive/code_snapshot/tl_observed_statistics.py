"""Unchanged v3 equal-recording then equal-participant summaries.

content_id MUST be the unique viewing-record key, participant the observer.
Source-video identifiers are separate provenance fields.
"""
import numpy as np
import pandas as pd
from tl_observed_bootstrap import ParticipantBootstrap

GROUP = ["scope", "scenario", "task", "stage", "source"]
PERSON = GROUP + ["participant"]
OCC = ["L_T", "L_P", "L_O", "L_N", "L_target", "L_gripper", "D", "C",
       "C_U", "C_excess", "E_target", "E_gripper", "B", "B_excess"]
AREA = ["area_T", "area_P", "area_O", "area_N"]
TIME = ["direct_rate", "via_rate", "dwell_T_ms", "dwell_P_ms", "dwell_O_ms"]

def present(columns, df):
    return [column for column in columns if column in df]


def group_items(df, keys):
    for key, part in df.groupby(keys, sort=True, observed=True, dropna=False):
        yield dict(zip(keys, key if isinstance(key, tuple) else (key,))), part


def participant_summary(trials, boot):
    metrics = present(OCC + AREA + TIME, trials)
    # A missing preference denominator does not invalidate the observed
    # occupancy or area for that trial. Keep per-metric support counts.
    grouped = trials.groupby(PERSON, dropna=False, sort=True, observed=True)
    means = grouped[metrics].mean()
    valid_rows = trials.copy()
    if "valid_n" in trials:
        valid_rows = valid_rows[valid_rows.valid_n.fillna(0) > 0]
    n_valid = valid_rows.groupby(PERSON, dropna=False, observed=True).content_id.nunique()
    means["trials_total"] = grouped.content_id.nunique()
    means["trials_valid"] = n_valid.reindex(means.index, fill_value=0)
    if "valid_n" in trials:
        means["valid_n"] = grouped.valid_n.sum()
    if "valid_s" in trials:
        means["valid_s"] = grouped.valid_s.sum()
    means = means.join(grouped[metrics].count().add_prefix("n_trials_")).reset_index()
    rows = []
    for key, g in group_items(means, GROUP):
        for m in metrics:
            v = g[m].dropna().to_numpy(float)
            rows.append({**key, "metric": m, "n_participants": len(v),
                         "n_trials": int(g["n_trials_" + m].sum()),
                         "mean": np.mean(v) if len(v) else np.nan, **boot.mean(v)})
    return means, pd.DataFrame(rows)

