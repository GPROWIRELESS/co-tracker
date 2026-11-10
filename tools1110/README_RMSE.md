# RMSE analysis

All four arrange scripts call `RMSE.rmse` through the shared loading/grouping
module `rmse_arrange_common.py`. NumPy is the only third-party dependency.

```text
RMSE = sqrt(mean((dx**2 + dy**2) / 2))
```

The squared x/y errors are averaged over both dimensions (denominator 2*N).
Group results pool original coordinate pairs;
they do not average record-level RMSEs. Each metric has its own paired validity
mask. Filtering never shifts one coordinate stream relative to another.

## Run

From the tools1110 directory:

```powershell
python record_level_arrange.py
python task_level_arrange.py
python user_stage_arrange.py
python all_stage_arrange.py
```

Defaults:

- Index: `C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\dvss_pytorch\Index\index_novice_active.json`
- Dataset root: `D:\Dataset\dvss`
- Include only index `meta.task_name` A/C. Current index: A=33, C=51.
- Gaze path comes from `gaze_npy_path`, relative to the dataset root.
- DLC path: `<user>/sam/<record_id>/<record_id>_DLC.csv`.
- Output files are saved next to the scripts, irrespective of working directory.

All commands accept `--index`, `--data-root`, and `--output` overrides.

## Confirmed data rules

- Match NPY and DLC by original row position. Unequal lengths cause an error;
  no truncation, interpolation, resampling, lag shifting, or smoothing.
- Gaze: normalized x/y multiplied by 1280/1024; keep finite coordinates strictly
  inside the screen. The NPY `valid` field is not used.
- DLC: first x/y/likelihood triple is left PSM, second is right. Require finite
  coordinates strictly inside the screen and finite likelihood >=0.2.
- Screen interior: `0 < x < 1280` and `0 < y < 1024`.
- Target: filter each point_1..5 separately, then average the remaining points
  per frame. No remaining point means no point-based target for that frame.
- A valid per-frame `purple_string_knot_center` takes precedence over the point
  average. An empty/absent center falls back to the point average. A nonempty
  malformed center causes an error; a nonfinite/out-of-screen center is rejected.
- Current A/C target points and knot centers were audited as pixels. For
  normalized target input, explicitly pass `--point-units normalized` and/or
  `--center-units normalized`. Pixel defaults avoid guessing units from small
  coordinate values. Conversion occurs before screen filtering and averaging.
- Record/task analyses include every row, independent of stage labels.
- Stage analyses include only: grab ring, transfer ring, place ring, reach ring,
  reach string, wrap around psm, pull through and cinch, threading string through
  tower. `transport ring` maps to `transfer ring`. Case/whitespace are normalized.
- Repeated occurrences of the same stage in a record are pooled together.
- Empty, none and drop ring labels are excluded only from stage analysis.
  Unexpected stage labels stop stage analysis for explicit review.

## Outputs

| Script | CSV | Row definition |
| --- | --- | --- |
| record_level_arrange.py | record_level.csv | One per record |
| task_level_arrange.py | task_level.csv | One per task, all users/records pooled |
| user_stage_arrange.py | user_stage_arrange.csv | One per record and present whitelisted stage |
| all_stage_arrange.py | all_stage_arrange.csv | One per task and present whitelisted stage, all users/records pooled |

Each CSV contains `gaze_psm_l_rmse_px`, `gaze_psm_r_rmse_px`, and
`gaze_target_rmse_px`, plus independent `*_n_pairs` and `*_valid_seconds` columns.
`n_frames`/`duration_seconds` describe the full grouping scope before coordinate
filtering. Durations use the user-specified 60 Hz sample rate.

`record_level.csv` and `user_stage_arrange.csv` append every field from the
record's complete index `meta` object at the right, using the original field
names and first-seen JSON key order. Keys appearing in only some records are
still included as columns; absent/null values are blank. Existing `user` and
`task_name` columns move into this right-hand metadata block without duplicate
headers. The final two columns are absolute `gaze_npy_path` and `dlc_csv_path`.
Structured metadata values are serialized as JSON cells. Metadata keys that
collide with computed output names cause an explicit error rather than overwrite
metrics. Task/all-stage aggregate CSVs do not append record-specific metadata.

No valid pairs produces an empty RMSE cell and n_pairs=0, rather than a zero
error. An existing stage is retained even if one metric has no valid pairs.

Each command also writes a matching `.audit.json` containing source paths,
settings, skipped B/D records, raw scope frame counts, canonical stage counts,
source validity counts, rejected finite target point counts, and knot-center
usage. CSV output is written only after all records are successfully computed.

## Checks

```powershell
python -m unittest test_RMSE test_rmse_arrange -v
```

Tests cover the exact formula, frame pooling, target averaging/precedence,
normalized target conversion, independent pairing, likelihood boundaries,
record-wide stage inclusion, repeated-stage merging, aliases, and row mismatch.
