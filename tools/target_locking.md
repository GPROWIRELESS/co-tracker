Target-locking annotation and tracking use two separate scripts. Both default to novice active Task A only, one to five points per stage, and the dataset root `/mnt/nas/Users/yxl3527/Dataset/dvss`.

1. Manually annotate all stages on a desktop with a display:

   ```bash
   python tools/label_target_locking.py
   ```

   The UI shows the first frame on the left and the last frame on the right for each consecutive run of the `stage` column (`event` is also supported). Choose **First frame (left)** or **Last frame (right)** above the images, then click one to five target pixels in a consistent order on the selected green view. The other view is reference-only and ignores clicks. Switching views discards unsaved clicks and loads any saved manual points on the chosen endpoint. Right Arrow or Enter saves only these manual points, then opens the next stage. No model or PyTorch is loaded by the UI. Right click or Backspace undoes a point; R clears points; Left Arrow revisits the previous stage in the current recording; N opens the next video; P opens the previous video; Esc exits. N/P also accept uppercase keys, open the video at its first stage, and discard unsaved clicks. At the first or last video, P or N respectively does nothing. Navigation backwards or exiting discards unsaved clicks. After the last stage the next recording opens. Every launch opens each recording from its first stage, including recordings that are fully labeled. Saved points are loaded for editing. Manual points are stored on the corresponding first or last CSV row, marked `manual`. Each stage has one seed endpoint; saving replaces the other endpoint and clears old predictions.

   To replace saved points, press R to clear them, click replacement points, then Right Arrow or Enter to save. Saving clears that stage's old tracking predictions, so rerun tracking afterward. No additional command-line option is needed.

2. Check all selected inputs and manual seeds before tracking:

   ```bash
   python tools/track_target_locking.py --check-only
   ```

3. Propagate the completed annotations with CoTracker:

   ```bash
   python tools/track_target_locking.py
   ```

   Run in the CUDA-enabled `dvss-pytorch` environment. The script validates every selected recording and stage seed before loading the model or writing files. Each stage starts with a fresh online tracking state. First-frame seeds track forward; last-frame seeds track through reversed video frames, and predictions map back to the original chronological CSV rows. Tracking stays within the stage. It uses the notebook's support grid and overlapping `2 * model.step` windows, keeping only decoded windows in CPU memory. It preserves exact manual seed coordinates and saves the other frames’ predicted coordinates into that same gaze CSV. Already completed stages are skipped; `--overwrite` recomputes them. Each stage is saved immediately so interrupted runs can resume. A one-frame stage needs no inference. No visualization video is generated.

Common options for both scripts:

```bash
python tools/label_target_locking.py \
  --items Index/items/novice_active_taskA \
  --data-root /mnt/nas/Users/yxl3527/Dataset/dvss \
  --points 5
```

The dataset root is the **dvss directory**, containing `Active_Data`. `/dvss/Active_Data/...`, `dvss/Active_Data/...`, `Active_Data/...`, and legacy absolute paths containing `Active_Data` all resolve under this root. Gaze CSV files resolve as `Active_Data/UserN/npy/RECORD_gaze.csv`. Index JSON directories or individual list files are accepted; summaries are ignored and repeated videos are deduplicated. All active entries in the supplied lists are selected, regardless of task or skill level; use `--items` to choose a different group, such as `Index/items/novice_active_taskC`. Entries without a domain are treated as active. `python tools/label_target_locking.py --list` prints resolved local video paths without opening the UI.

Coordinates are original-resolution pixels with origin at the top left. Existing CSV columns are preserved. Added columns are `point_1_x`, `point_1_y`, through `point_5_x`, `point_5_y`, plus `point_label_mode` (`manual` on the selected endpoint, `tracked` on propagated rows). Before tracking, all other stage rows remain blank. `point_n_visible` columns are neither created nor required; existing visibility columns are removed whenever the CSV is saved. No dataset files are migrated just by opening the UI. Relabeling a stage clears that stage's old predictions so the batch script regenerates them. The first save creates `.csv.before_point_labels.bak`; later saves preserve this original backup. Saves replace the CSV atomically and normalize CSV formatting. Avoid editing these CSV files in another application during either operation.

A stage is a consecutive run of an event, so later repetitions remain separate stages. Empty, `none`, and `nan` events are skipped unless `--include-none` is passed to both scripts. Video frames are `Frame_Number` minus the first CSV frame number, matching the inspected User1_1_1 recording. If a recording uses another origin, set `--frame-offset NUMBER` consistently in both scripts (`0` for absolute video indices). Bounds and strictly increasing frame numbers are checked; gaps are allowed.

Tracking options: `--checkpoint checkpoints/scaled_online.pth`, `--device cuda` (defaults), or `--device cpu`. The UI needs OpenCV, Pillow, and Tkinter. Only the batch script requires PyTorch and CoTracker dependencies. The former `label_stages.py` and `--no-track` option have been replaced by the manual-only `label_target_locking.py`.

Verification:

```bash
python -m unittest discover -s tests -p test_target_locking.py -v
```

Each stage accepts **1–5 points**. Press Right Arrow or Enter after any valid count; five is the maximum, not a required count. The tracker automatically reads the number of complete coordinate pairs in that stage’s manual seed. Unused point columns remain blank. `--points` optionally lowers the maximum (range 1–5). Relabeling with fewer points clears old unused coordinates.

The UI lists all stage types found across the novice Task A recordings, each with a checkbox in **Skip stage types throughout the labeling queue**. Check types such as `reach ring` or `grab ring` once to skip every occurrence in this and following recordings. Press Right Arrow or Enter to save the choices with the current annotation and advance; selected stage types are passed automatically without point clicks. Their rows are recorded as `point_label_mode=skipped` and coordinates are cleared. The batch tracker ignores these stages, including with `--overwrite`.

Choices remain active across recordings during the current UI session. Skipped decisions already saved in CSV persist across restarts; checkbox selections themselves are session-only. Unchecking a type enables future annotation for that type. Use Left Arrow to revisit a previously skipped occurrence in the current recording, then label and save to restore it. Stage types excluded by the default `none` filter appear when using `--include-none`.

At startup the UI validates allocation CSVs and reports/excludes recordings with missing files, missing `event`/`Frame_Number` columns, or empty data. Excluded recordings remain unmodified; they need stage annotations before they can be labeled. UTF-8 BOM headers are supported. This exclusion is separate from intentionally skipping a stage type. The batch tracker still validates the entire supplied queue and reports missing stage allocations instead of silently omitting them.

Tracked points with false model visibility, nonfinite coordinates, or coordinates outside the image are saved as blank x/y pairs. The CSV row remains `tracked`, so these dropped points do not cause repeated tracking on resume. Manual seed points are preserved. Use `--overwrite` to apply this filtering to previously completed stages.
