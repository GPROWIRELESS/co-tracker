# Gaze versus PSM position cross-correlation

中文详细说明：[代码、公式和全部输出字段](../docs/gaze_psm_code_and_formulas_zh.md)；
[2026-10-30 结果复核报告](../docs/gaze_psm_results_audit_zh.md)。
[图片可读性与结论图选择](../docs/gaze_psm_figures_guide_zh.md)。

The primary analysis estimates whether gaze positions precede corresponding
left/right PSM mask-centroid positions. It uses normalized 2-D **position xcorr**,
spatial-distance checks, and novice-level inference. Velocity xcorr is secondary.
Dataset files are read only.

Run from the repository root with Python, NumPy, SciPy, OpenCV, and Matplotlib:

```bash
python tools/gaze_psm_xcorr.py
```

To parallelize extraction and correlation on a multicore machine:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/gaze_psm_xcorr.py \
  --within-stage --workers 8
```

The random pairings are drawn in the parent process and results retain their
recording/replicate order. Changing `--workers` does not change statistical
weights, randomization, or bootstrap results.

The default selection is novice Task A/C records in `Index/content_index.csv`:
33 Task A records and 51 Task C records from the same 11 novices. Outputs go to
`outputs/gaze_psm_xcorr`. A complete run decodes both mask videos for each record;
bootstrap and shuffled controls add analysis time. OpenCV image operations use
one thread by default to limit per-frame overhead; `--opencv-threads` overrides it.
Shuffled replicates are processed sequentially so their recording arrays do not
accumulate in memory.

A partial-record smoke test is:

```bash
python tools/gaze_psm_xcorr.py \
  --records User1_1_1 --max-frames 600 \
  --bootstrap 20 --shuffles 0 --output /tmp/gaze_psm_smoke
```

One recording cannot support group inference with the default minimum novice
count. Its recording curves and extraction checks are still useful. Partial
runs are not full-record results and must not be pooled with the main analysis.
Use separate output directories for smoke tests and sensitivity analyses.

## What the result means

At positive lag k, the comparison is `gaze[t]` versus `PSM[t+k]`.

- Positive peak lag: gaze positions precede corresponding PSM positions.
- Negative peak lag: gaze positions follow PSM positions.
- Near zero: approximately synchronous at the available timing resolution.

The lag search is the same +/-2 seconds for all conditions. The peak is the
largest **signed** normalized correlation, not the largest absolute correlation.
The implementation pools x/y covariance and variance after centering each
coordinate separately. Correlation is normalized using valid overlapping pairs
at each lag. A weak, broad, multimodal, or boundary peak cannot support a precise
lead estimate. The reported shift is a time delay, not a frequency-specific
phase angle or proof of causality.

Position correlation can be high despite a fixed spatial offset. The analysis
therefore also measures actual Euclidean distances in the original coordinates,
median distance, and the fraction of pairs within `--spatial-tolerance`. The
50-pixel default is exploratory: calibrate it using gaze accuracy and instrument
scale, then check sensitivity. Distances are reported over lag and at the same
group peak and zero lag; they are not optimized separately for every recording.
The group distance curve averages recording median distances within each novice,
then averages novices. It is not a median over all pooled samples.

The supplied gaze is raw gaze, not classified fixations. Calling the result a
fixation result requires separately validated fixation labels. PSM position is
the mask centroid, not a tool tip. Xcorr estimates typical trajectory alignment,
not the exact first arrival of each individual fixation or action.

## Existing pixel gaze coordinates

For pixel-valued gaze columns in the allocation CSV, use:

```bash
python tools/gaze_psm_xcorr.py \
  --gaze-source allocation --gaze-units pixels \
  --gaze-x-column gaze_x --gaze-y-column gaze_y
```

Use this only when those columns really contain pixels. Some allocation files
contain normalized values despite the `gaze_x/gaze_y` names; select
`--gaze-units normalized` for those. Default input is raw normalized gaze.
`--gaze-valid-values` configures accepted raw `Status` or allocation `valid` codes;
the default accepts code 2, including numeric spellings `2.0` and `2e0`.
Invalid codes such as `0` are still excluded. Column names can be overridden for
either source. Records with no valid gaze or no valid gaze overlapping the video
are reported as errors, rather than successful records with empty gaze arrays.

## Alignment and extraction

For each record, the pipeline reads:

```text
Active_Data/<user>/csv/<id>_video_ts.csv
Active_Data/<user>/csv/<id>_gaze.csv
Active_Data/<user>/npy/<id>_gaze.csv
Active_Data/<user>/sam/<id>/mask/obj_psm_l.mp4
Active_Data/<user>/sam/<id>/mask/obj_psm_r.mp4
```

Original integer nanosecond timestamps are differenced before conversion to
seconds. Scientific notation is parsed through exact decimal arithmetic, without
an intermediate float that could lose nanosecond precision. Nonfinite and
fractional-nanosecond values are rejected. Raw gaze uses `Capture Unix timestamp`
and `norm_x/norm_y`, scaled to the
mask image dimensions. Only configured valid gaze statuses and coordinates in
[0,1] are accepted. Stage labels are matched by `Frame_Number`; rounded stage-CSV
timestamps are not used. Deleted video timestamp rows are handled consistently.
An ambiguous decoded-mask/timestamp alignment fails the record rather than
silently truncating it.

Each grayscale mask is thresholded at 127, and image moments give the average
foreground pixel location. Areas smaller than 10 pixels are missing. All
foreground components contribute to the centroid. Left/right refers to the
instrument, not camera eyes.

The default `--hz auto` uses the rounded median observed video rate across the
selected records; it does not assume 25 fps. `--hz` can override it. Upsampling cannot improve acquisition timing precision.
Interpolation does not bridge long gaps. Smoothing is symmetric and confined to
valid stretches. Changes in mask shape, occlusion, camera motion, or gaze/video
registration can produce apparent movement; inspect representative overlays
before interpreting results. `--flip-gaze-y` is available if gaze Y increases
upward; default image coordinates increase downward.

## Stage analysis

The main stage analysis is anchored to **PSM time**:

```text
corr(gaze[u-lag], PSM[u] | stage(PSM[u]) == selected_stage)
```

Gaze can therefore occur in an earlier stage. Looking toward a placement location
during transfer can contribute to the subsequent PSM place stage.

Contiguous occurrences are not concatenated. At every lag, the code centers each
occurrence's valid overlap and sums centered cross-products and squared
variations before normalizing once. Short occurrences contribute their valid
pairs; the minimum overlap applies to the pooled curve, not to each occurrence.
The default pooled stage minimum is 2 seconds, versus 5 seconds for whole tasks.
Pair counts and contributing occurrences are exported for every lag.

`--within-stage` adds a sensitivity analysis requiring both samples to belong to
the same contiguous occurrence. This helps identify how boundary-crossing pairs
change the result. The lag range remains identical. Stage curves with too little
pooled support or movement variation are missing rather than zero correlation.
Sparse stages, especially Task C threading, should be treated as exploratory
when few novices contribute.

## Statistical unit and controls

The novice is the independent analysis unit:

1. Obtain recording curves, pooling occurrences within recordings for stages.
2. Fisher-z average available recording curves equally within each novice.
3. Fisher-z average novice curves equally across novices.
4. Locate the peak of the group curve, rather than averaging recording peaks.

The primary group estimate uses the same supported novice set throughout the lag
window; all-available curves and contributor counts are also saved as diagnostics.
The recordings contributing within a novice can still change across lag. The
`record_support_changes` flag identifies this. `r_fixed_records` uses only attempts
supported throughout the full lag window, and `r_primary_fixed_record_users`
repeats the primary averaging on that sensitivity curve's novice subset. Compare
those two to isolate recording-composition effects from changes in the novice
subset. Stable recording IDs/counts and the fixed-record novice set are exported.
In `group_curves.csv`, `n_users` is the **all-available** contributor count at that
lag; `n_stable_users` is the constant novice count supporting the primary `r`.
`n_fixed_record_users` gives the fixed-record sensitivity count. These counts
can differ and should not be substituted for one another.
When recording support changes and a fixed-record curve is available, figures
show it as an orange dashed sensitivity curve with its novice count.
Bootstrap replicates resample novices and rebuild curves and peaks. Bootstrap
bands are pointwise, not simultaneous bands. With only 11 novices, intervals can
be wide or multimodal. Inspect the exported bootstrap peak distribution and
boundary frequency instead of treating an interval alone as evidence of a lead.

Unpaired controls combine gaze from a different novice with PSM from the same
task. The pipeline repeats the lag search and hierarchical aggregation, producing
both a curve envelope and a distribution of maximum correlations. This is an
empirical **unpaired reference**, not a universal chance level or causal test.
Inspect its support and interpretation when task structure or recording lengths
differ. `--shuffles 0` disables it for fast diagnostics.

A/C contrasts resample the shared novice identities jointly. Stage contrasts are
within task and use their available shared novices; differently named stages
across A/C are not treated as equivalent. Unsupported comparisons are explicitly
flagged. Many stage contrasts are exploratory; formal multiple-comparison claims
require a prespecified testing family and correction.

## Output files

| File | Contents |
|---|---|
| `record_curves.csv` | Recording curves, spatial metrics, and support at each lag |
| `record_summary.csv` | Recording peaks, quality status, and boundary flags |
| `novice_curves.csv` | Within-novice averages and recording support |
| `novice_summary.csv` | Novice-level peak estimates |
| `group_curves.csv` | Primary and all-available curves, bands, spatial metrics, counts |
| `group_summary.csv` | Group peaks, stable novices, uncertainty, spatial checks, reference comparison |
| `contrasts.csv` | Paired task and within-task stage comparisons |
| `bootstrap_peaks.csv` | Full bootstrap group peak-lag distributions |
| `bootstrap_contrasts.csv` | Bootstrap paired-difference distributions |
| `unpaired_peaks.csv` | Shuffled maximum-correlation distributions |
| `quality.csv` | Extraction/coverage and recording quality information |
| `config.json` | Run settings and reproducibility information |
| `results.json` | Full recording, novice, group, and contrast data |
| `quality.json` | Structured extraction and quality diagnostics |
| `cache/<id>.npz` | Fingerprinted aligned trajectories and native centroids |
| `figures/` | Whole-task/stage curves, spatial checks, record peaks, paired novice plots, bootstrap peak-lag histograms |

Main correlation figures show equal-novice group curves, pointwise bootstrap
bands, and unpaired envelopes. Whole-task panels share a data-derived y range
per signal; stage panels share a y range within each stage row. Individual
curves are retained separately in `novice_curves_*.png`. Proximity axes use
percentages; numerical tables retain fractions. Boundary record peaks use hollow
markers. Paired novice plots include boundary estimates in orange for visibility;
these are diagnostic, not reliable delays. Insufficient-data panels remain
explicitly labeled instead of being filled with zero estimates. Bootstrap
peak-lag histograms expose broad or multimodal timing estimates. Every figure
from a run marked `partial_record` in `config.json` is watermarked
`PARTIAL RECORD / SMOKE TEST`.

The readability revision adds `conclusion_position.png` for group peak intervals,
paired task contrasts and spatial proximity, `stage_peak_lags_position.png` for
exploratory stage intervals, and `spatial_changes.png` for changes from zero lag.
PNG and vector PDF versions are saved together. Replot saved results without
the original NAS inputs or rewriting statistical tables using:

```bash
python tools/replot_gaze_psm_outputs.py --output outputs/gaze_psm_xcorr
```

## Reproducibility and sensitivity checks

Keep the primary configuration fixed before examining group peaks. Repeat the
analysis in separate output directories using reasonable smoothing durations,
spatial tolerances, movement-variation thresholds, and overlap criteria. Check
whether the primary delay is stable and whether spatial proximity improves at
that delay compared with zero lag. Do not interpret only the most favorable
configuration.

The extraction and analysis phases can be run separately with `--phase extract`
and `--phase analyze`. Cache reuse is enabled by default; `--no-reuse-cache` forces
re-extraction. `--phase analyze` requires compatible cached trajectories in the
same output directory, selected records, and matching extraction arguments.
Changing the selected records can change the auto-selected rate, so use an
explicit `--hz` when sharing caches across subsets. Cache fingerprints include
input sizes/modification times and extraction settings. `--no-plots` exports tables without importing
Matplotlib. Use `--help` for the full parameter list.

Cache format version 4 invalidates version 3 gaze trajectories affected by the
timestamp/status parsing bugs. When all source fingerprints, extraction settings,
native frame numbers, and native times match, native mask centroids/areas from a
version 3 cache can be reused. Gaze, alignment, smoothing, and quality statistics
are regenerated; the old gaze arrays are never reused during migration.
