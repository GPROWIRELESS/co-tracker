# DLC versions of gaze/PSM and score correlation scripts

These scripts copy the existing analyses and replace mask centroids with the
`left_psm` and `right_psm` x/y predictions from
`Active_Data/User*/sam/<clip>/<clip>_DLC.csv`. Each instrument is analyzed
separately; positions are not averaged between instruments. The three DLC
header rows identify the columns. Likelihoods are saved without a cutoff.

The default selection remains novice tasks A and C. Read-only inspection found
all 84 CSVs (33 task A, 51 task C), all with the same column schema and one row
per retained video frame. DLC indices are zero-based retained-frame positions,
not the original `Frame_Number` values. The loader validates this alignment.

The source left-camera video supplies width/height metadata for gaze-to-pixel
conversion. Both PSM predictions occupy that image coordinate system. Video
frames and mask videos are not decoded. Mask-area QC is omitted because DLC
provides point predictions; the old mask options remain accepted but unused.

Gaze handling, time alignment, interpolation, smoothing, position/velocity
correlations, peak selection, stage rules, bootstrap, unpaired controls, score
matching, inference, and plotting retain the existing implementations.
The DLC pipeline has its own cache fingerprint/version and default output
directories ending in `_dlc`.

## Score inputs

Performance scores are supplied by `Index/Task_A_Score.csv` and
`Index/Task_C_Score.csv`. No new performance-score calculation or score file
is needed for DLC. The score-correlation scripts match those existing scores
to the newly calculated DLC peaks; they do not derive performance scores from
PSM coordinates. The original score scripts also accept explicit `--input`
and `--output` paths; the DLC copies provide convenient DLC defaults.

## Run commands

Run from the repository root, in this order. For the DLC calculation requested
on October 31, 2026, the existing analysis settings are retained: 41 Hz inferred
from the timestamps, both signals and stage modes, 1,000 gaze/PSM bootstrap
draws and 100 shuffled controls; score associations use 10,000 participant
bootstrap draws and 19,999 permutations with at least six qualifying users.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python tools/gaze_psm_xcorr_dlc.py --both-in-stage --workers 8
python tools/gaze_psm_xcorr_max_dlc.py
python tools/analyze_score_corr_dlc.py
python tools/analyze_component_score_corr_dlc.py
python tools/analyze_raw_score_corr_dlc.py
python tools/analyze_velocity_score_corr_dlc.py
python tools/analyze_score_corr_max_dlc.py
python tools/analyze_penalty_score_corr_dlc.py
```

The component/raw/velocity scripts supply the existing Score, Time, Eco,
TimeRaw, and EcoRaw analyses for `score_corr_dlc`; the max score script includes
all five fields for `score_corr_max_dlc`. The penalty script adds PenaltySum and
PenaltyRawSum to both DLC score directories, preserving the existing workflow.
All scripts accept their original command-line options (`--help` lists them).
Shared plotting/layout helpers still accept explicit DLC input/output paths.

The four output roots are `outputs/gaze_psm_xcorr_dlc`,
`outputs/gaze_psm_xcorr_max_dlc`, `outputs/score_corr_dlc`, and
`outputs/score_corr_max_dlc`. Both score roots provide matching
`metrics/<position|velocity>/<outcome>/` CSV tables and
`plots/<position|velocity>/<outcome>/` figures, with `peak_r`, `signed_lag_s`,
and `absolute_lag_s` predictor subfolders. Their `README.md` files link to
each result set. Logs and the final run audit are saved under `logs/dlc_*`
and `logs/*dlc*.log`.

All 84 recordings are included in the gaze/PSM calculations. Total Score is
available for 83: `User8_7_2` has no score and is marked `NotAttempted`.
Time, Eco, TimeRaw, EcoRaw, PenaltySum, and PenaltyRawSum each have 66 usable
values before phase-specific exclusions. Missing scores are not imputed.

Input tests inspect CSV loading, frame alignment, and unchanged analysis
function bodies without executing metrics or creating analysis outputs:

```bash
python -m unittest discover -s tests -p 'test_gaze_psm_dlc.py'
```
