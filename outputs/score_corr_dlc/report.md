# Peak gaze/PSM timing and performance scores

Analysis date: 2026-10-31T01:29:52-04:00. All analyses are exploratory.

## Main finding

No signed peak lead/lag association with performance survives the specified multiple-comparison correction. The data do not establish that a larger gaze lead improves performance.

Across the three primary analysis methods and all three predictors, 0 estimates have BH q < 0.05. Correction is performed separately for each method, pooling its valid whole-task and phase tests and all three predictors; methods and sensitivity families are not jointly corrected.

Of 19 primary between-participant signed-lag tests, 1 reach uncorrected p < 0.05 (smallest p = 0.006). Task A transfer/right: r = -0.784, 95% CI [-1.000, -0.306], p = 0.006, q = 0.1625; transfer/left has r = 0.000. Whole-task signed-lag correlations range from -0.239 to 0.296.

Selected correlation-strength estimates: in the position analysis, Task C wrap/right has r = 0.563, 95% CI [-0.123, 0.898], p = 0.076, q = 0.4921, and Task A whole-task/left has r = 0.664, 95% CI [0.183, 0.904], p = 0.030, q = 0.3454. In the separate secondary velocity analysis, Task A grab/right strength between participants has r = 0.400, 95% CI [-0.317, 0.972], p = 0.226, q = 0.7120; grab/left strength within participants has r = 0.385, 95% CI [-0.081, 0.719], p = 0.168, q = 0.6468, and after order adjustment r = 0.191, 95% CI [-0.121, 0.498], p = 0.426, q = 0.8018. These are strength results and do not establish a beneficial lead/lag.

## Coverage and matching

The saved correlation run contains 84 recordings from 11 novices. Exact UserID + TaskNum + Attempt matching yields 83 usable scores. Task letter, participant type and index metadata are also checked. Expert scores have no corresponding curves in this novice-only run and are not analyzed.

| Task | Recordings | Usable scores | Score = 0 | Failed, scored zero | Completed |
|---|---:|---:|---:|---:|---:|
| A | 33 | 33 | 7 | 0 | 33 |
| C | 51 | 50 | 35 | 16 | 34 |

- Excluded `User8_7_2`: score status `NotAttempted`; missing_score;unscored_attempt_status. Missing scores are never replaced with zero.

The supplied failed-attempt zeros are retained in the primary analysis. Completed-only sensitivity removes these failures. `User6_1_1` has a supplied Completed score of zero but missing score components; its total score is retained as supplied. The score is for an entire attempt, not a phase-specific performance rating.

## Validity rules

1. Use the existing full-record correlation run (41 Hz, ±2 s, 80 ms smoothing). Position is primary; velocity is a separate secondary analysis. Positive lag means gaze precedes PSM; negative means gaze follows PSM. Lag is a time shift, not a frequency phase angle.
2. Admit a task/side/phase/mode condition only if its saved group status is `ok`, its peak correlation is positive, and at least three stable participants support it. Drop-ring conditions and group peaks at either search limit are excluded before looking at score associations.
3. A recording must have a usable score, quality status ok, an interior positive correlation peak, and finite correlation across the entire lag grid. This avoids estimating a peak from a partially supported search window. Selection may itself depend on skill, so exclusion counts are provided.
4. Require at least 6 participants for between-participant tests, and that many participants with both predictor and score variation for within-participant tests. A valid computational peak does not imply a precise individual delay: broad curves, weak correlations and near-boundary bootstrap intervals remain limitations.

| Phase / side | Group status | Eligible recordings / users | Median peak gain over zero |
|---|---|---:|---:|
| A: Reach Ring (L) | ok | 27 / 11 | 0.0315 |
| A: Reach Ring (R) | ok | 33 / 11 | 0.0171 |
| A: Grab Ring (L) | ok | 25 / 11 | 0.0842 |
| A: Grab Ring (R) | ok | 28 / 11 | 0.0498 |
| A: Transfer Ring (L) | ok | 31 / 11 | 0.0450 |
| A: Transfer Ring (R) | ok | 31 / 11 | 0.0890 |
| A: Place Ring (L) | ok | 30 / 11 | 0.0862 |
| A: Place Ring (R) | ok | 31 / 11 | 0.0496 |
| A: Drop Ring (L) | no_valid_curve | 0 / 0 | NA |
| A: Drop Ring (R) | no_valid_curve | 0 / 0 | NA |
| C: Reach String (L) | ok | 48 / 11 | 0.0201 |
| C: Reach String (R) | ok | 46 / 11 | 0.0323 |
| C: Threading (L) | ok | 16 / 7 | 0.1676 |
| C: Threading (R) | ok | 16 / 8 | 0.2117 |
| C: Wrap around PSM (L) | ok | 36 / 11 | 0.0297 |
| C: Wrap around PSM (R) | ok | 37 / 11 | 0.0167 |
| C: Pull/cinch (L) | ok | 18 / 10 | 0.0639 |
| C: Pull/cinch (R) | boundary_peak | 0 / 0 | NA |

## How the association was estimated

- **Between participants:** for each task/side/phase, take each participant's median recording peak lag, median absolute lag, and Fisher-z mean recording peak r. Relate each feature to the mean score of those same eligible recordings using Spearman correlation. This asks whether higher-scoring participants have different timing. These medians summarize recording peaks; they are not peaks of participant-averaged correlation curves from `novice_summary.csv`.
- **Within participant:** globally rank recording features and scores within each condition (average ranks for ties), subtract each participant's mean ranks, and compute a weighted correlation. Each participant has total weight one. This asks whether a participant's better attempts have different timing. The adjusted version additionally removes linear effects of TaskNum (session order in the index) and Attempt number from both rank variables, using the same weights. It is a rank-linear partial correlation, not ordinary Spearman correlation; the order adjustment is a sensitivity check, not full control of learning.
- **Uncertainty:** 10,000 whole-participant bootstrap resamples; percentile 95% intervals. Between-participant p values use 19,999 random score permutations across participants, two-sided with the +1 correction. Within-participant p values use a participant-cluster CR1 sandwich and t(G−1), with absorbed participant intercepts counted in the finite-sample correction. With only 6–11 contributing participants these p values and bootstrap intervals are approximate and can disagree. Whole-participant resampling preserves the repeated-attempt structure; within-participant bootstrap retains the original rank scores. No individual time samples are treated as independent observations.
- **Multiple comparisons:** BH correction includes signed lag, absolute lag and strength, both sides, both tasks, and all admitted whole-task/phase conditions, separately for each scenario and method. Tests with insufficient variation are omitted from that family's denominator and explicitly labeled in the CSV. The primary phase analysis is anchored to PSM time; strict same-occurrence pairing is a sensitivity analysis. `stronger_peaks` additionally requires recording peak r ≥ 0.2; this exploratory cutoff is not a validated reliability threshold.

## Signed timing: participant-level results

| Condition | Predictor | N recordings / users | Association [95% CI] | p | BH q |
|---|---|---:|---:|---:|---:|
| A: Transfer Ring (R) | signed_lag_s | 31 / 11 | -0.784 [-1.000, -0.306] | 0.006 | 0.162 |
| C: Wrap around PSM (L) | signed_lag_s | 36 / 11 | -0.405 [-0.841, 0.236] | 0.217 | 0.774 |
| C: Pull/cinch (L) | signed_lag_s | 18 / 10 | -0.356 [-0.981, 0.461] | 0.313 | 0.851 |
| A: Whole task (R) | signed_lag_s | 33 / 11 | 0.296 [-0.423, 0.856] | 0.378 | 0.851 |
| C: Wrap around PSM (R) | signed_lag_s | 37 / 11 | 0.296 [-0.451, 0.847] | 0.381 | 0.851 |
| A: Grab Ring (R) | signed_lag_s | 28 / 11 | -0.273 [-0.827, 0.505] | 0.420 | 0.851 |
| A: Reach Ring (L) | signed_lag_s | 27 / 11 | 0.260 [-0.521, 0.904] | 0.433 | 0.851 |
| C: Whole task (L) | signed_lag_s | 44 / 11 | -0.239 [-0.825, 0.470] | 0.481 | 0.857 |
| A: Place Ring (R) | signed_lag_s | 31 / 11 | 0.182 [-0.495, 0.750] | 0.592 | 0.912 |
| A: Reach Ring (R) | signed_lag_s | 33 / 11 | -0.164 [-0.702, 0.585] | 0.625 | 0.927 |
| C: Reach String (R) | signed_lag_s | 46 / 11 | -0.119 [-0.728, 0.591] | 0.728 | 0.988 |
| C: Reach String (L) | signed_lag_s | 48 / 11 | -0.083 [-0.600, 0.549] | 0.811 | 1.000 |
| C: Threading (L) | signed_lag_s | 16 / 7 | -0.204 [-0.871, 0.454] | 0.854 | 1.000 |
| C: Whole task (R) | signed_lag_s | 48 / 11 | 0.055 [-0.567, 0.778] | 0.877 | 1.000 |
| A: Whole task (L) | signed_lag_s | 30 / 11 | -0.046 [-0.737, 0.640] | 0.897 | 1.000 |
| A: Grab Ring (L) | signed_lag_s | 25 / 11 | 0.009 [-0.692, 0.857] | 0.990 | 1.000 |
| A: Place Ring (L) | signed_lag_s | 30 / 11 | 0.009 [-0.720, 0.786] | 0.990 | 1.000 |
| A: Transfer Ring (L) | signed_lag_s | 31 / 11 | 0.000 [-0.549, 0.589] | 1.000 | 1.000 |
| C: Threading (R) | signed_lag_s | 16 / 8 | 0.082 [-0.513, 0.595] | 1.000 | 1.000 |

Positive association means higher score accompanies a more positive lag (larger gaze lead or smaller gaze lag); negative association means higher score accompanies a more negative lag. For absolute lag, a negative association means higher performance nearer synchrony.

## Strongest primary associations across features

The following are sorted by uncorrected p to make candidate effects inspectable. Conclusions should use q and sensitivity, not the ranking alone.

Between participants:

| Condition | Predictor | N recordings / users | Association [95% CI] | p | BH q |
|---|---|---:|---:|---:|---:|
| A: Transfer Ring (R) | absolute_lag_s | 31 / 11 | -0.795 [-0.998, -0.316] | 0.004 | 0.162 |
| A: Transfer Ring (R) | signed_lag_s | 31 / 11 | -0.784 [-1.000, -0.306] | 0.006 | 0.162 |
| C: Wrap around PSM (L) | absolute_lag_s | 36 / 11 | -0.734 [-0.927, -0.295] | 0.014 | 0.268 |
| C: Pull/cinch (L) | absolute_lag_s | 18 / 10 | 0.707 [0.126, 0.974] | 0.028 | 0.345 |
| A: Whole task (L) | peak_r | 30 / 11 | 0.664 [0.183, 0.904] | 0.030 | 0.345 |
| C: Whole task (R) | peak_r | 48 / 11 | 0.624 [0.029, 0.953] | 0.044 | 0.423 |
| A: Transfer Ring (L) | peak_r | 31 / 11 | 0.573 [-0.010, 0.944] | 0.073 | 0.492 |
| C: Wrap around PSM (R) | peak_r | 37 / 11 | 0.563 [-0.123, 0.898] | 0.076 | 0.492 |

Within participant, after order adjustment:

| Condition | Predictor | N recordings / users | Association [95% CI] | p | BH q |
|---|---|---:|---:|---:|---:|
| A: Transfer Ring (L) | peak_r | 31 / 11 | 0.465 [0.219, 0.665] | 0.005 | 0.219 |
| A: Transfer Ring (L) | absolute_lag_s | 31 / 11 | 0.396 [0.025, 0.716] | 0.034 | 0.649 |
| A: Whole task (L) | signed_lag_s | 30 / 11 | 0.512 [0.158, 0.802] | 0.048 | 0.649 |
| A: Grab Ring (L) | signed_lag_s | 25 / 11 | -0.475 [-0.829, -0.088] | 0.064 | 0.649 |
| A: Whole task (L) | absolute_lag_s | 30 / 11 | 0.481 [0.124, 0.737] | 0.068 | 0.649 |
| A: Transfer Ring (L) | signed_lag_s | 31 / 11 | 0.347 [-0.119, 0.666] | 0.124 | 0.923 |
| A: Place Ring (R) | peak_r | 31 / 11 | -0.331 [-0.688, -0.018] | 0.192 | 0.923 |
| C: Whole task (L) | peak_r | 44 / 11 | 0.168 [-0.131, 0.436] | 0.205 | 0.923 |

## Timing sensitivity for the strongest between-participant candidates

Each row below is a separate analysis. The strict within-stage mode can differ in phase admissibility and retained recordings; a missing result is not a zero effect.

### A: Transfer Ring (R)

| Scenario / method | N recordings / users | Association [95% CI] | p | q |
|---|---:|---:|---:|---:|
| primary / between_participants | 31 / 11 | -0.784 [-1.000, -0.306] | 0.006 | 0.162 |
| primary / within_participant | 31 / 11 | 0.471 [0.072, 0.731] | 0.049 | 0.637 |
| primary / within_adjusted_order | 31 / 11 | 0.158 [-0.153, 0.438] | 0.448 | 0.923 |
| completed_only / between_participants | 31 / 11 | -0.784 [-0.991, -0.310] | 0.007 | 0.200 |
| completed_only / within_participant | 31 / 11 | 0.471 [0.079, 0.734] | 0.049 | 0.638 |
| completed_only / within_adjusted_order | 31 / 11 | 0.158 [-0.149, 0.444] | 0.448 | 0.895 |
| stronger_peaks / between_participants | 30 / 11 | -0.674 [-0.972, -0.077] | 0.026 | 0.396 |
| stronger_peaks / within_participant | 30 / 11 | 0.547 [0.209, 0.779] | 0.019 | 0.362 |
| stronger_peaks / within_adjusted_order | 30 / 11 | 0.088 [-0.211, 0.477] | 0.727 | 0.960 |
| within_stage / between_participants | 31 / 11 | 0.064 [-0.682, 0.702] | 0.848 | 0.950 |
| within_stage / within_participant | 31 / 11 | 0.210 [-0.155, 0.530] | 0.417 | 0.904 |
| within_stage / within_adjusted_order | 31 / 11 | -0.083 [-0.627, 0.428] | 0.815 | 0.926 |

### C: Wrap around PSM (L)

| Scenario / method | N recordings / users | Association [95% CI] | p | q |
|---|---:|---:|---:|---:|
| primary / between_participants | 36 / 11 | -0.405 [-0.841, 0.236] | 0.217 | 0.774 |
| primary / within_participant | 36 / 11 | 0.062 [-0.263, 0.454] | 0.789 | 0.987 |
| primary / within_adjusted_order | 36 / 11 | -0.076 [-0.409, 0.325] | 0.737 | 0.946 |
| completed_only / between_participants | 27 / 10 | -0.018 [-0.692, 0.761] | 0.963 | 1.000 |
| completed_only / within_participant | 27 / 10 | 0.006 [-0.397, 0.463] | 0.982 | 0.982 |
| completed_only / within_adjusted_order | 27 / 10 | -0.141 [-0.464, 0.413] | 0.591 | 0.945 |
| stronger_peaks / between_participants | 33 / 11 | -0.233 [-0.802, 0.448] | 0.486 | 0.901 |
| stronger_peaks / within_participant | 33 / 11 | 0.058 [-0.287, 0.466] | 0.817 | 0.942 |
| stronger_peaks / within_adjusted_order | 33 / 11 | -0.099 [-0.404, 0.276] | 0.650 | 0.960 |
| within_stage / between_participants | 41 / 11 | 0.339 [-0.330, 0.805] | 0.299 | 0.731 |
| within_stage / within_participant | 41 / 11 | -0.268 [-0.510, 0.009] | 0.216 | 0.904 |
| within_stage / within_adjusted_order | 41 / 11 | -0.308 [-0.523, 0.001] | 0.104 | 0.926 |

### C: Pull/cinch (L)

| Scenario / method | N recordings / users | Association [95% CI] | p | q |
|---|---:|---:|---:|---:|
| primary / between_participants | 18 / 10 | -0.356 [-0.981, 0.461] | 0.313 | 0.851 |
| primary / within_participant | 18 / 10 | NA [NA, NA] | NA | NA |
| primary / within_adjusted_order | 18 / 10 | NA [NA, NA] | NA | NA |
| completed_only / between_participants | 17 / 10 | -0.356 [-0.980, 0.450] | 0.309 | 0.863 |
| completed_only / within_participant | 17 / 10 | NA [NA, NA] | NA | NA |
| completed_only / within_adjusted_order | 17 / 10 | NA [NA, NA] | NA | NA |
| stronger_peaks / between_participants | 4 / 3 | NA [NA, NA] | NA | NA |
| stronger_peaks / within_participant | 4 / 3 | NA [NA, NA] | NA | NA |
| stronger_peaks / within_adjusted_order | 4 / 3 | NA [NA, NA] | NA | NA |
| within_stage / between_participants | 0 / 0 | NA [NA, NA] | NA | NA |
| within_stage / within_participant | 0 / 0 | NA [NA, NA] | NA | NA |
| within_stage / within_adjusted_order | 0 / 0 | NA [NA, NA] | NA | NA |

### A: Place Ring (R)

| Scenario / method | N recordings / users | Association [95% CI] | p | q |
|---|---:|---:|---:|---:|
| primary / between_participants | 31 / 11 | 0.182 [-0.495, 0.750] | 0.592 | 0.912 |
| primary / within_participant | 31 / 11 | 0.028 [-0.501, 0.587] | 0.943 | 0.987 |
| primary / within_adjusted_order | 31 / 11 | -0.211 [-0.608, 0.366] | 0.475 | 0.923 |
| completed_only / between_participants | 31 / 11 | 0.182 [-0.498, 0.747] | 0.598 | 0.902 |
| completed_only / within_participant | 31 / 11 | 0.028 [-0.487, 0.595] | 0.943 | 0.982 |
| completed_only / within_adjusted_order | 31 / 11 | -0.211 [-0.613, 0.361] | 0.475 | 0.912 |
| stronger_peaks / between_participants | 15 / 8 | -0.048 [-0.684, 0.795] | 0.936 | 0.988 |
| stronger_peaks / within_participant | 15 / 8 | NA [NA, NA] | NA | NA |
| stronger_peaks / within_adjusted_order | 15 / 8 | NA [NA, NA] | NA | NA |
| within_stage / between_participants | 31 / 11 | -0.516 [-0.808, 0.118] | 0.107 | 0.731 |
| within_stage / within_participant | 31 / 11 | -0.092 [-0.566, 0.478] | 0.804 | 0.904 |
| within_stage / within_adjusted_order | 31 / 11 | -0.132 [-0.563, 0.485] | 0.715 | 0.926 |

## Secondary velocity correlation strength

Task A grab-ring velocity estimates are shown below. The complete results for all phases are in the association tables.

| Side / method | N recordings / users | Rank association [95% CI] | p | BH q |
|---|---:|---:|---:|---:|
| L / between_participants | 32 / 11 | +0.636 [+0.037, +0.946] | 0.04045 | 0.4854 |
| L / within_participant | 32 / 11 | +0.385 [-0.081, +0.719] | 0.16769 | 0.6468 |
| L / within_adjusted_order | 32 / 11 | +0.191 [-0.121, +0.498] | 0.42628 | 0.8018 |
| R / between_participants | 33 / 11 | +0.400 [-0.317, +0.972] | 0.22630 | 0.7120 |
| R / within_participant | 33 / 11 | +0.453 [+0.037, +0.764] | 0.06459 | 0.4335 |
| R / within_adjusted_order | 33 / 11 | +0.264 [-0.256, +0.708] | 0.44256 | 0.8018 |

The right-side between-participant strength correlation has a leave-one-participant-out range of 0.200 to 0.624. Compare between-participant estimates with within-participant and order-adjusted estimates before interpreting performance differences. Recording peak velocity correlations in this phase have median L = 0.072, R = 0.083. A high correlation between that strength metric and scores does not make gaze–PSM velocity correlation itself large. Signed velocity timing estimates with BH q < 0.05: 0.

The strict within-stage position estimate for A place/right is r = -0.516, 95% CI [-0.808, 0.118], p = 0.107, q = 0.7306. Its main PSM-anchored estimate is r = 0.182, 95% CI [-0.495, 0.750], p = 0.592, q = 0.9118, and its within-participant order-adjusted strict-stage association is r = -0.132, 95% CI [-0.563, 0.485], p = 0.715, q = 0.9262. These estimates use different stage-support rules and should be compared with their uncertainty.

## Interpretation and limitations

- A/C are analyzed separately because their score distributions and task behaviors differ. Task C's many zeros create heavy ties and some participants have no score variation. A null result is not evidence that timing never matters, and these data cannot resolve small effects reliably.
- Individual timing is often weakly identified even when a peak is interior. `phase_validity.csv` records group bootstrap boundary frequencies, gain relative to zero lag and descriptive widths within 0.01 of the maximum. That width is a shape diagnostic, not a confidence interval. C threading has fewer supported participants and broad group peak intervals; its surviving computational peaks require particular caution.
- The median over only eligible attempts can use a different subset for different phases or sides. Exclusion may be related to performance. Participant leave-one-out ranges are supplied to expose single-participant sensitivity; no per-recording peak uncertainty is propagated into the score association.
- Between-participant differences can reflect general skill; within-participant changes can reflect practice, task difficulty, failed attempts or score floor effects. Even an association surviving correction would not establish that changing gaze timing causes better performance. The measurements are raw gaze and DLC left/right PSM coordinates, not validated fixation arrivals or tool-tip positions.

## Outputs and reproduction

- [Figure booklet](score_corr_figures.pdf), [association overview](figures/association_overview.png), [all valid phase timing scatters](figures/phase_signed_lag_scatter.png), [whole-task feature scatters](figures/whole_task_scatter.png), [timing sensitivity](figures/signed_lag_sensitivity.png), [score coverage](figures/score_distribution.png), [secondary velocity strength](figures/velocity_grab_ring_strength.png). PNG and vector PDF versions are provided.
- `associations.csv`: all primary, completed-only, stronger-peak, strict within-stage and secondary velocity estimates, including excluded conditions and BH family sizes.
- `participant_summaries.csv`: exact participant inputs to between-participant analysis, with retained recording IDs.
- `matched_scores.csv`, `score_inventory.csv`, `record_phase_audit.csv`, `phase_validity.csv`: joins, unused score rows, per-record exclusions and phase diagnostics.
- `provenance.json`, `validation.json`, and `code_snapshot/analyze_score_corr_dlc.py`: settings, input hashes, numerical checks and the analysis source.

From the repository root:

```bash
python tools/analyze_score_corr_dlc.py --input outputs/gaze_psm_xcorr_dlc --index Index --output outputs/score_corr_dlc --bootstrap 10000 --permutations 19999 --seed 20261030 --min-users 6
```

The original correlation outputs and input scores are read only. Statistical associations were computed from saved numerical curves, without access to the NAS source videos.
