# Performance associations with maximum-xcorr PSM stage peaks

Each recording/stage uses its previously selected PSM. Score matching never changes that selection. Position is primary and velocity is secondary; whole-task curves are not included.

| Outcome | Available recordings before stage filtering | Successful tests, including sensitivities | Family-specific q < 0.05 rows (not independent discoveries) |
|---|---:|---:|---:|
| Score | 83 / 84 | 279 | 1 |
| Time | 66 / 84 | 120 | 1 |
| Eco | 66 / 84 | 120 | 0 |
| TimeRaw | 66 / 84 | 120 | 1 |
| EcoRaw | 66 / 84 | 120 | 0 |

TimeRaw and EcoRaw are approximately affine, sign-reversed versions of Time and Eco within each task. Their rank associations are near-mirrors and are retained for traceability, not counted as independent evidence. Rounding and aggregation can cause small differences.

## Methods

Outcomes are supplied Score, Time, Eco, TimeRaw and EcoRaw values, without sign reversal or imputation. Failed attempts with supplied zero scores remain in primary analyses. Available-score counts are before stage exclusions.

Eligible groups have status ok, positive peak correlation and at least three stable users. Eligible recordings have usable outcomes and quality, positive interior peaks and complete lag support. Tests require 6 qualifying participants; within-participant tests require within-user variation.

Between-participant summaries use mean outcomes, median signed lag, median absolute lag and Fisher-z mean peak r from exactly the same eligible recordings. Within-participant analyses use equal participant weights and rank-based fixed effects, with an additional model adjusting for task order and attempt. Positive signed lag means gaze leads PSM.

Uncertainty uses 10,000 whole-participant bootstrap draws. Between-participant p values use 19,999 two-sided participant score permutations with a +1 correction. Within-participant p values use the existing participant-cluster CR1 t approximation.

BH families pool all three predictors and stages separately within each outcome, scenario and method. Outcomes, methods and sensitivity scenarios are not jointly corrected. Total Score includes within-stage, completed-only and stronger-peaks sensitivities. Other outcomes use primary position and secondary velocity analyses.

Selected PSM means the instrument with higher signed correlation, not an independently identified operated instrument. All associations are exploratory; score is measured for the whole attempt, not for each stage.

## Results

See [all associations](all_associations.csv), [record selection audit](record_phase_audit.csv), [association overviews](association_overviews.pdf), and the `peak_correlation_vs_*`, `lag_correlation_vs_*`, and `absolute_lag_correlation_vs_*` figure directories. Scatter points show participants; Pearson fits are descriptive, while Spearman p/q and confidence intervals come from the saved inference. Intervals with fewer than 97.5% defined bootstrap draws are labeled unreliable.


## Interpretation and limitations

The table counts significant rows within separate BH families; it does not count independent discoveries. No overall FDR claim across outcomes, scenarios and methods is made. The existing position-primary and velocity-secondary labels describe the analysis plan; all findings remain exploratory. A small number of discoveries does not establish a global null, and q values do not identify which findings are false.

Task C Score has a substantial floor at zero. Completed-only filtering does not remove this floor. PenaltySum <= -100 is a severe-total-penalty proxy, not a validated definition of a critical event; individual penalty fields must be checked. Excluding such attempts changes the target population and is supplementary sensitivity analysis, not a replacement for the primary analysis.

TimeRaw-versus-recording-duration discrepancies and repeated score vectors warrant source verification. Neither proves incorrect identity matching or authorizes replacement of source scores. See [correction review](corrections/review_zh.md) and [diagnostics](corrections/availability_diagnostics.csv) for the audit.

Selecting the larger side peak raises the selected correlation by construction. A positive selected peak and a score association do not establish meaningful or statistically significant gaze-PSM coupling. Signed lag describes the selected side for that recording/stage; side switching can mix instrument-specific timing. These observations do not by themselves prove that the peak or lag is noise.

Small-cluster CR1 t inference is approximate. Within-participant bootstrap draws retain the original full-sample ranks. A percentile interval touching -1 or +1 indicates boundary sensitivity, not by itself a degenerate bootstrap. An unrestricted within-participant score permutation requires exchangeability and can be invalid in the presence of practice/order effects.

### Score / Task A / transfer ring / velocity / peak r

| Method | Association | p | Family q | Participants |
|---|---:|---:|---:|---:|
| between_participants | 0.636 | 0.0389 | 0.5058 | 11.0 |
| within_participant | 0.579 | 0.0001611 | 0.003383 | 11.0 |
| within_adjusted_order | 0.388 | 0.07388 | 0.5172 | 11.0 |

The within-participant Score association passes its family correction before order adjustment but not after adjustment. This weakens evidence for an association independent of task order; it does not prove that practice caused the entire association.

In the recorded Task A sample Attempt is constant at 1, so attempt adjustment contributes no additional covariate information; task-order adjustment still operates.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/analyze_score_corr_max_dlc.py --input outputs/gaze_psm_xcorr_max_dlc --index Index --output outputs/score_corr_max_dlc --bootstrap 10000 --permutations 19999 --seed 20261030 --min-users 6
```

## 2026-11-05 audit additions

Including penalties, the comparison tables contain 1,377 rows, 999 finite-p tests and 51 BH families. See [point-by-point Chinese review](corrections/review_zh.md). Three family-significant rows concern Score, Time and TimeRaw in the same A/transfer-ring/velocity/within condition; TimeRaw mirrors Time. A supplementary global BH calculation yields q=0.15864 for all three. Excluding the Raw mirrors gives global q=0.12227 for Score and 0.15576 for Time. These post-hoc scopes do not replace the original family q values.

The known-PenaltySum > -100 subset retains A=33 and C=18 recordings before stage filtering; it is not a validated critical-error-free subset. It yields 111 estimable tests and no new Task C family-significant findings. C duration discrepancies are flagged in 22/33 complete cases at a descriptive 20% tolerance (A: 0/33); missing timing fields remain unavailable. Source scores and identity matches were not changed.

The revision diagnostics builder is archived in corrections/code_snapshot/build_corrections.py. Run it with --repo, --output (a separate review directory), and --patched-tools pointing to the current tools directory. The two original generators regenerate their respective legacy and comparison overviews; supplementary diagnostics require the archived builder.
