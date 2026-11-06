# Performance associations with maximum-xcorr PSM stage peaks

Each recording/stage uses its previously selected PSM. Score matching never changes that selection. Position is primary and velocity is secondary; whole-task curves are not included.

| Outcome | Available recordings before stage filtering | Successful tests, including sensitivities | BH q < 0.05 |
|---|---:|---:|---:|
| Score | 83 / 84 | 279 | 1 |
| Time | 66 / 84 | 120 | 1 |
| Eco | 66 / 84 | 120 | 0 |
| TimeRaw | 66 / 84 | 120 | 1 |
| EcoRaw | 66 / 84 | 120 | 0 |

## Methods

Outcomes are supplied Score, Time, Eco, TimeRaw and EcoRaw values, without sign reversal or imputation. Failed attempts with supplied zero scores remain in primary analyses. Available-score counts are before stage exclusions.

Eligible groups have status ok, positive peak correlation and at least three stable users. Eligible recordings have usable outcomes and quality, positive interior peaks and complete lag support. Tests require 6 qualifying participants; within-participant tests require within-user variation.

Between-participant summaries use mean outcomes, median signed lag, median absolute lag and Fisher-z mean peak r from exactly the same eligible recordings. Within-participant analyses use equal participant weights and rank-based fixed effects, with an additional model adjusting for task order and attempt. Positive signed lag means gaze leads PSM.

Uncertainty uses 10,000 whole-participant bootstrap draws. Between-participant p values use 19,999 two-sided participant score permutations with a +1 correction. Within-participant p values use the existing participant-cluster CR1 t approximation.

BH families pool all three predictors and stages separately within each outcome, scenario and method. Outcomes, methods and sensitivity scenarios are not jointly corrected. Total Score includes within-stage, completed-only and stronger-peaks sensitivities. Other outcomes use primary position and secondary velocity analyses.

Selected PSM means the instrument with higher signed correlation, not an independently identified operated instrument. All associations are exploratory; score is measured for the whole attempt, not for each stage.

## Results

See [all associations](all_associations.csv), [record selection audit](record_phase_audit.csv), [association overviews](association_overviews.pdf), and the `peak_correlation_vs_*`, `lag_correlation_vs_*`, and `absolute_lag_correlation_vs_*` figure directories. Scatter points show participants; Pearson fits are descriptive, while Spearman p/q and confidence intervals come from the saved inference. Intervals with fewer than 97.5% defined bootstrap draws are labeled unreliable.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/analyze_score_corr_max_dlc.py --input outputs/gaze_psm_xcorr_max_dlc --index Index --output outputs/score_corr_max_dlc --bootstrap 10000 --permutations 19999 --seed 20261030 --min-users 6
```
