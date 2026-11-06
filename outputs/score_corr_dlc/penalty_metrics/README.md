# PenaltySum and PenaltyRawSum associations

Uses the supplied penalty columns directly. PenaltySum may be negative; no sign reversal is applied. Zero is a valid observed value; missing values are excluded and never replaced with zero. The outcome is measured for the entire attempt.

Position is primary and velocity secondary. Peak r, signed peak lag and absolute peak lag are analyzed between participants, within participants and with task-order/attempt adjustment. Existing peak, group-support and participant-count eligibility rules are unchanged.

10,000 participant bootstrap samples; 19,999 between-participant score permutations; minimum 6 qualifying participants. Within-participant p values use the existing cluster CR1 approximation. BH correction pools all three predictors and stages separately per outcome, scenario and method; the two penalty outcomes and prior score outcomes are not jointly corrected.

| Outcome | Available A / C | Successful tests | BH q < .05 |
|---|---:|---:|---:|
| PenaltySum | 33 / 33 | 324 | 0 |
| PenaltyRawSum | 33 / 33 | 324 | 0 |

Browse the matching `../metrics/<signal>/<outcome>/` and `../plots/<signal>/<outcome>/` folders, or use the [main navigation](../README.md). Existing five-outcome analyses remain unchanged.

Reproduce both roots:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/analyze_penalty_score_corr_dlc.py
```
