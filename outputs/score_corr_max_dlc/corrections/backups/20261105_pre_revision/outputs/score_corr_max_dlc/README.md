# Score correlation comparison layout

Selected PSM per recording/stage.

Use the same relative paths in `score_corr` and `score_corr_max` to compare results.

```text
metrics/
  all_associations.csv
  all_participant_summaries.csv
  position|velocity/
    Score|Time|Eco|TimeRaw|EcoRaw|PenaltySum|PenaltyRawSum/
      associations.csv
      participant_summaries.csv
      matched_outcomes.csv
      record_phase_audit.csv
      phase_validity.csv
      provenance.json
plots/
  association_overviews.pdf
  position|velocity/
    Score|Time|Eco|TimeRaw|EcoRaw|PenaltySum|PenaltyRawSum/
      overview.png / overview.pdf
      peak_r/
      signed_lag_s/
      absolute_lag_s/
```

Each predictor folder contains the corresponding Task A/C figures, association table, plotted participant points and axis audit. Whole-task figures are present only for analyses with whole-task data.

| Signal | Outcome | Tables | Overview | Peak r | Signed lag | Absolute lag |
|---|---|---|---|---|---|---|
| position | Score | [CSV](metrics/position/Score/) | [PDF](plots/position/Score/overview.pdf) | [plots](plots/position/Score/peak_r/) | [plots](plots/position/Score/signed_lag_s/) | [plots](plots/position/Score/absolute_lag_s/) |
| position | Time | [CSV](metrics/position/Time/) | [PDF](plots/position/Time/overview.pdf) | [plots](plots/position/Time/peak_r/) | [plots](plots/position/Time/signed_lag_s/) | [plots](plots/position/Time/absolute_lag_s/) |
| position | Eco | [CSV](metrics/position/Eco/) | [PDF](plots/position/Eco/overview.pdf) | [plots](plots/position/Eco/peak_r/) | [plots](plots/position/Eco/signed_lag_s/) | [plots](plots/position/Eco/absolute_lag_s/) |
| position | TimeRaw | [CSV](metrics/position/TimeRaw/) | [PDF](plots/position/TimeRaw/overview.pdf) | [plots](plots/position/TimeRaw/peak_r/) | [plots](plots/position/TimeRaw/signed_lag_s/) | [plots](plots/position/TimeRaw/absolute_lag_s/) |
| position | EcoRaw | [CSV](metrics/position/EcoRaw/) | [PDF](plots/position/EcoRaw/overview.pdf) | [plots](plots/position/EcoRaw/peak_r/) | [plots](plots/position/EcoRaw/signed_lag_s/) | [plots](plots/position/EcoRaw/absolute_lag_s/) |
| position | PenaltySum | [CSV](metrics/position/PenaltySum/) | [PDF](plots/position/PenaltySum/overview.pdf) | [plots](plots/position/PenaltySum/peak_r/) | [plots](plots/position/PenaltySum/signed_lag_s/) | [plots](plots/position/PenaltySum/absolute_lag_s/) |
| position | PenaltyRawSum | [CSV](metrics/position/PenaltyRawSum/) | [PDF](plots/position/PenaltyRawSum/overview.pdf) | [plots](plots/position/PenaltyRawSum/peak_r/) | [plots](plots/position/PenaltyRawSum/signed_lag_s/) | [plots](plots/position/PenaltyRawSum/absolute_lag_s/) |
| velocity | Score | [CSV](metrics/velocity/Score/) | [PDF](plots/velocity/Score/overview.pdf) | [plots](plots/velocity/Score/peak_r/) | [plots](plots/velocity/Score/signed_lag_s/) | [plots](plots/velocity/Score/absolute_lag_s/) |
| velocity | Time | [CSV](metrics/velocity/Time/) | [PDF](plots/velocity/Time/overview.pdf) | [plots](plots/velocity/Time/peak_r/) | [plots](plots/velocity/Time/signed_lag_s/) | [plots](plots/velocity/Time/absolute_lag_s/) |
| velocity | Eco | [CSV](metrics/velocity/Eco/) | [PDF](plots/velocity/Eco/overview.pdf) | [plots](plots/velocity/Eco/peak_r/) | [plots](plots/velocity/Eco/signed_lag_s/) | [plots](plots/velocity/Eco/absolute_lag_s/) |
| velocity | TimeRaw | [CSV](metrics/velocity/TimeRaw/) | [PDF](plots/velocity/TimeRaw/overview.pdf) | [plots](plots/velocity/TimeRaw/peak_r/) | [plots](plots/velocity/TimeRaw/signed_lag_s/) | [plots](plots/velocity/TimeRaw/absolute_lag_s/) |
| velocity | EcoRaw | [CSV](metrics/velocity/EcoRaw/) | [PDF](plots/velocity/EcoRaw/overview.pdf) | [plots](plots/velocity/EcoRaw/peak_r/) | [plots](plots/velocity/EcoRaw/signed_lag_s/) | [plots](plots/velocity/EcoRaw/absolute_lag_s/) |
| velocity | PenaltySum | [CSV](metrics/velocity/PenaltySum/) | [PDF](plots/velocity/PenaltySum/overview.pdf) | [plots](plots/velocity/PenaltySum/peak_r/) | [plots](plots/velocity/PenaltySum/signed_lag_s/) | [plots](plots/velocity/PenaltySum/absolute_lag_s/) |
| velocity | PenaltyRawSum | [CSV](metrics/velocity/PenaltyRawSum/) | [PDF](plots/velocity/PenaltyRawSum/overview.pdf) | [plots](plots/velocity/PenaltyRawSum/peak_r/) | [plots](plots/velocity/PenaltyRawSum/signed_lag_s/) | [plots](plots/velocity/PenaltyRawSum/absolute_lag_s/) |

## Interpretation and compatibility

- Original coefficients, confidence intervals, p/q values, score matching and exclusions are unchanged. No inference was rerun.
- `score_corr` contains left/right PSMs and whole-task conditions; `score_corr_max` contains one selected PSM per stage. These differences are retained.
- Original BH families remain intact. Filtering a table into a signal folder does not recalculate its q values.
- Existing file paths, reports, source snapshots and historical audits remain available. Predictor directories reuse existing plots through relative symbolic links where possible.
- Penalties also use the standard top-level `peak_correlation_vs_PenaltySum/`, `peak_correlation_vs_PenaltyRawSum/` and matching `lag_correlation_vs_*` / `absolute_lag_correlation_vs_*` entries, each with position and velocity folders.
- `metrics` contains filtered copies for comparison. After regenerating numerical results or legacy plots, rerun the organizer to refresh these copies and fill missing plot sets.
- [Original analysis report](report.md), [layout manifest](layout.json), [layout validation](layout_validation.json).

Rebuild both layouts from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/organize_score_corr_outputs.py
```
