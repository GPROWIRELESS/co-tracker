# Maximum-xcorr DLC model comparison: figure and table notes

The main analysis follows `gaze_psm_xcorr_max_dlc`: each gaze source independently selects the left or right DLC instrument with the largest signed correlation peak for each recording, signal, stage and mode. The winner's entire curve is retained. This is not a pointwise maximum envelope. GT and a model can select different instruments.

`position` with `psm_stage` is the primary descriptive comparison. `velocity` and `within_stage` results remain available. Positive lag means gaze leads PSM. Peak-lag differences compare independently selected maximum-xcorr statistics and should not automatically be interpreted as timing error against the same physical instrument.

## Tables

- `paired_summary.csv`: task × stage × signal × mode × model summaries.
- `paired_overview.csv`: task × signal × mode × model summaries across stages. Conditions are first averaged within each participant, then participants receive equal weight for fields prefixed `participant_equal_`.
- `common5/position_psm_stage_summary.csv`: the primary common-five comparison and the all-five-same-GT-side sensitivity summary, distinguished by `scope`.
- Every summary contains `own_eligible`, `common5`, and `common5_same_gt_side` scopes. `own_eligible` uses each model's eligible subset. `common5` fixes exactly the same eligible recording-stage conditions across all five models. `common5_same_gt_side` further requires every model to select the same instrument as GT.
- `common5_pairs.csv` and `common5_same_gt_side_pairs.csv` are long tables retaining all five models' values. The corresponding `_conditions.csv` files list unique condition identities.
- `side_selection_concordance.csv` gives selection confusion counts and agreement. `all_defined_selections` includes defined L/R selections before peak-eligibility filtering; other scopes are listed separately.
- `participant_equal_mean_abs_max_lag_difference_s` is the participant-weighted mean of absolute recording-stage max-lag differences. `condition_median_abs_max_lag_difference_s` is a condition-level median. These are different estimands and are named explicitly.
- Direction agreement uses the saved synchronous band. A small negative and positive lag within that band both count as synchronous.

## Figures

- Four group overlays: Task A/C × position/velocity. Rows are all stages; columns are PSM-stage and within-occurrence modes. Group overlays show each source's own stable-user count and may not share identical support.
- Two lag-agreement scatter figures: own eligible and common five-model support.
- Four difference figures: peak lag and peak correlation, each on own eligible and common five-model support.
- Two selected-side confusion/selection-agreement figures: Task A/C, position PSM-stage, all defined selections before eligibility filtering.
- Two paired group-curve difference figures: shared complete-lag participants within each model/GT pair, with paired participant-bootstrap 95% intervals. These support sets can differ between model pairs. Crosses mark nonpositive or boundary group peaks; unestimable rows are marked NA. These exploratory intervals are not adjusted for multiple comparisons.
- Ten per-video position curve figures. Each panel records every source's selected instrument. Boundary peaks use triangle markers.
- PNG and PDF versions are in `figures/`. `figures/comparison_report.pdf` combines the comparison figures. `comparison_figure_manifest.json` records the complete list.

Repeated stages from the same person are not independent observations. Boxes and scatter points describe the observed sample; no independent-condition confidence interval or score correlation is computed by this script. Group-difference intervals use the saved paired participant bootstrap. The same-side subset may be small and is a sensitivity analysis, not a replacement for the original independent-selection method.

Unique common-five conditions across tasks/signals/modes: 117. Common-five same-GT-side conditions: 51.
