# TimeRaw and EcoRaw association overviews

These add two figures matching the original position-association overview: same rows, three analysis panels, three peak features and a −1 to +1 color scale.

- [TimeRaw PDF](../figures/association_overview_TimeRaw.pdf) / [PNG](../figures/association_overview_TimeRaw.png)
- [EcoRaw PDF](../figures/association_overview_EcoRaw.pdf) / [PNG](../figures/association_overview_EcoRaw.png)
- [Both figures in one PDF](../raw_metric_overviews.pdf)

The outcome is the named raw CSV field, used directly with no sign reversal. Positive coefficients mean a larger raw outcome, and negative coefficients mean a smaller raw outcome. For TimeRaw, positive means longer duration and negative means shorter duration. EcoRaw is labeled as the raw economy-of-motion measure; no unit conversion is applied.

Peak signed lag, absolute lag, peak r, participant aggregation, rank transformations, equal participant weighting, order/attempt adjustment, 10,000 participant bootstrap replicates and 19,999 participant permutations use the same methods as the original analysis. Between-participant outcomes are means of the exact same eligible recordings used for the peak features. Each within-participant outcome is the recording's own raw value. Missing raw values are excluded and never imputed. The `score_usable` column in these component-specific audit files indicates availability of that file's named raw outcome.

BH correction is separate for each outcome and analysis method, pooling all three peak features and all admitted whole-task/phase conditions. It is not jointly corrected across overall Score, TimeRaw and EcoRaw. Asterisks mark q < 0.05 in that outcome/method family. Only the primary position analysis is shown; no additional sensitivity or velocity analyses were run for these two outcomes.

| Outcome | Available A / C | Missing raw outcomes | Successful tests | q < 0.05 |
|---|---:|---:|---:|---:|
| TimeRaw | 33 / 33 | 18 | 156 | 6 |
| EcoRaw | 33 / 33 | 18 | 156 | 0 |

All available raw outcomes in this run come from completed attempts. The raw-outcome sample differs from the overall-Score analysis, which included scored failures. Differences across figures can therefore reflect both the outcome definition and the retained attempts. Phase-specific exclusions can reduce these counts further.

The CSV files contain estimates, uncertainty, p/q values, participant counts and exact matched recording IDs. Every successful correlation and cluster-adjusted p value was checked independently using SciPy and full weighted regressions with participant dummy variables. All BH values were checked against SciPy. Input and original overall-Score output hashes remained unchanged.

Reproduce from the repository root:

```bash
python tools/analyze_raw_score_corr_dlc.py --input outputs/gaze_psm_xcorr_dlc --index Index --output outputs/score_corr_dlc --bootstrap 10000 --permutations 19999 --seed 20261030 --min-users 6
```
