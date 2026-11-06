# Velocity correlations versus performance outcomes

Velocity-based peak strength and signed peak lag are related separately to the supplied Score, Time, Eco, TimeRaw and EcoRaw fields. The saved gaze/PSM velocity curves are used; velocity comes from consecutive smoothed position differences at the recorded sampling rate. No new extraction or parameter tuning was performed.

Each outcome has peak-correlation and signed-lag figures under its own `velocity/` subfolder. Each folder contains a combined PDF, whole-task and phase PNG/PDF figures, plotted participant values, association statistics and validation.

| Outcome | Available A / C | Peak strength | Signed peak lag |
|---|---:|---|---|
| Score | 33 / 50 | [PDF](../peak_correlation_vs_score/velocity/peak_correlation_vs_score.pdf) | [PDF](../lag_correlation_vs_score/velocity/peak_lag_vs_score.pdf) |
| Time | 33 / 33 | [PDF](../peak_correlation_vs_Time/velocity/peak_correlation_vs_Time.pdf) | [PDF](../lag_correlation_vs_Time/velocity/peak_lag_vs_Time.pdf) |
| Eco | 33 / 33 | [PDF](../peak_correlation_vs_Eco/velocity/peak_correlation_vs_Eco.pdf) | [PDF](../lag_correlation_vs_Eco/velocity/peak_lag_vs_Eco.pdf) |
| TimeRaw | 33 / 33 | [PDF](../peak_correlation_vs_TimeRaw/velocity/peak_correlation_vs_TimeRaw.pdf) | [PDF](../lag_correlation_vs_TimeRaw/velocity/peak_lag_vs_TimeRaw.pdf) |
| EcoRaw | 33 / 33 | [PDF](../peak_correlation_vs_EcoRaw/velocity/peak_correlation_vs_EcoRaw.pdf) | [PDF](../lag_correlation_vs_EcoRaw/velocity/peak_lag_vs_EcoRaw.pdf) |

Phase eligibility and peak filtering are checked for the velocity signal: positive interior peaks, complete lag support and eligible group status. Recording subsets and admitted phases can therefore differ from position results. Participant means of each outcome use exactly the same attempts as the predictor summary. Mean correlation strength uses Fisher-z averaging; signed lag uses the median of recording peak lags. Positive lag means gaze velocity leads instrument velocity. Outcomes are used directly, including negative component scores; missing values are excluded.

The analysis uses 10,000 participant bootstrap draws and 19,999 participant score permutations. The complete association tables include between-participant, within-participant, and within-participant order/attempt-adjusted results. The scatter plots show between-participant associations. BH correction is separate for each outcome and method, including signed lag, absolute lag and strength, both tasks, both sides and admitted whole-task/phase conditions. It is not jointly corrected across the five outcomes. Displayed bootstrap intervals are exploratory and unadjusted; intervals with too many undefined bootstrap draws are flagged as unreliable. Velocity results remain secondary and do not establish causation.

All successful estimates, cluster-adjusted p values and BH q values were checked independently. Overall-Score velocity estimates reproduce the original saved analysis at the same settings. Position result files were verified unchanged.

Reproduce from the repository root:

```bash
python tools/analyze_velocity_score_corr_dlc.py --input outputs/gaze_psm_xcorr_dlc --index Index --output outputs/score_corr_dlc --bootstrap 10000 --permutations 19999 --seed 20261030 --min-users 6
```
