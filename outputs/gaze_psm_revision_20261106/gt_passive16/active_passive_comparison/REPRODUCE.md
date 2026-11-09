# Reproduction

Run in an environment with Python, NumPy, pandas, SciPy and Matplotlib. Original source directories and input caches are recorded with SHA256 in comparison_manifest.json and support_audit.json. Use a NEW output directory.

```powershell
python code_snapshot/support_audit.py --output NEW_OUTPUT
python code_snapshot/build_comparison.py --output NEW_OUTPUT
python code_snapshot/validate_comparison.py --output NEW_OUTPUT
python code_snapshot/build_report.py --output NEW_OUTPUT
```

The supplement reuses frozen original per-record results; common-time sensitivity additionally needs the referenced original caches. These paths are explicit in support_audit.json. All frozen numerical Python modules used by the audit are included in code_snapshot/frozen_kernels. Existing source data and baseline result files are not regenerated.

The bootstrap archive stores scalar paired-metric difference draws and paired group-curve peak-lag draws; bootstrap_index.json maps arrays to conditions. Pointwise difference-curve ranges are exported in paired_group_curves.csv and reproducible from the documented deterministic seeds. These ranges are exploratory, not simultaneous or calibrated confidence bands.
