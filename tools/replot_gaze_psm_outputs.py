"""Regenerate figures from saved results, preserving statistical artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gaze_psm_figures import generate_figures


def _statistical_hashes(output):
    paths = [p for p in output.iterdir() if p.is_file() and p.suffix in {'.csv', '.json'}
             and not p.name.startswith(('figure_refresh_', 'figure_readability_'))]
    paths += list((output/'cache').glob('*.npz'))
    paths += list((output/'code_snapshot').glob('*.py'))
    return {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gaze_psm_xcorr'))
    parser.add_argument('--backup-name', default='figures_before_readability_20261030')
    args = parser.parse_args()
    output = args.output
    before = _statistical_hashes(output)
    backup = output/args.backup_name
    backup.mkdir(parents=True, exist_ok=True)
    # An existing backup is the original view, not a collection of later plots.
    if not any(backup.iterdir()):
        for p in (output/'figures').glob('*'):
            if p.is_file():
                shutil.copy2(p, backup/p.name)
    results = json.loads((output/'results.json').read_text(encoding='utf-8'))
    # Restore only numerical arrays, leaving metadata lists such as IDs intact.
    arrays = {'lag_s', 'r', 'pairs', 'occurrences', 'gaze_rms', 'psm_rms',
              'distance_median_px', 'proximity', 'n_records', 'r_fixed_records',
              'r_primary_fixed_record_users', 'r_available', 'n_users',
              'r_ci_low', 'r_ci_high', 'bootstrap_peak_lag_s', 'null_r_low',
              'null_r_high', 'null_peak_r', 'bootstrap_difference_s'}
    for rows in results.values():
        for row in rows:
            for key in arrays & row.keys():
                if isinstance(row[key], list):
                    row[key] = np.asarray(row[key], dtype=float)
    generate_figures(output, results['record_curves'], results['user_curves'],
                     results['group_curves'], results['contrasts'])
    after = _statistical_hashes(output)
    if before != after:
        raise RuntimeError('Statistical artifacts changed while plotting')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'statistics_unchanged': True, 'statistical_files_verified': len(before),
              'statistical_sha256': after, 'backup': backup.as_posix(),
              'png_count': len(list((output/'figures').glob('*.png'))),
              'pdf_count': len(list((output/'figures').glob('*.pdf'))),
              'plotting_code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [ROOT/'tools/gaze_psm_figures.py',
                                                 ROOT/'tools/gaze_psm_report.py', Path(__file__)]}}
    (output/'figure_readability_20261030.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Saved {report['png_count']} PNG/PDF figures; {len(before)} statistical files unchanged.")


if __name__ == '__main__':
    main()
