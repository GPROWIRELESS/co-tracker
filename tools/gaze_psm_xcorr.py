#!/usr/bin/env python3
"""Position xcorr of gaze versus PSM masks; positive lag means gaze leads.

Run ``python tools/gaze_psm_xcorr.py --help`` for extraction, pooled stages,
novice bootstrap, unpaired references, cached reanalysis and plotting options.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def xcorr(gaze, psm, hz, max_lag, min_pairs):
    """Compatibility API: return lag seconds, normalized 2-D r, pair counts."""
    from tools.gaze_psm_core import analyze_curve
    result = analyze_curve(gaze, psm, hz, max_lag=max_lag,
                           min_overlap=min_pairs/hz, min_gaze_rms=0,
                           min_psm_rms=0, compute_spatial=False)
    return result["lag_s"], result["r"], result["pairs"]


def interpolate(t, xy, target, max_gap):
    from tools.gaze_psm_pipeline import interpolate_valid
    return interpolate_valid(t, xy, target, max_gap)


def centroids(path, threshold, min_area, limit=None):
    from tools.gaze_psm_pipeline import decode_mask
    xy, _, width, height, _ = decode_mask(path, threshold, min_area, limit)
    return xy, width, height


def main(argv=None):
    from tools.gaze_psm_pipeline import main as run
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
