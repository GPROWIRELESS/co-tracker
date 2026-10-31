#!/usr/bin/env python3
"""Position xcorr of gaze versus DLC PSM coordinates; positive lag means gaze leads.

Run ``python tools/gaze_psm_xcorr_dlc.py --help`` for extraction, pooled stages,
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
    from tools.gaze_psm_pipeline_dlc import interpolate_valid
    return interpolate_valid(t, xy, target, max_gap)


def centroids(path, threshold, min_area, limit=None):
    """Read separate DLC left/right coordinates; legacy mask options are unused.

    ``path`` is sam/<clip>/<clip>_DLC.csv. Both sides share the source
    left-camera resolution. The returned first value is a side-to-xy dict.
    """
    from tools.gaze_psm_pipeline_dlc import read_dlc, video_metadata
    path = Path(path)
    points, _ = read_dlc(path, limit=limit)
    cid = path.parent.name
    width, height, _ = video_metadata(path.parents[2]/"stimuli"/f"{cid}.mp4")
    return points, width, height


def main(argv=None):
    from tools.gaze_psm_pipeline_dlc import main as run
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
