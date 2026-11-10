"""Pixel RMSE for two already aligned and filtered coordinate arrays.

Formula: sqrt(mean(((x_a-x_b)**2 + (y_a-y_b)**2) / 2)).
The denominator is twice the number of frames (both coordinate dimensions).
Data loading, coordinate scaling, validity masks and grouping belong to the
arrange scripts. This module never smooths or changes sample alignment.
"""

import numpy as np


def rmse(coordinates_a, coordinates_b):
    """Return pixel RMSE for matching finite (N, 2) arrays.

    Empty matching arrays return NaN because there are no valid pairs.
    Nonfinite inputs raise ValueError: callers must explicitly apply their
    chosen validity mask before calling, preserving the original row pairing.
    """
    a = np.asarray(coordinates_a, dtype=np.float64)
    b = np.asarray(coordinates_b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape:
        raise ValueError("Coordinates must have identical (N, 2) shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Coordinates must be finite; apply a paired validity mask first")
    if len(a) == 0:
        return float("nan")
    difference = a - b
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1) / 2.0)))
