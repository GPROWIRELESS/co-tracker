"""Spatial probability operations shared by datasets and conditional VAEs."""

import torch


def coordinate_grid(hw, *, device=None, dtype=torch.float32):
    h, w = hw
    if min(h, w) < 2:
        raise ValueError("Spatial grids need at least two cells per axis")
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], -1)


def gaze_heatmap(coords, hw, sigma=1.0):
    """Normalized Gaussian label uncertainty, sigma measured in grid cells."""
    if sigma <= 0 or coords.shape[-1] != 2:
        raise ValueError("Expected coordinates[...,2] and sigma>0")
    grid = coordinate_grid(hw, device=coords.device, dtype=coords.dtype).flatten(0, 1)
    target = coords * coords.new_tensor([hw[1] - 1, hw[0] - 1])
    distance = (grid - target[..., None, :]).square().sum(-1)
    return (
        (-distance / (2 * sigma * sigma)).softmax(-1).reshape(*coords.shape[:-1], *hw)
    )


def probability_mean(probability):
    h, w = probability.shape[-2:]
    xy = coordinate_grid(
        (h, w), device=probability.device, dtype=probability.dtype
    ) / probability.new_tensor([w - 1, h - 1])
    return probability.flatten(-2) @ xy.flatten(0, 1)
