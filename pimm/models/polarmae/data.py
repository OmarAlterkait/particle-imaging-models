"""
Data format conversion between pimm (packed) and PoLAr-MAE (padded) formats.

pimm uses packed tensors: (N_total, C) with an offset tensor of shape (B,)
where offset[i] = cumulative number of points up to and including batch i.

PoLAr-MAE uses padded tensors: (B, N_max, C) with a lengths tensor of shape (B,)
where lengths[i] = number of valid points in batch element i.
"""

import torch


def packed_to_batched(feat: torch.Tensor, offset: torch.Tensor):
    """Convert pimm packed format to PoLAr-MAE padded/batched format.

    Args:
        feat: (N_total, C) packed features (e.g. [x, y, z, energy]).
        offset: (B,) cumulative sum of point counts per sample.

    Returns:
        points: (B, N_max, C) zero-padded tensor.
        lengths: (B,) number of valid points per sample.
    """
    lengths = torch.diff(offset, prepend=offset.new_zeros(1))
    B = lengths.shape[0]
    N_max = lengths.max().item()
    C = feat.shape[1]

    points = feat.new_zeros(B, N_max, C)
    start = 0
    for i in range(B):
        n = lengths[i].item()
        points[i, :n] = feat[start : start + n]
        start += n

    return points, lengths


def batched_to_packed(points: torch.Tensor, lengths: torch.Tensor):
    """Convert PoLAr-MAE padded/batched format to pimm packed format.

    Args:
        points: (B, N_max, C) padded tensor.
        lengths: (B,) number of valid points per sample.

    Returns:
        feat: (N_total, C) packed features.
        offset: (B,) cumulative sum of point counts.
    """
    B = points.shape[0]
    parts = []
    for i in range(B):
        n = lengths[i].item()
        parts.append(points[i, :n])
    feat = torch.cat(parts, dim=0)
    offset = torch.cumsum(lengths, dim=0)
    return feat, offset
