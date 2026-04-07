"""
CNMS (Centrality-based Non-Maximum Suppression) for point cloud centroid selection.

Provides both packed and padded tensor format implementations.
"""
import torch
from typing import Tuple, Optional

from . import _ext

try:
    from pointops import ball_query as pointops_ball_query
    HAS_POINTOPS = True
except ImportError:
    HAS_POINTOPS = False

try:
    from pytorch3d_ops.ops import ball_query as pytorch3d_ball_query
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False


def _offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """Convert offset tensor to batch indices."""
    batch_size = offset.shape[0]
    counts = torch.zeros(batch_size, dtype=torch.long, device=offset.device)
    counts[0] = offset[0]
    counts[1:] = offset[1:] - offset[:-1]
    return torch.repeat_interleave(
        torch.arange(batch_size, device=offset.device),
        counts
    )


@torch.no_grad()
def cnms_packed(
    coord: torch.Tensor,
    offset: torch.Tensor,
    radius: float,
    overlap_factor: float = 0.5,
    max_neighbors: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CNMS for packed tensor format.

    Args:
        coord: (N, 3) packed point coordinates
        offset: (B,) cumulative point counts per batch
        radius: ball radius for centroid coverage
        overlap_factor: overlap factor (0.5 = 50% diameter overlap)
        max_neighbors: max neighbors to consider for overlap

    Returns:
        centroids: (M, 3) selected centroid coordinates
        centroid_offset: (B,) cumulative centroid counts
        retain_mask: (N,) boolean mask of retained points
    """
    if not HAS_POINTOPS:
        raise ImportError("pointops is required for cnms_packed()")

    device = coord.device
    N = coord.shape[0]
    B = offset.shape[0]

    query_radius = 2 * radius * overlap_factor

    idx, _ = pointops_ball_query(
        max_neighbors,
        query_radius,
        0.0,
        coord,
        offset,
    )

    overlap_counts = (idx >= 0).sum(dim=1)
    batch = _offset2batch(offset)

    max_count = overlap_counts.max() + 1
    sort_key = batch * max_count + (max_count - overlap_counts)
    sorted_idx = torch.argsort(sort_key)

    idx_int = idx.int()
    sorted_idx_int = sorted_idx.int()
    offset_int = offset.int()

    retain = _ext.greedy_reduction_packed(
        sorted_idx_int,
        idx_int,
        offset_int,
        -1
    )

    centroids = coord[retain]
    centroid_batch = batch[retain]
    centroid_counts = torch.bincount(centroid_batch, minlength=B)
    centroid_offset = torch.cumsum(centroid_counts, dim=0)
    return centroids, centroid_offset, retain


@torch.no_grad()
def cnms_padded(
    centroids: torch.Tensor,
    radius: float,
    overlap_factor: float = 0.5,
    K: Optional[int] = None,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CNMS for padded tensor format (original PoLAr-MAE interface).

    Args:
        centroids: (B, P, 3) padded point coordinates
        radius: ball radius
        overlap_factor: overlap factor (0.5 = 50% diameter overlap)
        K: max neighbors for overlap checking (default: P)
        lengths: (B,) actual lengths per batch (default: all P)

    Returns:
        centroids: (B, P, 3) reordered centroids (retained first)
        lengths: (B,) number of retained centroids per batch
        reorder_idx: (B, P) reordering indices
    """
    if not HAS_PYTORCH3D:
        raise ImportError("pytorch3d is required for cnms_padded")

    B, P, D = centroids.shape
    device = centroids.device

    if lengths is None:
        lengths = torch.full((B,), P, dtype=torch.long, device=device)

    if K is None:
        K = P

    query_radius = 2 * radius * overlap_factor

    _, idx, _ = pytorch3d_ball_query(
        p1=centroids,
        p2=centroids,
        K=K,
        radius=query_radius,
        lengths1=lengths,
        lengths2=lengths,
        return_nn=False,
    )

    overlap_counts = (idx != -1).sum(dim=-1)
    _, sorted_indices = overlap_counts.sort(dim=-1, descending=True)

    retain = _ext.greedy_reduction_padded(sorted_indices, idx, lengths, -1)

    reorder_idx = torch.argsort((~retain).float(), dim=1)
    centroids_reordered = centroids.gather(
        dim=1,
        index=reorder_idx.unsqueeze(-1).expand(-1, -1, D)
    )
    new_lengths = retain.sum(dim=1)

    return centroids_reordered, new_lengths, reorder_idx


# Backward-compatible alias: `from cnms import cnms` works as the padded API.
cnms = cnms_padded
