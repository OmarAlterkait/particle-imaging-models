"""Building blocks for Volt-MAE.

Contains the Volt backbone transformer primitives (RoPE, RoPE_Attention, Block)
copied from libs/Volt/pointcept/models/volt/volt_base.py — kept verbatim to
preserve checkpoint compatibility with vanilla Volt downstream configs — plus
Volt-MAE-specific utilities: point<->token alignment, target construction,
random patch-level masking, and the reconstruction head.
"""

from __future__ import annotations

import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale


# ---------------------------------------------------------------------------
# Volt backbone primitives (verbatim from libs/Volt/pointcept/models/volt/volt_base.py)
# ---------------------------------------------------------------------------
class RoPE(nn.Module):
    def __init__(
        self,
        theta: float = 100.0,
        freq_split: tuple = (12, 12, 8),
        max_grid_size: tuple = (1024, 1024, 512),
    ) -> None:
        super().__init__()
        freqs_x = 1.0 / theta ** torch.linspace(0, 1, freq_split[0])
        freqs_y = 1.0 / theta ** torch.linspace(0, 1, freq_split[1])
        freqs_z = 1.0 / theta ** torch.linspace(0, 1, freq_split[2])

        self.register_buffer(
            "cis_cache_x", self._precompute(freqs_x, max_grid_size[0]), persistent=False
        )
        self.register_buffer(
            "cis_cache_y", self._precompute(freqs_y, max_grid_size[1]), persistent=False
        )
        self.register_buffer(
            "cis_cache_z", self._precompute(freqs_z, max_grid_size[2]), persistent=False
        )

    def _precompute(self, freqs, max_pos):
        freqs_pos = torch.outer(torch.arange(max_pos).float(), freqs)
        return torch.polar(torch.ones_like(freqs_pos), freqs_pos)

    def compute_axial_cis_efficient(self, indices):
        cis_x = self.cis_cache_x[indices[:, 0]]
        cis_y = self.cis_cache_y[indices[:, 1]]
        cis_z = self.cis_cache_z[indices[:, 2]]
        return torch.cat([cis_x, cis_y, cis_z], dim=-1).unsqueeze(0)


class RoPE_Attention(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.h_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(2)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(2)

        return q_out.type_as(q), k_out.type_as(k)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        N, C = x.shape
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.h_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q).to(q.dtype), self.k_norm(k).to(k.dtype)
        q, k = self.apply_rotary_emb(q, k, freqs_cis)
        qkv = torch.stack([q, k, v], dim=0).permute(2, 0, 1, 3)

        qkv_dtype = qkv.dtype
        x = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half(),
            cu_seqlens,
            max_seqlen=max_seqlen,
        )

        x = x.reshape(-1, C).to(qkv_dtype)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPE_Attention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), freqs_cis, cu_seq_lens, max_seqlen))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Volt-MAE specific utilities
# ---------------------------------------------------------------------------
def _pack_indices(batch: torch.Tensor, ijk: torch.Tensor, shape_hash: int) -> torch.Tensor:
    """Pack (batch, i, j, k) into a single int64 hash per row."""
    b = batch.to(torch.int64)
    i = ijk[..., 0].to(torch.int64)
    j = ijk[..., 1].to(torch.int64)
    k = ijk[..., 2].to(torch.int64)
    return ((b * shape_hash + i) * shape_hash + j) * shape_hash + k


def sort_tokens_by_batch(
    token_features: torch.Tensor,
    token_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort sparse-conv tokens by (batch, i, j, k).

    spconv does not guarantee batch-contiguous output indices, while
    flash_attn_varlen_qkvpacked_func requires tokens for each sequence to be
    laid out contiguously according to cu_seqlens.
    """
    if token_indices.numel() == 0:
        return token_features, token_indices

    shape_hash = int(token_indices[:, 1:].max().item()) + 2
    sort_key = _pack_indices(token_indices[:, 0], token_indices[:, 1:], shape_hash)
    order = torch.argsort(sort_key)
    return token_features.index_select(0, order), token_indices.index_select(0, order)


def build_point_to_token(
    grid_coord: torch.Tensor,
    batch: torch.Tensor,
    token_indices: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Map each input point to its parent token id.

    Args:
        grid_coord: (N, 3) int, point voxel indices at the fine resolution.
        batch: (N,) int, batch index per point.
        token_indices: (T, 4) int, (batch, i, j, k) from x.indices after the tokenizer.
        stride: tokenizer stride (e.g. 5). With SparseConv3d(padding=0) the parent
            token index of a fine voxel `p` is `p // stride`.

    Returns:
        point_to_token: (N,) int64, index into `token_indices` for each point.
    """
    parent = grid_coord // stride  # (N, 3)
    shape_hash = int(max(
        parent.max().item() if parent.numel() else 0,
        token_indices[:, 1:].max().item() if token_indices.numel() else 0,
    )) + 2

    point_hash = _pack_indices(batch, parent, shape_hash)
    token_hash = _pack_indices(token_indices[:, 0], token_indices[:, 1:], shape_hash)

    sorted_token_hash, token_order = token_hash.sort()
    pos = torch.searchsorted(sorted_token_hash, point_hash)
    # Guard against out-of-range before gather
    pos_clamped = pos.clamp(max=sorted_token_hash.numel() - 1)
    matched_hash = sorted_token_hash[pos_clamped]
    ok = matched_hash == point_hash
    if not ok.all():
        missing = (~ok).sum().item()
        raise RuntimeError(
            f"build_point_to_token: {missing}/{point_hash.numel()} points could not "
            f"be matched to any token. This indicates the tokenizer did not produce "
            f"an output voxel for every input parent — check stride/padding."
        )
    return token_order[pos_clamped]


def build_targets(
    point_to_token: torch.Tensor,
    grid_coord: torch.Tensor,
    token_indices: torch.Tensor,
    energy: torch.Tensor,
    stride: int,
    num_tokens: int,
):
    """Build per-token dense sub-voxel energy + occupancy targets.

    After GridSample at the fine resolution, at most one point occupies each
    sub-voxel. Occupancy is tracked by a dedicated scatter of ones so we
    don't have to infer it from energy (log-transformed energies can be
    negative, so `energy > 0` is unreliable).

    Args:
        point_to_token: (N,) int64 token id per point.
        grid_coord: (N, 3) fine-resolution voxel indices.
        token_indices: (T, 4) post-tokenizer indices.
        energy: (N, 1) or (N,) energy per point (may already be log-transformed).
        stride: tokenizer stride.
        num_tokens: T.

    Returns:
        energy_target: (T, stride**3) float, summed energies per sub-voxel.
        occ_target: (T, stride**3) float in {0, 1}, occupancy mask.
    """
    s3 = stride ** 3
    parent_ijk = token_indices[point_to_token, 1:]  # (N, 3)
    sub = grid_coord - parent_ijk * stride  # (N, 3) in [0, stride)
    sub_idx = (sub[:, 0] * stride + sub[:, 1]) * stride + sub[:, 2]  # (N,)

    eng = energy.squeeze(-1) if energy.dim() == 2 else energy
    eng = eng.to(torch.float32)

    flat_idx = point_to_token * s3 + sub_idx.to(point_to_token.dtype)
    energy_target = torch.zeros(num_tokens * s3, dtype=torch.float32, device=eng.device)
    energy_target.scatter_add_(0, flat_idx, eng)

    occ_target = torch.zeros(num_tokens * s3, dtype=torch.float32, device=eng.device)
    occ_target.scatter_add_(0, flat_idx, torch.ones_like(eng))
    occ_target.clamp_(max=1.0)

    return energy_target.view(num_tokens, s3), occ_target.view(num_tokens, s3)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    alpha: float | None = None,
    reduction: str = "none",
) -> torch.Tensor:
    """Numerically-stable focal BCE.

    gamma=0 recovers plain BCE. `alpha` (optional) reweights classes.
    Accepts soft targets in [0, 1] so future label smoothing works unchanged.
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma > 0.0:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        ce = (1.0 - p_t).pow(gamma) * ce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        ce = alpha_t * ce
    if reduction == "mean":
        return ce.mean()
    if reduction == "sum":
        return ce.sum()
    return ce


def occ_supervision_mask(
    occ_target: torch.Tensor,
    stride: int,
    dilate: int = 0,
    empty_beta: float = 1.0,
    generator: torch.Generator | None = None,
):
    """Build the per-sub-voxel supervision mask for occupancy.

    Args:
        occ_target: (M, stride**3) {0, 1} — raw per-sub-voxel occupancy for masked tokens.
        stride: tokenizer stride (patch side length).
        dilate: radius of the positive dilation; 0 disables.
        empty_beta: fraction of empties to supervise (1.0 = keep all, 0.0 = drop all).
        generator: optional torch.Generator for reproducible negative sampling.

    Returns:
        sup_mask    (M, s**3) bool  — entries to include in the occupancy loss
        sup_targ    (M, s**3) float — 1.0 on positives+border, 0.0 on sampled empties
        pos_mask    (M, s**3) bool  — raw positives (occ_target == 1)
        border_mask (M, s**3) bool  — dilated shell (not counting the positives themselves)
        neg_mask    (M, s**3) bool  — sampled empties
    """
    M, s3 = occ_target.shape
    assert s3 == stride ** 3, f"occ_target shape {occ_target.shape} mismatched to stride {stride}"

    pos_mask = occ_target > 0  # (M, s³)

    if dilate > 0:
        k = 2 * dilate + 1
        occ_grid = pos_mask.view(M, 1, stride, stride, stride).float()
        dilated = F.max_pool3d(occ_grid, kernel_size=k, stride=1, padding=dilate) > 0
        dilated = dilated.view(M, s3)
        border_mask = dilated & ~pos_mask
    else:
        border_mask = torch.zeros_like(pos_mask)

    positive_region = pos_mask | border_mask  # (M, s³)

    if empty_beta >= 1.0:
        neg_mask = ~positive_region
    elif empty_beta <= 0.0:
        neg_mask = torch.zeros_like(pos_mask)
    else:
        rand = torch.rand(
            occ_target.shape,
            device=occ_target.device,
            generator=generator,
            dtype=torch.float32,
        )
        neg_mask = (~positive_region) & (rand < empty_beta)

    sup_mask = positive_region | neg_mask
    sup_targ = positive_region.float()
    return sup_mask, sup_targ, pos_mask, border_mask, neg_mask


@torch.no_grad()
def reconstruction_diagnostics(
    occ_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    border_mask: torch.Tensor,
    thresholds: tuple[float, ...] = (0.7,),
    prefix: str = "recon",
) -> dict[str, torch.Tensor]:
    """Scalar diagnostics for masked-token occupancy reconstruction."""
    prob = torch.sigmoid(occ_logits.detach())
    true_target = pos_mask
    dilated_target = pos_mask | border_mask
    neg_mask = ~dilated_target

    zero = prob.new_zeros(())

    def masked_mean(mask):
        return prob[mask].mean() if mask.any() else zero

    def prf(pred, target):
        tp = (pred & target).sum().float()
        fp = (pred & ~target).sum().float()
        fn = (~pred & target).sum().float()
        precision = tp / (tp + fp + 1.0e-10)
        recall = tp / (tp + fn + 1.0e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1.0e-10)
        return precision, recall, f1

    def topk_recall():
        counts = true_target.sum(dim=1).long()
        valid = counts > 0
        if not valid.any():
            return zero
        order = prob.argsort(dim=1, descending=True)
        ranked_hits = true_target.gather(1, order).float()
        cumulative_hits = ranked_hits.cumsum(dim=1)
        gather_idx = (counts.clamp(min=1) - 1).unsqueeze(1)
        per_token = cumulative_hits.gather(1, gather_idx).squeeze(1)
        per_token = per_token / counts.clamp(min=1).float()
        return per_token[valid].mean()

    metrics = {
        f"{prefix}_prob_true": masked_mean(true_target),
        f"{prefix}_prob_border": masked_mean(border_mask),
        f"{prefix}_prob_negative": masked_mean(neg_mask),
        f"{prefix}_topk_true_occ_recall": topk_recall(),
    }

    for threshold in thresholds:
        tag = f"t{int(round(float(threshold) * 100)):02d}"
        pred = prob >= float(threshold)
        true_precision, true_recall, true_f1 = prf(pred, true_target)
        dil_precision, dil_recall, dil_f1 = prf(pred, dilated_target)
        metrics.update({
            f"{prefix}_true_precision_{tag}": true_precision,
            f"{prefix}_true_recall_{tag}": true_recall,
            f"{prefix}_true_f1_{tag}": true_f1,
            f"{prefix}_dilated_precision_{tag}": dil_precision,
            f"{prefix}_dilated_recall_{tag}": dil_recall,
            f"{prefix}_dilated_f1_{tag}": dil_f1,
            f"{prefix}_pred_per_masked_{tag}": pred.sum(dim=1).float().mean(),
        })

    return metrics


def random_token_mask(
    token_batch_ids: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
):
    """Per-batch random patch-level masking.

    For each event, randomly pick `mask_ratio` fraction of its tokens to mask.
    Returns index tensors plus cu_seqlens tensors (int32) sized for
    `flash_attn_varlen_qkvpacked_func`.
    """
    device = token_batch_ids.device
    T = token_batch_ids.numel()
    rand = torch.rand(T, device=device, generator=generator)

    # Sort by (batch_id, rand). Integer batch id dominates, fractional rand
    # shuffles within each batch.
    batch_counts = torch.bincount(token_batch_ids)
    B = batch_counts.numel()

    sort_key = token_batch_ids.to(torch.float64) + rand.to(torch.float64)
    order = torch.argsort(sort_key)
    # `order` lays out tokens grouped by batch, shuffled within each group.
    batch_offsets = torch.zeros(B + 1, dtype=torch.int64, device=device)
    batch_offsets[1:] = torch.cumsum(batch_counts, dim=0)

    keep_flags = torch.zeros(T, dtype=torch.bool, device=device)
    mask_flags = torch.zeros(T, dtype=torch.bool, device=device)
    kept_counts = torch.zeros(B, dtype=torch.int64, device=device)
    for b in range(B):
        start = batch_offsets[b].item()
        end = batch_offsets[b + 1].item()
        n = end - start
        n_mask = int(round(n * mask_ratio))
        n_mask = min(max(n_mask, 0), n)
        batch_order = order[start:end]
        mask_flags[batch_order[:n_mask]] = True
        keep_flags[batch_order[n_mask:]] = True
        kept_counts[b] = n - n_mask

    ids_kept = torch.nonzero(keep_flags, as_tuple=False).squeeze(1)
    ids_masked = torch.nonzero(mask_flags, as_tuple=False).squeeze(1)

    # Keep ids sorted by original token order so that scatter + flash_attn cu_seqlens
    # stay aligned with token_batch_ids ordering.
    ids_kept, _ = ids_kept.sort()
    ids_masked, _ = ids_masked.sort()

    # cu_seqlens: int32, shape (B+1,), prefix sum of per-batch counts
    cu_kept = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_kept[1:] = torch.cumsum(
        torch.bincount(token_batch_ids[ids_kept], minlength=B).to(torch.int32), dim=0
    )
    cu_full = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_full[1:] = torch.cumsum(batch_counts.to(torch.int32), dim=0)

    max_kept = int((cu_kept[1:] - cu_kept[:-1]).max().item()) if B else 0
    max_full = int(batch_counts.max().item()) if B else 0

    return ids_kept, ids_masked, cu_kept, max_kept, cu_full, max_full


class ReconHead(nn.Module):
    """Per-token dense sub-voxel head emitting (occ_logits, energy_pred).

    Single shared MLP with output width `num_targets * kernel**3`. Returns a
    tensor of shape `(..., num_targets, kernel**3)` so callers can index the
    occupancy and energy channels cleanly.
    """

    def __init__(
        self,
        dim: int,
        kernel: int = 5,
        hidden_mult: int = 2,
        num_targets: int = 2,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.kernel = kernel
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * hidden_mult, num_targets * kernel ** 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(self.norm(x))))
        return out.view(*out.shape[:-1], self.num_targets, self.kernel ** 3)
