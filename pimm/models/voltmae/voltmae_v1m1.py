"""Volt-MAE: MAE pretraining with the Volt backbone.

Tokenizes a point cloud via a strided sparse conv into non-overlapping
voxel patches, masks 60% of patches, encodes only unmasked patches with a
Volt-style transformer, and predicts per-patch dense sub-voxel energy
volumes for masked patches. Because each sub-voxel holds at most one
point after GridSample and energies are non-negative, the target doubles
as an occupancy signal (zero ⇒ empty).
"""

from __future__ import annotations

from typing import Optional

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from pimm.models.builder import MODELS
from pimm.models.modules import PointModel
from pimm.models.utils.misc import offset2batch
from pimm.models.voltmae.layers import (
    Block,
    RoPE,
    ReconHead,
    build_point_to_token,
    build_targets,
    focal_bce_with_logits,
    occ_supervision_mask,
    random_token_mask,
    reconstruction_diagnostics,
    sort_tokens_by_batch,
)
from pimm.utils.logger import get_logger

logger = get_logger(__name__)


@MODELS.register_module("Volt-MAE-v1m1")
class VoltMAE(PointModel):
    """Volt backbone + MAE pretext task on voxel patches."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 384,
        enc_depth: int = 12,
        dec_depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        qk_norm: bool = True,
        drop_path: float = 0.3,
        stride: int = 5,
        kernel_size: int = 5,
        mask_ratio: float = 0.6,
        increase_drop_path: bool = True,
        energy_key: str = "energy",
        rope_max_grid_size: tuple = (1024, 1024, 1024),
        rope_freq_split: tuple = (5, 5, 6),
        occ_loss_weight: float = 1.0,
        energy_loss_weight: float = 1.0,
        occ_focal_gamma: float = 0.0,
        occ_focal_alpha: Optional[float] = None,
        occ_dilate: int = 0,
        occ_empty_beta: float = 1.0,
        diag_thresholds: tuple[float, ...] = (0.7,),
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.stride = stride
        self.kernel_size = kernel_size
        self.mask_ratio = mask_ratio
        self.energy_key = energy_key
        self.sub_voxels = stride**3
        self.occ_loss_weight = occ_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.occ_focal_gamma = occ_focal_gamma
        self.occ_focal_alpha = occ_focal_alpha
        self.occ_dilate = occ_dilate
        self.occ_empty_beta = occ_empty_beta
        self.diag_thresholds = tuple(diag_thresholds)

        # Tokenizer — named `tokenizer` to match Volt's checkpoint keys so the
        # encoder is portable to downstream Volt fine-tune configs.
        self.tokenizer = spconv.SparseConv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            indice_key="embedding",
        )

        # Encoder blocks — stored as `blocks` to match Volt's state dict layout.
        total_depth = enc_depth
        if increase_drop_path:
            enc_dp = torch.linspace(0, drop_path, total_depth).tolist()
        else:
            enc_dp = [drop_path] * total_depth

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=enc_dp[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )

        # Decoder blocks — MAE-only, discarded at fine-tune time.
        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=0.0,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(dec_depth)
            ]
        )

        # RoPE: sum(freq_split) must equal h_dim // 2 so each head's
        # h_dim channels pair up into `sum(freq_split)` complex rotary channels.
        h_dim = embed_dim // num_heads
        assert sum(rope_freq_split) * 2 == h_dim, (
            f"rope_freq_split {tuple(rope_freq_split)} sums to {sum(rope_freq_split)}, "
            f"but h_dim//2 = {h_dim // 2} for embed_dim={embed_dim}, num_heads={num_heads}. "
            f"Adjust rope_freq_split so that sum(..)*2 == embed_dim // num_heads."
        )
        self.pos_enc = RoPE(
            freq_split=tuple(rope_freq_split),
            max_grid_size=tuple(rope_max_grid_size),
        )
        self.recon_head = ReconHead(embed_dim, kernel=stride)

        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

        logger.info(
            f"Volt-MAE: in_channels={in_channels}, embed_dim={embed_dim}, "
            f"enc_depth={enc_depth}, dec_depth={dec_depth}, stride={stride}, "
            f"mask_ratio={mask_ratio}, "
            f"w_occ={occ_loss_weight}, w_energy={energy_loss_weight}, "
            f"focal(gamma={occ_focal_gamma}, alpha={occ_focal_alpha}), "
            f"dilate={occ_dilate}, empty_beta={occ_empty_beta}"
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    @staticmethod
    def _compute_cu_seqlens(batch_indices: torch.Tensor):
        counts = torch.bincount(batch_indices)
        cu = torch.zeros(
            counts.numel() + 1, dtype=torch.int32, device=batch_indices.device
        )
        cu[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
        max_seqlen = int(counts.max().item()) if counts.numel() else 0
        return cu, max_seqlen

    def forward(self, data_dict):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        if "batch" in data_dict:
            batch = data_dict["batch"]
        else:
            batch = offset2batch(data_dict["offset"])
        energy = data_dict[self.energy_key]

        # 1. Build sparse tensor. The +96 buffer mirrors the Volt reference
        #    implementation (libs/Volt/pointcept/models/volt/volt_base.py:240).
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=int(batch[-1].item()) + 1,
        )

        # 2. Tokenize → (T, D) patch tokens at indices (T, 4).
        x = self.tokenizer(x)
        token_features, token_indices = sort_tokens_by_batch(
            x.features, x.indices.long()
        )
        T = token_features.shape[0]

        # 3. Align input points to their parent token, build dense targets.
        point_to_token = build_point_to_token(
            grid_coord.long(), batch.long(), token_indices, self.stride
        )
        energy_target, occ_target = build_targets(
            point_to_token,
            grid_coord.long(),
            token_indices,
            energy,
            self.stride,
            T,
        )  # each (T, stride**3)

        # 4. RoPE position codes for every token.
        freqs_cis_full = self.pos_enc.compute_axial_cis_efficient(token_indices[:, 1:])

        # 5. Patch-level random masking.
        token_batch_ids = token_indices[:, 0].to(torch.int64)
        (
            ids_kept,
            ids_masked,
            cu_kept,
            max_kept,
            cu_full,
            max_full,
        ) = random_token_mask(token_batch_ids, self.mask_ratio)

        if ids_masked.numel() == 0:
            # Degenerate case (tiny batch / mask_ratio=0) — return zero loss but
            # keep the graph connected so DDP doesn't complain.
            loss = token_features.sum() * 0.0
            zero = loss.detach()
            return {
                "loss": loss,
                "occ_loss": zero,
                "occ_loss_pos": zero,
                "occ_loss_neg": zero,
                "energy_loss": zero,
                "mean_tokens": torch.tensor(
                    float(T) / max(1, int(batch[-1].item()) + 1),
                    device=feat.device,
                ),
                "mean_masked": torch.tensor(0.0, device=feat.device),
                "mean_occupied_per_masked": torch.tensor(0.0, device=feat.device),
                "mean_border_per_masked": torch.tensor(0.0, device=feat.device),
            }

        # 6. Encoder on unmasked tokens only.
        enc_tokens = token_features.index_select(0, ids_kept)
        enc_freqs = freqs_cis_full[:, ids_kept]
        for blk in self.blocks:
            enc_tokens = blk(enc_tokens, enc_freqs, cu_kept, max_kept)

        # 7. Reassemble full token sequence: encoder outputs in kept slots,
        #    learnable mask token in masked slots. Use enc_tokens dtype so
        #    AMP/autocast-produced dtypes match.
        full_tokens = torch.zeros(
            T, enc_tokens.shape[1], dtype=enc_tokens.dtype, device=enc_tokens.device
        )
        full_tokens.index_copy_(0, ids_kept, enc_tokens)
        mask_tok = self.mask_token.to(full_tokens.dtype).expand(ids_masked.numel(), -1)
        full_tokens.index_copy_(0, ids_masked, mask_tok)

        # 8. Decoder on full token set.
        dec_tokens = full_tokens
        for blk in self.decoder_blocks:
            dec_tokens = blk(dec_tokens, freqs_cis_full, cu_full, max_full)

        # 9. Dense sub-voxel prediction: (T, 2, stride**3) → (occ_logits, energy).
        recon = self.recon_head(dec_tokens)  # (T, 2, stride**3)
        occ_logits = recon[..., 0, :].float()  # (T, s^3)
        energy_pred = recon[..., 1, :].float()  # (T, s^3)

        # Slice to masked tokens only.
        occ_logits_m = occ_logits.index_select(0, ids_masked)
        energy_pred_m = energy_pred.index_select(0, ids_masked)
        occ_target_m = occ_target.index_select(0, ids_masked)
        energy_target_m = energy_target.index_select(0, ids_masked)

        # Occupancy: supervision mask (optional dilation + negative subsampling)
        #     focal BCE (optional).
        (
            sup_mask_m,
            sup_targ_m,
            pos_mask_m,
            border_mask_m,
            neg_mask_m,
        ) = occ_supervision_mask(
            occ_target_m,
            stride=self.stride,
            dilate=self.occ_dilate,
            empty_beta=self.occ_empty_beta,
        )

        occ_per_elem = focal_bce_with_logits(
            occ_logits_m,
            sup_targ_m,
            gamma=self.occ_focal_gamma,
            alpha=self.occ_focal_alpha,
            reduction="none",
        )
        if sup_mask_m.any():
            occ_loss = occ_per_elem[sup_mask_m].mean()
        else:
            occ_loss = occ_per_elem.sum() * 0.0

        # Energy: MSE restricted to sub-voxels that are actually occupied
        # (pos_mask_m — NOT the dilated border, which has no ground-truth energy).
        # Teacher-forced occupancy means the MSE only has to model the energy
        # distribution where points exist, not pull toward zero everywhere else —
        # that job belongs to the occ head.
        if pos_mask_m.any():
            energy_loss = F.mse_loss(
                energy_pred_m[pos_mask_m],
                energy_target_m[pos_mask_m],
                reduction="mean",
            )
        else:
            energy_loss = energy_pred_m.sum() * 0.0

        loss = self.occ_loss_weight * occ_loss + self.energy_loss_weight * energy_loss

        # Diagnostic breakdown (guarded for empty masks).
        def _safe_mean(t, mask):
            return t[mask].mean() if mask.any() else t.new_zeros(())

        pos_or_border_m = pos_mask_m | border_mask_m
        occ_loss_pos = _safe_mean(occ_per_elem, pos_or_border_m).detach()
        occ_loss_neg = _safe_mean(occ_per_elem, neg_mask_m).detach()

        B = int(batch[-1].item()) + 1
        M = max(1, ids_masked.numel())
        output = {
            "loss": loss,
            "occ_loss": occ_loss.detach(),
            "occ_loss_pos": occ_loss_pos,
            "occ_loss_neg": occ_loss_neg,
            "energy_loss": energy_loss.detach(),
            "mean_tokens": torch.tensor(float(T) / max(1, B), device=feat.device),
            "mean_masked": torch.tensor(
                float(ids_masked.numel()) / max(1, B), device=feat.device
            ),
            "mean_occupied_per_masked": (pos_mask_m.sum().float() / M).detach(),
            "mean_border_per_masked": (border_mask_m.sum().float() / M).detach(),
        }
        if self.diag_thresholds:
            output.update(
                reconstruction_diagnostics(
                    occ_logits_m,
                    pos_mask_m,
                    border_mask_m,
                    thresholds=self.diag_thresholds,
                )
            )
        return output

    @torch.no_grad()
    def encode(self, data_dict):
        """Per-point encoder features for downstream linear probing.

        Runs the tokenizer and the (unmasked) encoder, then broadcasts each
        token's output to every input point that falls inside its 5x5x5
        parent patch. This is consumed by PretrainEvaluator, which expects
        packed (N, C) per-point features paired with point-level labels.
        """
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        if "batch" in data_dict:
            batch = data_dict["batch"]
        else:
            batch = offset2batch(data_dict["offset"])

        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=int(batch[-1].item()) + 1,
        )
        x = self.tokenizer(x)
        tokens, token_indices = sort_tokens_by_batch(
            x.features, x.indices.long()
        )

        cu_full, max_full = self._compute_cu_seqlens(token_indices[:, 0])
        freqs = self.pos_enc.compute_axial_cis_efficient(token_indices[:, 1:])
        for blk in self.blocks:
            tokens = blk(tokens, freqs, cu_full, max_full)

        p2t = build_point_to_token(
            grid_coord.long(), batch.long(), token_indices, self.stride
        )
        return tokens.index_select(0, p2t)  # (N_points, embed_dim)
