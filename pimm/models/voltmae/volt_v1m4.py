"""Volt-v1m4: DINO-capable Volt backbone with continuous RoPE jitter.

This is a standalone copy of the Volt-v1m3 backbone with one intentional
change: RoPE is computed from token-level continuous coordinates, with optional
PT-v3m8-style train-time coordinate jitter/rescaling. It does not inherit from
Volt-v1m3, so this file is self-contained for ablation/debugging.
"""

from __future__ import annotations

from typing import Optional

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter
from torch.nn.init import trunc_normal_

from pimm.models.builder import MODELS
from pimm.models.modules import PointModel
from pimm.models.utils.structure import Point
from pimm.models.voltmae.layers import (
    Block,
    _pack_indices,
    build_point_to_token,
    sort_tokens_by_batch,
)


class ContinuousRoPE(nn.Module):
    def __init__(
        self,
        theta: float = 10.0,
        freq_split: tuple = (11, 11, 10),
        jitter_degree: Optional[float] = None,
        rescale_degree: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.jitter_degree = jitter_degree
        self.rescale_degree = rescale_degree

        freqs_x = 1.0 / theta ** torch.linspace(0, 1, freq_split[0])
        freqs_y = 1.0 / theta ** torch.linspace(0, 1, freq_split[1])
        freqs_z = 1.0 / theta ** torch.linspace(0, 1, freq_split[2])
        self.register_buffer("freqs_x", freqs_x, persistent=False)
        self.register_buffer("freqs_y", freqs_y, persistent=False)
        self.register_buffer("freqs_z", freqs_z, persistent=False)

    def _should_augment(self) -> bool:
        if not self.training:
            return False
        return (
            self.jitter_degree is not None
            and self.jitter_degree > 1
            or self.rescale_degree is not None
            and self.rescale_degree > 1
        )

    @staticmethod
    def _compute_cis(coord: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        freqs = freqs.to(device=coord.device, dtype=coord.dtype)
        angles = coord[:, None] * freqs[None, :]
        return torch.polar(torch.ones_like(angles), angles)

    @torch.no_grad()
    def _augment_coords(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(dtype=self.freqs_x.dtype)

        if self.jitter_degree is not None and self.jitter_degree > 1:
            log_gamma = coords.new_tensor(self.jitter_degree).log()
            jitter = torch.empty(3, device=coords.device, dtype=coords.dtype).uniform_(
                -log_gamma,
                log_gamma,
            )
            coords = coords * jitter.exp()

        if self.rescale_degree is not None and self.rescale_degree > 1:
            log_eta = coords.new_tensor(self.rescale_degree).log()
            rescale = torch.empty(1, device=coords.device, dtype=coords.dtype).uniform_(
                -log_eta,
                log_eta,
            )
            coords = coords * rescale.exp()

        return coords

    def compute_axial_cis_efficient(self, coords):
        coords = coords.to(dtype=self.freqs_x.dtype)
        if self._should_augment():
            coords = self._augment_coords(coords)

        cis_x = self._compute_cis(coords[:, 0], self.freqs_x)
        cis_y = self._compute_cis(coords[:, 1], self.freqs_y)
        cis_z = self._compute_cis(coords[:, 2], self.freqs_z)
        return torch.cat([cis_x, cis_y, cis_z], dim=-1).unsqueeze(0)


@MODELS.register_module("Volt-v1m4")
class VoltV1M4Backbone(PointModel):
    def __init__(
        self,
        in_channels: int = 4,
        stem_channels: int = 32,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        qk_norm: bool = True,
        drop_path: float = 0.3,
        increase_drop_path: bool = True,
        stride: int = 5,
        kernel_size: int = 5,
        mask_token: bool = True,
        final_norm: bool = True,
        rope_max_grid_size: tuple = (1024, 1024, 1024),
        rope_freq_split: tuple = (11, 11, 10),
        direct_tokenizer: bool = True,
        token_mask_token: bool = True,
        num_cls_tokens: int = 1,
        num_register_tokens: int = 4,
        rope_theta: float = 10.0,
        rope_jitter: Optional[float] = None,
        rope_rescale: Optional[float] = None,
        token_mask_rope_jitter: Optional[float] = None,
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.direct_tokenizer = bool(direct_tokenizer)
        self.num_cls_tokens = int(num_cls_tokens)
        self.num_register_tokens = int(num_register_tokens)
        self.num_special_tokens = self.num_cls_tokens + self.num_register_tokens
        self.token_mask_rope_jitter = token_mask_rope_jitter

        if self.direct_tokenizer:
            self.stem = nn.Identity()
            self.point_mask_token = None
            self.tokenizer = spconv.SparseConv3d(
                in_channels,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
                indice_key="embedding",
            )
        else:
            self.stem = spconv.SubMConv3d(
                in_channels,
                stem_channels,
                kernel_size=1,
                bias=True,
                indice_key="volt_stem",
            )
            if mask_token:
                self.point_mask_token = nn.Parameter(torch.zeros(stem_channels))
                trunc_normal_(self.point_mask_token, std=0.02)
            else:
                self.point_mask_token = None
            self.tokenizer = spconv.SparseConv3d(
                stem_channels,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
                indice_key="embedding",
            )

        if increase_drop_path:
            dp = torch.linspace(0, drop_path, depth).tolist()
        else:
            dp = [drop_path] * depth

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=dp[i],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim) if final_norm else nn.Identity()

        h_dim = embed_dim // num_heads
        assert sum(rope_freq_split) * 2 == h_dim, (
            f"rope_freq_split {tuple(rope_freq_split)} sums to {sum(rope_freq_split)}, "
            f"but h_dim//2 = {h_dim // 2} for embed_dim={embed_dim}, num_heads={num_heads}. "
            f"Adjust rope_freq_split so that sum(..)*2 == embed_dim // num_heads."
        )
        self.pos_enc = ContinuousRoPE(
            theta=rope_theta,
            freq_split=tuple(rope_freq_split),
            jitter_degree=rope_jitter,
            rescale_degree=rope_rescale,
        )

        self.apply(self._init_weights)
        if self.tokenizer.bias is not None:
            self.tokenizer.bias.data.zero_()
            self.tokenizer.bias.requires_grad_(False)

        if token_mask_token:
            self.token_mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
            trunc_normal_(self.token_mask_token, std=0.02)
        else:
            self.token_mask_token = None

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, self.embed_dim)
            )
            trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, self.embed_dim)
            )
            trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

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

    @staticmethod
    def _membership_mask(
        query_indices: torch.Tensor,
        selected_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if selected_indices is None or selected_indices.numel() == 0:
            return torch.zeros(
                query_indices.shape[0], device=query_indices.device, dtype=torch.bool
            )
        shape_hash = (
            int(
                max(
                    query_indices[:, 1:].max().item() if query_indices.numel() else 0,
                    selected_indices[:, 1:].max().item()
                    if selected_indices.numel()
                    else 0,
                )
            )
            + 2
        )
        query_hash = _pack_indices(query_indices[:, 0], query_indices[:, 1:], shape_hash)
        selected_hash = _pack_indices(
            selected_indices[:, 0], selected_indices[:, 1:], shape_hash
        )
        selected_hash = torch.unique(selected_hash)
        selected_hash, _ = selected_hash.sort()
        pos = torch.searchsorted(selected_hash, query_hash)
        pos = pos.clamp(max=selected_hash.numel() - 1)
        return selected_hash[pos] == query_hash

    def _jitter_masked_rope_coords(
        self,
        rope_coords: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.token_mask_rope_jitter is None
            or self.token_mask_rope_jitter <= 0
            or not token_mask.any()
        ):
            return rope_coords

        jitter = float(self.token_mask_rope_jitter)
        rope_coords = rope_coords.clone()
        noise = torch.randn_like(rope_coords[token_mask]).mul(jitter)
        noise = noise.clamp(min=-2 * jitter, max=2 * jitter)
        rope_coords[token_mask] = rope_coords[token_mask] + noise
        return rope_coords

    def _insert_special_tokens(
        self,
        patch_tokens: torch.Tensor,
        rope_coords: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        if self.num_special_tokens == 0:
            cu, max_seq = self._compute_cu_seqlens(token_indices[:, 0])
            patch_positions = torch.arange(
                patch_tokens.shape[0], device=patch_tokens.device
            )
            return patch_tokens, rope_coords, cu, max_seq, patch_positions, None, None

        batch_indices = token_indices[:, 0].long()
        counts = torch.bincount(batch_indices)
        starts = torch.cat(
            [counts.new_zeros(1), torch.cumsum(counts, dim=0)[:-1]], dim=0
        )
        seq_tokens = []
        seq_rope_coords = []
        patch_positions = []
        cls_positions = []
        register_positions = []
        cursor = 0
        device = patch_tokens.device

        for b, count in enumerate(counts.tolist()):
            start = int(starts[b].item())
            end = start + int(count)
            specials = []
            if self.cls_token is not None:
                specials.append(
                    self.cls_token.expand(1, -1, -1)
                    .reshape(self.num_cls_tokens, self.embed_dim)
                    .to(dtype=patch_tokens.dtype)
                )
            if self.register_tokens is not None:
                specials.append(
                    self.register_tokens.expand(1, -1, -1)
                    .reshape(self.num_register_tokens, self.embed_dim)
                    .to(dtype=patch_tokens.dtype)
                )
            specials = torch.cat(specials, dim=0)
            seq_tokens.append(torch.cat([specials, patch_tokens[start:end]], dim=0))
            seq_rope_coords.append(
                torch.cat(
                    [
                        torch.zeros(
                            self.num_special_tokens,
                            3,
                            device=device,
                            dtype=rope_coords.dtype,
                        ),
                        rope_coords[start:end],
                    ],
                    dim=0,
                )
            )

            if self.num_cls_tokens > 0:
                cls_positions.append(
                    torch.arange(
                        cursor,
                        cursor + self.num_cls_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                )
            if self.num_register_tokens > 0:
                reg_start = cursor + self.num_cls_tokens
                register_positions.append(
                    torch.arange(
                        reg_start,
                        reg_start + self.num_register_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                )
            patch_positions.append(
                torch.arange(
                    cursor + self.num_special_tokens,
                    cursor + self.num_special_tokens + count,
                    device=device,
                    dtype=torch.long,
                )
            )
            cursor += self.num_special_tokens + count

        seq_counts = counts + self.num_special_tokens
        cu = torch.zeros(seq_counts.numel() + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.cumsum(seq_counts.to(torch.int32), dim=0)
        return (
            torch.cat(seq_tokens, dim=0),
            torch.cat(seq_rope_coords, dim=0),
            cu,
            int(seq_counts.max().item()) if seq_counts.numel() else 0,
            torch.cat(patch_positions, dim=0),
            torch.cat(cls_positions, dim=0) if cls_positions else None,
            torch.cat(register_positions, dim=0) if register_positions else None,
        )

    def forward(self, point, return_tokens: bool = False):
        if not isinstance(point, Point):
            point = Point(point)
        if "grid_coord" not in point.keys():
            assert {"coord", "grid_size"}.issubset(point.keys()), (
                "Volt-v1m4 requires either `grid_coord` or both `coord` and "
                "`grid_size` on the input Point."
            )
            point["grid_coord"] = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()

        grid_coord = point.grid_coord
        batch = point.batch
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()
        x = spconv.SparseConvTensor(
            features=point.feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=int(batch[-1].item()) + 1,
        )

        x = self.stem(x)
        if (
            self.point_mask_token is not None
            and "mask" in point.keys()
            and point.mask is not None
        ):
            x = x.replace_feature(
                torch.where(
                    point.mask.unsqueeze(-1),
                    self.point_mask_token.to(x.features.dtype),
                    x.features,
                )
            )
        x = self.tokenizer(x)

        patch_tokens, token_indices = sort_tokens_by_batch(x.features, x.indices.long())
        point_to_token = build_point_to_token(
            grid_coord.long(), batch.long(), token_indices, self.stride
        )
        num_tokens = patch_tokens.shape[0]
        token_coord = torch_scatter.scatter_mean(
            point.coord, point_to_token, dim=0, dim_size=num_tokens
        )

        token_mask = self._membership_mask(
            token_indices,
            point.token_mask_grid.long() if "token_mask_grid" in point.keys() else None,
        )
        if self.token_mask_token is not None and token_mask.any():
            patch_tokens = torch.where(
                token_mask.unsqueeze(-1),
                self.token_mask_token.to(dtype=patch_tokens.dtype),
                patch_tokens,
            )
        token_coord = self._jitter_masked_rope_coords(token_coord, token_mask)

        (
            sequence_tokens,
            sequence_rope_coords,
            cu,
            max_seq,
            patch_positions,
            cls_positions,
            register_positions,
        ) = self._insert_special_tokens(patch_tokens, token_coord, token_indices)

        for blk in self.blocks:
            freqs = self.pos_enc.compute_axial_cis_efficient(sequence_rope_coords)
            sequence_tokens = blk(sequence_tokens, freqs, cu, max_seq)
        sequence_tokens = self.norm(sequence_tokens)

        patch_tokens = sequence_tokens[patch_positions]
        cls_tokens = (
            sequence_tokens[cls_positions].view(-1, self.num_cls_tokens, self.embed_dim)
            if cls_positions is not None
            else None
        )
        register_tokens = (
            sequence_tokens[register_positions].view(
                -1, self.num_register_tokens, self.embed_dim
            )
            if register_positions is not None
            else None
        )

        coarse_dict = dict(
            feat=patch_tokens,
            coord=token_coord,
            grid_coord=token_indices[:, 1:].int(),
            batch=token_indices[:, 0].long(),
            grid_size=point.grid_size * self.stride
            if "grid_size" in point.keys()
            else None,
            pooling_parent=point,
            pooling_inverse=point_to_token,
        )
        if "origin_coord" in point.keys():
            coarse_dict["origin_coord"] = torch_scatter.scatter_mean(
                point.origin_coord, point_to_token, dim=0, dim_size=num_tokens
            )
        if "segment_motif" in point.keys() or "segment" in point.keys():
            input_idx = torch.arange(point.feat.shape[0], device=point.feat.device)
            head_indices, _ = torch_scatter.scatter_min(
                input_idx, point_to_token, dim=0, dim_size=num_tokens
            )
            for label_key in ("segment_motif", "segment"):
                if label_key in point.keys():
                    coarse_dict[label_key] = point[label_key][head_indices]

        coarse_point = Point({k: v for k, v in coarse_dict.items() if v is not None})
        if not return_tokens:
            return coarse_point
        return dict(
            point=coarse_point,
            cls_tokens=cls_tokens,
            register_tokens=register_tokens,
            patch_tokens=patch_tokens,
            token_indices=token_indices,
            token_mask=token_mask,
            point_to_token=point_to_token,
        )
