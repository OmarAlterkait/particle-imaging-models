"""Sanity checks for build_point_to_token + build_targets.

Run with:
    conda run -n pointcept-torch2.5.0-cu12.4 python -m pimm.models.voltmae._test_alignment
"""

from __future__ import annotations

import spconv.pytorch as spconv
import torch

from pimm.models.voltmae.layers import (
    build_point_to_token,
    build_targets,
    sort_tokens_by_batch,
)


def test_alignment_basic():
    """Handcrafted 3 batches, 20 points. Verify tokenizer outputs
    `input_idx // stride` and that we can recover every point's token."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Points: (batch, x, y, z) in voxel indices. Chosen so several points
    # share a parent 5-voxel patch, and some patches contain 1 vs many points.
    raw = torch.tensor(
        [
            # batch 0, patch (0,0,0)  →  sub-voxels (0,0,0), (1,2,3), (4,4,4)
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 4, 4, 4],
            # batch 0, patch (1,0,0)  →  sub (0,0,0) and (3,3,3)
            [0, 5, 0, 0],
            [0, 8, 3, 3],
            # batch 0, patch (2,3,4)  →  sub (2,2,1)
            [0, 12, 17, 21],
            # batch 1, patch (0,0,0)  →  sub (0,0,0)
            [1, 0, 0, 0],
            # batch 1, patch (1,1,1)  →  sub (0,0,0), (4,4,4)
            [1, 5, 5, 5],
            [1, 9, 9, 9],
            # batch 2, patch (10, 10, 10) → sub (2, 2, 2)
            [2, 52, 52, 52],
        ],
        dtype=torch.int64,
        device=device,
    )
    batch = raw[:, 0].int()
    grid_coord = raw[:, 1:].int()
    N = raw.shape[0]

    # Energy per point (unique positive values so we can check scatter is right)
    energy = torch.arange(1, N + 1, dtype=torch.float32, device=device).unsqueeze(-1) * 0.1

    # Construct a 4-channel feat like the real pipeline would.
    feat = torch.cat([grid_coord.float(), energy], dim=-1)

    # Build SparseConvTensor + run a strided tokenizer.
    stride = 5
    sparse_shape = (int(grid_coord.max().item()) + 96,) * 3
    indices = torch.cat([batch.unsqueeze(-1), grid_coord], dim=1).contiguous().int()
    st = spconv.SparseConvTensor(
        features=feat,
        indices=indices,
        spatial_shape=list(sparse_shape),
        batch_size=int(batch.max().item()) + 1,
    )

    tokenizer = spconv.SparseConv3d(
        in_channels=4, out_channels=8,
        kernel_size=5, stride=5, bias=True, indice_key="embedding",
    ).to(device)

    out = tokenizer(st)
    raw_token_indices = out.indices.long()
    raw_is_batch_contiguous = bool(
        (raw_token_indices[:-1, 0] <= raw_token_indices[1:, 0]).all()
    )
    print(f"raw tokenizer output is batch-contiguous: {raw_is_batch_contiguous}")
    _, token_indices = sort_tokens_by_batch(out.features, raw_token_indices)
    assert bool((token_indices[:-1, 0] <= token_indices[1:, 0]).all())
    T = token_indices.shape[0]
    print(f"N points = {N}, T tokens = {T}")
    print("token_indices:", token_indices.tolist())

    # Check assumption: each point's parent token (batch, coord//5) must be
    # in token_indices.
    point_parents = torch.cat(
        [batch.long().unsqueeze(-1), (grid_coord.long() // stride)], dim=1
    )
    print("point parent ids:", point_parents.tolist())

    # Use the alignment helper.
    p2t = build_point_to_token(
        grid_coord.long(), batch.long(), token_indices, stride
    )
    print("point→token:", p2t.tolist())

    # Verify mapping is correct
    for i in range(N):
        expected = point_parents[i]
        got = token_indices[p2t[i]]
        assert torch.equal(expected, got), f"mismatch at point {i}: {expected} vs {got}"
    print("✓ build_point_to_token: every point maps to its parent token")

    # Target construction
    energy_target, occ_target = build_targets(
        p2t, grid_coord.long(), token_indices, energy, stride, T,
    )
    print(
        f"energy_target shape: {energy_target.shape}, "
        f"nonzero: {(energy_target != 0).sum().item()}/{energy_target.numel()}"
    )
    print(
        f"occ_target shape: {occ_target.shape}, "
        f"occupied: {int(occ_target.sum().item())}/{occ_target.numel()}"
    )

    # Hand check: sum of all energies conserved through the energy scatter.
    assert torch.allclose(energy_target.sum(), energy.sum()), (
        f"energy_target sum {energy_target.sum().item()} != energy sum {energy.sum().item()}"
    )
    # Hand check: occupancy count matches number of input points (assuming no
    # sub-voxel collisions — true for this handcrafted fixture).
    assert int(occ_target.sum().item()) == N, (
        f"occ_target sum {int(occ_target.sum().item())} != N points {N}"
    )
    print("✓ build_targets: total energy conserved; occupancy sum = N points")

    # Check one specific point ended up in the right sub-voxel slot.
    # Point 1 is at (0, 1, 2, 3), parent patch (0, 0, 0, 0), sub-voxel (1,2,3),
    # linear index = 1*25 + 2*5 + 3 = 38. Energy = 0.2.
    tok_00 = torch.where(
        (token_indices[:, 0] == 0) & (token_indices[:, 1:].eq(torch.tensor(
            [0, 0, 0], device=device, dtype=torch.int64
        )).all(-1))
    )[0]
    assert tok_00.numel() == 1
    assert torch.isclose(energy_target[tok_00[0], 38], torch.tensor(0.2, device=device))
    assert occ_target[tok_00[0], 38].item() == 1.0
    print("✓ build_targets: specific sub-voxel slot verified (energy + occupancy)")


def test_occ_supervision_mask():
    """Hand-check dilation + negative subsampling on a synthetic 5³ patch."""
    from pimm.models.voltmae.layers import occ_supervision_mask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = 5
    s3 = stride ** 3

    # One patch, two positives: corner (0,0,0) and center (2,2,2).
    occ = torch.zeros(1, s3, device=device)
    def ijk_to_flat(i, j, k):
        return i * stride * stride + j * stride + k
    occ[0, ijk_to_flat(0, 0, 0)] = 1
    occ[0, ijk_to_flat(2, 2, 2)] = 1

    # dilate=1 → each positive becomes a 3×3×3 block; check counts + ids.
    sup_mask, sup_targ, pos_mask, border_mask, neg_mask = occ_supervision_mask(
        occ, stride=stride, dilate=1, empty_beta=1.0
    )
    assert pos_mask.sum().item() == 2
    # Corner (0,0,0) dilated → 2*2*2 = 8 voxels (the corner gets clipped);
    # center (2,2,2) dilated → full 3*3*3 = 27 voxels.
    # Borders exclude the original positives themselves.
    # Corner contribution to border: 8 - 1 = 7
    # Center contribution to border: 27 - 1 = 26
    # They don't overlap (corner block is {0,1}×{0,1}×{0,1}; center block is
    # {1,2,3}×{1,2,3}×{1,2,3} — they share (1,1,1) only).
    # Shared border voxel: (1,1,1) is in both dilated blocks and in neither positive.
    # So total unique border = 7 + 26 - 1 = 32.
    expected_border = 7 + 26 - 1
    assert border_mask.sum().item() == expected_border, (
        f"border count {border_mask.sum().item()} vs expected {expected_border}"
    )
    # Positives and border are disjoint
    assert not (pos_mask & border_mask).any()
    # sup_targ is 1 on pos+border, 0 elsewhere
    assert torch.equal(sup_targ, (pos_mask | border_mask).float())
    # empty_beta=1.0 → every non-positive non-border sub-voxel is supervised negative
    assert neg_mask.sum().item() == s3 - 2 - expected_border
    assert sup_mask.sum().item() == s3  # all supervised
    print("✓ occ_supervision_mask: dilation=1, empty_beta=1.0 — shapes + ids verified")

    # empty_beta=0.5 should drop ~half of the negatives (stochastic).
    torch.manual_seed(42)
    _, _, pm2, bm2, nm2 = occ_supervision_mask(
        occ, stride=stride, dilate=1, empty_beta=0.5
    )
    empties = s3 - 2 - expected_border
    n_sampled = int(nm2.sum().item())
    assert abs(n_sampled - empties * 0.5) < empties * 0.25, (
        f"empty_beta=0.5 sampled {n_sampled}/{empties}, expected ~{empties/2}"
    )
    print(f"✓ occ_supervision_mask: empty_beta=0.5 sampled {n_sampled}/{empties} empties")

    # dilate=0 → border should be empty
    _, _, pm0, bm0, _ = occ_supervision_mask(
        occ, stride=stride, dilate=0, empty_beta=1.0
    )
    assert bm0.sum().item() == 0
    assert torch.equal(pm0, pos_mask)
    print("✓ occ_supervision_mask: dilate=0 — no border")


def test_focal_bce():
    """Focal BCE: gamma=0 recovers plain BCE; gamma>0 reduces negatives."""
    from pimm.models.voltmae.layers import focal_bce_with_logits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5], device=device)
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)

    import torch.nn.functional as F
    bce_plain = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce_via_focal = focal_bce_with_logits(logits, targets, gamma=0.0, reduction="none")
    assert torch.allclose(bce_plain, bce_via_focal, atol=1e-6)
    print("✓ focal_bce_with_logits: gamma=0 matches plain BCE")

    # With gamma=2.0, confident-and-correct predictions are down-weighted.
    bce_focal = focal_bce_with_logits(logits, targets, gamma=2.0, reduction="none")
    # The confidently-correct pair (logit=2, target=1) should drop the most.
    ratio_confident = (bce_focal[0] / bce_plain[0]).item()
    ratio_uncertain = (bce_focal[3] / bce_plain[3]).item()  # logit=-0.5, target=1 (wrong direction)
    assert ratio_confident < 0.1, f"confident-correct should be heavily downweighted, got {ratio_confident}"
    assert ratio_uncertain > ratio_confident, (
        f"uncertain should be less downweighted than confident, got {ratio_uncertain} vs {ratio_confident}"
    )
    print(f"✓ focal_bce_with_logits: gamma=2 downweights confident (ratio={ratio_confident:.3f}) vs uncertain (ratio={ratio_uncertain:.3f})")


def test_encode_batch_invariance():
    """Encoding an event alone must match encoding it inside a larger batch."""
    if not torch.cuda.is_available():
        print("skipping encode batch-invariance check: CUDA is not available")
        return

    from pimm.models.utils.misc import offset2batch
    from pimm.models.voltmae.voltmae_v1m2 import VoltMAE

    device = "cuda"
    torch.manual_seed(7)
    model = VoltMAE(
        in_channels=4,
        embed_dim=64,
        enc_depth=2,
        dec_depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path=0.0,
        stride=5,
        kernel_size=5,
        stem_channels=8,
        stem_layers=1,
        mask_ratio=0.5,
        rope_max_grid_size=(64, 64, 64),
        rope_freq_split=(3, 3, 2),
        occ_dilate=0,
    ).to(device).eval()

    def make_event(seed: int, n_tokens: int = 96):
        gen = torch.Generator(device=device).manual_seed(seed)
        flat = torch.randperm(8 * 8 * 8, generator=gen, device=device)[:n_tokens]
        parent = torch.stack(
            [flat // 64, (flat // 8) % 8, flat % 8], dim=1
        ).to(torch.int64)
        sub = torch.randint(0, 5, (n_tokens, 3), generator=gen, device=device)
        grid = parent * 5 + sub
        feat = torch.randn(n_tokens, 4, generator=gen, device=device)
        return grid.int(), feat

    grid0, feat0 = make_event(11)
    grid1, feat1 = make_event(23)
    n0, n1 = grid0.shape[0], grid1.shape[0]

    alone = {
        "grid_coord": grid0,
        "feat": feat0,
        "offset": torch.tensor([n0], dtype=torch.long, device=device),
    }
    batched = {
        "grid_coord": torch.cat([grid0, grid1], dim=0),
        "feat": torch.cat([feat0, feat1], dim=0),
        "offset": torch.tensor([n0, n0 + n1], dtype=torch.long, device=device),
    }

    # Show the raw tokenizer invariant that motivated the sort.
    batch = offset2batch(batched["offset"])
    patch_ids, num_patches = model._compute_patch_ids(
        batched["grid_coord"], batch, model.stride
    )
    mixed = model.stem(batched["feat"], patch_ids, num_patches)
    indices = torch.cat(
        [batch.unsqueeze(-1).int(), batched["grid_coord"].int()], dim=1
    ).contiguous()
    st = spconv.SparseConvTensor(
        features=mixed,
        indices=indices,
        spatial_shape=torch.add(torch.max(batched["grid_coord"], dim=0).values, 96).tolist(),
        batch_size=2,
    )
    raw = model.tokenizer(st).indices.long()
    raw_is_batch_contiguous = bool((raw[:-1, 0] <= raw[1:, 0]).all())
    print(f"raw batched tokenizer output is batch-contiguous: {raw_is_batch_contiguous}")

    with torch.inference_mode():
        encoded_alone = model.encode(alone)
        encoded_batched = model.encode(batched)[:n0]

    diff = (encoded_alone - encoded_batched).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    print(f"encode batch-invariance diff: max={max_diff:.6f}, mean={mean_diff:.6f}")
    assert max_diff < 5e-2 and mean_diff < 5e-3, (
        "encode(event) changed when another event was added to the batch; "
        f"max diff={max_diff}, mean diff={mean_diff}"
    )
    print("✓ encode batch-invariance: event features are stable across batching")


if __name__ == "__main__":
    test_alignment_basic()
    test_occ_supervision_mask()
    test_focal_bce()
    test_encode_batch_invariance()
    print("\nAll alignment tests passed.")
