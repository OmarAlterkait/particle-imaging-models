"""Volt-MAE pre-training on PILArNet.

Tokenizer: spconv.SparseConv3d(kernel=5, stride=5) — non-overlapping 5×5×5
voxel patches. Pretext task: dense per-patch sub-voxel energy prediction
with MSE loss on masked patches only. Energy doubles as occupancy signal
(zero = empty sub-voxel).
"""

_base_ = ["../_base_/default_runtime.py"]

# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------
batch_size = 48
num_worker = 12
batch_size_val = 32
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
clip_grad = 3.0
find_unused_parameters = False
seed = 0
num_events = 100_000

use_wandb = True
wandb_project = "Pretraining-VoltMAE-PILArNet"

warmup_ratio = 0.05
grid_size = 0.002  # → grid_coord max ≈ 665 per axis in the unit-sphere frame

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
model = dict(
    type="Volt-MAE",
    in_channels=4,          # (x, y, z, log-energy)
    embed_dim=768,
    enc_depth=12,
    dec_depth=4,
    num_heads=12,  # 12 × 32-dim heads → h_dim//2=16; z gets +1 freq (see rope_freq_split)
    mlp_ratio=4,
    init_values=None,
    qk_norm=True,
    drop_path=0.3,
    stride=5,
    kernel_size=5,
    mask_ratio=0.6,
    increase_drop_path=True,
    energy_key="energy",
    rope_max_grid_size=(1024, 1024, 1024),
    rope_freq_split=(11, 11, 10),   # sum must = embed_dim // num_heads = 64

    # Occupancy-loss refinements (see pimm/models/voltmae/layers.py::occ_supervision_mask
    # and ::focal_bce_with_logits). Combat the 4% positive-rate imbalance:
    occ_focal_gamma=2.0,     # focal BCE down-weights easy negatives
    occ_focal_alpha=None,    # no class reweighting (gamma alone carries it)
    occ_dilate=1,            # 1-voxel shell around each point is soft-positive
    occ_empty_beta=0.5,      # supervise 50% of empty sub-voxels per patch
)

# --------------------------------------------------------------------------
# Optimizer & scheduler
# --------------------------------------------------------------------------
epoch = 100
optimizer = dict(type="AdamW", lr=0.0004, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0004,
    pct_start=warmup_ratio,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# --------------------------------------------------------------------------
# Dataset
#
# Pipeline: LogTransform energy → NormalizeCoord → RandomRotate → GridSample
# (produces grid_coord + dedup + energy summation per voxel) → Collect.
# --------------------------------------------------------------------------
_log = dict(type="LogTransform", min_val=0.01, max_val=20.0, log=True, keys=("energy",))
_norm = dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=665.1076)
_rot_x = dict(type="RandomRotate", angle=[-1, 1], axis="x", always_apply=True, center=[0, 0, 0])
_rot_y = dict(type="RandomRotate", angle=[-1, 1], axis="y", always_apply=True, center=[0, 0, 0])
_rot_z = dict(type="RandomRotate", angle=[-1, 1], axis="z", always_apply=True, center=[0, 0, 0])
_grid = dict(
    type="GridSample",
    grid_size=grid_size,
    hash_type="fnv",
    mode="train",
    return_grid_coord=True,
    sum_keys=("energy",),
)

data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="train",
        transform=[
            _norm, _rot_x, _rot_y, _rot_z, _grid, _log,
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "energy"),
                feat_keys=("coord", "energy"),
            ),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=num_events,
        remove_low_energy_scatters=True,
        loop=1,
    ),
    val=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="val",
        transform=[
            _norm, _grid, _log, 
            dict(type="Copy", keys_dict={"segment_motif": "segment"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "energy", "segment"),
                feat_keys=("coord", "energy"),
            ),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=5000,
        remove_low_energy_scatters=False,
        loop=1,
    ),
    test=dict(
        type="PILArNetH5Dataset",
        revision="v1",
        split="val",
        transform=[
            _norm, _grid, _log, 
            dict(type="Copy", keys_dict={"segment_motif": "segment"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "energy", "segment"),
                feat_keys=("coord", "energy"),
            ),
        ],
        test_mode=False,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1000,
        remove_low_energy_scatters=False,
        loop=1,
    ),
)

# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------
class_freqs = [1926651899, 2038240940, 34083197, 92015482, 1145363125]
class_weights = [sum(class_freqs) / f for f in class_freqs]

hooks = [
    dict(type="WandbNamer", keys=("model.type", "data.train.max_len")),
    dict(
        type="ParameterCounter",
        show_details=True, show_gradients=False,
        sort_by_params=True, min_params=1,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=1000, save_iter_checkpoints=False),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True, exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True, exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(type="CheckpointLoader"),
    dict(
        type="PretrainEvaluator",
        write_cls_iou=True,
        every_n_steps=1000,
        train_config=dict(
            criteria=[dict(type="CrossEntropyLoss", weight=class_weights)],
        ),
    ),
]
