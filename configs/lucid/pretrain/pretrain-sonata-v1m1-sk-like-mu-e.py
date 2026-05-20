"""
Sonata v1m1 SSL pretraining on LUCiD SK-like raw sensor events.

Uses config_000001 (mu-) and config_000003 (e-) with a deterministic per-config
holdout for event-level linear probing.
"""

__import__("pimm.datasets.lucid_event_ssl")
__import__("pimm.engines.hooks.lucid_event_probe")

_base_ = ["../../_base_/default_runtime.py"]

# misc custom setting
batch_size = 128
num_worker = 8
batch_size_val = 16
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
find_unused_parameters = False
detect_anomaly = False
matmul_precision = "high"
deterministic = False
seed = 0

use_wandb = True
wandb_project = "Pretraining-Sonata-LUCiD-SKLike"

grid_size = 0.04
warmup_ratio = 0.05

data_root = "/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like"
lucid_configs = [
    dict(name="config_000001", label=0, label_name="mu-"),
    dict(name="config_000003", label=1, label_name="e-"),
]

# Roughly 2% of each config based on the current ~100k events/config.
# Override from CLI if you want a larger/smaller probe set.
holdout_events_per_config = 2000

coord_center = [0.0, 0.0, 0.0]
coord_scale = 18.1 # m
pe_log_min = 0.01
pe_log_max = 50.0
time_log_scale = 50.0
time_log_max = 4000.0
min_points = 1024

# model settings
model = dict(
    type="Sonata-v1m1",
    backbone=dict(
        type="PT-v3m8",
        in_channels=5,  # [x, y, z, log_PE, relative_log_time]
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        enc_cpe_channels=(54, 108, 108, 108, 108),
        mlp_ratio=4,
        qk_norm=False,
        qkv_bias=True,
        qk_scale=None,
        layer_scale=1e-5,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        enable_cpe=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
        cpe_first_layer_only=False,
        rope_base=10,
        rope_jitter=1.1,
        rope_rescale=1.2,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=576,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=6,
    mask_size_start=0.2,
    mask_size_base=0.5,
    mask_size_warmup_ratio=warmup_ratio,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=warmup_ratio,
    mask_jitter=grid_size / 2,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=warmup_ratio,
    student_temp=0.10,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1.0,
    match_max_r=2 * grid_size,
    up_cast_level=0,
)

# scheduler settings
epoch = 100
base_lr = 0.0026
lr_decay = 0.9

base_wd = 0.04
final_wd = 0.2

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=warmup_ratio,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

base_event_transform = [
    dict(type="NormalizeCoord", center=coord_center, scale=coord_scale),
    dict(
        type="Update",
        keys_dict={"index_valid_keys": ["coord", "energy", "time", "sensor_idx"]},
    ),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        sum_keys=("energy",),
        min_keys=("time",),
    ),
    dict(
        type="LogTransform",
        min_val=pe_log_min,
        max_val=pe_log_max,
        log=True,
        clip=True,
        keys=("energy",),
    ),
    dict(
        type="RelativeLogNormalize",
        scale=time_log_scale,
        max_val=time_log_max,
        out_min=-1.0,
        out_max=1.0,
        keys=("time",),
    ),
]

transform = base_event_transform + [
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "energy", "time"),
        global_view_num=2,
        global_view_scale=(0.55, 1.0),
        local_view_num=6,
        local_view_scale=(0.15, 0.45),
        global_shared_transform=[
            dict(
                type="MultiplicativeRandomJitter",
                sigma=0.05,
                clip=0.05,
                keys=("energy",),
                p=0.8,
            ),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=grid_size / 4,
                clip=grid_size,
                keys=("coord",),
            ),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
            dict(
                type="RandomJitter",
                sigma=grid_size / 4,
                clip=grid_size,
                keys=("coord",),
            ),
        ],
        max_size=30000,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_energy",
            "global_time",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_energy",
            "local_time",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=(
            "global_coord",
            "global_energy",
            "global_time",
        ),
        local_feat_keys=(
            "local_coord",
            "local_energy",
            "local_time",
        ),
    ),
]

val_transform = base_event_transform + [
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=("coord", "energy", "time", "event_label", "grid_size", "name"),
        feat_keys=("coord", "energy", "time"),
    ),
]

data = dict(
    num_classes=2,
    names=["mu-", "e-"],
    ignore_index=-1,
    train=dict(
        type="LUCiDEventSSLDataset",
        data_root=data_root,
        configs=lucid_configs,
        split="train",
        dataset_name="wc",
        holdout_events_per_config=holdout_events_per_config,
        holdout_seed=seed,
        holdout_strategy="random",
        min_points=min_points,
        aggregate_sensor_hits=True,
        time_aggregation="earliest",
        transform=transform,
        loop=1,
    ),
    val=dict(
        type="LUCiDEventSSLDataset",
        data_root=data_root,
        configs=lucid_configs,
        split="holdout",
        dataset_name="wc",
        holdout_events_per_config=holdout_events_per_config,
        holdout_seed=seed,
        holdout_strategy="random",
        min_points=min_points,
        aggregate_sensor_hits=True,
        time_aggregation="earliest",
        transform=val_transform,
        loop=1,
    ),
)

hooks = [
    dict(
        type="WandbNamer",
        keys=(
            "model.type",
            "data.train.holdout_events_per_config",
            "amp_dtype",
            "seed",
        ),
        sep="-",
    ),
    dict(
        type="ParameterCounter",
        show_details=False,
        show_gradients=False,
        sort_by_params=True,
        min_params=1,
    ),
    dict(type="CheckpointLoader"),
    dict(
        type="DtypeOverrider",
        class_patterns=["LayerNorm"],
        dtype="float32",
        override_parameters=True,
        methods_to_override=["forward"],
    ),
    dict(type="ModelHook"),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(
        type="WeightDecayScheduler",
        base_value=base_wd,
        final_value=final_wd,
        warmup_ratio=1.0,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=1000),
    dict(type="GradientNormLogger", log_frequency=10, log_per_layer=False),
    dict(
        type="EventLinearProbeEvaluator",
        every_n_steps=1000,
        label_key="event_label",
        train_fraction=0.5,
        seed=seed,
        prefix="event_probe",
        class_names=data["names"],
        require_heldout_data=True,
        train_config=dict(
            epochs=20,
            batch_size=8192,
            weight_decay=0.01,
            criteria=[dict(type="CrossEntropyLoss")],
        ),
    ),
    dict(
        type="PrototypeUsageLogger",
        log_frequency=10,
        prefix="prototypes",
    ),
    dict(
        type="FeatureStdMonitor",
        log_frequency=10,
        prefix="feature_std",
        monitor_student=True,
        monitor_teacher=True,
        track_channels=False,
    ),
    dict(
        type="ResourceUtilizationLogger",
        log_frequency=10,
        prefix="resources",
        log_per_gpu=True,
        log_cpu=True,
        log_system_memory=True,
    ),
]
