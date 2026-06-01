# Detector Datasets

pimm's detector dataset loaders live in the standalone **pimm-data** package
(vendored as the `libs/pimm-data` submodule). `pimm/datasets/__init__.py` is a
re-export shim that registers them into pimm's `DATASETS`/`TRANSFORMS`
registries, so configs reference `type="JAXTPCDataset"` etc. unchanged.

This page covers pimm-side config recipes. The authoritative reference for the
**full output schema, modality combinations, per-dataset API, and transform
pipeline** is **[`libs/pimm-data/README.md`](../libs/pimm-data/README.md)**.

## Output model (read this first)

Datasets return a **nested** dict — one sub-dict per loaded modality — not a
flat point cloud:

```python
data = ds.get_data(idx)   # raw numpy, no transforms
# { 'name': str, 'split': str,
#   'edep':   {...},   # when 'edep' in modalities
#   'sensor': {...},   # when 'sensor' in modalities
#   'hits':   {...},   # when 'hits' in modalities
#   'labl':   {...},   # when 'labl' in modalities  (dimension tables)
#   'bridges':{...} }  # JAXTPC only, when 'hits' loaded (edep↔hits↔labl FKs)
```

Transforms are **stream-scoped**: `ApplyToStream(stream=…)` dispatches inner
transforms onto one modality's sub-dict, and a terminal `Collect(stream=…)`
lifts the chosen keys to the flat `{coord, feat, grid_coord, segment, offset,
…}` dict the model consumes (tensorizing as it goes — no separate `ToTensor`).

```
get_data() → ApplyToStream(stream='edep', [...]) → Collect(stream='edep', ...)
nested dict      mutates edep sub-dict (numpy)        flat dict (tensors)
```

Modalities are `edep` / `sensor` / `hits` / `labl` (the four production file
types). Readers are joint-indexed: one global `idx` resolves to the **same
physics event** in every loaded modality (events not present in all loaded
modalities, or filtered out, are dropped to keep the streams aligned).

---

## JAXTPCDataset

Liquid Argon TPC (JAXTPC production output). Auto-detects **wire** (U/V/Y
planes, sensor/hits `coord` is `(M, 2)`) vs **pixel** (`coord` is `(M, 3)`).

### Data layout

```
data_root/
├── edep/    sim_edep_0000.h5    — 3D truth deposits
├── sensor/  sim_sensor_0000.h5  — sparse readout per plane
├── hits/    sim_hits_0000.h5    — per-particle sensor decomposition (+ bridges)
└── labl/    sim_labl_0000.h5    — per-volume track_id → label tables
```

`split` is a sub-directory under each modality (`edep/{split}/…`); pass
`split=""` for a flat layout. Real `doraemon` output is nested one level deeper
(`edep/run_NNNNNNNN/sim_edep_*.h5`) — point at one run with
`split="run_NNNNNNNN"`.

### Task → config

| Task | `modalities` | What `Collect(stream='edep')` yields |
|------|-------------|--------------------------------------|
| 3D segmentation | `("edep", "labl")` | `coord (N,3)`, `feat`, `segment (N,)` (via `RemapSegment`) |
| 3D self-supervised | `("edep",)` | `coord (N,3)`, `feat` (no labels) |
| Instance seg on hits | `("hits", "labl")` | `Collect(stream='hits', keys=(…,'segment','instance'))` |

`segment`/`instance` exist only when `labl` is also loaded. `("labl",)` alone
and `("sensor", "labl")` are rejected (labl needs an instance-bearing modality
— `edep` or `hits` — to join against).

### Config parameters

```python
data = dict(train=dict(
    type="JAXTPCDataset",
    data_root="/path/to/jaxtpc",
    split="train",              # "" for flat, "run_NNNN" for run-nested
    dataset_name="sim",         # file prefix → sim_edep_*.h5
    modalities=("edep", "labl"),
    volume=None,                # None=all volumes, 0=volume_0 only
    label_key="pdg",            # 'pdg' (default), 'cluster', 'interaction', 'ancestor'
    min_deposits=1024,          # drop events with fewer edep deposits
    transform=[...],
))
```

### Transform recipe (3D segmentation, from `configs/detector/_base_/jaxtpc_seg.py`)

```python
transform = [
    dict(type="ApplyToStream", stream="edep", transforms=[
        dict(type="NormalizeCoord", center=[0,0,0], scale=2160.0*3**0.5),
        dict(type="LogTransform", min_val=0.01, max_val=20.0),
        dict(type="RemapSegment", scheme="motif_5cls"),   # label_key='pdg' → 5 classes
        dict(type="GridSample", grid_size=0.001, hash_type="fnv",
             mode="train", return_grid_coord=True),
        dict(type="RandomRotate", angle=[-1,1], axis="z", center=[0,0,0], p=0.8),
        dict(type="RandomFlip", p=0.5),
    ]),
    dict(type="Collect", stream="edep",
         keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]
```

### Label chain

- **edep**: `deposit → track_id → labl.track_{label_key}` → `segment`.
- **hits**: `group_id → group_to_track → track_id → labl.track_{label_key}`
  (the FK arrays live in `data['bridges']`).

---

## LUCiDDataset

Water Cherenkov / PhotonSim output (PMT-based). Same modality names; `sensor`
is per-PMT readout. `coord` uses 3D PMT positions when available (via
`pmt_positions` / `pmt_positions_file`, or the file's `config/pmt_positions`),
else falls back to `(N, 1)` sensor indices.

### Config parameters

```python
data = dict(train=dict(
    type="LUCiDDataset",
    data_root="/path/to/wc",
    split="",
    dataset_name="wc",          # file prefix → wc_sensor_*.h5
    modalities=("sensor",),     # ('sensor',), ('edep',), ('sensor','labl'), ...
    min_segments=0,             # drop events with fewer edep segments
    pe_threshold=0.0,           # drop hits entries with PE ≤ threshold
    transform=[...],
))
```

There is **no** `output_mode` / `include_labels` parameter — load labels by
adding `"labl"` to `modalities`; `segment`/`instance` then attach to the
sensor/hits/edep sub-dict.

---

## MultiModalEventDataset

A detector-agnostic **event-selection wrapper**: it composes one single-source
dataset per `source` (a `LUCiDDataset`/`JAXTPCDataset` config) and adds source
mixture (per-source `label`/`config_id`/`weight`) plus a deterministic, hash-keyed
train/val/test `holdout` (reproducible, invariant to shard add/remove/reorder).
This is what the WAND SSL configs use.

```python
data = dict(train=dict(
    type="MultiModalEventDataset",
    source_dataset=dict(type="LUCiDDataset", modalities=("sensor",), dataset_name="wc"),
    sources=[dict(name="config_000001", label=0, config_id=0),
             dict(name="config_000003", label=1, config_id=1)],
    data_root="/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like",
    split="train",                       # 'train' | 'val' | 'test' | 'all'
    holdout=dict(seed=0, n_per_config=2000),
    min_points=1024,
    transform=[...],
))
```

See `configs/lucid/pretrain/pretrain-sonata-v1m1-sk-like-mu-e.py` for the full
sensor-aggregation + multi-view transform pipeline.

---

## Adding a new detector

Datasets and readers now live in **pimm-data**, not this repo:

1. Add reader(s) under `pimm-data/src/pimm_data/readers/` (lightweight
   convention: `__init__` builds the event index, `h5py_worker_init` lazily
   opens fork-safe HDF5 handles, `read_event(idx)` returns
   `dict[str, np.ndarray]`).
2. Add a dataset class under `pimm-data/src/pimm_data/` (inherit
   `ShardEventDataset` for the joint-index / multimodal machinery, or
   `torch.utils.data.Dataset` directly), registered via
   `@DATASETS.register_module()`, and export it from `pimm_data/__init__.py`.
3. Nothing changes in pimm: the shim re-exports it automatically. No changes to
   transforms, collation, models, or training infrastructure.

## Running tests

```bash
# data layer (pimm-data) — synthetic fixtures
pytest libs/pimm-data/tests

# pimm shim resolves to the de-fork pimm_data
pytest tests/test_shim.py
```
