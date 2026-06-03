<div align="center">

# Particle Imaging Models (pimm)
### Foundation model research for particle imaging detectors

</div>

A codebase for perception research for time projection chambers (TPCs), with a focus on liquid argon TPCs, built on the [Pointcept](https://github.com/Pointcept/Pointcept) training and inference framework.

This repository currently deals with 3D charge clouds only, with plans to incorporate 2D
images (e.g., wireplane waveforms) and other modalities in the near future.

## Overview

**pimm** adapts methods in deep learning and computer vision for event reconstruction in LArTPC detectors. This repository provides:

- **Self-supervised pre-training**: discriminative pre-training ([Sonata](https://arxiv.org/abs/2503.16429)) for learning good representations
of LArTPC images.
- **Panoptic segmentation** (PointGroup, Panda Detector) models for particle and interaction instance/semantic segmentation
- **Semantic segmentation** models for per-pixel segmentation.

In sum, **pimm** integrates the following works:  
**Backbone**: 
[MinkUNet](https://github.com/NVIDIA/MinkowskiEngine), [SpUNet](https://github.com/traveller59/spconv) (see [SparseUNet](#sparseunet)),
[PTv1](https://arxiv.org/abs/2012.09164), [PTv2](https://arxiv.org/abs/2210.05666), [PTv3](https://arxiv.org/abs/2312.10035) (see [Point Transformers](#point-transformers)),
**Instance Segmentation**: 
[PointGroup](https://github.com/dvlab-research/PointGroup) (see [PointGroup](#pointgroup)),  
[Panda Detector](https://arxiv.org/abs/2512.01324) (see [Panda Detector](#detector));  
**Pre-training**: 
[Sonata](https://arxiv.org/abs/2503.16429) (see [Sonata](#sonata)),
[PoLAr-MAE](https://arxiv.org/abs/2502.02558) (see [PoLAr-MAE](#polar-mae));  
**Datasets**:
[PILArNet-M](https://arxiv.org/abs/2502.02558) (see [PILArNet-M](#pilarnet-m)) 

### TODO

We are looking at including the following models/modalities in the future:
- [ ] [SPINE](https://arxiv.org/abs/2102.01033), up until postprocessing module
- [x] [PoLAr-MAE](https://arxiv.org/abs/2502.02558) pre-training and fine-tuning
- [ ] 2D TPC waveforms/networks, e.g., [NuGraph](https://arxiv.org/abs/2403.11872)
- [ ] Optical waveforms

## Quick Start

### Using the container (recommended)

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

#### Train (single GPU):

```bash
apptainer exec --nv --bind XXX /path/to/pimm.sif \
  sh scripts/train.sh -g 1 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask
```

where XXX is a directory (not including your home path) you'd like to ensure that pimm will be able to see inside the container. This is not needed unless you are working on an HPC with directories organized in a non-standard way. For example, at SLAC National Laboratory's S3DF cluster, you must `--bind /sdf,/lscratch`.

#### Multi-GPU:

Change `-g 1` (1 GPU) to `-g 4` (4 GPUs), or omit `-g` to use all available GPUs.

#### Multi-Node:

For training on Slurm configurations, you can use the `multinode.slurm.sbatch` file in `scripts/slurm/` to submit your sbatch job.

To get started just adjust the number of nodes and GPUs

```
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
```

Then modify `-m` and `-g` to the number of nodes and number of tasks (i.e., GPUs) per node: `-m 2 -g 4`

> See [Dataset Preparation](#dataset-preparation) to download PILArNet-M.

### From source

```bash
git clone https://github.com/youngsm/particle-imaging-models.git
cd particle-imaging-models
conda env create -f environment.yml
conda activate pimm-torch2.5.0-cu12.4
sh scripts/train.sh -g 1 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask
```

Requires CUDA 11.6+ for FlashAttention (set `enable_flash=False` in configs if unavailable).

## Multi-Node Training

SLURM templates are provided for multi-node training:

```bash
cp scripts/slurm/multinode.slurm.sbatch my_job.sh   # generic / SLAC cluster
cp scripts/slurm/multinode.nersc.sbatch my_job.sh   # NERSC Perlmutter
# edit SBATCH headers + experiment section, then:
sbatch my_job.sh
```

The key rule: `--ntasks-per-node` must equal the number of GPUs per node. The training script handles distributed setup automatically via SLURM environment variables.

## HPC Launching

For routine Slurm runs, prefer the site-aware launcher over copying and editing
one-off Slurm scripts. It composes common defaults, a site profile, and
optionally a run recipe into the final Slurm job.

If you just made a normal Python config and want to submit it with site
defaults:

```bash
scripts/launch.py submit --site s3df \
  --config-dir panda/pretrain_geometry_combos \
  --config pretrain-sonata-v1m1-pilarnet-e050-head512-tail-wd20
```

For a saved run recipe with launch-time state such as checkpoint weights,
resource overrides, or W&B naming:

```bash
scripts/launch.py submit --site s3df launch/runs/e050_tail.yaml
scripts/launch.py submit --site nersc launch/runs/e050_tail.yaml
```

Always dry-run first when changing sites or resources:

```bash
scripts/launch.py dry-run --site s3df launch/runs/e050_tail.yaml
```

`--site s3df` submits through `ssh iana`, activates the
`pointcept-torch2.5.0-cu12.4` mamba environment, and runs `sbatch` from the
configured `paths.repo_root`. `--site nersc` currently assumes the launcher is run on a
NERSC login node and renders a Shifter/Perlmutter job. See
`launch/README.md` for the YAML layer details and override syntax.

## Training & Testing

The entry point is `scripts/train.sh`:

```bash
sh scripts/train.sh -d <dataset> -c <config> [options]
```

| Flag | Description |
|------|-------------|
| `-d` | Config directory (e.g., `panda/pretrain`, `panda/semseg`) |
| `-c` | Config name without `.py` |
| `-n` | Experiment name (default: auto-generated) |
| `-g` | GPUs per machine (default: all available) |
| `-m` | Number of machines (default: 1) |
| `-w` | Path to checkpoint (to be used by CheckpointLoader) |
| `-r true` | Resume training from last checkpoint |
| `-C` | Dev mode: skip code snapshot, run from repo source |
| `-h` | Show full help |

```bash
# Override config values
sh scripts/train.sh -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask \
  -- --options epoch=10 data.train.max_len=1000

# Fine-tune from pre-trained checkpoint
sh scripts/train.sh -g 4 -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin \
  -w /path/to/checkpoint.pth

# Resume
sh scripts/train.sh -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask \
  -n my_experiment -r true
```

See [Config Structure](#config-structure) for more on how configs work.

By default, `train.sh` snapshots the codebase into `exp/<dataset>/<name>/code/` and runs the code from this snapshot for reproducibility. Use `-C` to skip this during development.

Model checkpoints, which can be quite large, are saved to `exp/<dataset>/<name>/model/`. To redirect to a separate disk, set `MODEL_DIR` in your `.env` file or environment; this will save the checkpoint to `MODEL_DIR` and symlink it to `exp/<dataset>/<name>/model`.

## Configuration System

Configurations are Python dictionary-based files located in the `configs/` directory. Each config file defines the model architecture, dataset settings, training hyperparameters, and different hooks to run during training (checkpoint saving, logging, evaluation).

### Config Structure

Configs use a hierarchical structure with `_base_` inheritance:

```python
_base_ = ["../../_base_/default_runtime.py"]

# Override or add settings
model = dict(type="PT-v3m2", ...)
data = dict(train=dict(...), val=dict(...))
```

### Modifying Configs

You can modify configs in two ways:

1. Edit the config file directly
2. Override via command line using `--options`:
   ```bash
   sh scripts/train.sh ... -- --options epoch=50 data.train.max_len=500000
   ```

Example configs can be found in:
- `configs/panda/pretrain/` - Pre-training configurations
- `configs/panda/semseg/` - Semantic segmentation configurations  
- `configs/panda/panseg/` - Panoptic segmentation configurations

## Dataset Preparation

### PILArNet-M

Download the 168GB dataset from Hugging Face:

```bash
python tools/download_pilarnet.py --version v2 --output_dir /path/to/dir
```

Data saves to `~/.cache/pimm/pilarnet/v2` if `output_dir` is not provided. After downloading the dataset, run `cp example.env .env` and set `PILARNET_DATA_ROOT_V2`. This will allow the dataloader to automatically find the data.

PILArNet has two revisions. **v2** is recommended for new models (adds PID, momentum, and vertex information). **v1** is the original dataset from the PoLAr-MAE paper. Events differ between splits, so models trained on v1 should be evaluated on v1.

### Detector datasets (pimm-data)

The detector dataset loaders — `JAXTPCDataset` (LArTPC), `LUCiDDataset` (water
Cherenkov), `MultiModalEventDataset` (source-mixture / holdout wrapper), and
`PILArNetH5Dataset` — live in the standalone **[pimm-data](https://github.com/OmarAlterkait/pimm-data)**
package, vendored here as the `libs/pimm-data` git submodule:

```bash
git submodule update --init libs/pimm-data
pip install -e libs/pimm-data
```

`pimm/datasets/__init__.py` is a thin re-export shim: it registers everything
from `pimm_data` into pimm's `DATASETS`/`TRANSFORMS` registries, so configs keep
referencing `type="JAXTPCDataset"` etc. unchanged. `tools/train.py` logs the
resolved `pimm_data.__file__` at startup and asserts the de-fork dataset is
present, so a stale editable install fails loudly instead of silently loading an
old tree.

The full modality schema (`step`/`sensor`/`hits`/`labl`), nested-dict output,
transform pipeline (`ApplyToStream` → `Collect`), and per-dataset API are
documented in **[`libs/pimm-data/README.md`](libs/pimm-data/README.md)**; see
[`docs/DETECTOR_DATASET.md`](docs/DETECTOR_DATASET.md) for pimm-side config
recipes.

## Data Format

Point cloud data should be organized with the following structure:

```python
{
    'coord': (N, 3),           # 3D hit positions [x, y, z]
    'feat': (N, C),            # Hit features (charge, time, etc.)
    'segment': (N,1),          # Semantic labels (optional, for training)
    'instance': (N,1),         # Instance IDs (optional, for training)
    ...                        # Extra attributes
}
```

The data often needs to be re-scaled to new domains that lead to more efficient training
(e.g., centering/scaling of coordinates to [-1,1]$^3$). This can be done within the Dataset class, or from a Transform. See the transform sections of configuration files for more details.

### Packed Data Format

This library works with packed data, where all batched quantities are in two dimensions instead of three, i.e. `(N, 3)` instead of `(B, N, 3)`. This is because point clouds are variable length, and getting to a 3 dimensional tensor would require padding. Instead of padding, there is an `offset` tensor, which is of length `B` and gives the indices in the packed tensors at which a point cloud ends and a new one starts.

`Offset` is conceptually similar to the concept of `Batch` in PyG, and can be seen as the cumulative sum of a `lengths` tensor. A visual illustration of batch and offset is as follows:

<p align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png">
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png" width="480">
    </picture><br>
</p>

### Docker / Apptainer

Pre-built images are available on Docker Hub:

| Image | Description |
|-------|-------------|
| `youngsm/pimm:pytorch2.5.0-cuda12.4` | Standard image |
| `youngsm/pimm-nersc:pytorch2.5.0-cuda12.4` | NERSC variant with extra dependencies |

```bash
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

## Model Zoo

### Model Versioning

Models use `vXmY` naming (version X, mode Y). Different modes indicate small architecture variants, while versions indicate large architectural changes.

### Backbones

- **[PTv3](https://arxiv.org/abs/2312.10035)** (Point Transformer V3) — efficient backbone with FlashAttention. Requires `spconv` and CUDA 11.6+ for FlashAttention (which is optional)
- **SparseUNet** — SpConv-based UNet.
- **[PTv2](https://arxiv.org/abs/2210.05666)**, **[PTv1](https://arxiv.org/abs/2012.09164)** — earlier Point Transformer versions.

### Pre-training

- **[Panda/Sonata](https://arxiv.org/abs/2503.16429)** — DINO-style self-supervised learning with teacher-student framework and online prototype clustering.
- **[PoLAr-MAE](https://arxiv.org/abs/2502.02558)** — masked autoencoder with chamfer + energy reconstruction losses.

### Instance / Panoptic Segmentation

- **[PointGroup](https://github.com/dvlab-research/PointGroup)** — clustering-based instance segmentation.
- **[Panda Detector](https://arxiv.org/abs/2512.01324)** — Mask2Former-style detection modified to take low energy deposits into account.

## Logging

Both TensorBoard and Weights & Biases are enabled by default. Set `use_wandb=False` to disable W&B. To authenticate, either run `wandb login` or add `WANDB_API_KEY=your_key` to your `.env` file (see `example.env`).

```python
hooks = [
    dict(type="WandbNamer", keys=("model.type", "data.train.max_len", "amp_dtype", "seed")),
    ...
]
```


## Acknowledgements

Built on [Pointcept](https://github.com/Pointcept/Pointcept). Thanks to them!

## License

MIT (inherited from Pointcept).
