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
git clone https://github.com/youngsm/particle-imaging-models.git
cd particle-imaging-models
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

#### Train (single GPU):

```bash
apptainer exec --nv --bind XXX /path/to/pimm.sif \
  sh scripts/train.sh -g 1 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask
```

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
cp scripts/slurm/multinode.slurm.sbatch my_job.sh  # generic cluster
cp scripts/slurm/multinode.nersc.sbatch my_job.sh   # NERSC Perlmutter
# edit SBATCH headers + experiment section, then:
sbatch my_job.sh
```

The key rule: `--ntasks-per-node` must equal the number of GPUs per node. The training script handles distributed setup automatically via SLURM environment variables.

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
| `-w` | Path to checkpoint (for fine-tuning or resuming) |
| `-r true` | Resume training from last checkpoint |
| `-C` | Dev mode: skip code snapshot, run from repo source |
| `-h` | Show full help |

```bash
# List available configs
sh scripts/train.sh --list
sh scripts/train.sh --list panda/pretrain

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

Testing:

```bash
sh scripts/test.sh -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin \
  -n my_experiment -w model_best
```

By default, `train.sh` snapshots the codebase into `exp/<dataset>/<name>/code/` for reproducibility. Use `-C` to skip this during development.

Checkpoints save to `exp/<dataset>/<name>/model/`. To redirect to a separate disk, set `MODEL_DIR` in your `.env` file or environment.

## Configuration

Configs are Python dicts in `configs/` with `_base_` inheritance:

```python
_base_ = ["../../_base_/default_runtime.py"]
model = dict(type="PT-v3m2", ...)
data = dict(train=dict(...), val=dict(...))
```

Override any value from the command line with `-- --options key=value nested.key=value`.

## Dataset Preparation

### PILArNet-M

Download the dataset from Hugging Face:

```bash
python tools/download_pilarnet.py --version v2
```

Data saves to `~/.cache/pimm/pilarnet/v2` by default and is auto-detected. To use a custom path, create a `.env` file in the repo root (see `example.env`):

```
PILARNET_DATA_ROOT_V2=/path/to/pilarnet/v2
```

PILArNet has two revisions. **v2** is recommended for new models (adds PID, momentum, and vertex information). **v1** is the original dataset from the PoLAr-MAE paper. Events differ between splits, so models trained on v1 should be evaluated on v1.

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

- **[Panda/Sonata](https://arxiv.org/abs/2503.16429)** — DINO-style self-supervised learning with teacher-student framework and online prototype clustering. Used in Panda.
- **[PoLAr-MAE](https://arxiv.org/abs/2502.02558)** — masked autoencoder with chamfer + energy reconstruction losses.

### Instance / Panoptic Segmentation

- **[PointGroup](https://github.com/dvlab-research/PointGroup)** — clustering-based instance segmentation.
- **[Panda Detector](https://arxiv.org/abs/2512.01324)** — Mask2Former-style detection modified to take low energy deposits into account.

## Data Format

Point clouds use a **packed tensor format**: `(N, 3)` instead of `(B, N, 3)` to avoid padding variable-length clouds. An `offset` tensor of length `B` gives the cumulative sum of point counts per sample.

```python
{"coord": (N, 3), "feat": (N, C), "segment": (N, 1), "instance": (N, 1)}
```

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