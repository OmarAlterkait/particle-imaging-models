#!/bin/bash
# Multi-node training template (Apptainer).
# sbatch scripts/slurm/multinode.slurm.sbatch

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4       # must match gpu count
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=myaccount
#SBATCH --output=slurm_logs/%j_%x.txt

set -e
mkdir -p slurm_logs
export NCCL_SOCKET_IFNAME=^docker0,lo

# ── Edit these ──────────────────────────────────────────────────
SIF=pimm.sif
DATASET="panda/pretrain"
CONFIG="pretrain-sonata-v1m1-pilarnet-smallmask"
EXTRA=""  # e.g. "-- --options epoch=100 data.train.max_len=100000"
# ────────────────────────────────────────────────────────────────

CMD="sh scripts/train.sh -m $SLURM_NNODES -g $SLURM_GPUS_ON_NODE -d $DATASET -c $CONFIG $EXTRA"
srun apptainer exec --nv $SIF bash -c "$CMD"
