#!/bin/sh
cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

# Load .env if present
[ -f "$ROOT_DIR/.env" ] && set -a && . "$ROOT_DIR/.env" && set +a

TRAIN_CODE=train.py

# ─── Help ───────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
Usage: sh scripts/train.sh [OPTIONS] [-- --options key=val ...]

Options:
  -d DATASET    Dataset/config directory (e.g., panda/pretrain, panda/semseg)
  -c CONFIG     Config name without .py (e.g., pretrain-sonata-v1m1-pilarnet-smallmask)
  -n NAME       Experiment name (default: auto-generated from CONFIG + timestamp)
  -g GPUS       GPUs per machine (default: auto-detect all available)
  -m MACHINES   Number of machines (default: 1)
  -w WEIGHT     Path to pretrained checkpoint
  -r true       Resume training from last checkpoint
  -a NAME       Override Weights & Biases run name
  -p PYTHON     Python interpreter (default: python)
  -C            Dev mode: skip code copy, run directly from repo source
  -h            Show this help

Examples:
  # Single-GPU Sonata pre-training
  sh scripts/train.sh -g 1 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask

  # 4-GPU with custom experiment name
  sh scripts/train.sh -g 4 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask -n my_exp

  # Override config values
  sh scripts/train.sh -g 1 -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin -- --options epoch=10

  # Dev mode (no code copy, changes to repo take effect immediately)
  sh scripts/train.sh -C -g 1 -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin -n dev

  # List available configs
  sh scripts/train.sh --list
  sh scripts/train.sh --list panda/pretrain
EOF
  exit 0
}

# ─── Config listing ─────────────────────────────────────────────
list_configs() {
  filter="$1"
  if [ -n "$filter" ]; then
    dir="configs/$filter"
    if [ ! -d "$dir" ]; then
      echo "No config directory: $dir"
      echo ""
      echo "Available directories:"
      find configs -name "*.py" -not -path "*/_base_/*" -not -name "__*" -exec dirname {} \; | \
        sort -u | sed 's|^configs/||' | sed 's/^/  /'
      exit 1
    fi
    find "$dir" -maxdepth 1 -name "*.py" -not -name "__*" | sort | while read f; do
      basename "$f" .py
    done | sed 's/^/  /'
  else
    find configs -name "*.py" -not -path "*/_base_/*" -not -name "__*" -exec dirname {} \; | \
      sort -u | while read dir; do
        rel=$(echo "$dir" | sed 's|^configs/||')
        echo "$rel/"
        find "$dir" -maxdepth 1 -name "*.py" -not -name "__*" | sort | while read f; do
          echo "  $(basename "$f" .py)"
        done
        echo ""
      done
  fi
  exit 0
}

# Handle --list and --help before getopts (they use -- prefix)
case "$1" in
  --list) shift; list_configs "$1" ;;
  --help) usage ;;
esac

# ─── Defaults ───────────────────────────────────────────────────
DATASET=scannet
CONFIG="None"
EXP_NAME=""
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"
NO_COPY=false
MODEL_DIR=""  # User may set this externally, otherwise empty by default

while getopts "p:d:c:n:w:g:m:r:a:Ch" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    a)
      WANDB_NAME=$OPTARG
      ;;
    C)
      NO_COPY=true
      ;;
    h)
      usage
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      echo "Run 'sh scripts/train.sh -h' for usage."
      exit 1
      ;;
  esac
done

# shift past processed options to get extra args (e.g., --options key=val)
shift $((OPTIND-1))
EXTRA_ARGS="$@"

# ─── Auto-generate experiment name if not provided ──────────────
if [ -z "${EXP_NAME}" ]; then
  if [ "${CONFIG}" != "None" ]; then
    CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
    EXP_NAME="${CONFIG}-${CURRENT_DATETIME}"
  else
    EXP_NAME="debug"
  fi
fi

# ─── Validate config exists ────────────────────────────────────
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
if [ "${CONFIG}" != "None" ] && [ ! -f "$CONFIG_DIR" ]; then
  echo "Error: Config not found: $CONFIG_DIR"
  PARENT_DIR="configs/${DATASET}"
  if [ -d "$PARENT_DIR" ]; then
    echo ""
    echo "Available configs in ${DATASET}/:"
    find "$PARENT_DIR" -maxdepth 1 -name "*.py" -not -name "__*" | sort | while read f; do
      echo "  $(basename "$f" .py)"
    done
  else
    echo ""
    echo "Dataset directory not found: $PARENT_DIR"
    echo ""
    echo "Available datasets:"
    find configs -name "*.py" -not -path "*/_base_/*" -not -name "__*" -exec dirname {} \; | \
      sort -u | sed 's|^configs/||' | sed 's/^/  /'
  fi
  exit 1
fi

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

EXP_DIR=exp/${DATASET}/${EXP_NAME}

# Build MODEL_SAVE_DIR and symlink if MODEL_DIR is set
if [ -n "$MODEL_DIR" ]; then
  # If MODEL_DIR is set, checkpoints go to MODEL_DIR/.../model
  MODEL_SAVE_DIR=${MODEL_DIR%/}/$EXP_DIR/model
  MODEL_LINK_DIR=${EXP_DIR}/model
  echo "MODEL_SAVE_DIR: $MODEL_SAVE_DIR"
else
  # If not set, checkpoints go to EXP_DIR/model
  MODEL_SAVE_DIR=${EXP_DIR}/model
  MODEL_LINK_DIR=""
fi

if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_SAVE_DIR/model_last.pth
fi

# ─── Code snapshot vs dev mode ──────────────────────────────────
if [ "$NO_COPY" = true ]; then
  CODE_DIR="."
  export PYTHONPATH=.
  echo "Dev mode: running from repo source (no code copy)"
elif [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]; then
  CODE_DIR=${EXP_DIR}/code
  export PYTHONPATH=./$CODE_DIR
  echo "Resuming: running from codebase snapshot $CODE_DIR"
else
  RESUME=false
  CODE_DIR=${EXP_DIR}/code
  mkdir -p "$CODE_DIR"

  # Determine if this is rank 0 (master process)
  # Check SLURM_PROCID first (SLURM), then RANK (PyTorch), default to 0 if not set
  RANK=${SLURM_PROCID:-${RANK:-0}}

  echo " =========> CREATE EXP DIR <========="
  echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
  cp -r scripts tools pimm "$CODE_DIR" 2>/dev/null

  # Ensure physical checkpoint dir exists
  mkdir -p "$MODEL_SAVE_DIR"

  if [ -n "$MODEL_LINK_DIR" ]; then
    # Link local 'model' folder to physical checkpoint dir
    ln -sfn "$(realpath "$MODEL_SAVE_DIR")" "$MODEL_LINK_DIR"
  fi

  export PYTHONPATH=./$CODE_DIR
  echo "[pimm] Running from snapshot: $CODE_DIR"
  echo "[pimm] NOTE: Edits to repo source won't affect this run. Use -C for dev mode."
fi

echo "Loading config in:" $CONFIG_DIR

sleep 0.5

echo " =========> RUN TASK <========="
ulimit -n 65536

# Slurm Native Mode - Script handles distributed setup automatically
COMMON_ARGS="--config-file $CONFIG_DIR --options save_path=$EXP_DIR"

if [ -n "$WANDB_NAME" ]; then
  COMMON_ARGS="$COMMON_ARGS wandb_run_name=$WANDB_NAME"
fi

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE $COMMON_ARGS $EXTRA_ARGS
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE $COMMON_ARGS resume="$RESUME" weight="$WEIGHT" $EXTRA_ARGS
fi
