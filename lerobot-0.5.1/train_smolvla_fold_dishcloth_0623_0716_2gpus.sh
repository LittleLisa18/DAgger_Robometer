#!/bin/bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# Hugging Face cache directory (model already downloaded here)
export HF_HOME=/home/ma-user/work/dataset/huggingface

# Force offline mode to ensure local model is used
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp/yantong_tmp
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before starting training}"

# activate conda
cd /home/ma-user/work/users/yantong
source conda_env.sh smolvla
cd lerobot-0.5.1

# ============================================
# Training
# ============================================
echo ">>> Running full 2-GPU training mode (200k steps)"
echo ">>> Using merged dataset: fold_dishcloth_merged_v30"

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=local/fold_dishcloth_merged_v30 \
  --dataset.root=/home/ma-user/work/users/yantong/data \
  --batch_size=64 \
  --steps=200000 \
  --save_freq=10000 \
  --output_dir=outputs/train/smolvla_fold_dishcloth_0623_0716_$(date +%Y%m%d_%H%M%S) \
  --job_name=smolvla_fold_dishcloth_0623_0716 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.repo_id=local/smolvla_fold_dishcloth_0623_0716 \
  --policy.push_to_hub=false \
  --wandb.enable=true

# ============================================
# Notes
# ============================================
# 1. batch_size=64 is the per-GPU batch size.
#    Effective batch size = 64 * 2 = 128
# 2. If you encounter OOM (Out of Memory), reduce batch_size to 32 or 16.
# 3. --mixed_precision=bf16 speeds up training and saves memory (A800 supports bf16).
