#!/bin/bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# Hugging Face cache directory (model already downloaded here)
export HF_HOME=/home/ma-user/work/dataset/huggingface

# Force offline mode to ensure local model is used
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp/yantong_tmp
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

export WANDB_API_KEY="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"

# activate conda
cd /home/ma-user/work/users/yantong
source conda_env.sh smolvla
cd lerobot-0.5.1

# ============================================
# Training Modes (uncomment the one you need)
# ============================================

# ---------- Mode 1: Single-GPU Test (10 steps) ----------
# Purpose: Quick validation of environment and dataset
# echo ">>> Running single-GPU test mode (10 steps)"
# lerobot-train \
#   --dataset.repo_id=local/pick_pen_0807_v30 \
#   --dataset.root=/home/ma-user/work/users/yantong/data/pick_pen_0807_v30 \
#   --batch_size=16 \
#   --steps=10 \
#   --output_dir=outputs/train/test_"$(date +%Y%m%d_%H%M%S)" \
#   --job_name=test \
#   --policy.type=smolvla \
#   --policy.device=cuda \
#   --policy.repo_id=local/test_smolvla \
#   --wandb.enable=false \
#   --policy.push_to_hub=false

# ---------- Mode 2: Full 8-GPU Training (200k steps) ----------
# Purpose: Main training run using 8 GPUs
echo ">>> Running full 8-GPU training mode (200k steps)"
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --dataset.repo_id=local/pick_pen_0807_v30 \
  --dataset.root=/home/ma-user/work/users/yantong/data/pick_pen_0807_v30 \
  --batch_size=64 \
  --steps=200000 \
  --save_freq=10000 \
  --output_dir=outputs/train/smolvla_pick_pen_0807_$(date +%Y%m%d_%H%M%S) \
  --job_name=smolvla_pick_pen_0807 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.repo_id=local/smolvla_pick_pen \
  --policy.push_to_hub=false \
  --wandb.enable=true

# ============================================
# Notes
# ============================================
# 1. batch_size=32 is the per-GPU batch size.
#    Effective batch size = 32 * 8 = 256
# 2. If you encounter OOM (Out of Memory), reduce batch_size to 16 or 8.
# 3. If you have spare memory, increase batch_size to 64.
# 4. --mixed_precision=bf16 speeds up training and saves memory (A800 supports bf16).
# 5. $(date +%Y%m%d_%H%M%S) adds a timestamp to the output directory
#    to prevent overwriting previous runs.
# 6. To run the test mode, comment out Mode 2 and uncomment Mode 1.