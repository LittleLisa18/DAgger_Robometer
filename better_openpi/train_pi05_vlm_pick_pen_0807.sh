#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

# 8-GPU training environment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/ma-user/work/users/yantong/cache/openpi}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1500}"
export WANDB_API_KEY="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"

mkdir -p "$OPENPI_DATA_HOME"
chmod 700 "$OPENPI_DATA_HOME"

cd /home/ma-user/work/users/yantong
source conda_env.sh better_openpi /home/ma-user/work/users/yantong/cuda-12.8

cd DAgger_Robometer/better_openpi

CONFIG_NAME="pi05_vlm_agilex"
EXP_NAME="pi05_vlm_pick_pen_0807"
NORM_STATS="assets/$CONFIG_NAME/pick_pen_0807/norm_stats.json"

else
  echo "Using existing normalization statistics: $NORM_STATS"
fi
echo "Starting full 8-GPU training..."
uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  --fsdp-devices=8 \
  --overwrite
