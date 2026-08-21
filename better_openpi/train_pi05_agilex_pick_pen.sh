#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

# 8-GPU training environment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/ma-user/work/users/yantong/cache/openpi}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1500}"

mkdir -p "$OPENPI_DATA_HOME"
chmod 700 "$OPENPI_DATA_HOME"

CONFIG_NAME="pi05_agilex"
EXP_NAME="pi05_pick_pen_0807"

source /userhome/cs3/u3013043/code/better_openpi_lyt/.venv/bin/activate

echo "Computing normalization statistics..."
uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

echo "Starting full 8-GPU training..."
uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  --fsdp-devices=8 \
  --overwrite