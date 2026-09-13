#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Edit paths and training settings in this JSON first. Teacher is started separately.
CONFIG="${1:-configs/distill_smolvla_autodagger.json}"
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"

# Single GPU student; put the teacher on a separate GPU when possible.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" -m lerobot.scripts.lerobot_train \
  --config_path="$CONFIG"

# Multi-GPU alternative (replace the single-GPU command above):
# CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -m accelerate.commands.launch \
#   --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=bf16 \
#   --dynamo_backend=no --module lerobot.scripts.lerobot_train --config_path="$CONFIG"
