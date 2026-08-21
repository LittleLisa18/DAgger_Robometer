#!/bin/bash
set -e
cd "$(dirname "$0")"

CHECKPOINT="/path/to/checkpoints/last/pretrained_model"

PYTHONPATH=src python -m lerobot.scripts.lerobot_serve_smolvla_piper \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --actions-per-chunk 50 \
  --local-files-only
