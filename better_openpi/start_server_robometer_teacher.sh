#!/bin/bash
TEACHER_CHECKPOINT="${1:-/media/sail/jinghua4T/ckpt_yantong/pi05_fold_dishcloth_0716/49999}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
uv run scripts/serve_policy_with_robometer.py \
  --policy-config pi05_vlm_agilex \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --policy-port 8003 \
  --robometer-url http://127.0.0.1:8002 \
  --dashboard-port 8081 \
  --camera cam_high \
  --failure-timeout 16 \
  --success-threshold 0.5 \
  --max-frames 8 \
  --monitor-interval 1.0 \
  --output-dir robometer_live_runs/teacher
