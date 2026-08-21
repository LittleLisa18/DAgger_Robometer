#!/bin/bash
STUDENT_CHECKPOINT="${1:-/media/sail/jinghua4T/ckpt_yantong/pi05_vlm_fold_dishcloth_0716/49999}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
uv run scripts/serve_policy_with_robometer.py \
  --policy-config pi05_vlm_agilex \
  --checkpoint "${STUDENT_CHECKPOINT}" \
  --policy-port 8000 \
  --robometer-url http://127.0.0.1:8002 \
  --dashboard-port 8080 \
  --camera cam_high \
  --failure-timeout 6 \
  --success-threshold 0.5 \
  --max-frames 8 \
  --monitor-interval 1.0 \
  --output-dir robometer_live_runs/student