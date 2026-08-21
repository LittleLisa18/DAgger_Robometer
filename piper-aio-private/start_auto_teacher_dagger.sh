#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate piperaio

exec python inference/infer_teacher_dagger_ensemble.py \
  --model openpi \
  --host 127.0.0.1 \
  --port 8000 \
  --student_dashboard_port 8080 \
  --teacher_model openpi \
  --teacher_host 127.0.0.1 \
  --teacher_port 8003 \
  --teacher_dashboard_port 8081 \
  --ctrl_type joint \
  --student_success_duration 4 \
  --teacher_success_duration 4 \
  --robometer_poll_interval 0.05 \
  --save_rollout \
  "$@"
