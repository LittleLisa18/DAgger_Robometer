#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-/home/ma-user/work/users/luyuxiang/code}"
STUDENT_PYTHON="${STUDENT_PYTHON:-$PACKAGE_ROOT/lerobot-0.5.1/.venv/bin/python}"
COLLECTOR_PYTHON="${COLLECTOR_PYTHON:-/home/ma-user/work/users/luyuxiang/envs/libero/bin/python}"
ROBOMETER_PYTHON="${ROBOMETER_PYTHON:-$PACKAGE_ROOT/robometer/.venv/bin/python}"
ROBOMETER_MODEL="${ROBOMETER_MODEL:-/home/ma-user/work/model/robometer-4b-fft-libero}"
export STUDENT_PYTHON COLLECTOR_PYTHON ROBOMETER_PYTHON
export PYTHONPATH="$PACKAGE_ROOT:$CODE_ROOT/better_openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
 student)
  export PYTHONPATH="$PACKAGE_ROOT/lerobot-0.5.1/src:$PYTHONPATH"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  exec "$STUDENT_PYTHON" -m autodagger_libero.serve_student "$@" ;;
 robometer)
  "$COLLECTOR_PYTHON" -m autodagger_libero.check_robometer_files \
    --model "$ROBOMETER_MODEL"
  if [[ ! -x "$ROBOMETER_PYTHON" ]]; then
    echo 'Robometer needs its own Python 3.10 environment with project dependencies; set ROBOMETER_PYTHON.' >&2
    exit 1
  fi
  cd "$PACKAGE_ROOT/robometer"
  export PYTHONPATH="$PACKAGE_ROOT/robometer:$PYTHONPATH"
  exec "$ROBOMETER_PYTHON" robometer/evals/eval_server.py \
    "model_path=$ROBOMETER_MODEL" \
    server_url=127.0.0.1 server_port=8102 num_gpus=1 "$@" ;;
 collect)
  export PYTHONPATH="${LIBERO_ROOT:-$CODE_ROOT/LIBERO}:$PYTHONPATH"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PACKAGE_ROOT/autodagger_libero/.libero}"
  "$COLLECTOR_PYTHON" -m autodagger_libero.prepare_libero --root "${LIBERO_ROOT:-$CODE_ROOT/LIBERO}" --config-dir "$LIBERO_CONFIG_PATH"
  exec "$COLLECTOR_PYTHON" -m autodagger_libero.collect "$@" ;;
 export)
  export PYTHONPATH="$PACKAGE_ROOT/lerobot-0.5.1/src:$PYTHONPATH"
  exec "$STUDENT_PYTHON" -m autodagger_libero.export_dataset "$@" ;;
 view)
  exec "$COLLECTOR_PYTHON" -m autodagger_libero.view_run "$@" ;;
 test)
  exec "$COLLECTOR_PYTHON" -m unittest discover -s "$PACKAGE_ROOT/autodagger_libero/tests" -v "$@" ;;
 check-export|check-student|check-browser|check-services)
  export PYTHONPATH="$PACKAGE_ROOT/lerobot-0.5.1/src:$PYTHONPATH"
  TEST_MODULE="${MODE//-/_}"
  exec "$STUDENT_PYTHON" -m "autodagger_libero.tests.$TEST_MODULE" "$@" ;;
 *) echo 'Usage: bash autodagger_libero/run_hw.sh {student|robometer|collect|export|view|test|check-export|check-student|check-browser|check-services} [args]' ;;
esac
