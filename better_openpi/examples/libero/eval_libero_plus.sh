#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <TASK_SUITE|summary> <SAVE_NAME> [--args.resume] [extra main.py args...]"
  exit 1
fi

source examples/libero/.venv/bin/activate

export LIBERO_CONFIG_PATH=$PWD/third_party/LIBERO-plus
export PYTHONPATH="$PWD/third_party/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"

MODE_OR_SUITE=$1
SAVE_NAME=$2
shift 2

if [[ "$MODE_OR_SUITE" == "summary" ]]; then
  python examples/libero/main.py \
    --args.plus \
    --args.plus_summary_only \
    --args.save_name="${SAVE_NAME}" \
    --args.out_path="data/libero_plus_eval" \
    "$@"
else
  python examples/libero/main.py \
    --args.plus \
    --args.task_suite_name="${MODE_OR_SUITE}" \
    --args.save_name="${SAVE_NAME}" \
    --args.num_trials_per_task=1 \
    --args.out_path="data/libero_plus_eval" \
    "$@"
fi
