#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 修改这里即可：单卡 (0)，双卡 (0 1)，四卡 (0 1 2 3)。
GPUS=(0 1 2 3 4 5 6 7)
TEACHER_PORT=18700                 # 各卡依次使用 18700、18701……
MASTER_PORT=29500
CONFIG="configs/distill_smolvla_mixed.json"
OPENPI="/home/ma-user/work/users/luyuxiang/code/better_openpi"
TEACHER_CONFIG="pi05_libero"
CHECKPOINT="checkpoints/pi05_libero_bl/29999"

PYTHON="$PWD/.venv/bin/python"
export HF_HOME=/home/ma-user/work/hf HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
N=${#GPUS[@]}
# 检查端口，避免误连已有 teacher；生成按 local rank 排列的地址。
URLS=$("$PYTHON" - "$TEACHER_PORT" "$MASTER_PORT" "${GPUS[@]}" <<'PY'
import json, socket, sys
base, master = map(int, sys.argv[1:3])
gpus = sys.argv[3:]
assert gpus and len(set(gpus)) == len(gpus), 'GPU list must be nonempty and unique'
ports = list(range(base, base + len(gpus)))
assert master not in ports and all(0 < p < 65536 for p in [*ports, master])
for port in [*ports, master]:
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', port))
print(json.dumps([f'ws://127.0.0.1:{p}' for p in ports]))
PY
)
mkdir -p outputs/teacher_servers
LOGS=$(mktemp -d "$PWD/outputs/teacher_servers/run-XXXXXXXX")
echo "Teacher logs: $LOGS"
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for rank in "${!GPUS[@]}"; do
    echo "rank=$rank GPU=${GPUS[$rank]} teacher_port=$((TEACHER_PORT + rank))"
    (
        cd "$OPENPI"
        exec setsid env -u VIRTUAL_ENV -u PYTHONPATH CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" \
            "$HOME/.local/bin/uv" run scripts/serve_policy.py --port="$((TEACHER_PORT + rank))" \
            policy:checkpoint --policy.config="$TEACHER_CONFIG" --policy.dir="$CHECKPOINT"
    ) > "$LOGS/teacher_rank$rank.log" 2>&1 &
    PIDS+=("$!")
done

deadline=$((SECONDS + 600))
for rank in "${!GPUS[@]}"; do
    until curl --noproxy '*' -fs --max-time 2 "http://127.0.0.1:$((TEACHER_PORT + rank))/healthz" >/dev/null; do
        for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "Teacher exited: $LOGS" >&2; exit 1; }; done
        (( SECONDS < deadline )) || { echo "Teacher startup timeout: $LOGS" >&2; exit 1; }
        sleep 1
    done
done

LAUNCH=()
if (( N > 1 )); then LAUNCH+=(--multi_gpu); fi
setsid env PYTHONPATH="$PWD/src" "$PYTHON" -m accelerate.commands.launch \
    "${LAUNCH[@]}" --num_processes="$N" --gpu_ids="$CUDA_VISIBLE_DEVICES" \
    --num_machines=1 --machine_rank=0 --mixed_precision=bf16 --dynamo_backend=no \
    --main_process_port="$MASTER_PORT" --module lerobot.scripts.lerobot_train \
    --config_path="$CONFIG" --distillation.teacher_urls="$URLS" &
PIDS+=("$!")
wait "${PIDS[$N]}"
