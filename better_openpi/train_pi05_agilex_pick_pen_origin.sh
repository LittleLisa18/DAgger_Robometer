#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-train}"
EXP_NAME="${2:-pick_pen_50k}"

export HF_HOME="${HF_HOME:-/home/ma-user/work/users/yantong/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/ma-user/work/users/yantong/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

case "$MODE" in
  stats)
    exec uv run scripts/compute_norm_stats.py --config-name pi05_agilex
    ;;
  test)
    exec uv run scripts/train.py pi05_agilex \
      --exp-name="$EXP_NAME" \
      --num-train-steps=100 \
      --batch-size=8 \
      --save-interval=100 \
      --wandb-enabled=false \
      --overwrite
    ;;
  train)
    exec uv run scripts/train.py pi05_agilex \
      --exp-name="$EXP_NAME" \
      --num-train-steps=50000 \
      --batch-size=32 \
      --save-interval=5000 \
      --keep-period=5000 \
      --wandb-enabled=false
    ;;
  resume)
    exec uv run scripts/train.py pi05_agilex \
      --exp-name="$EXP_NAME" \
      --resume \
      --wandb-enabled=false
    ;;
  *)
    echo "Usage: $0 {stats|test|train|resume} [experiment_name]" >&2
    exit 2
    ;;
esac


#!/bin/bash
#SBATCH --partition=q-hgpu-batch
#SBATCH --time=48:00:00
#SBATCH --job-name=pi05_agilex_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h800:4
#SBATCH --mem=200G
#SBATCH --output=logs/pi05_agilex_%j.out
#SBATCH --error=logs/pi05_agilex_%j.err

set -euo pipefail

# 可选：指定要用的节点（如果你确实要请求特定节点）
# SBATCH --nodelist=gpucluster-g3

# 如果集群要求使用 --gres 格式，请将上面的 --gpus-per-node=4 替换为：
# #SBATCH --gres=gpu:4

# W&B API key (set from user). Consider storing this more securely (e.g., Slurm secrets).
export WANDB_API_KEY="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"
# 将 OpenPI 缓存重定向到用户可写目录，避免默认的 /mnt/models 权限错误
export OPENPI_DATA_HOME=/userhome/cs3/u3013043/.cache/openpi
mkdir -p "$OPENPI_DATA_HOME"
chmod 700 "$OPENPI_DATA_HOME"

# 激活 Python 虚拟环境（按实际路径调整）
source /userhome/cs3/u3013043/code/better_openpi_lyt/.venv/bin/activate

# 进入代码目录（按实际路径调整）
cd /userhome/cs3/u3013043/code/better_openpi_lyt

# 保存 CPU 线程环境变量
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# 启动训练（命令行格式可能按 tyro 配置不同，下面是常用调用方式）
uv run scripts/compute_norm_stats.py --config-name pi05_agilex
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_agilex --exp-name=pi05_fold_dishcloth_0623_0716 --overwrite
# 如果你更习惯键值风格，也可以尝试：
# python scripts/train.py pi05_agilex exp_name=pi05_fold_dishcloth_0623_0716 batch_size=128 num_workers=8
