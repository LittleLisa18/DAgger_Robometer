#!/bin/bash
set -e
cd "$(dirname "$0")"

DATA=/home/ma-user/work/users/yantong/data/fold_dishcloth_0623

export UV_CACHE_DIR=/home/ma-user/work/users/yantong/cache/uv
export TMPDIR=/home/ma-user/work/users/yantong/tmp
export UV_HTTP_TIMEOUT=1500
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

echo "Checking and converting dataset if necessary..."
python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id=local/fold_dishcloth_0623 \
    --root="$DATA" \
    --push-to-hub=false