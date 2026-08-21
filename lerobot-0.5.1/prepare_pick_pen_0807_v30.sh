#!/bin/bash
set -e
cd "$(dirname "$0")"

DATA=/home/ma-user/work/users/yantong/data/pick_pen_0807

export UV_CACHE_DIR=/home/ma-user/work/users/yantong/cache/uv
export TMPDIR=/home/ma-user/work/users/yantong/tmp
export UV_HTTP_TIMEOUT=1500
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

echo "Checking and converting dataset if necessary..."
python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id=local/pick_pen_0807 \
    --root="$DATA" \
    --push-to-hub=false