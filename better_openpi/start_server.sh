XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_calvin --policy.dir=checkpoints/pi05_calvin/pi05_calvin/5000

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
#     --use-custom-sample-kwargs \
#     --infer-time-schedule=HAS \
#     --alpha=0.6 \
#     --u0=0.9 \
#     policy:checkpoint \
#     --policy.config=pi05_faster_libero --policy.dir=checkpoints/pi05_faster_libero/pi05_faster_libero/29999