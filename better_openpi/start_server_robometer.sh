XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/server_policy_with_robometer.py \
    --policy-config pi05_agilex \
    --checkpoint /media/sail/jinghua4T/ckpt_yantong/pi05_fold_dishcloth_0623/49999 \
    --policy-port 8000 \
    --robometer-url http://127.0.0.1:8002 \
    --dashboard-port 8080 \
    --camera cam_high \
    --max-frames 8 \
    --monitor-interval 1.0


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
#     --use-custom-sample-kwargs \
#     --infer-time-schedule=HAS \
#     --alpha=0.6 \
#     --u0=0.9 \
#     policy:checkpoint \
#     --policy.config=pi05_faster_libero --policy.dir=checkpoints/pi05_faster_libero/pi05_faster_libero/29999