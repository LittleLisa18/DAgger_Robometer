# OpenPI inference with live Robometer monitoring

This integration adds a new server entry point and leaves all existing OpenPI and
Robometer source files unchanged. Robot clients continue to use the normal OpenPI
WebSocket protocol. Robometer runs as a separate HTTP service and is queried by a
background worker, so reward-model latency does not block action inference.

## 1. Start Robometer

Use a different port from OpenPI. From the `robometer` directory:

```bash
uv run python robometer/evals/eval_server.py \
  model_path=Robometer-4B \
  server_url=0.0.0.0 \
  server_port=8001 \
  num_gpus=1
```

`model_path` can instead be a local Robometer checkpoint. Ideally put OpenPI and
Robometer on different GPUs (or different machines) because both models are large.

Check the reward server before continuing:

```bash
curl http://127.0.0.1:8001/health
```

## 2. Start the monitored OpenPI server

From `better_openpi`:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 uv run scripts/serve_policy_with_robometer.py \
  --policy-config pi05_agilex \
  --checkpoint checkpoints/pi05_agilex/pick_beverage/49999 \
  --prompt "pick up the beverage" \
  --policy-port 8000 \
  --robometer-url http://127.0.0.1:8001 \
  --dashboard-port 8080 \
  --camera cam_high \
  --max-frames 16 \
  --monitor-interval 1.0
```

For RTC/FASTER checkpoints, select their matching config, for example
`--policy-config pi05_faster_agilex`. The checkpoint argument points to the step
directory containing `params/` and `assets/`, not to `params/` itself.

If Robometer is on another machine, set `--robometer-url http://<reward-server-ip>:8001`.

To enable failure highlighting when progress does not increase, add (for example)
`--failure-timeout 5`. If omitted, the feature is disabled. Progress changes of
0.05 or less are treated as model jitter rather than an increase. Failure is reported
only when that timeout is reached and the current success probability is at or below
`--success-threshold`. A probability above the threshold clears the accumulated
stall duration; if it later drops, the timeout starts again from that high-probability
sample.

## 3. Connect the robot and view the dashboard

The existing robot client needs no protocol change:

```python
client = websocket_client_policy.WebsocketClientPolicy(
    host="<openpi-server-ip>", port=8000
)
result = client.infer(observation)
actions = result["actions"]
```

For AgileX, `observation` should contain `images.cam_high`, `images.cam_left_wrist`,
`images.cam_right_wrist`, a 14-D `state`, and `prompt`. Open in a browser:

```text
http://<openpi-server-ip>:8080
```

The page shows the latest selected camera image, current progress, binary success,
success probability, live history curves, model latency/status, policy request count,
and dropped stale monitor-job count. Colors and the three Robometer metrics follow
`robometer/scripts/visualize_robometer_video.py`.

The Piper `OpenpiClient` also sends an independent dashboard preview from the original
top-camera frame. It uses 640x480 JPEG at quality 90 and is rate-limited to 5 FPS.
The sender has a one-frame queue, a short timeout, and silently drops preview frames
when the dashboard is slow or unavailable, so it cannot block policy inference. Set
`dashboard_port=None` when constructing `OpenpiClient` to disable this preview. The
normal 224x224 OpenPI and Robometer inputs are unchanged.

Pressing `r` in the Piper inference reset menu resets this timeline automatically.
Click **Reset episode** when using another client. A changed prompt also starts a new
timeline automatically. A custom client may alternatively send
`"robometer_reset": true` in its first observation; this key is removed before the
observation reaches OpenPI.

Each completed Robometer estimate is appended to
`better_openpi/robometer_live_runs/live_<timestamp>.jsonl`. Each record includes
progress, success probability, thresholded success, latency, task, episode, and the
full per-request progress/success traces.

## Operational notes

- Monitoring uses only the camera selected by `--camera`; the default is `cam_high`.
- The worker queue retains only the newest pending snapshot. A slow or unavailable
  Robometer therefore degrades the dashboard, not robot action inference.
- `--monitor-interval 1.0` means at most one reward request per second. Raise it if
  GPU load is high.
- `--max-frames 16` retains an evenly sampled rolling trajectory. This matches the
  default Robometer context size; change it only if the selected checkpoint expects
  another value.
- `--use-frame-steps` reproduces Robometer's prefix/frame-step evaluation, but costs
  substantially more compute and is usually unsuitable for low-latency monitoring.
- If both models share one GPU, reduce OpenPI's XLA memory fraction and expect lower
  throughput. Separate processes do not by themselves prevent GPU-memory contention.
