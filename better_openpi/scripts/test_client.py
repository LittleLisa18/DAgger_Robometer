"""Example client for the streaming WebSocket policy server.

Usage:
    1. Start the streaming server:
       python scripts/serve_policy.py --streaming --use-custom-sample-kwargs \
           --infer-time-schedule adaptive --policy.config <CONFIG> --policy.dir <CKPT_DIR>

    2. Run this client:
       python test_client.py
"""

import time

import numpy as np
from openpi_client import websocket_client_policy


HOST = "0.0.0.0"
PORT = 8000

ACTION_DIM = 14
ACTION_HORIZON = 50
DELAY = 4


def make_random_observation(*, rtc: bool = False) -> dict:
    """Build a random observation dict for testing."""
    obs = {
        "state": np.random.randn(ACTION_DIM).astype(np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }
    if rtc:
        obs["action_prefix"] = np.random.randn(DELAY, ACTION_DIM).astype(np.float32)
        obs["delay"] = np.array(DELAY)
    return obs


def demo_non_streaming():
    """Standard (non-streaming) inference — single request / single response."""
    print("=" * 60)
    print("Non-streaming inference")
    print("=" * 60)

    client = websocket_client_policy.WebsocketClientPolicy(HOST, PORT)
    obs = make_random_observation(rtc=True)

    t0 = time.perf_counter()
    result = client.infer(obs)
    elapsed = time.perf_counter() - t0

    actions = result["actions"]
    print(f"  Received full chunk: shape={actions.shape}, time={elapsed:.3f}s")
    print()


NUM_RUNS = 20
NUM_PARTIALS = 10


def demo_streaming():
    """Streaming inference — partial actions arrive before full chunk is ready."""
    print("=" * 60)
    print("Streaming inference")
    print("=" * 60)

    client = websocket_client_policy.WebsocketClientPolicy(HOST, PORT)
    obs = make_random_observation(rtc=False)

    # warmup
    for _ in range(2):
        client.infer_streaming(obs)

    # For each of the first NUM_PARTIALS positions, collect arrival times across runs
    arrival_times: list[list[float]] = [[] for _ in range(NUM_PARTIALS)]
    action_shapes: list[tuple[int, ...] | None] = [None] * NUM_PARTIALS

    for run in range(NUM_RUNS):
        partial_log: list[tuple[float, np.ndarray]] = []

        def on_actions_ready(actions: np.ndarray):
            ts = time.time()
            partial_log.append((ts, actions))

        t0 = time.time()
        client.infer_streaming(obs, on_actions_ready=on_actions_ready)

        for i in range(min(NUM_PARTIALS, len(partial_log))):
            arrival_times[i].append(partial_log[i][0] - t0)
            if action_shapes[i] is None:
                action_shapes[i] = partial_log[i][1].shape

        # time.sleep(10)

    print(f"\nRan {NUM_RUNS} times. Mean arrival time (s) for first {NUM_PARTIALS} partial messages:\n")
    for i in range(NUM_PARTIALS):
        if arrival_times[i]:
            mean_t = np.mean(arrival_times[i])
            std_t = np.std(arrival_times[i])
            n = len(arrival_times[i])
            shape_str = f", shape={action_shapes[i]}" if action_shapes[i] else ""
            print(f"  Partial {i + 1:2d}: mean = {mean_t*1000:.1f}ms  (std = {std_t*1000:.1f}ms, n = {n}{shape_str})")
        else:
            print(f"  Partial {i + 1:2d}: no data")
    print()


if __name__ == "__main__":
    demo_streaming()
    # demo_non_streaming()
