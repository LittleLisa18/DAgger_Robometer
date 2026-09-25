#!/usr/bin/env python3
"""Benchmark end-to-end Robometer inference latency through its HTTP API."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Video used to construct the Robometer input")
    parser.add_argument("--task", required=True, help="Language instruction sent to Robometer")
    parser.add_argument("--robometer-url", default="http://127.0.0.1:8001", help="Robometer server base URL")
    parser.add_argument("--max-frames", type=int, default=8, help="Number of evenly sampled input frames")
    parser.add_argument("--runs", type=int, default=20, help="Number of measured inference requests")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warm-up requests excluded from the statistics")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument("--use-frame-steps", action="store_true", help="Enable Robometer frame-step prompting")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    if args.max_frames <= 0 or args.runs <= 0 or args.warmup_runs < 0 or args.timeout <= 0:
        parser.error("max-frames, runs, and timeout must be positive; warmup-runs must be non-negative")
    return args


def sample_video_frames(video_path: Path, count: int) -> np.ndarray:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Could not determine frame count for {video_path}")
        if frame_count < count:
            raise ValueError(f"Video contains {frame_count} frames, fewer than --max-frames={count}")

        indices = np.linspace(0, frame_count - 1, count, dtype=int)
        frames: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to decode frame {index} from {video_path}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return np.stack(frames).astype(np.uint8, copy=False)


def build_multipart(frames: np.ndarray, task: str, use_frame_steps: bool) -> tuple[bytes, str]:
    boundary = "----robometer-benchmark-" + uuid.uuid4().hex
    sample = {
        "sample_type": "progress",
        "trajectory": {
            "frames": {"__numpy_file__": "sample_0_trajectory_frames"},
            "frames_shape": list(frames.shape),
            "task": task,
            "id": "latency-benchmark",
            "metadata": {"subsequence_length": int(len(frames))},
            "video_embeddings": None,
        },
    }
    array_buffer = io.BytesIO()
    np.save(array_buffer, frames, allow_pickle=False)
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    add_field("sample_0", json.dumps(sample))
    add_field("use_frame_steps", "true" if use_frame_steps else "false")
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="sample_0_trajectory_frames"; filename="frames.npy"\r\n',
            b"Content-Type: application/octet-stream\r\n\r\n",
            array_buffer.getvalue(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def run_request(endpoint: str, body: bytes, content_type: str, timeout: float) -> tuple[float, dict[str, Any], int]:
    request = urllib.request.Request(endpoint, data=body, method="POST", headers={"Content-Type": content_type})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Robometer returned HTTP {exc.code}: {detail}") from exc
    latency_s = time.perf_counter() - started
    return latency_s, json.loads(response_body), len(response_body)


def latest_predictions(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    progress_rows = (payload.get("outputs_progress") or {}).get("progress_pred") or []
    success_rows = (payload.get("outputs_success") or {}).get("success_probs") or []
    progress = float(progress_rows[0][-1]) if progress_rows and progress_rows[0] else None
    success = float(success_rows[0][-1]) if success_rows and success_rows[0] else None
    return progress, success


def main() -> None:
    args = parse_args()
    frames = sample_video_frames(args.video, args.max_frames)
    body, content_type = build_multipart(frames, args.task, args.use_frame_steps)
    endpoint = args.robometer_url.rstrip("/") + "/evaluate_batch_npy"

    print(f"Endpoint: {endpoint}")
    print(f"Input: {args.max_frames} frames with shape {tuple(frames.shape[1:])}")
    print(f"Warm-up requests: {args.warmup_runs}; measured requests: {args.runs}")

    for index in range(args.warmup_runs):
        latency_s, _, _ = run_request(endpoint, body, content_type, args.timeout)
        print(f"Warm-up {index + 1:02d}/{args.warmup_runs}: {latency_s * 1000:.2f} ms")

    records: list[dict[str, Any]] = []
    for index in range(args.runs):
        latency_s, result, response_bytes = run_request(endpoint, body, content_type, args.timeout)
        progress, success = latest_predictions(result)
        record = {
            "run": index + 1,
            "latency_s": latency_s,
            "latency_ms": latency_s * 1000,
            "progress": progress,
            "success_probability": success,
            "response_bytes": response_bytes,
        }
        records.append(record)
        print(f"Run {index + 1:02d}/{args.runs}: {record['latency_ms']:.2f} ms")

    latencies_ms = np.asarray([record["latency_ms"] for record in records], dtype=np.float64)
    summary = {
        "video": str(args.video.expanduser().resolve()),
        "task": args.task,
        "endpoint": endpoint,
        "max_frames": args.max_frames,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms, ddof=1)) if args.runs > 1 else 0.0,
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "requests_per_second": float(1000.0 / np.mean(latencies_ms)),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"robometer_latency_{timestamp}.csv"
    json_path = output_dir / f"robometer_latency_{timestamp}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print("\nLatency summary (20 measured requests unless overridden):")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-run CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
