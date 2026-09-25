#!/usr/bin/env python3
"""Benchmark Robometer locally without starting an HTTP inference server."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local checkpoint path or Hugging Face model id")
    parser.add_argument("--video", type=Path, required=True, help="Input video")
    parser.add_argument("--task", required=True, help="Language instruction")
    parser.add_argument("--max-frames", type=int, default=8, help="Number of evenly sampled frames")
    parser.add_argument("--runs", type=int, default=20, help="Number of measured forward passes")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warm-up passes excluded from statistics")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    if args.max_frames <= 0 or args.runs <= 0 or args.warmup_runs < 0:
        parser.error("max-frames and runs must be positive; warmup-runs must be non-negative")
    return args


def sample_video_frames(video_path: Path, count: int) -> np.ndarray:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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


def prepare_inference(
    model_path: str,
    frames: np.ndarray,
    task: str,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], bool, int]:
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(model_path=model_path, device=device)
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    trajectory = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id="latency-benchmark",
        metadata={"subsequence_length": int(len(frames))},
        video_embeddings=None,
    )
    batch = batch_collator([ProgressSample(trajectory=trajectory, sample_type="progress")])
    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete" if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    return reward_model, tokenizer, progress_inputs, is_discrete, num_bins


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_once(
    reward_model: Any,
    tokenizer: Any,
    progress_inputs: dict[str, Any],
    is_discrete: bool,
    num_bins: int,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )
    synchronize(device)
    return time.perf_counter() - started, result


def latest_predictions(result: dict[str, Any]) -> tuple[float | None, float | None]:
    progress_rows = result.get("progress_pred") or []
    success_rows = (result.get("outputs_success") or {}).get("success_probs") or []
    progress = float(progress_rows[0][-1]) if progress_rows and progress_rows[0] else None
    success = float(success_rows[0][-1]) if success_rows and success_rows[0] else None
    return progress, success


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    frames = sample_video_frames(args.video, args.max_frames)
    print(f"Loading Robometer from {args.model_path} on {device} ...")
    reward_model, tokenizer, progress_inputs, is_discrete, num_bins = prepare_inference(
        args.model_path, frames, args.task, device
    )
    print(f"Input: {args.max_frames} frames with shape {tuple(frames.shape[1:])}")
    print(f"Warm-up passes: {args.warmup_runs}; measured passes: {args.runs}")

    for index in range(args.warmup_runs):
        latency_s, _ = forward_once(
            reward_model, tokenizer, progress_inputs, is_discrete, num_bins, device
        )
        print(f"Warm-up {index + 1:02d}/{args.warmup_runs}: {latency_s * 1000:.2f} ms")

    records: list[dict[str, Any]] = []
    for index in range(args.runs):
        latency_s, result = forward_once(
            reward_model, tokenizer, progress_inputs, is_discrete, num_bins, device
        )
        progress, success = latest_predictions(result)
        records.append(
            {
                "run": index + 1,
                "latency_s": latency_s,
                "latency_ms": latency_s * 1000,
                "progress": progress,
                "success_probability": success,
            }
        )
        print(f"Run {index + 1:02d}/{args.runs}: {latency_s * 1000:.2f} ms")

    latencies_ms = np.asarray([record["latency_ms"] for record in records], dtype=np.float64)
    summary = {
        "measurement": "local_model_forward",
        "model_path": args.model_path,
        "video": str(args.video.expanduser().resolve()),
        "task": args.task,
        "device": str(device),
        "max_frames": args.max_frames,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms, ddof=1)) if args.runs > 1 else 0.0,
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "inferences_per_second": float(1000.0 / np.mean(latencies_ms)),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"robometer_local_latency_{timestamp}.csv"
    json_path = output_dir / f"robometer_local_latency_{timestamp}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print("\nLatency summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-run CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
