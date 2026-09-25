#!/usr/bin/env python3
"""Run offline Robometer inference and Live-Robometer-style failure detection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Robometer checkpoint or Hugging Face model id")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input videos")
    parser.add_argument("--pattern", default="*.mp4", help="Video filename pattern")
    parser.add_argument("--task", required=True, help="Language instruction shared by the videos")
    parser.add_argument("--sampling-fps", type=float, default=1.0, help="Target Robometer evaluation frequency")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum history frames per monitoring step")
    parser.add_argument("--progress-epsilon", type=float, default=0.05)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--failure-timeout", type=float, default=4.0, help="Required stagnation duration in seconds")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, mps, or cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("offline_failure_results"))
    args = parser.parse_args()
    if args.sampling_fps <= 0 or args.max_frames <= 0 or args.failure_timeout <= 0:
        parser.error("sampling-fps, max-frames, and failure-timeout must be positive")
    if args.progress_epsilon < 0:
        parser.error("progress-epsilon must be non-negative")
    if not 0 <= args.success_threshold <= 1:
        parser.error("success-threshold must be in [0, 1]")
    return args


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
    return device


def video_schedule(video_path: Path, sampling_fps: float) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        native_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if native_fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid video metadata: fps={native_fps}, frames={total_frames}")
        duration = (total_frames - 1) / native_fps
        desired = max(1, int(round(duration * sampling_fps)) + 1)
        desired = min(desired, total_frames)
        monitor_indices = np.linspace(0, total_frames - 1, desired, dtype=int)
    finally:
        capture.release()
    return monitor_indices, monitor_indices.astype(np.float64) / native_fps


def load_history(capture: cv2.VideoCapture, end_index: int, max_frames: int) -> np.ndarray:
    history_size = min(max_frames, end_index + 1)
    history_indices = np.linspace(0, end_index, history_size, dtype=int)
    frames: list[np.ndarray] = []
    for index in history_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to decode frame {index}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames).astype(np.uint8, copy=False)


def prepare_model(model_path: str, device: torch.device) -> tuple[Any, Any, Any, bool, int]:
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(model_path=model_path, device=device)
    reward_model.eval()
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete" if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    return tokenizer, reward_model, collator, is_discrete, num_bins


def infer_video(
    frames: np.ndarray,
    task: str,
    video_id: str,
    tokenizer: Any,
    reward_model: Any,
    collator: Any,
    is_discrete: bool,
    num_bins: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id=video_id,
        metadata={"subsequence_length": int(len(frames))},
        video_embeddings=None,
    )
    batch = collator([ProgressSample(trajectory=trajectory, sample_type="progress")])
    inputs = batch["progress_inputs"]
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    with torch.inference_mode():
        result = compute_batch_outputs(
            reward_model,
            tokenizer,
            inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )
    progress_rows = result.get("progress_pred") or []
    success_rows = (result.get("outputs_success") or {}).get("success_probs") or []
    if not progress_rows or not progress_rows[0]:
        raise RuntimeError("Robometer returned no progress predictions")
    if not success_rows or not success_rows[0]:
        raise RuntimeError("Robometer returned no success probabilities")
    progress = np.clip(np.asarray(progress_rows[0], dtype=np.float64), 0.0, 1.0)
    success = np.clip(np.asarray(success_rows[0], dtype=np.float64), 0.0, 1.0)
    if len(progress) != len(frames) or len(success) != len(frames):
        raise RuntimeError(
            f"Prediction length mismatch: frames={len(frames)}, progress={len(progress)}, success={len(success)}"
        )
    return progress, success


def detect_failure(
    times: np.ndarray,
    progress: np.ndarray,
    success: np.ndarray,
    progress_epsilon: float,
    success_threshold: float,
    failure_timeout: float,
) -> dict[str, np.ndarray | int | None]:
    """Mirror LiveState: reset the timer on progress or confident success."""
    best_progress: float | None = None
    last_valid_time = float(times[0])
    reference = np.empty(len(times), dtype=np.float64)
    stall_seconds = np.empty(len(times), dtype=np.float64)
    failure_condition = np.zeros(len(times), dtype=bool)

    for index, (sample_time, progress_value, success_probability) in enumerate(
        zip(times, progress, success, strict=True)
    ):
        if best_progress is None or progress_value > best_progress + progress_epsilon:
            best_progress = float(progress_value)
            last_valid_time = float(sample_time)
        if success_probability > success_threshold:
            last_valid_time = float(sample_time)
        reference[index] = best_progress
        stall_seconds[index] = max(0.0, float(sample_time) - last_valid_time)
        failure_condition[index] = (
            success_probability <= success_threshold and stall_seconds[index] >= failure_timeout
        )

    failure_indices = np.flatnonzero(failure_condition)
    first_failure_index = int(failure_indices[0]) if failure_indices.size else None
    failure_latched = np.zeros(len(times), dtype=bool)
    if first_failure_index is not None:
        # Live execution stops on the first detected failure, so keep the
        # offline visualization latched from that event onward.
        failure_latched[first_failure_index:] = True
    return {
        "reference_progress": reference,
        "stall_seconds": stall_seconds,
        "failure_condition": failure_condition,
        "failure_latched": failure_latched,
        "first_failure_index": first_failure_index,
    }


def save_plot(
    output_path: Path,
    title: str,
    times: np.ndarray,
    progress: np.ndarray,
    success: np.ndarray,
    detection: dict[str, Any],
    success_threshold: float,
    failure_timeout: float,
) -> None:
    first_failure_index = detection["first_failure_index"]
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, progress, marker="o", label="Progress", color="#6699A1")
    axes[0].plot(times, detection["reference_progress"], linestyle="--", label="Reference progress", color="#B58D76")
    axes[0].set_ylabel("Progress")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(loc="best")

    axes[1].plot(times, success, marker="o", label="Success probability", color="#8D7AAE")
    axes[1].axhline(success_threshold, linestyle="--", color="#C6655A", label=r"$\theta_s$")
    axes[1].set_ylabel("Success probability")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(loc="best")

    axes[2].plot(times, detection["stall_seconds"], marker="o", label="Stagnation duration", color="#7A9E7E")
    axes[2].axhline(failure_timeout, linestyle="--", color="#C6655A", label=r"$T_f$")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xlabel("Video time (s)")
    axes[2].legend(loc="best")

    if first_failure_index is not None:
        failure_time = float(times[first_failure_index])
        for axis in axes:
            axis.axvline(failure_time, color="#B4423C", linewidth=2, label="Failure detected")
            axis.axvspan(failure_time, float(times[-1]), color="#D98C83", alpha=0.15)
        axes[0].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    model_components: tuple[Any, Any, Any, bool, int],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer, reward_model, collator, is_discrete, num_bins = model_components
    monitor_indices, times = video_schedule(video_path, args.sampling_fps)
    progress_values: list[float] = []
    success_values: list[float] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        for monitor_index, end_index in enumerate(monitor_indices):
            history = load_history(capture, int(end_index), args.max_frames)
            history_progress, history_success = infer_video(
                history,
                args.task,
                f"{video_path.stem}-{monitor_index}",
                tokenizer,
                reward_model,
                collator,
                is_discrete,
                num_bins,
                device,
            )
            progress_values.append(float(history_progress[-1]))
            success_values.append(float(history_success[-1]))
    finally:
        capture.release()
    progress = np.asarray(progress_values, dtype=np.float64)
    success = np.asarray(success_values, dtype=np.float64)
    detection = detect_failure(
        times,
        progress,
        success,
        args.progress_epsilon,
        args.success_threshold,
        args.failure_timeout,
    )
    first_index = detection["first_failure_index"]
    failure_time = float(times[first_index]) if first_index is not None else None

    np.save(output_dir / f"{video_path.stem}_progress.npy", progress)
    np.save(output_dir / f"{video_path.stem}_success_probability.npy", success)
    np.save(output_dir / f"{video_path.stem}_failure.npy", detection["failure_latched"])
    timeline_path = output_dir / f"{video_path.stem}_timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample",
                "video_time_s",
                "progress",
                "reference_progress",
                "success_probability",
                "stall_seconds",
                "failure_condition",
                "failure_latched",
            ]
        )
        for index in range(len(times)):
            writer.writerow(
                [
                    index,
                    times[index],
                    progress[index],
                    detection["reference_progress"][index],
                    success[index],
                    detection["stall_seconds"][index],
                    int(detection["failure_condition"][index]),
                    int(detection["failure_latched"][index]),
                ]
            )
    plot_path = output_dir / f"{video_path.stem}_failure_detection.png"
    save_plot(
        plot_path,
        f"{video_path.name} — Offline Robometer Failure Detection",
        times,
        progress,
        success,
        detection,
        args.success_threshold,
        args.failure_timeout,
    )
    return {
        "video": str(video_path),
        "num_monitoring_steps": len(times),
        "duration_s": float(times[-1]),
        "failure_detected": first_index is not None,
        "failure_time_s": failure_time,
        "final_progress": float(progress[-1]),
        "final_success_probability": float(success[-1]),
        "timeline_csv": str(timeline_path),
        "visualization": str(plot_path),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        raise FileNotFoundError(f"No videos matching {args.pattern!r} in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading Robometer once from {args.model_path} on {device} ...")
    model_components = prepare_model(args.model_path, device)

    summaries: list[dict[str, Any]] = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}")
        summary = process_video(video_path, args, model_components, device, output_dir)
        summaries.append(summary)
        status = f"FAILURE at {summary['failure_time_s']:.2f}s" if summary["failure_detected"] else "no failure"
        print(f"  {status}")

    summary_csv = output_dir / "failure_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary_json = output_dir / "failure_summary.json"
    with summary_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "parameters": {
                    "model_path": args.model_path,
                    "task": args.task,
                    "device": str(device),
                    "sampling_fps": args.sampling_fps,
                    "max_frames": args.max_frames,
                    "progress_epsilon": args.progress_epsilon,
                    "success_threshold": args.success_threshold,
                    "failure_timeout": args.failure_timeout,
                },
                "videos": summaries,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Visualizations: {output_dir}/*_failure_detection.png")


if __name__ == "__main__":
    main()
