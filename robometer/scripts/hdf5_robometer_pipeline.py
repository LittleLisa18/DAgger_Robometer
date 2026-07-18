#!/usr/bin/env python3
"""Run Robometer on an HDF5 episode and create exactly four result files.

Outputs:
  <episode>_<camera>_rewards.npy
  <episode>_<camera>_rewards_success_probs.npy
  <episode>_<camera>_rewards_progress_success.png
  <episode>_robometer_visualization.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from robometer.evals.eval_viz_utils import create_combined_progress_success_plot
from scripts.example_inference_local import compute_rewards_per_frame_local
from scripts.visualize_robometer_video import (
    BG,
    DAGGER,
    MUTED,
    PROBABILITY,
    PROGRESS,
    ROLLOUT,
    SUCCESS,
    TEXT,
    draw_chart,
    fit_frame,
    put_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, help="Input episode HDF5 file")
    parser.add_argument("--model-path", required=True, help="Local model path or Hugging Face model ID")
    parser.add_argument("--task", required=True, help="Task instruction given to Robometer")
    parser.add_argument("--output-dir", required=True, help="Directory for the four output files")
    parser.add_argument(
        "--camera-key",
        default="observations/images/cam_high",
        help="HDF5 RGB camera dataset key (default: observations/images/cam_high)",
    )
    parser.add_argument("--source-fps", type=float, default=30.0, help="HDF5 recording FPS")
    parser.add_argument("--inference-fps", type=float, default=3.0, help="Robometer sampling FPS")
    parser.add_argument("--max-frames", type=int, default=512, help="Maximum frames sent to Robometer")
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--title", default=None, help="Visualization title")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--codec", default="mp4v", help="FourCC output codec")
    return parser.parse_args()


def decode_label(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def make_sample_indices(total_frames: int, source_fps: float, inference_fps: float, max_frames: int) -> np.ndarray:
    desired = int(round(total_frames * inference_fps / source_fps))
    desired = max(1, min(desired, total_frames, max_frames))
    if desired == total_frames:
        return np.arange(total_frames, dtype=np.int64)
    # Same policy used by example_inference_local.py through extract_frames().
    return np.linspace(0, total_frames - 1, desired, dtype=np.int64)


def output_paths(hdf5_path: Path, output_dir: Path, camera_key: str):
    camera = camera_key.rstrip("/").split("/")[-1]
    base = f"{hdf5_path.stem}_{camera}"
    rewards = output_dir / f"{base}_rewards.npy"
    success = output_dir / f"{base}_rewards_success_probs.npy"
    plot = output_dir / f"{base}_rewards_progress_success.png"
    video = output_dir / f"{hdf5_path.stem}_robometer_visualization.mp4"
    return rewards, success, plot, video


def save_plot(progress, probability, threshold: float, title: str, path: Path) -> None:
    binary = (probability > threshold).astype(np.int32)
    show_success = len(probability) == len(progress) and len(progress) > 0
    figure = create_combined_progress_success_plot(
        progress_pred=progress,
        num_frames=len(progress),
        success_binary=binary if show_success else None,
        success_probs=probability if show_success else None,
        success_labels=None,
        title=title,
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def render_video(
    images,
    labels,
    progress,
    probability,
    sample_indices,
    output_path: Path,
    args: argparse.Namespace,
    title: str,
) -> None:
    total_frames = int(images.shape[0])
    binary = (probability >= args.success_threshold).astype(np.float32)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec),
        args.source_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4 with codec {args.codec}: {output_path}")

    chart_y1, chart_y2 = int(args.height * 0.705), args.height - 35
    margin, gap = 36, 20
    chart_w = (args.width - 2 * margin - 2 * gap) // 3
    boxes = [
        (margin + i * (chart_w + gap), chart_y1, margin + i * (chart_w + gap) + chart_w, chart_y2)
        for i in range(3)
    ]
    video_box = (150, 100, args.width - 150, chart_y1 - 22)

    try:
        for frame_index in range(total_frames):
            rgb = np.asarray(images[frame_index])
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            canvas = np.full((args.height, args.width, 3), BG, dtype=np.uint8)

            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)[0]
            put_text(canvas, title, ((args.width - title_size[0]) // 2, 50), 1.05, TEXT, 2)
            shown, vx, vy = fit_frame(frame, video_box)
            canvas[vy : vy + shown.shape[0], vx : vx + shown.shape[1]] = shown
            cv2.rectangle(canvas, (vx, vy), (vx + shown.shape[1], vy + shown.shape[0]), (82, 91, 106), 2)

            label_index = min(frame_index, len(labels) - 1)
            label = decode_label(labels[label_index]).strip().lower()
            label_color = DAGGER if label == "dagger" else ROLLOUT
            label_text = label.upper() if label else "UNKNOWN"
            badge_w = max(150, cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 34)
            cv2.rectangle(canvas, (vx + 14, vy + 14), (vx + 14 + badge_w, vy + 55), label_color, -1)
            put_text(canvas, label_text, (vx + 30, vy + 43), 0.7, (18, 25, 28), 2)

            elapsed = frame_index / args.source_fps
            duration = total_frames / args.source_fps
            put_text(
                canvas,
                f"{elapsed:06.2f}s / {duration:06.2f}s",
                (vx + shown.shape[1] - 245, vy + 43),
                0.65,
                TEXT,
                2,
            )
            visible = int(np.searchsorted(sample_indices, frame_index, side="right"))
            fraction = frame_index / max(1, total_frames - 1)
            draw_chart(canvas, boxes[0], "Task Progress", progress, PROGRESS, visible, fraction)
            draw_chart(
                canvas,
                boxes[1],
                f"Success (threshold {args.success_threshold:g})",
                binary,
                SUCCESS,
                visible,
                fraction,
                True,
            )
            draw_chart(canvas, boxes[2], "Success Probability", probability, PROBABILITY, visible, fraction)
            writer.write(canvas)
            if (frame_index + 1) % max(1, round(args.source_fps * 5)) == 0:
                print(f"Rendered {frame_index + 1}/{total_frames} frames")
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    if args.source_fps <= 0 or args.inference_fps <= 0:
        raise ValueError("--source-fps and --inference-fps must be positive")
    if args.inference_fps > args.source_fps:
        raise ValueError("--inference-fps cannot exceed --source-fps")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    hdf5_path = Path(args.hdf5).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rewards_path, success_path, plot_path, video_path = output_paths(hdf5_path, output_dir, args.camera_key)
    title = args.title or f"{hdf5_path.stem.replace('_', ' ').title()} — Robometer Evaluation"

    with h5py.File(hdf5_path, "r") as episode:
        if args.camera_key not in episode:
            raise KeyError(f"Camera dataset not found: {args.camera_key}")
        if "collect" not in episode:
            raise KeyError("HDF5 does not contain the required 'collect' dataset")
        images = episode[args.camera_key]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected RGB images [T,H,W,3], got {images.shape}")
        total_frames = int(images.shape[0])
        labels = episode["collect"][:]
        if len(labels) != total_frames:
            label_indices = np.linspace(0, len(labels) - 1, total_frames).round().astype(int)
            labels = labels[label_indices]
        sample_indices = make_sample_indices(
            total_frames, args.source_fps, args.inference_fps, args.max_frames
        )
        sampled_frames = images[sample_indices]

        print(f"Running Robometer on {len(sample_indices)} sampled frames...")
        progress, probability = compute_rewards_per_frame_local(
            model_path=args.model_path,
            video_frames=np.asarray(sampled_frames, dtype=np.uint8),
            task=args.task,
        )
        progress = np.asarray(progress, dtype=np.float32).reshape(-1)
        probability = np.asarray(probability, dtype=np.float32).reshape(-1)
        if len(progress) != len(sample_indices) or len(probability) != len(sample_indices):
            raise ValueError(
                "Model output length does not match sampled frames: "
                f"frames={len(sample_indices)}, progress={len(progress)}, success={len(probability)}"
            )

        np.save(rewards_path, progress)
        np.save(success_path, probability)
        save_plot(progress, probability, args.success_threshold, f"Progress/Success — {hdf5_path.stem}", plot_path)
        render_video(images, labels, progress, probability, sample_indices, video_path, args, title)

    print("Created exactly four result files:")
    for path in (plot_path, success_path, rewards_path, video_path):
        print(path)


if __name__ == "__main__":
    main()
