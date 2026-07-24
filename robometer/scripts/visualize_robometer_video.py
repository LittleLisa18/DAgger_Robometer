#!/usr/bin/env python3
"""Create an MP4 with source video, live Robometer curves, and collect labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


BG = (18, 22, 30)
PANEL = (29, 35, 46)
TEXT = (238, 241, 246)
MUTED = (157, 166, 181)
GRID = (62, 70, 84)
PROGRESS = (83, 190, 255)
SUCCESS = (104, 211, 145)
PROBABILITY = (218, 134, 255)
ROLLOUT = (255, 184, 77)
DAGGER = (92, 220, 132)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Original MP4 video")
    parser.add_argument("--results-dir", required=True, help="Directory containing Robometer .npy results")
    parser.add_argument("--hdf5", required=True, help="Matching episode HDF5 file")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--title", default="Dishcloth Folding — Robometer Timeline")
    parser.add_argument("--inference-fps", type=float, default=3.0)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--failure-timeout", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--progress-increase-epsilon", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def put_text(image, text, xy, scale=0.7, color=TEXT, thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def rounded_panel(image, p1, p2, color=PANEL, radius=18):
    x1, y1 = p1
    x2, y2 = p2
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def fit_frame(frame, box):
    x1, y1, x2, y2 = box
    max_w, max_h = x2 - x1, y2 - y1
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    x = x1 + (max_w - size[0]) // 2
    y = y1 + (max_h - size[1]) // 2
    return resized, x, y


def decode_labels(raw) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in raw
    ])


def resample_labels(labels: np.ndarray, video_frames: int) -> np.ndarray:
    if len(labels) == video_frames:
        return labels
    indices = np.linspace(0, len(labels) - 1, video_frames).round().astype(int)
    return labels[indices]


def load_results(results_dir: Path, video_stem: str):
    rewards_path = results_dir / f"{video_stem}_rewards.npy"
    probs_path = results_dir / f"{video_stem}_rewards_success_probs.npy"
    if not rewards_path.exists() or not probs_path.exists():
        raise FileNotFoundError(f"Expected {rewards_path.name} and {probs_path.name} in {results_dir}")
    progress = np.asarray(np.load(rewards_path), dtype=np.float32).reshape(-1)
    probability = np.asarray(np.load(probs_path), dtype=np.float32).reshape(-1)
    if len(progress) != len(probability):
        raise ValueError(f"Result length mismatch: progress={len(progress)}, success={len(probability)}")
    if not len(progress):
        raise ValueError("Robometer result arrays are empty")
    return progress, probability


def failure_timeline(
    progress: np.ndarray,
    probability: np.ndarray,
    inference_fps: float,
    failure_timeout: float | None,
    progress_increase_epsilon: float,
    success_threshold: float,
) -> np.ndarray:
    """Return failure state after each sample using progress and success confidence."""
    failure = np.zeros(len(progress), dtype=bool)
    if failure_timeout is None:
        return failure

    best_progress = float(progress[0])
    last_reset_time = 0.0
    for index, (progress_value, success_probability) in enumerate(zip(progress, probability)):
        sample_time = index / inference_fps
        if float(progress_value) > best_progress + progress_increase_epsilon:
            best_progress = float(progress_value)
            last_reset_time = sample_time
        if float(success_probability) > success_threshold:
            last_reset_time = sample_time
        failure[index] = (
            float(success_probability) <= success_threshold
            and sample_time - last_reset_time >= failure_timeout
        )
    return failure


def draw_chart(canvas, box, title, values, color, visible_count, current_x, binary=False):
    x1, y1, x2, y2 = box
    rounded_panel(canvas, (x1, y1), (x2, y2))
    put_text(canvas, title, (x1 + 18, y1 + 29), 0.62, TEXT, 2)
    left, right, top, bottom = x1 + 45, x2 - 15, y1 + 48, y2 - 32
    for level in (0.0, 0.5, 1.0):
        yy = round(bottom - level * (bottom - top))
        cv2.line(canvas, (left, yy), (right, yy), GRID, 1, cv2.LINE_AA)
        put_text(canvas, f"{level:g}", (x1 + 8, yy + 5), 0.38, MUTED)
    n = len(values)
    if visible_count > 0:
        shown = np.clip(values[:visible_count], 0, 1)
        xs = np.linspace(left, right, n)[:visible_count]
        ys = bottom - shown * (bottom - top)
        points = np.column_stack((xs, ys)).round().astype(np.int32)
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), 4, color, -1, cv2.LINE_AA)
        elif binary:
            for idx in range(1, len(points)):
                cv2.line(canvas, tuple(points[idx - 1]), (points[idx, 0], points[idx - 1, 1]), color, 3, cv2.LINE_AA)
                cv2.line(canvas, (points[idx, 0], points[idx - 1, 1]), tuple(points[idx]), color, 3, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
        # Mark every Robometer inference sample explicitly on the live curve.
        for point in points:
            cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(point), 4, (235, 239, 246), 1, cv2.LINE_AA)
        value = shown[-1]
        put_text(canvas, f"{value:.3f}" if not binary else str(int(value)), (x2 - 85, y1 + 29), 0.55, color, 2)
    cursor = round(left + np.clip(current_x, 0, 1) * (right - left))
    cv2.line(canvas, (cursor, top), (cursor, bottom), (220, 225, 235), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.inference_fps <= 0:
        raise ValueError("--inference-fps must be positive")
    if args.failure_timeout is not None and args.failure_timeout <= 0:
        raise ValueError("--failure-timeout must be positive")
    if args.progress_increase_epsilon < 0:
        raise ValueError("--progress-increase-epsilon must be non-negative")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain four characters")

    video_path = Path(args.video).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    hdf5_path = Path(args.hdf5).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    progress, probability = load_results(results_dir, video_path.stem)
    binary = (probability >= args.success_threshold).astype(np.float32)

    failure_by_sample = failure_timeline(
        progress,
        probability,
        args.inference_fps,
        args.failure_timeout,
        args.progress_increase_epsilon,
        args.success_threshold,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0 or total_frames <= 0:
        raise ValueError("Video has invalid FPS or frame count")

    with h5py.File(hdf5_path, "r") as episode:
        if "collect" not in episode:
            raise KeyError("HDF5 does not contain the required 'collect' dataset")
        labels = resample_labels(decode_labels(episode["collect"][:]), total_frames)

    # This exactly mirrors example_inference_local.extract_frames: linspace across
    # the complete video after determining the requested number of samples.
    desired = int(round(total_frames * args.inference_fps / video_fps))
    desired = max(1, min(desired, total_frames))
    if desired != len(progress):
        # Results may have been capped with --max-frames; their stored length is
        # authoritative, while linspace remains the inference sampling policy.
        desired = len(progress)
    sample_indices = np.linspace(0, total_frames - 1, desired, dtype=int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*args.codec), video_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video with codec {args.codec}: {output_path}")

    chart_y1, chart_y2 = int(args.height * 0.705), args.height - 35
    margin, gap = 36, 20
    chart_w = (args.width - 2 * margin - 2 * gap) // 3
    boxes = [(margin + i * (chart_w + gap), chart_y1, margin + i * (chart_w + gap) + chart_w, chart_y2) for i in range(3)]
    video_box = (150, 100, args.width - 150, chart_y1 - 22)

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            canvas = np.full((args.height, args.width, 3), BG, dtype=np.uint8)
            title_size = cv2.getTextSize(args.title, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)[0]
            put_text(canvas, args.title, ((args.width - title_size[0]) // 2, 50), 1.05, TEXT, 2)

            shown, vx, vy = fit_frame(frame, video_box)
            canvas[vy:vy + shown.shape[0], vx:vx + shown.shape[1]] = shown
            visible_count = int(np.searchsorted(sample_indices, frame_index, side="right"))
            failed = visible_count > 0 and failure_by_sample[visible_count - 1]
            border_color = (69, 69, 255) if failed else (82, 91, 106)
            border_width = 10 if failed else 2
            cv2.rectangle(canvas, (vx, vy), (vx + shown.shape[1], vy + shown.shape[0]), border_color, border_width)
            if failed:
                cv2.rectangle(canvas, (vx + 14, vy + 67), (vx + 245, vy + 112), border_color, -1)
                put_text(canvas, "FAILURE: STALLED", (vx + 27, vy + 98), 0.67, (255, 255, 255), 2)

            label = labels[min(frame_index, len(labels) - 1)].strip().lower()
            label_color = DAGGER if label == "dagger" else ROLLOUT
            label_text = label.upper() if label else "UNKNOWN"
            badge_w = max(150, cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 34)
            cv2.rectangle(canvas, (vx + 14, vy + 14), (vx + 14 + badge_w, vy + 55), label_color, -1)
            put_text(canvas, label_text, (vx + 30, vy + 43), 0.7, (18, 25, 28), 2)
            elapsed = frame_index / video_fps
            duration = total_frames / video_fps
            time_text = f"{elapsed:06.2f}s / {duration:06.2f}s"
            put_text(canvas, time_text, (vx + shown.shape[1] - 245, vy + 43), 0.65, TEXT, 2)

            fraction = frame_index / max(1, total_frames - 1)
            draw_chart(canvas, boxes[0], "Task Progress", progress, PROGRESS, visible_count, fraction)
            draw_chart(canvas, boxes[1], f"Success (threshold {args.success_threshold:g})", binary, SUCCESS, visible_count, fraction, True)
            draw_chart(canvas, boxes[2], "Success Probability", probability, PROBABILITY, visible_count, fraction)
            writer.write(canvas)
            frame_index += 1
            if frame_index % max(1, round(video_fps * 5)) == 0:
                print(f"Rendered {frame_index}/{total_frames} frames")
    finally:
        cap.release()
        writer.release()

    print(f"Saved synchronized visualization: {output_path}")
    print(f"Video: {total_frames} frames at {video_fps:g} FPS")
    print(f"Robometer samples: {len(progress)} at requested {args.inference_fps:g} FPS")


if __name__ == "__main__":
    main()
