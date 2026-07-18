#!/usr/bin/env python3
"""Convert an RGB camera stream in an HDF5 episode to an MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", help="Input HDF5 episode (single-file mode)")
    parser.add_argument("--output", help="Output MP4 path (single-file mode)")
    parser.add_argument("--input-dir", help="Folder containing HDF5 files (batch mode)")
    parser.add_argument("--output-dir", help="Folder for generated MP4 files (batch mode)")
    parser.add_argument(
        "--camera-key",
        default="observations/images/cam_high",
        help="HDF5 RGB image dataset key",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        required=True,
        help="Original HDF5 recording frame rate",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=3.0,
        help="Output video frame rate (default: 3)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive original-frame end index (default: end of episode)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC video codec (default: mp4v)",
    )
    return parser.parse_args()


def convert_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as episode:
        if args.camera_key not in episode:
            available = []
            episode.visititems(
                lambda name, obj: available.append(name)
                if isinstance(obj, h5py.Dataset)
                else None
            )
            raise KeyError(
                f"Camera dataset '{args.camera_key}' not found. "
                f"Available datasets: {available}"
            )

        images = episode[args.camera_key]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB images shaped [T,H,W,3], got {images.shape}"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"Expected uint8 images, got {images.dtype}")

        total_frames, height, width, _ = images.shape
        start = max(0, args.start_frame)
        end = total_frames if args.end_frame is None else min(args.end_frame, total_frames)
        if start >= end:
            raise ValueError(f"Invalid frame range [{start}, {end})")

        # Floating-point sampling preserves the requested rate even when the
        # source/output FPS ratio is not an integer.
        step = args.source_fps / args.output_fps
        source_indices = np.unique(
            np.minimum(
                np.round(np.arange(start, end, step)).astype(np.int64),
                end - 1,
            )
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*args.codec),
            args.output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open video writer for {output_path} with codec {args.codec}"
            )

        try:
            for index in source_indices:
                rgb = images[int(index)]
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    duration = len(source_indices) / args.output_fps
    print(f"Input: {input_path}")
    print(f"Camera: {args.camera_key}")
    print(f"Original frames: [{start}, {end})")
    print(f"Written frames: {len(source_indices)}")
    print(f"Output FPS: {args.output_fps:g}")
    print(f"Output duration: {duration:.2f} seconds")
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    if args.source_fps <= 0 or args.output_fps <= 0:
        raise ValueError("--source-fps and --output-fps must be positive")
    if args.output_fps > args.source_fps:
        raise ValueError("--output-fps cannot exceed --source-fps")
    if len(args.codec) != 4:
        raise ValueError("--codec must be a four-character FourCC code")

    single_mode = args.hdf5 is not None or args.output is not None
    batch_mode = args.input_dir is not None or args.output_dir is not None
    if single_mode and batch_mode:
        raise ValueError("Use either single-file mode or batch mode, not both")

    if single_mode:
        if not args.hdf5 or not args.output:
            raise ValueError("Single-file mode requires both --hdf5 and --output")
        convert_file(
            Path(args.hdf5).expanduser().resolve(),
            Path(args.output).expanduser().resolve(),
            args,
        )
        return

    if not args.input_dir or not args.output_dir:
        raise ValueError("Batch mode requires both --input-dir and --output-dir")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_dir}")

    input_files = sorted(input_dir.glob("*.hdf5")) + sorted(input_dir.glob("*.h5"))
    if not input_files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found in {input_dir}")

    print(f"Found {len(input_files)} HDF5 files")
    for number, input_path in enumerate(input_files, start=1):
        output_path = output_dir / f"{input_path.stem}_cam_high.mp4"
        print(f"\n[{number}/{len(input_files)}]")
        convert_file(input_path, output_path, args)


if __name__ == "__main__":
    main()
