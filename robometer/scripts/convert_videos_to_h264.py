#!/usr/bin/env python3
"""Batch-convert videos to OpenCV-compatible H.264 MP4 files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/Users/liuyantong/Desktop/Research/AutoDagger/DAgger_Robometer/test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert videos to H.264/yuv420p without modifying the originals."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>_h264).",
    )
    parser.add_argument("--pattern", default="*.mp4", help="Input glob pattern.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality (lower is better).")
    parser.add_argument("--preset", default="medium", help="libx264 encoding preset.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing converted files."
    )
    return parser.parse_args()


def convert_video(
    ffmpeg: str,
    source: Path,
    destination: Path,
    crf: int,
    preset: str,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return "skipped"

    temporary = destination.with_name(f".{destination.stem}.converting.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "none",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temporary),
    ]

    try:
        subprocess.run(command, check=True)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "converted"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir.with_name(f"{input_dir.name}_h264")
    )

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not 0 <= args.crf <= 51:
        raise SystemExit("--crf must be between 0 and 51")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "ffmpeg was not found. Install it first, for example: "
            "conda install -c conda-forge ffmpeg libdav1d"
        )

    videos = sorted(path for path in input_dir.glob(args.pattern) if path.is_file())
    if not videos:
        raise SystemExit(f"No files matching {args.pattern!r} in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = skipped = failed = 0
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    for index, source in enumerate(videos, start=1):
        destination = output_dir / f"{source.stem}.mp4"
        print(f"[{index}/{len(videos)}] {source.name}", end=" ... ", flush=True)
        try:
            status = convert_video(
                ffmpeg,
                source,
                destination,
                args.crf,
                args.preset,
                args.overwrite,
            )
        except subprocess.CalledProcessError as error:
            failed += 1
            print(f"FAILED (ffmpeg exit code {error.returncode})")
            continue

        if status == "converted":
            converted += 1
        else:
            skipped += 1
        print(status)

    print(f"Done: converted={converted}, skipped={skipped}, failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
