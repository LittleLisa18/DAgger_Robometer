"""CLI for serving a SmolVLA checkpoint to the existing Piper OpenPI client."""

from __future__ import annotations

import argparse
import logging

from lerobot.remote_inference.openpi_compat import (
    OpenPICompatibleWebSocketServer,
    SmolVLAPiperAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local SmolVLA checkpoint directory or Hub model ID")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind address")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket port used by Piper")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device (auto, cuda, cuda:0, mps, or cpu)",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help="Number of actions returned per request (defaults to checkpoint n_action_steps)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access Hugging Face Hub while loading the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    adapter = SmolVLAPiperAdapter(
        args.checkpoint,
        device=args.device,
        actions_per_chunk=args.actions_per_chunk,
        local_files_only=args.local_files_only,
    )
    OpenPICompatibleWebSocketServer(adapter, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
