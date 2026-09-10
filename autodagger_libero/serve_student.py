"""SmolVLA LIBERO adapter, reusing LeRobot's OpenPI-compatible transport."""

import argparse
import numpy as np
import torch
from lerobot.remote_inference.openpi_compat import (
    SmolVLAPiperAdapter,
    OpenPICompatibleWebSocketServer,
)
from .core import config_hash


class LiberoStudentServer(OpenPICompatibleWebSocketServer):
    async def run(self):
        from websockets.asyncio.server import serve

        async with serve(
            self._handler, self.host, self.port, compression=None, max_size=None
        ) as server:
            # Announce readiness only after the socket has successfully bound.
            port = server.sockets[0].getsockname()[1]
            print(
                f"[Student READY] ws://{self.host}:{port} — "
                "模型已加载，服务已开始监听，等待采集器请求。Ctrl+C 停止服务。",
                flush=True,
            )
            await server.serve_forever()


class SmolVLALiberoAdapter(SmolVLAPiperAdapter):
    def __init__(
        self, checkpoint, *, device="cuda", actions_per_chunk=1, local_files_only=True
    ):
        from lerobot.remote_inference.openpi_compat import (
            _load_smolvla_config,
            _resolve_device,
        )
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.checkpoint, self.device = checkpoint, _resolve_device(device)
        config = _load_smolvla_config(checkpoint, local_files_only=local_files_only)
        config.device = self.device
        # The full fine-tuned checkpoint contains VLM weights. Construct its
        # architecture without downloading/loading a second base-weight copy.
        config.load_vlm_weights = False
        self.config = config
        self.actions_per_chunk = actions_per_chunk
        if not 1 <= actions_per_chunk <= config.chunk_size:
            raise ValueError("actions_per_chunk outside checkpoint horizon")
        self._validate_checkpoint_features()
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint, config=config, local_files_only=local_files_only
        )
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def _validate_checkpoint_features(self):
        if self.config.n_obs_steps != 1:
            raise ValueError("LIBERO adapter requires n_obs_steps=1")
        expected = {"observation.images.image", "observation.images.image2"}
        if set(self.config.image_features) != expected:
            raise ValueError(f"LIBERO requires camera features {expected}")
        if self.config.robot_state_feature.shape != (
            8,
        ) or self.config.action_feature.shape != (7,):
            raise ValueError("LIBERO requires 8D state and 7D action")

    def prepare_observation(self, payload):
        if not isinstance(payload["prompt"], str):
            raise ValueError("Task prompt must be a string")
        state = np.asarray(payload["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite LIBERO state (8,)")
        result = {
            "observation.state": torch.from_numpy(state.copy()),
            "task": payload["prompt"],
        }
        for source, target in (
            ("observation/image", "image"),
            ("observation/wrist_image", "image2"),
        ):
            image = np.asarray(payload[source])
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError("LIBERO student requires uint8 HWC 256x256 RGB images")
            result["observation.images." + target] = self._prepare_image(image, source)
        return result

    def infer(self, payload):
        # n_obs_steps=1: each request is independent; no Piper action queue.
        self.policy.reset()
        observation = self.preprocessor(self.prepare_observation(payload))
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        processed = [
            self.postprocessor(chunk[:, i, :]) for i in range(self.actions_per_chunk)
        ]
        # NumPy cannot represent bfloat16; cast after checkpoint postprocessing.
        actions = torch.stack(processed, dim=1).squeeze(0).float().cpu().numpy()
        if (
            actions.shape != (self.actions_per_chunk, 7)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid postprocessed actions: {actions.shape}")
        return {"actions": actions}

    @property
    def metadata(self):
        return {
            "policy_type": "smolvla_libero",
            "checkpoint": self.checkpoint,
            "config_sha256": config_hash(self.checkpoint + "/config.json"),
            "action_dim": 7,
            "state_dim": 8,
            "actions_per_chunk": self.actions_per_chunk,
        }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", default="/home/ma-user/work/model/smolvla_libero")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--actions-per-chunk", type=int, default=1)
    args = p.parse_args()
    print(f"[Student] 正在加载检查点：{args.checkpoint}", flush=True)
    adapter = SmolVLALiberoAdapter(
        args.checkpoint,
        device=args.device,
        actions_per_chunk=args.actions_per_chunk,
        local_files_only=True,
    )
    try:
        LiberoStudentServer(adapter, host=args.host, port=args.port).serve_forever()
    except KeyboardInterrupt:
        print("\n[Student STOPPED] 服务已停止。", flush=True)


if __name__ == "__main__":
    main()
