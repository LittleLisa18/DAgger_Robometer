"""OpenPI-compatible WebSocket serving for SmolVLA.

The Piper runtime already talks to OpenPI's ``WebsocketClientPolicy``.  This
module implements the same wire format so that the unchanged robot client can
use a LeRobot SmolVLA checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import torch
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

LOGGER = logging.getLogger(__name__)


def _load_smolvla_config(checkpoint: str, *, local_files_only: bool) -> Any:
    """Load a policy config through LeRobot's type-aware config registry."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=local_files_only)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError(
            f"Expected a SmolVLA checkpoint, got policy type {config.type!r} from {checkpoint!r}"
        )
    return config


def _pack_numpy(value: Any) -> Any:
    """Encode NumPy values exactly as ``openpi_client.msgpack_numpy`` does."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot encode value of type {type(value).__name__}")


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=tuple(value[b"shape"]),
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_numpy)


def unpack_message(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_unpack_numpy)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class PiperObservationKeys:
    """Mapping from Piper's OpenPI payload to the AgileX training features."""

    cam_high: str = f"{OBS_IMAGES}.cam_high"
    cam_left_wrist: str = f"{OBS_IMAGES}.cam_left_wrist"
    cam_right_wrist: str = f"{OBS_IMAGES}.cam_right_wrist"

    @property
    def by_payload_key(self) -> dict[str, str]:
        return {
            "cam_high": self.cam_high,
            "cam_left_wrist": self.cam_left_wrist,
            "cam_right_wrist": self.cam_right_wrist,
        }


class SmolVLAPiperAdapter:
    """Turn Piper/OpenPI observations into postprocessed SmolVLA action chunks."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "auto",
        actions_per_chunk: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        # Keep model imports lazy so protocol tooling can run without loading the
        # complete transformers stack.
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.checkpoint = checkpoint
        self.device = _resolve_device(device)
        self.keys = PiperObservationKeys()

        config = _load_smolvla_config(checkpoint, local_files_only=local_files_only)
        config.device = self.device
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=local_files_only,
        )
        self.policy.to(self.device)
        self.policy.eval()

        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.config = config
        self.actions_per_chunk = actions_per_chunk or config.n_action_steps
        if not 1 <= self.actions_per_chunk <= config.chunk_size:
            raise ValueError(
                f"actions_per_chunk must be between 1 and {config.chunk_size}, got {self.actions_per_chunk}"
            )
        self._validate_checkpoint_features()

    def _validate_checkpoint_features(self) -> None:
        expected_images = set(self.config.image_features)
        provided_images = set(self.keys.by_payload_key.values())
        missing = expected_images - provided_images
        if missing:
            raise ValueError(
                "The checkpoint expects camera features that the Piper payload does not provide: "
                f"{sorted(missing)}. Expected compatible AgileX features are {sorted(provided_images)}."
            )

        state_feature = self.config.robot_state_feature
        action_feature = self.config.action_feature
        if state_feature is None or action_feature is None:
            raise ValueError("The checkpoint must define observation.state and action features")
        if state_feature.shape != (14,) or action_feature.shape != (14,):
            raise ValueError(
                "Piper requires 14-dimensional state and action features, got "
                f"state={state_feature.shape}, action={action_feature.shape}"
            )

    @staticmethod
    def _prepare_image(image: Any, name: str) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Image {name!r} must have 3 dimensions, got {array.shape}")
        if array.shape[0] == 3:
            chw = array
        elif array.shape[-1] == 3:
            chw = array.transpose(2, 0, 1)
        else:
            raise ValueError(f"Image {name!r} must be CHW or HWC RGB, got {array.shape}")
        return torch.as_tensor(np.ascontiguousarray(chw), dtype=torch.float32).div_(255.0)

    def prepare_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(payload["state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Piper state must have shape (14,), got {state.shape}")

        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("Piper observation must contain an 'images' mapping")

        observation: dict[str, Any] = {OBS_STATE: torch.from_numpy(state.copy())}
        for payload_key, feature_key in self.keys.by_payload_key.items():
            if feature_key not in self.config.image_features:
                continue
            if payload_key not in images:
                raise ValueError(f"Piper observation is missing image {payload_key!r}")
            observation[feature_key] = self._prepare_image(images[payload_key], payload_key)

        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError(f"Piper prompt must be a string, got {type(prompt).__name__}")
        observation["task"] = prompt
        observation["robot_type"] = "agilex"
        return observation

    def infer(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        observation = self.preprocessor(self.prepare_observation(payload))
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        if chunk.ndim != 3 or chunk.shape[0] != 1:
            raise RuntimeError(f"Expected SmolVLA action chunk shape (1,T,D), got {tuple(chunk.shape)}")

        chunk = chunk[:, : self.actions_per_chunk]
        processed = [self.postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])]
        actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"Expected postprocessed Piper actions with shape (T,14), got {actions.shape}")
        return {"actions": actions}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy_type": "smolvla",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "actions_per_chunk": self.actions_per_chunk,
            "action_dim": self.config.output_features[ACTION].shape[0],
            "image_features": sorted(self.config.image_features),
            "protocol": "openpi-websocket-v1",
        }


class OpenPICompatibleWebSocketServer:
    """Serve one or more unchanged OpenPI clients with serialized inference."""

    def __init__(self, adapter: SmolVLAPiperAdapter, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.adapter = adapter
        self.host = host
        self.port = port
        self._inference_lock = asyncio.Lock()

    async def _handler(self, websocket: ServerConnection) -> None:
        LOGGER.info("Client %s connected", websocket.remote_address)
        await websocket.send(pack_message(self.adapter.metadata))
        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    raise TypeError("The OpenPI-compatible protocol accepts binary MessagePack frames only")
                observation = unpack_message(message)
                if not isinstance(observation, dict):
                    raise TypeError("Decoded observation must be a mapping")
                async with self._inference_lock:
                    started = asyncio.get_running_loop().time()
                    result = await asyncio.to_thread(self.adapter.infer, observation)
                    result["server_timing"] = {
                        "infer_ms": (asyncio.get_running_loop().time() - started) * 1000
                    }
                await websocket.send(pack_message(result))
        except ConnectionClosed:
            pass
        except Exception:
            LOGGER.exception("Inference request failed")
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011, reason="SmolVLA inference failed")
        finally:
            LOGGER.info("Client %s disconnected", websocket.remote_address)

    async def run(self) -> None:
        LOGGER.info("Serving SmolVLA OpenPI-compatible endpoint on ws://%s:%d", self.host, self.port)
        async with serve(self._handler, self.host, self.port, compression=None, max_size=None) as server:
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())
