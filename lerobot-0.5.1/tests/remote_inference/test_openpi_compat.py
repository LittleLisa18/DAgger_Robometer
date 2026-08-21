from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.remote_inference.openpi_compat import (
    SmolVLAPiperAdapter,
    _load_smolvla_config,
    pack_message,
    unpack_message,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_message_codec_matches_numpy_shapes_and_dtypes() -> None:
    value = {
        "state": np.arange(14, dtype=np.float32),
        "image": np.arange(18, dtype=np.uint8).reshape(3, 2, 3),
        "scalar": np.int64(7),
    }
    decoded = unpack_message(pack_message(value))
    np.testing.assert_array_equal(decoded["state"], value["state"])
    np.testing.assert_array_equal(decoded["image"], value["image"])
    assert decoded["scalar"] == np.int64(7)
    assert decoded["state"].dtype == np.float32


def test_load_config_accepts_saved_smolvla_type_field(tmp_path) -> None:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    SmolVLAConfig().save_pretrained(tmp_path)

    config = _load_smolvla_config(str(tmp_path), local_files_only=True)

    assert isinstance(config, SmolVLAConfig)


class _FakePolicy:
    def predict_action_chunk(self, observation):
        assert observation[OBS_STATE].shape == (1, 14)
        assert observation[f"{OBS_IMAGES}.cam_high"].shape == (1, 3, 4, 5)
        return torch.arange(3 * 14, dtype=torch.float32).reshape(1, 3, 14)


def _feature(shape: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


def _adapter() -> SmolVLAPiperAdapter:
    adapter = SmolVLAPiperAdapter.__new__(SmolVLAPiperAdapter)
    adapter.checkpoint = "test"
    adapter.device = "cpu"
    from lerobot.remote_inference.openpi_compat import PiperObservationKeys

    adapter.keys = PiperObservationKeys()
    image_features = {
        f"{OBS_IMAGES}.cam_high": _feature((3, 4, 5)),
        f"{OBS_IMAGES}.cam_left_wrist": _feature((3, 4, 5)),
        f"{OBS_IMAGES}.cam_right_wrist": _feature((3, 4, 5)),
    }
    adapter.config = SimpleNamespace(
        image_features=image_features,
        robot_state_feature=_feature((14,)),
        action_feature=_feature((14,)),
        output_features={ACTION: _feature((14,))},
    )
    adapter.actions_per_chunk = 2
    adapter.policy = _FakePolicy()

    def preprocess(observation):
        return {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
            for key, value in observation.items()
        }

    adapter.preprocessor = preprocess
    adapter.postprocessor = lambda action: action
    return adapter


def test_adapter_maps_piper_payload_and_returns_action_chunk() -> None:
    adapter = _adapter()
    image = np.zeros((3, 4, 5), dtype=np.uint8)
    payload = {
        "state": np.arange(14, dtype=np.float32),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": "pick up the pen",
    }
    result = adapter.infer(payload)
    assert result["actions"].shape == (2, 14)
    assert result["actions"].dtype == np.float32


@pytest.mark.parametrize("shape", [(13,), (14, 1)])
def test_adapter_rejects_invalid_piper_state(shape: tuple[int, ...]) -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="shape"):
        adapter.prepare_observation({"state": np.zeros(shape), "images": {}})
