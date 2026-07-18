import concurrent.futures
import logging
import pathlib
import time
from collections.abc import Sequence
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("num_steps", "infer_time_schedule", "alpha", "u0")
            )

            # Explicitly split and JIT the Streaming API to avoid io_callback bottlenecks
            if hasattr(model, "sample_actions_streaming_init"):
                self._sample_actions_streaming_init = nnx_utils.module_jit(
                    model.sample_actions_streaming_init, static_argnames=("num_steps", "alpha", "u0")
                )
                self._sample_actions_streaming_step = nnx_utils.module_jit(model.sample_actions_streaming_step)

            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        if "action_prefix" in inputs or "delay" in inputs:
            assert "action_prefix" in inputs and "delay" in inputs, "action_prefix and delay must be present"
            # action_prefix (delay, action_dim)
            assert (
                inputs["action_prefix"].shape[0] == inputs["delay"]
            ), f"{inputs['action_prefix'].shape[0]} != {inputs['delay']}"
            assert not self._is_pytorch_model, "FASTER is not supported for PyTorch models"
            prefix_mode = True
        else:
            prefix_mode = False

        if "debug" in inputs and inputs["debug"]:
            debug = True
        else:
            debug = False

        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        all_x_t = None
        all_v_t = None
        if self._is_pytorch_model:
            actions, prefix_attn_weights, suffix_attn_weights = self._sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            # Convert full list of per-layer attention weights to numpy (all layers)
            prefix_attn_weights = np.concatenate(
                [w.detach().float().cpu().numpy() for w in prefix_attn_weights], axis=0
            )
            suffix_attn_weights = np.concatenate(
                [w.detach().float().cpu().numpy() for w in suffix_attn_weights], axis=0
            )
        else:
            if prefix_mode:
                sample_kwargs["delay"] = inputs["delay"]
                sample_kwargs["action_prefix"] = inputs["action_prefix"]

            model_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            if isinstance(model_output, tuple):
                actions, all_x_t, all_v_t = model_output
            else:
                actions = model_output
            outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if self._is_pytorch_model:
            outputs["prefix_attn_weights"] = prefix_attn_weights
            outputs["suffix_attn_weights"] = suffix_attn_weights

        if debug:
            outputs["all_x_t"] = all_x_t if all_x_t is not None else None
            outputs["all_v_t"] = all_v_t if all_v_t is not None else None
            if "actions" in inputs:
                outputs["targets"] = inputs["actions"]
        return outputs

    @override
    def infer_streaming(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        on_actions_ready=None,
        early_stop_actions: int | None = None,
    ) -> dict:  # type: ignore[misc]
        """Streaming inference via Python-unrolled loop.

        Massively improves performance by breaking JAX Host-Device sync barriers
        and utilizing asynchronous execution without `io_callback`.
        """
        assert not self._is_pytorch_model, "Streaming inference is only supported for JAX models"
        assert (
            self._sample_actions_streaming_init is not None
        ), "Model does not support streaming (no sample_actions_streaming_init method)"

        inputs = jax.tree.map(lambda x: x, obs)
        if "action_prefix" in inputs or "delay" in inputs:
            assert "action_prefix" in inputs and "delay" in inputs, "action_prefix and delay must be present"
            assert (
                inputs["action_prefix"].shape[0] == inputs["delay"]
            ), f"{inputs['action_prefix'].shape[0]} != {inputs['delay']}"
            prefix_mode = True
        else:
            prefix_mode = False

        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = {k: v for k, v in self._sample_kwargs.items() if k != "infer_time_schedule"}
        if noise is not None:
            noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        state_np = np.asarray(inputs["state"][0])

        if prefix_mode:
            sample_kwargs["delay"] = inputs["delay"]
            sample_kwargs["action_prefix"] = inputs["action_prefix"]

        num_steps = sample_kwargs.get("num_steps", 10)

        callback_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        start_time = time.monotonic()
        emitted_action_count = 0

        # Step 1: Pre-compute static data and kv_cache
        (
            x_t,
            already_output,
            t_starts,
            dt_schedule,
            is_ready_after_step,
            kv_cache,
            prefix_mask,
            prefix_action_mask,
            action_prefix_init,
            observation_preprocessed,
        ) = self._sample_actions_streaming_init(sample_rng, observation, **sample_kwargs)

        # Step 2: High-speed Asynchronous loop
        for i in range(num_steps):
            x_next, already_output, newly_ready = self._sample_actions_streaming_step(
                x_t,
                already_output,
                t_starts[i],
                dt_schedule[i],
                is_ready_after_step[i],
                kv_cache,
                prefix_mask,
                prefix_action_mask,
                action_prefix_init,
                observation_preprocessed,
            )

            newly_ready_np = np.asarray(newly_ready[0])
            newly_ready_count = int(np.count_nonzero(newly_ready_np))
            selected_count = newly_ready_count

            if early_stop_actions is not None:
                remaining = early_stop_actions - emitted_action_count
                selected_count = min(newly_ready_count, remaining)
                emitted_action_count += selected_count

            if on_actions_ready is not None and selected_count > 0:
                ready_actions_np = np.asarray(x_next[0][newly_ready_np])[:selected_count]

                def process_and_send(actions_data, state_data):
                    temp_output = {"state": state_data, "actions": actions_data}
                    temp_output = self._output_transform(temp_output)
                    on_actions_ready(temp_output["actions"])

                callback_executor.submit(process_and_send, ready_actions_np, state_np)

            x_t = x_next
            if early_stop_actions is not None and emitted_action_count >= early_stop_actions:
                break

        callback_executor.shutdown(wait=True)
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": x_t}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
