"""Online teacher labels, flow-matching batch construction, and DDP-safe reduction."""

import time
from contextlib import suppress

import numpy as np
import torch
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from lerobot.remote_inference.openpi_compat import pack_message, unpack_message


class OpenPITeacherClient:
    """One synchronous connection per training rank; no OpenPI model dependency."""

    def __init__(self, config):
        self.config = config
        self.ws = None
        self.metadata = None
        self.retry_count = 0

    def close(self):
        if self.ws is not None:
            ws, self.ws = self.ws, None
            with suppress(Exception):
                ws.close()

    def infer(self, observation):
        for attempt in range(self.config.retries + 1):
            try:
                if self.ws is None:
                    self.ws = connect(
                        self.config.teacher_url,
                        compression=None,
                        max_size=16 * 1024 * 1024,
                        open_timeout=self.config.timeout_s,
                        close_timeout=2,
                    )
                    metadata = self.ws.recv(timeout=self.config.timeout_s)
                    if not isinstance(metadata, bytes):
                        raise ValueError(f"Invalid teacher handshake: {metadata}")
                    self.metadata = unpack_message(metadata)
                self.ws.send(pack_message(observation))
                response = self.ws.recv(timeout=self.config.timeout_s)
                if not isinstance(response, bytes):
                    raise ValueError(f"Teacher server error: {response}")
                response = unpack_message(response)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise ValueError(f"Expected finite teacher actions (10, 7), got {actions.shape}")
                return np.clip(actions, -1, 1)
            except (OSError, TimeoutError, ConnectionClosed):
                self.close()
                if attempt == self.config.retries:
                    raise
                self.retry_count += 1
            except Exception:
                self.close()
                raise


def teacher_observation(batch, index):
    """Convert unnormalized LeRobot tensors; exported camera orientation is retained."""
    payload = {}
    for source, target in (
        ("observation.images.image", "observation/image"),
        ("observation.images.image2", "observation/wrist_image"),
    ):
        image = batch[source][index].detach().cpu()
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.shape != (3, 256, 256) or not torch.isfinite(image).all():
            raise ValueError(f"Invalid raw image: {source}, {image.shape}")
        if image.dtype != torch.uint8:
            if image.min() < 0 or image.max() > 1:
                raise ValueError("Teacher images must be raw pixels, not normalized student inputs")
            image = (image * 255).round().to(torch.uint8)
        payload[target] = image.permute(1, 2, 0).contiguous().numpy()
    state = batch["observation.state"][index].detach().cpu().numpy().reshape(-1)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Teacher state must contain 8 finite values")
    task = batch["task"][index]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Teacher prompt must be nonempty")
    payload.update({"observation/state": state.astype(np.float32), "prompt": task})
    return payload


def prepare_distillation_batch(raw_batch, preprocessor, client, accelerator):
    """Return [KD; BC] rows in one forward, with masks kept outside the processors.

    All rank-local preparation failures are synchronized before entering DDP.
    """
    error = None
    result = None
    started = time.perf_counter()
    retries_before = client.retry_count
    try:
        batch_size, horizon, action_dim = raw_batch["action"].shape
        if not 1 <= horizon <= 10 or action_dim != 7:
            raise ValueError("Expected student action batch (B, H, 7), 1 <= H <= 10")
        targets = np.stack(
            [client.infer(teacher_observation(raw_batch, i))[:horizon] for i in range(batch_size)]
        )
        teacher_actions = torch.as_tensor(
            targets, device=raw_batch["action"].device, dtype=raw_batch["action"].dtype
        )
        # Only observations/task/actions go through processors. Masks are attached afterwards.
        doubled = {}
        for key, value in raw_batch.items():
            if key.startswith("observation."):
                doubled[key] = torch.cat((value, value), dim=0)
        doubled["task"] = list(raw_batch["task"]) * 2
        doubled["action"] = torch.cat((teacher_actions, raw_batch["action"]), dim=0)
        processed = preprocessor(doubled)
        bc_mask = raw_batch["distill_bc_mask"].to(accelerator.device, dtype=torch.bool)
        processed["distill_bc_mask"] = bc_mask
        result = processed
    except Exception as exc:
        error = exc
    failed = torch.tensor(int(error is not None), device=accelerator.device)
    failed = accelerator.reduce(failed, reduction="sum")
    if failed.item():
        client.close()
        raise RuntimeError(
            f"Teacher/batch preparation failed on {failed.item()} rank(s); local error: {error!r}"
        ) from error
    result["distill_request_metrics"] = {
        "teacher_batch_s": time.perf_counter() - started,
        "teacher_retries": client.retry_count - retries_before,
    }
    return result


def distillation_loss(errors, bc_mask, accelerator, config):
    """Global masked means with gradients scaled for DDP's rank averaging."""
    kd, bc = errors.float().chunk(2, dim=0)
    if bc_mask.shape != bc.shape[:2] or kd.shape[-1] != 7:
        raise ValueError("Distillation errors/masks have incompatible shapes")
    kd_sum = kd.sum()
    bc_sum = (bc * bc_mask.unsqueeze(-1)).sum()
    counts = torch.stack(
        (
            torch.tensor(kd.numel(), device=errors.device, dtype=torch.float32),
            bc_mask.sum().float() * bc.shape[-1],
        )
    )
    totals = accelerator.reduce(
        torch.stack((kd_sum.detach(), bc_sum.detach(), counts[0], counts[1])), reduction="sum"
    )
    denom = totals[2:].clamp_min(1)
    loss = accelerator.num_processes * (
        config.kd_weight * kd_sum / denom[0] + config.bc_weight * bc_sum / denom[1]
    )
    means = totals[:2] / denom
    return loss, {
        "kd_loss": means[0].item(),
        "bc_loss": means[1].item(),
        "loss": (config.kd_weight * means[0] + config.bc_weight * means[1]).item(),
        "kd_valid_steps": (totals[2] / 7).item(),
        "bc_valid_steps": (totals[3] / 7).item(),
    }
