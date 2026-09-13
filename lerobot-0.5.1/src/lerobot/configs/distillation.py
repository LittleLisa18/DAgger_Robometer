"""Configuration for final-action distillation through an OpenPI server."""

import math
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    teacher_url: str
    teacher_id: str
    timeout_s: float = 60.0
    retries: int = 1
    kd_weight: float = 1.0
    bc_weight: float = 1.0
    # Populated on first run and checked against the dataset when resuming.
    dataset_fingerprint: str | None = None

    def validate(self, cfg) -> None:
        if not self.teacher_url.startswith(("ws://", "wss://")) or not self.teacher_id.strip():
            raise ValueError("Distillation requires a WebSocket teacher_url and nonempty teacher_id")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0 or self.retries < 0:
            raise ValueError("Teacher timeout must be positive and retries nonnegative")
        if any(not math.isfinite(w) or w < 0 for w in (self.kd_weight, self.bc_weight)):
            raise ValueError("Distillation weights must be finite and nonnegative")
        if self.kd_weight + self.bc_weight <= 0:
            raise ValueError("At least one distillation loss weight must be positive")
        p = cfg.policy
        if p.type != "smolvla" or p.n_obs_steps != 1:
            raise ValueError("Online distillation requires SmolVLA with n_obs_steps=1")
        if not 1 <= p.chunk_size <= 10 or not 1 <= p.n_action_steps <= p.chunk_size:
            raise ValueError("Require 1 <= n_action_steps <= student chunk_size <= 10")
        if p.adapt_to_pi_aloha or p.use_delta_joint_actions_aloha or p.empty_cameras:
            raise ValueError("LIBERO distillation does not use Aloha transforms or empty cameras")
        if p.rtc_config is not None and p.rtc_config.enabled:
            raise ValueError("RTC is not supported during distillation")
        if cfg.use_rabc or cfg.peft is not None or p.use_peft or cfg.rename_map:
            raise ValueError("Distillation cannot be combined with RA-BC, PEFT, or observation renaming")
        if cfg.dataset.streaming or cfg.dataset.image_transforms.enable:
            raise ValueError("Distillation requires a non-streaming dataset without image augmentation")
        sources = getattr(cfg.dataset, "sources", None)
        if not sources and not cfg.dataset.root:
            raise ValueError("Specify dataset.root for the local AutoDAgger export and audit sidecars")
        if sources and any(mode in ("QUANTILES", "QUANTILE10") for mode in p.normalization_mapping.values()):
            raise ValueError(
                "Multi-dataset distillation supports MEAN_STD, MIN_MAX, or IDENTITY normalization"
            )
        if not p.pretrained_path:
            raise ValueError("Specify a pretrained SmolVLA checkpoint")
