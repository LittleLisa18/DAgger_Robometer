"""Testable control rules and crash-safe episode storage (no ML imports)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np


@dataclasses.dataclass
class Config:
    student_url: str = "ws://127.0.0.1:8100"
    teacher_url: str = "ws://127.0.0.1:8101"
    robometer_url: str = "http://127.0.0.1:8102"
    suite: str = "libero_spatial"
    task_ids: list[int] | None = None
    episodes: int = 50
    initial_state_start: int = 0
    seed: int = 7
    output: str = "runs/autodagger_spatial"
    student_replan: int = 1
    teacher_replan: int = 5
    monitor_every: int = 10
    max_frames: int = 16
    stall_steps: int = 50
    progress_epsilon: float = 0.05
    regression_enabled: bool = True
    regression_threshold: float = 0.1
    regression_checks: int = 2
    success_threshold: float = 0.5
    timeout: float = 60
    retries: int = 1
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8088
    force_teacher_step: int | None = None

    def validate(self):
        if not isinstance(self.regression_enabled, bool):
            raise ValueError("regression_enabled must be a JSON boolean")
        for key in (
            "timeout",
            "progress_epsilon",
            "regression_threshold",
            "success_threshold",
        ):
            if not np.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        for key in (
            "episodes",
            "student_replan",
            "teacher_replan",
            "monitor_every",
            "max_frames",
            "stall_steps",
            "regression_checks",
            "timeout",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.max_frames < 2 or self.retries < 0 or self.initial_state_start < 0:
            raise ValueError(
                "max_frames >= 2, retries and initial_state_start >= 0 required"
            )
        if not 0 <= self.success_threshold <= 1:
            raise ValueError("success_threshold must be in [0,1]")
        if self.progress_epsilon < 0 or self.regression_threshold <= 0:
            raise ValueError("invalid progress thresholds")
        if self.force_teacher_step is not None and self.force_teacher_step < 0:
            raise ValueError("force_teacher_step must be nonnegative")


class Gate:
    def __init__(self, config):
        self.config = config
        self.best = None
        self.improvement_baseline = None
        self.last_improved = 0
        self.regressions = 0

    def update(self, step, progress, success):
        if not np.isfinite([progress, success]).all() or not 0 <= success <= 1:
            raise ValueError("Invalid Robometer scores")
        c = self.config
        # Small improvements accumulate toward the stall threshold, while every
        # new peak counts when measuring a later regression.
        if (
            self.improvement_baseline is None
            or progress > self.improvement_baseline + c.progress_epsilon
        ):
            self.improvement_baseline, self.last_improved = progress, step
        self.best = progress if self.best is None else max(self.best, progress)
        if success > c.success_threshold:
            self.last_improved, self.regressions = step, 0
            return None
        self.regressions = (
            self.regressions + 1
            if c.regression_enabled and self.best - progress >= c.regression_threshold
            else 0
        )
        if c.regression_enabled and self.regressions >= c.regression_checks:
            return "progress_regression"
        if step - self.last_improved >= c.stall_steps:
            return "progress_stalled"
        return None


def sample_prefix(frames, max_frames):
    indices = np.linspace(0, len(frames) - 1, min(len(frames), max_frames), dtype=int)
    return np.stack([frames[i] for i in indices]), indices.tolist()


def validate_actions(actions, count):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < count:
        raise ValueError(f"Expected at least ({count},7) actions, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Actions must be finite")
    return np.clip(actions[:count], -1, 1)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunStore:
    def __init__(self, root, config, models):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": dataclasses.asdict(config),
            "models": models,
            "schema_version": 1,
        }
        # Resume only when the recorded configuration and model metadata match.
        path = self.root / "run.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(
                "Output belongs to a different config/model run; choose a new output directory"
            )
        atomic_json(path, manifest)
        for pending in self.root.glob("*/pending.json"):
            if not (pending.parent / "metadata.json").exists():
                meta = json.loads(pending.read_text())
                records = []
                for frame in sorted(pending.parent.glob("frame_*.npz")):
                    # Only renamed, committed frames belong to the journal.
                    # A killed writer can leave a truncated *.tmp.npz beside them.
                    if re.fullmatch(r"frame_[0-9]{6}\.npz", frame.name) is None:
                        continue
                    with np.load(frame, allow_pickle=False) as data:
                        records.append({k: data[k] for k in data.files})
                meta.update(
                    end_reason="interrupted",
                    error="Recovered incomplete episode after process exit",
                    accepted_for_distillation=False,
                    steps=len(records),
                )
                self.save(meta["episode_id"], records, meta)

    def completed(self, episode_id):
        return (self.root / episode_id / "metadata.json").exists()

    def begin(self, episode_id):
        directory = self.root / episode_id
        directory.mkdir(exist_ok=True)
        return directory

    def save(self, episode_id, records, metadata):
        directory = self.begin(episode_id)
        if records:
            tmp = directory / "trajectory.tmp.npz"
            np.savez_compressed(
                tmp, **{k: np.asarray([r[k] for r in records]) for k in records[0]}
            )
            os.replace(tmp, directory / "trajectory.npz")
        # Metadata is the commit marker: publish it only after the trajectory.
        atomic_json(directory / "metadata.json", metadata)
        for frame in directory.glob("frame_*.npz"):
            frame.unlink()
        (directory / "pending.json").unlink(missing_ok=True)

    def checkpoint(self, episode_id, record, metadata):
        directory = self.begin(episode_id)
        if record is not None:
            name = f"frame_{int(record['step']):06d}"
            tmp = directory / (name + ".tmp.npz")
            np.savez_compressed(tmp, **record)
            os.replace(tmp, directory / (name + ".npz"))
        atomic_json(directory / "pending.json", metadata)

    def summaries(self):
        summaries = []
        for p in sorted(self.root.glob("*/metadata.json")):
            item = json.loads(p.read_text())
            item.pop("samples", None)
            summaries.append(item)
        return summaries


def config_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
