"""Shared storage and configuration for the independent student evaluation."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .core import RunStore, atomic_json


@dataclasses.dataclass
class EvaluationConfig:
    output: str
    suite: str
    task_ids: list
    episodes: int
    initial_state_start: int
    seed: int
    control_freq: int
    max_steps: int
    settling_steps: int
    actions_per_chunk: int
    max_frames: int
    tail_offsets: list
    success_threshold: float
    timeout: float
    retries: int
    startup_timeout: float
    test_only: bool
    gpu: str
    student_checkpoint: str
    student_port: int
    score_port: int
    score_models: dict
    student_python: str
    robometer_python: str
    libero_root: str
    hf_home: str

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        fields = {f.name for f in dataclasses.fields(cls)}
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError("Evaluation JSON must specify exactly all config fields")
        c = cls(**values)
        for name in (
            "episodes",
            "control_freq",
            "max_steps",
            "actions_per_chunk",
            "max_frames",
            "student_port",
            "score_port",
        ):
            if type(getattr(c, name)) is not int or getattr(c, name) <= 0:
                raise ValueError("Expected positive integer: " + name)
        for name in ("initial_state_start", "seed", "settling_steps", "retries"):
            if type(getattr(c, name)) is not int or getattr(c, name) < 0:
                raise ValueError("Expected nonnegative integer: " + name)
        if (
            c.max_frames < 2
            or c.student_port == c.score_port
            or max(c.student_port, c.score_port) > 65535
        ):
            raise ValueError("Invalid frame count or ports")
        if (
            c.suite
            not in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
            or not c.task_ids
            or len(set(c.task_ids)) != len(c.task_ids)
            or any(type(t) is not int or not 0 <= t < 10 for t in c.task_ids)
        ):
            raise ValueError(
                "Specify a supported LIBERO suite and unique task IDs in [0,9]"
            )
        if (
            not c.tail_offsets
            or 0 not in c.tail_offsets
            or any(type(t) is not int or t < 0 for t in c.tail_offsets)
        ):
            raise ValueError("tail_offsets must contain 0 and nonnegative integers")
        if set(c.score_models) != {"original", "finetuned"} or any(
            not isinstance(p, str) for p in c.score_models.values()
        ):
            raise ValueError("score_models must specify original and finetuned paths")
        if (
            type(c.test_only) is not bool
            or not isinstance(c.gpu, str)
            or not c.gpu.isdigit()
        ):
            raise ValueError("Invalid test_only or single GPU index")
        if (
            not np.isfinite([c.success_threshold, c.timeout, c.startup_timeout]).all()
            or not 0 <= c.success_threshold <= 1
            or min(c.timeout, c.startup_timeout) <= 0
        ):
            raise ValueError("Invalid thresholds or timeouts")
        c.output = str(Path(c.output).expanduser().resolve())
        return c


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def model_identity(directory):
    """Hash actual checkpoint files, not just a mutable path or model name."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".cache" not in p.parts
        and p.suffix in (".safetensors", ".json", ".yaml")
        and p.name
        not in ("trainer_state.json", "dataset_random_state.json", "metrics.json")
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("No model weights in " + str(root))
    hashes = {str(p.relative_to(root)): file_hash(p) for p in files}
    return {"path": str(root), "sha256": digest(hashes), "files": hashes}


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class EvaluationStore:
    """Never overwrite interrupted attempts; a pointer commits normal episodes."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)

    def completed(self, episode_id):
        return (self.root / "episodes" / episode_id / "completed.json").is_file()

    def begin(self, episode_id, metadata):
        directory = self.root / "episodes" / episode_id
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("attempt_*"):
            pending = old / "pending.json"
            if pending.exists() and not (old / "metadata.json").exists():
                recovered = json.loads(pending.read_text())
                recovered.update(
                    end_reason="interrupted",
                    env_success=None,
                    error="Process exited before attempt commit",
                )
                atomic_json(old / "metadata.json", recovered)
                pending.rename(old / "recovered_pending.json")
        numbers = [
            int(p.name.split("_")[-1])
            for p in directory.glob("attempt_*")
            if p.is_dir()
        ]
        attempt = "attempt_%04d" % (max(numbers, default=0) + 1)
        store = object.__new__(RunStore)
        store.root = directory
        store.checkpoint(attempt, None, metadata)
        return store, attempt

    def finish(self, episode_id, store, attempt, records, metadata, terminal):
        directory = store.root / attempt
        if terminal is not None:
            atomic_npz(
                directory / "terminal.npz", **terminal, step=np.int64(len(records))
            )
        store.save(attempt, records, metadata)
        if metadata["end_reason"] in ("success", "step_limit"):
            if (
                not records
                or terminal is None
                or type(metadata["env_success"]) is not bool
            ):
                raise ValueError(
                    "Normal episode must have actions, terminal observation and label"
                )
            hashes = {
                name: file_hash(directory / name)
                for name in ("trajectory.npz", "terminal.npz", "metadata.json")
            }
            atomic_json(
                store.root / "completed.json",
                {"attempt": attempt, "files": hashes, "identity": digest(hashes)},
            )

    def episodes(self, verify=False, exclude=()):
        for pointer in sorted((self.root / "episodes").glob("*/completed.json")):
            if pointer.parent.name in exclude:
                continue
            entry = json.loads(pointer.read_text())
            directory = pointer.parent / entry["attempt"]
            if verify and any(
                file_hash(directory / name) != value
                for name, value in entry["files"].items()
            ):
                raise ValueError("Episode changed after commit: " + str(directory))
            meta = json.loads((directory / "metadata.json").read_text())
            if verify:
                steps = meta["steps"]
                if (
                    meta["end_reason"] not in ("success", "step_limit")
                    or type(meta["env_success"]) is not bool
                ):
                    raise ValueError("Invalid committed episode outcome")
                with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
                    for key, width in (
                        ("action", 7),
                        ("policy_action", 7),
                        ("state", 8),
                    ):
                        value = data[key]
                        if (
                            value.shape != (steps, width)
                            or not np.isfinite(value).all()
                        ):
                            raise ValueError("Invalid trajectory array: " + key)
                    if (
                        not np.array_equal(data["step"], np.arange(steps))
                        or data["collect"].shape != (steps,)
                        or not (data["collect"] == "rollout").all()
                    ):
                        raise ValueError("Invalid rollout frame alignment")
                    if not np.array_equal(
                        data["action"], np.clip(data["policy_action"], -1, 1)
                    ):
                        raise ValueError("Executed action does not match clipping")
                with np.load(directory / "terminal.npz", allow_pickle=False) as data:
                    if (
                        int(data["step"]) != steps
                        or data["state"].shape != (8,)
                        or not np.isfinite(data["state"]).all()
                    ):
                        raise ValueError("Invalid terminal state")
                    if any(
                        data[key].shape != (256, 256, 3) or data[key].dtype != np.uint8
                        for key in ("image", "image2")
                    ):
                        raise ValueError("Invalid terminal cameras")
            yield directory, meta, entry["identity"]


def score_protocol(config):
    return {
        "version": 1,
        "camera": "image",
        "rotation": "already_rotated_once",
        "sampling": "uniform_prefix_including_terminal",
        "max_frames": config.max_frames,
        "tail_offsets": config.tail_offsets,
        "success_threshold": config.success_threshold,
        "prediction": "success_probs[-1]",
        "timeout": config.timeout,
        "retries": config.retries,
    }


def prefix_inputs(directory, steps, config):
    """Yield T+1-observation prefixes without loading wrist images for scoring."""
    with np.load(Path(directory) / "trajectory.npz", allow_pickle=False) as data:
        images = data["image"]
        if len(images) != steps or not np.array_equal(data["step"], np.arange(steps)):
            raise ValueError("Trajectory step alignment is invalid")
    with np.load(Path(directory) / "terminal.npz", allow_pickle=False) as data:
        if int(data["step"]) != steps:
            raise ValueError("Terminal observation step mismatch")
        terminal = data["image"]
    for end in sorted({max(0, steps - offset) for offset in config.tail_offsets}):
        indices = np.linspace(0, end, min(end + 1, config.max_frames), dtype=int)
        frames = np.stack([terminal if i == steps else images[i] for i in indices])
        yield end, indices.tolist(), frames
