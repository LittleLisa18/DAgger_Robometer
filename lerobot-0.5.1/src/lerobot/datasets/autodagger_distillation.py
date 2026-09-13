"""Read-only AutoDAgger supervision masks and dataset provenance."""

import hashlib
import json
import logging
from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.compute_stats import aggregate_stats


class AutoDAggerDistillationDataset(Dataset):
    def __init__(self, dataset, chunk_size: int):
        if not 1 <= chunk_size <= 10:
            raise ValueError("Student chunk_size must be between 1 and 10")
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.meta = dataset.meta
        self.root = Path(dataset.root)
        self.episodes = dataset.episodes
        self.num_frames = dataset.num_frames
        self.num_episodes = dataset.num_episodes
        features = self.meta.features
        expected = {
            "observation.images.image": (3, 256, 256),
            "observation.images.image2": (3, 256, 256),
            "observation.state": (8,),
            "action": (7,),
        }
        if self.meta.fps != 10:
            raise ValueError("AutoDAgger LIBERO data must use 10 FPS")
        for key, shape in expected.items():
            if tuple(features.get(key, {}).get("shape", ())) != shape:
                raise ValueError(f"Expected {key} shape {shape}")
        if "collect" not in features:
            raise ValueError("AutoDAgger dataset is missing collect labels")
        if dataset.delta_timestamps.get("action") != [i / 10 for i in range(chunk_size)]:
            raise ValueError("Action delta timestamps must match the student chunk at 10 FPS")

        # Read columns only: never decode every camera image to construct masks.
        columns = dataset.hf_dataset.select_columns(["index", "episode_index", "collect"]).with_format(None)[
            :
        ]
        self.indices = np.asarray(columns["index"], dtype=np.int64)
        self.episode_indices = np.asarray(columns["episode_index"], dtype=np.int64)
        labels = columns["collect"]
        if not labels or any(label not in ("rollout", "teacher") for label in labels):
            raise ValueError("collect must contain only rollout or teacher")
        self.teacher = np.asarray([label == "teacher" for label in labels])
        if len(set(self.indices.tolist())) != len(self.indices):
            raise ValueError("Duplicate absolute frame indices")

        audit_path = self.root / "meta/autodagger_episodes.json"
        audits = json.loads(audit_path.read_text())
        by_episode = {row["dataset_episode_index"]: row for row in audits}
        if len(by_episode) != len(audits):
            raise ValueError("Duplicate episode indices in AutoDAgger audit")
        for episode, count in Counter(self.episode_indices.tolist()).items():
            row = by_episode.get(episode, {})
            if not (
                row.get("accepted_for_distillation") is True
                and row.get("took_over") is True
                and row.get("robometer_success") is True
                and row.get("test_only") is False
                and row.get("end_reason") in ("environment_terminated", "step_limit")
                and row.get("steps") == count
            ):
                raise ValueError(f"Episode {episode} does not pass the AutoDAgger export audit")
            positions = np.flatnonzero(self.episode_indices == episode)
            if np.any(np.diff(positions) != 1) or np.any(np.diff(self.indices[positions]) != 1):
                raise ValueError(f"Episode {episode} must contain contiguous frames")
            bounds = self.meta.episodes[episode]
            if (
                bounds["dataset_from_index"] != self.indices[positions[0]]
                or bounds["dataset_to_index"] != self.indices[positions[-1]] + 1
            ):
                raise ValueError(f"Episode {episode} metadata bounds disagree with its data rows")
            if not self.teacher[positions].any():
                raise ValueError(f"Episode {episode} has no teacher frames despite its takeover audit")

        # Content hashes also catch in-place data/normalization changes on resume.
        digest = hashlib.sha256()
        paths = [
            self.root / f"meta/{name}"
            for name in (
                "info.json",
                "stats.json",
                "autodagger_episodes.json",
                "autodagger_run.json",
                "tasks.parquet",
            )
        ]
        paths += sorted((self.root / "data").rglob("*.parquet"))
        paths += sorted((self.root / "meta/episodes").rglob("*.parquet"))
        paths += sorted((self.root / "videos").rglob("*.mp4"))
        for path in paths:
            digest.update(path.relative_to(self.root).as_posix().encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(self.indices.tobytes())
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        future = index + np.arange(self.chunk_size)
        safe = np.minimum(future, len(self) - 1)
        valid = (
            self.teacher[index]
            & (future < len(self))
            & (self.episode_indices[safe] == self.episode_indices[index])
            & (self.indices[safe] == self.indices[index] + np.arange(self.chunk_size))
            & self.teacher[safe]
        )
        valid = torch.from_numpy(valid.copy()) & ~item["action_is_pad"]
        return {**item, "distill_bc_mask": valid}


def selected_normalization_stats(dataset):
    """Read stats for selected episodes, once per metadata file and without decoding video."""
    keys = ["action", "observation.state", *dataset.meta.camera_keys]
    if dataset.episodes is None:
        return {key: dataset.meta.stats[key] for key in keys}

    import pyarrow.parquet as pq

    selected = set(dataset.episodes)
    found = set()
    stats = []
    for path in sorted((Path(dataset.root) / "meta/episodes").rglob("*.parquet")):
        names = pq.read_schema(path).names
        columns = ["episode_index"] + [
            name for name in names if any(name.startswith(f"stats/{key}/") for key in keys)
        ]
        for row in pq.read_table(path, columns=columns).to_pylist():
            episode = row["episode_index"]
            if episode not in selected:
                continue
            if episode in found:
                raise ValueError(f"Duplicate episode statistics for {episode}")
            found.add(episode)
            episode_stats = {}
            for key in keys:
                prefix = f"stats/{key}/"
                values = {
                    name[len(prefix) :]: np.atleast_1d(np.asarray(value, dtype=np.float64))
                    for name, value in row.items()
                    if name.startswith(prefix)
                    and value is not None
                    and not name[len(prefix) :].startswith("q")
                }
                if not {"min", "max", "mean", "std", "count"}.issubset(values):
                    raise ValueError(f"Missing normalization statistics for episode {episode}: {key}")
                if key in dataset.meta.camera_keys:
                    values = {
                        name: value if name == "count" else value.reshape(3, 1, 1)
                        for name, value in values.items()
                    }
                episode_stats[key] = values
            stats.append(episode_stats)
    if found != selected:
        raise ValueError(f"Missing statistics for selected episodes: {sorted(selected - found)}")
    return aggregate_stats(stats)


class MultiAutoDAggerDistillationDataset(Dataset):
    """Concatenate audited sources, leaving each source's action-window lookup independent."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("At least one AutoDAgger dataset is required")
        self.datasets = datasets
        first = datasets[0]
        for source in datasets[1:]:
            if source.meta.features != first.meta.features or source.meta.fps != first.meta.fps:
                raise ValueError("AutoDAgger sources must have matching feature schemas and FPS")
            if source.chunk_size != first.chunk_size:
                raise ValueError("AutoDAgger sources must use the same student chunk_size")
        self.cumulative_sizes = np.cumsum([len(source) for source in datasets]).tolist()
        self.num_frames = self.cumulative_sizes[-1]
        self.num_episodes = sum(source.num_episodes for source in datasets)
        self.episodes = None
        self.episode_maps = []
        bounds = []
        frame_offset = 0
        for source in datasets:
            mapping = {}
            for episode in dict.fromkeys(source.episode_indices.tolist()):
                positions = np.flatnonzero(source.episode_indices == episode)
                mapping[episode] = len(bounds)
                bounds.append(
                    {
                        "episode_index": len(bounds),
                        "dataset_from_index": frame_offset + int(positions[0]),
                        "dataset_to_index": frame_offset + int(positions[-1]) + 1,
                    }
                )
            self.episode_maps.append(mapping)
            frame_offset += len(source)

        import datasets as hf_datasets

        self.meta = SimpleNamespace(
            features=first.meta.features,
            fps=first.meta.fps,
            camera_keys=first.meta.camera_keys,
            stats=aggregate_stats([selected_normalization_stats(source.dataset) for source in datasets]),
            episodes=hf_datasets.Dataset.from_list(bounds),
            total_frames=self.num_frames,
            total_episodes=self.num_episodes,
        )
        # The order affects sampling and global indices. Paths may change when relocating identical data.
        provenance = {
            "format": "autodagger-multi-v1",
            "sources": [
                {"repo_id": source.dataset.repo_id, "fingerprint": source.fingerprint} for source in datasets
            ],
        }
        self.fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect_right(self.cumulative_sizes, index)
        offset = self.cumulative_sizes[source_index - 1] if source_index else 0
        source = self.datasets[source_index]
        local_index = index - offset
        item = source[local_index]
        episode = int(source.episode_indices[local_index])
        return {
            **item,
            "dataset_index": torch.tensor(source_index),
            "source_index": torch.tensor(int(source.indices[local_index])),
            "source_episode_index": torch.tensor(episode),
            "index": torch.tensor(index),
            "episode_index": torch.tensor(self.episode_maps[source_index][episode]),
        }


def make_distillation_dataset(cfg):
    from lerobot.datasets.factory import IMAGENET_STATS, make_dataset

    sources = cfg.dataset.sources
    configs = (
        [
            replace(
                cfg.dataset,
                sources=None,
                repo_id=source.repo_id,
                root=source.root,
                episodes=source.episodes,
                revision=source.revision,
                use_imagenet_stats=False,
            )
            for source in sources
        ]
        if sources is not None
        else [cfg.dataset]
    )
    datasets = []
    for index, dataset_config in enumerate(configs):
        root = Path(dataset_config.root).expanduser()
        if not (root / "meta/info.json").is_file():
            raise ValueError(f"Dataset source {index} is not a local LeRobot export: {root}")
        source_cfg = replace(cfg, dataset=replace(dataset_config, root=str(root)))
        source = AutoDAggerDistillationDataset(make_dataset(source_cfg), cfg.policy.chunk_size)
        logging.info(
            "AutoDAgger source %d: %s (%s), %d frames, %d episodes",
            index,
            dataset_config.repo_id,
            root,
            source.num_frames,
            source.num_episodes,
        )
        datasets.append(source)
    if sources is None:
        # Retain the single-dataset fingerprint and normalization for existing checkpoints.
        return datasets[0]
    dataset = MultiAutoDAggerDistillationDataset(datasets)
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for name, value in IMAGENET_STATS.items():
                dataset.meta.stats[key][name] = np.asarray(value, dtype=np.float32)
    return dataset
