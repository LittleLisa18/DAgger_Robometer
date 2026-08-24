from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


_VALID_COLLECT_LABELS = frozenset({"teleop", "teacher", "rollout", "dagger"})
_NON_ROLLOUT_COLLECT_LABELS = _VALID_COLLECT_LABELS - {"rollout"}
DISTILL_GT_MASK_KEY = "distill_gt_mask"


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        preserve_keys: Sequence[str] = (),
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._preserve_keys = tuple(preserve_keys)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        item = self._dataset[index]
        preserved = {key: item[key] for key in self._preserve_keys}
        transformed = self._transform(item)
        return {**transformed, **preserved}

    def __len__(self) -> int:
        return len(self._dataset)


class FullActionChunkLeRobotDataset(Dataset[T_co]):
    """Filters LeRobot samples by action-chunk validity and, optionally, current-frame collect label."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        action_horizon: int,
        *,
        pad_at_episode_end: bool = False,
        allowed_collect_labels: frozenset[str] | None = None,
    ):
        self._dataset = dataset
        self._indices = (
            list(range(len(dataset)))
            if pad_at_episode_end
            else _valid_full_action_chunk_indices(dataset, action_horizon)
        )
        self._gt_masks: list[bool] | None = None
        if allowed_collect_labels is not None:
            collect_labels = _read_and_validate_collect_labels(dataset)
            self._indices = [index for index in self._indices if collect_labels[index] in allowed_collect_labels]
            self._gt_masks = [collect_labels[index] != "rollout" for index in self._indices]

    def __getitem__(self, index: SupportsIndex) -> T_co:
        logical_index = index.__index__()
        item = self._dataset[self._indices[logical_index]]
        if self._gt_masks is None:
            return item
        return {**item, DISTILL_GT_MASK_KEY: np.asarray(self._gt_masks[logical_index], dtype=np.bool_)}

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def num_frames(self) -> int:
        return len(self)

    def __getattr__(self, name: str):
        try:
            dataset = object.__getattribute__(self, "_dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)


class _DatasetSize:
    def __init__(self, num_frames: int):
        self.num_frames = num_frames


class MultiLeRobotDatasetIndexFilter(Dataset[T_co]):
    """Applies per-dataset action-chunk and collect-label filtering on a MultiLeRobotDataset."""

    def __init__(
        self,
        dataset: lerobot_dataset.MultiLeRobotDataset,
        action_horizon: int,
        pad_at_episode_end: Sequence[bool],
        *,
        allowed_collect_labels: frozenset[str] | None = None,
    ):
        self._dataset = dataset
        self.repo_ids = dataset.repo_ids
        self._indices = []
        self._datasets = []
        self._gt_masks: list[bool] | None = [] if allowed_collect_labels is not None else None

        original_offset = 0
        for child_dataset, should_pad in zip(dataset._datasets, pad_at_episode_end, strict=True):  # noqa: SLF001
            if should_pad:
                local_indices = list(range(len(child_dataset)))
            else:
                local_indices = _valid_full_action_chunk_indices(child_dataset, action_horizon)
            if allowed_collect_labels is not None:
                collect_labels = _read_and_validate_collect_labels(child_dataset)
                local_indices = [index for index in local_indices if collect_labels[index] in allowed_collect_labels]
                self._gt_masks.extend(collect_labels[index] != "rollout" for index in local_indices)
            self._indices.extend(original_offset + index for index in local_indices)
            self._datasets.append(_DatasetSize(len(local_indices)))
            original_offset += len(child_dataset)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        logical_index = index.__index__()
        item = self._dataset[self._indices[logical_index]]
        if self._gt_masks is None:
            return item
        return {**item, DISTILL_GT_MASK_KEY: np.asarray(self._gt_masks[logical_index], dtype=np.bool_)}

    def __len__(self) -> int:
        return len(self._indices)

    def __getattr__(self, name: str):
        try:
            dataset = object.__getattribute__(self, "_dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)


def _as_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _read_and_validate_collect_labels(dataset: Dataset) -> list[str]:
    repo_id = getattr(dataset, "repo_id", "<unknown>")
    hf_dataset = getattr(dataset, "hf_dataset", None)
    features = getattr(hf_dataset, "features", {}) if hf_dataset is not None else {}
    if hf_dataset is None or "collect" not in features:
        raise ValueError(f"LeRobot dataset {repo_id!r} is missing required 'collect' feature for distillation.")

    try:
        raw_labels = hf_dataset["collect"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"LeRobot dataset {repo_id!r} is missing required 'collect' feature for distillation."
        ) from exc
    if len(raw_labels) != len(dataset):
        raise ValueError(
            f"LeRobot dataset {repo_id!r} has {len(raw_labels)} collect labels for {len(dataset)} frames."
        )

    labels = []
    invalid = set()
    for value in raw_labels:
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        label = value if isinstance(value, str) else None
        if label not in _VALID_COLLECT_LABELS:
            invalid.add(repr(value))
        else:
            labels.append(label)
    if invalid:
        raise ValueError(
            f"LeRobot dataset {repo_id!r} has invalid collect labels {sorted(invalid)}; "
            f"expected exactly {sorted(_VALID_COLLECT_LABELS)}."
        )
    return labels


def _valid_full_action_chunk_indices(dataset: Dataset, action_horizon: int) -> list[int]:
    if action_horizon <= 1:
        return list(range(len(dataset)))
    if not hasattr(dataset, "episode_data_index"):
        raise ValueError("Cannot filter episode-end action chunks because the dataset has no episode_data_index.")

    episode_data_index = dataset.episode_data_index
    episodes = getattr(dataset, "episodes", None)
    if episodes is None:
        total_episodes = getattr(getattr(dataset, "meta", None), "total_episodes", None)
        if total_episodes is None:
            total_episodes = len(episode_data_index["from"])
        episodes = range(total_episodes)

    indices = []
    for ep_idx in episodes:
        ep_start = _as_int(episode_data_index["from"][ep_idx])
        ep_end = _as_int(episode_data_index["to"][ep_idx])
        last_start = ep_end - action_horizon
        if last_start >= ep_start:
            indices.extend(range(ep_start, last_start + 1))
    return indices


def _resolve_pad_at_episode_end(repo_id: str | list[str], value: bool | Sequence[bool] | dict[str, bool]):
    if isinstance(repo_id, str):
        if isinstance(value, dict):
            return value.get(repo_id, True)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) != 1:
                raise ValueError("Single-dataset pad_at_episode_end must be a bool or a one-item list.")
            return bool(value[0])
        return bool(value)

    if isinstance(value, dict):
        missing = [dataset_id for dataset_id in repo_id if dataset_id not in value]
        if missing:
            raise ValueError(f"Missing pad_at_episode_end values for datasets: {missing}")
        return [bool(value[dataset_id]) for dataset_id in repo_id]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != len(repo_id):
            raise ValueError(
                "Number of pad_at_episode_end values "
                f"({len(value)}) must match number of repo_ids ({len(repo_id)})."
            )
        return [bool(v) for v in value]
    return [bool(value) for _ in repo_id]


def _filter_lerobot_episode_ends(
    dataset: Dataset,
    action_horizon: int,
    *,
    pad_at_episode_end: bool,
    allowed_collect_labels: frozenset[str] | None = None,
):
    if pad_at_episode_end and allowed_collect_labels is None:
        return dataset
    return FullActionChunkLeRobotDataset(
        dataset,
        action_horizon,
        pad_at_episode_end=pad_at_episode_end,
        allowed_collect_labels=allowed_collect_labels,
    )


def _filter_multi_lerobot_episode_ends(
    dataset: lerobot_dataset.MultiLeRobotDataset,
    action_horizon: int,
    pad_at_episode_end: Sequence[bool],
    *,
    allowed_collect_labels: frozenset[str] | None = None,
):
    if all(pad_at_episode_end) and allowed_collect_labels is None:
        return dataset
    return MultiLeRobotDatasetIndexFilter(
        dataset,
        action_horizon,
        pad_at_episode_end,
        allowed_collect_labels=allowed_collect_labels,
    )


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    distill_use_rollout_data: bool | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        if distill_use_rollout_data is not None:
            raise ValueError("Collect-aware distillation requires a LeRobot dataset; fake data is not supported.")
        return FakeDataset(model_config, num_samples=1024)

    allowed_collect_labels = None
    if distill_use_rollout_data is not None:
        allowed_collect_labels = (
            _VALID_COLLECT_LABELS if distill_use_rollout_data else _NON_ROLLOUT_COLLECT_LABELS
        )

    if isinstance(repo_id, str):
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
        )
        pad_at_episode_end = _resolve_pad_at_episode_end(
            repo_id, data_config.pad_at_episode_end
        )
        dataset = _filter_lerobot_episode_ends(
            dataset,
            action_horizon,
            pad_at_episode_end=pad_at_episode_end,
            allowed_collect_labels=allowed_collect_labels,
        )
    else:
        if len(repo_id) == 0:
            raise ValueError("Repo ID list is empty. Cannot create dataset.")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id[0])
        dataset = lerobot_dataset.MultiLeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
        )
        pad_at_episode_end = _resolve_pad_at_episode_end(
            repo_id, data_config.pad_at_episode_end
        )
        dataset = _filter_multi_lerobot_episode_ends(
            dataset,
            action_horizon,
            pad_at_episode_end,
            allowed_collect_labels=allowed_collect_labels,
        )

    if data_config.prompt_from_task:
        if isinstance(repo_id, str):
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        else:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromTaskField()])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def _unwrap_transformed_dataset(dataset: Dataset) -> Dataset:
    # TransformedDataset is our local wrapper; unwrap it so samplers can inspect the underlying dataset type.
    while isinstance(dataset, TransformedDataset):
        dataset = dataset._dataset  # noqa: SLF001
    return dataset


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    preserve_keys: Sequence[str] = (),
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        preserve_keys=preserve_keys,
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    include_distill_metadata: bool = False,
) -> DataLoader:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        include_distill_metadata: Whether to validate/filter LeRobot collect labels and return a GT mask.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    distill_use_rollout_data = None
    if include_distill_metadata:
        if config.distill_config is None:
            raise ValueError("include_distill_metadata requires config.distill_config to be set.")
        distill_use_rollout_data = config.distill_config.use_rollout_data

    if data_config.rlds_data_dir is not None:
        if include_distill_metadata:
            raise ValueError("Collect-aware distillation is only supported for LeRobot datasets, not RLDS datasets.")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        distill_use_rollout_data=distill_use_rollout_data,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    distill_use_rollout_data: bool | None = None,
) -> DataLoader:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        distill_use_rollout_data: None for a regular two-item batch; otherwise enables collect-aware distillation
            and determines whether rollout frames remain in the sampling index.
    """
    dataset = create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        distill_use_rollout_data=distill_use_rollout_data,
    )
    sampler_dataset = _unwrap_transformed_dataset(dataset)
    preserve_keys = (DISTILL_GT_MASK_KEY,) if distill_use_rollout_data is not None else ()
    dataset = transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
        preserve_keys=preserve_keys,
    )

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    # A single-dataset list has no meaningful weighting effect, so it uses the standard sampler.
    use_weighted_sampling = (
        isinstance(data_config.repo_id, list)
        and len(data_config.repo_id) > 1
        and data_config.dataset_weights is not None
    )
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if use_weighted_sampling:
                logging.warning(
                    "Weighted multi-dataset sampling is not supported with distributed PyTorch training. "
                    "Falling back to uniform distributed sampling."
                )
                use_weighted_sampling = False
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    if use_weighted_sampling:
        if not isinstance(sampler_dataset, lerobot_dataset.MultiLeRobotDataset | MultiLeRobotDatasetIndexFilter):
            raise ValueError("dataset_weights can only be used with a MultiLeRobotDataset.")
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedMultiDatasetSampler(
            sampler_dataset,
            weights=data_config.dataset_weights,
            num_samples=len(dataset),
            generator=generator,
        )
        shuffle = False

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        include_distill_metadata=distill_use_rollout_data is not None,
    )


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class WeightedMultiDatasetSampler(torch.utils.data.Sampler):
    """Samples from a MultiLeRobotDataset according to per-dataset weights."""

    def __init__(
        self,
        dataset: lerobot_dataset.MultiLeRobotDataset | MultiLeRobotDatasetIndexFilter,
        weights: dict[str, float] | list[float],
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ):
        if not isinstance(dataset, lerobot_dataset.MultiLeRobotDataset | MultiLeRobotDatasetIndexFilter):
            raise ValueError("WeightedMultiDatasetSampler only works with MultiLeRobotDataset")

        self.dataset = dataset
        self.generator = generator

        if isinstance(weights, dict):
            weight_list = [weights.get(repo_id, 1.0) for repo_id in dataset.repo_ids]
        else:
            if len(weights) != len(dataset.repo_ids):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of datasets ({len(dataset.repo_ids)})"
                )
            weight_list = list(weights)

        if any(weight < 0 for weight in weight_list):
            raise ValueError(f"Dataset weights must be non-negative, got: {weight_list}")
        total_weight = sum(weight_list)
        if total_weight <= 0:
            raise ValueError("At least one dataset weight must be positive")
        self.normalized_weights = [weight / total_weight for weight in weight_list]
        self._weights_tensor = torch.tensor(self.normalized_weights, dtype=torch.float32)

        # MultiLeRobotDataset does not expose per-dataset sizes publicly; this depends on lerobot internals.
        self.dataset_sizes = [ds.num_frames for ds in dataset._datasets]  # noqa: SLF001
        for repo_id, size, weight in zip(dataset.repo_ids, self.dataset_sizes, self.normalized_weights, strict=True):
            if size == 0 and weight > 0:
                raise ValueError(f"Dataset {repo_id!r} has positive sampling weight but contains no frames")
        self.dataset_cumsum = [0]
        for size in self.dataset_sizes:
            self.dataset_cumsum.append(self.dataset_cumsum[-1] + size)

        self.num_samples = num_samples if num_samples is not None else len(dataset)
        logging.info(
            "WeightedMultiDatasetSampler initialized with weights: "
            f"{dict(zip(dataset.repo_ids, self.normalized_weights, strict=True))}"
        )

    def __iter__(self):
        generator = self.generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        dataset_indices = torch.multinomial(
            self._weights_tensor,
            num_samples=self.num_samples,
            replacement=True,
            generator=generator,
        )
        for dataset_idx in dataset_indices.tolist():
            dataset_size = self.dataset_sizes[dataset_idx]
            sample_idx = torch.randint(0, dataset_size, (1,), generator=generator).item()
            yield self.dataset_cumsum[dataset_idx] + sample_idx

    def __len__(self):
        return self.num_samples


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
        *,
        include_distill_metadata: bool = False,
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self._include_distill_metadata = include_distill_metadata

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if self._include_distill_metadata:
                yield observation, batch["actions"], batch[DISTILL_GT_MASK_KEY]
            else:
                yield observation, batch["actions"]
