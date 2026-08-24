import dataclasses
import pathlib
import types

import jax
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


@dataclasses.dataclass(frozen=True)
class _MultiDatasetTestConfig(_config.DataConfigFactory):
    def create(self, assets_dirs: pathlib.Path, model_config):
        return self.create_base_config(assets_dirs, model_config)


def test_multi_dataset_config_factory_defaults():
    model = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    factory = _MultiDatasetTestConfig(repo_id=["dataset_a", "dataset_b"], dataset_weights=[0.25, 0.75])

    data_config = factory.create(pathlib.Path("/tmp/nonexistent-openpi-assets"), model)

    assert data_config.repo_id == ["dataset_a", "dataset_b"]
    assert data_config.asset_id is None
    assert data_config.dataset_weights == [0.25, 0.75]


def _fake_multi_lerobot_dataset(*, repo_ids=("dataset_a", "dataset_b"), sizes=(100, 100)):
    dataset = object.__new__(_data_loader.lerobot_dataset.MultiLeRobotDataset)
    dataset.repo_ids = list(repo_ids)
    dataset._datasets = [types.SimpleNamespace(num_frames=size) for size in sizes]  # noqa: SLF001
    return dataset


class _FakeHfDataset:
    def __init__(self, collect_labels, *, include_collect: bool = True):
        self._collect_labels = collect_labels
        self.features = {"collect": object()} if include_collect else {}

    def __getitem__(self, key):
        if key != "collect" or "collect" not in self.features:
            raise KeyError(key)
        return self._collect_labels


class _FakeLeRobotDataset:
    fps = 10
    collect_labels_by_repo = {}
    include_collect = True

    def __init__(self, repo_id, delta_timestamps=None):
        self.repo_id = repo_id
        self.delta_timestamps = delta_timestamps
        self.collect_labels = self.collect_labels_by_repo.get(
            repo_id,
            ["teleop", "teacher", "rollout", "dagger", "teleop", "teacher", "rollout", "dagger"],
        )
        self.hf_dataset = _FakeHfDataset(self.collect_labels, include_collect=self.include_collect)
        self.episode_data_index = {
            "from": [0, 5],
            "to": [5, 8],
        }

    @property
    def num_frames(self):
        return len(self)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"repo_id": self.repo_id, "index": index, "collect": self.collect_labels[index]}


def test_lerobot_dataset_wrapper_getattr_does_not_recurse_before_init():
    dataset = object.__new__(_data_loader.FullActionChunkLeRobotDataset)

    with pytest.raises(AttributeError):
        _ = dataset.missing_attribute

    dataset = object.__new__(_data_loader.MultiLeRobotDatasetIndexFilter)

    with pytest.raises(AttributeError):
        _ = dataset.missing_attribute


def test_lerobot_dataset_can_filter_episode_end_action_chunks(monkeypatch):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)

    data_config = _config.DataConfig(repo_id="dataset_a", pad_at_episode_end=False)
    model_config = pi0_config.Pi0Config(action_horizon=3)

    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)

    assert len(dataset) == 4
    assert [dataset[i]["index"] for i in range(len(dataset))] == [0, 1, 2, 5]


def test_lerobot_distillation_filters_rollout_and_uses_current_frame_label(monkeypatch):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)

    data_config = _config.DataConfig(repo_id="dataset_a", pad_at_episode_end=True)
    model_config = pi0_config.Pi0Config(action_horizon=3)

    without_rollout = _data_loader.create_torch_dataset(
        data_config,
        model_config.action_horizon,
        model_config,
        distill_use_rollout_data=False,
    )
    assert [without_rollout[i]["index"] for i in range(len(without_rollout))] == [0, 1, 3, 4, 5, 7]
    assert all(bool(without_rollout[i][_data_loader.DISTILL_GT_MASK_KEY]) for i in range(len(without_rollout)))

    with_rollout = _data_loader.create_torch_dataset(
        data_config,
        model_config.action_horizon,
        model_config,
        distill_use_rollout_data=True,
    )
    assert [bool(with_rollout[i][_data_loader.DISTILL_GT_MASK_KEY]) for i in range(len(with_rollout))] == [
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
    ]
    # Index 1 is teacher followed by rollout, while index 2 is rollout followed by dagger. Only the current frame wins.
    assert with_rollout[1]["collect"] == "teacher"
    assert bool(with_rollout[1][_data_loader.DISTILL_GT_MASK_KEY])
    assert with_rollout[2]["collect"] == "rollout"
    assert not bool(with_rollout[2][_data_loader.DISTILL_GT_MASK_KEY])
    assert "collect" not in with_rollout.delta_timestamps


@pytest.mark.parametrize("collect_labels", [["Teleop"] * 8, ["unknown"] * 8])
def test_lerobot_distillation_rejects_invalid_collect_labels(monkeypatch, collect_labels):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(_FakeLeRobotDataset, "collect_labels_by_repo", {"dataset_a": collect_labels})
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)

    data_config = _config.DataConfig(repo_id="dataset_a")
    model_config = pi0_config.Pi0Config(action_horizon=3)
    with pytest.raises(ValueError, match="invalid collect labels"):
        _data_loader.create_torch_dataset(
            data_config,
            model_config.action_horizon,
            model_config,
            distill_use_rollout_data=True,
        )


def test_lerobot_distillation_requires_collect_feature(monkeypatch):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    class MissingCollectDataset(_FakeLeRobotDataset):
        include_collect = False

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", MissingCollectDataset)

    data_config = _config.DataConfig(repo_id="dataset_a")
    model_config = pi0_config.Pi0Config(action_horizon=3)
    with pytest.raises(ValueError, match="missing required 'collect'"):
        _data_loader.create_torch_dataset(
            data_config,
            model_config.action_horizon,
            model_config,
            distill_use_rollout_data=True,
        )


def test_multi_lerobot_dataset_filters_episode_ends_per_dataset(monkeypatch):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    class FakeMultiLeRobotDataset:
        def __init__(self, repo_ids, delta_timestamps):
            self.repo_ids = list(repo_ids)
            self.delta_timestamps = delta_timestamps
            self._datasets = [_FakeLeRobotDataset(repo_id, delta_timestamps) for repo_id in repo_ids]
            self._offsets = [0]
            for dataset in self._datasets:
                self._offsets.append(self._offsets[-1] + len(dataset))

        def __len__(self):
            return self._offsets[-1]

        def __getitem__(self, index):
            for dataset_index, start in enumerate(self._offsets[:-1]):
                end = self._offsets[dataset_index + 1]
                if start <= index < end:
                    item = self._datasets[dataset_index][index - start].copy()
                    item["dataset_index"] = dataset_index
                    return item
            raise IndexError(index)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "MultiLeRobotDataset", FakeMultiLeRobotDataset)

    data_config = _config.DataConfig(
        repo_id=["dataset_a", "dataset_b"],
        pad_at_episode_end={"dataset_a": False, "dataset_b": True},
    )
    model_config = pi0_config.Pi0Config(action_horizon=3)

    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)

    assert len(dataset) == 12
    assert [child.num_frames for child in dataset._datasets] == [4, 8]  # noqa: SLF001
    assert [dataset[i]["repo_id"] for i in range(len(dataset))] == ["dataset_a"] * 4 + ["dataset_b"] * 8
    assert [dataset[i]["index"] for i in range(len(dataset))] == [0, 1, 2, 5, 0, 1, 2, 3, 4, 5, 6, 7]

    sampler = _data_loader.WeightedMultiDatasetSampler(dataset, weights=[0.5, 0.5], num_samples=10)
    assert sampler.dataset_sizes == [4, 8]


def test_multi_lerobot_distillation_updates_filtered_dataset_sizes(monkeypatch):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    class FakeMultiLeRobotDataset:
        def __init__(self, repo_ids, delta_timestamps):
            self.repo_ids = list(repo_ids)
            self.delta_timestamps = delta_timestamps
            self._datasets = [_FakeLeRobotDataset(repo_id, delta_timestamps) for repo_id in repo_ids]
            self._offsets = [0]
            for dataset in self._datasets:
                self._offsets.append(self._offsets[-1] + len(dataset))

        def __len__(self):
            return self._offsets[-1]

        def __getitem__(self, index):
            for dataset_index, start in enumerate(self._offsets[:-1]):
                end = self._offsets[dataset_index + 1]
                if start <= index < end:
                    item = self._datasets[dataset_index][index - start].copy()
                    item["dataset_index"] = dataset_index
                    return item
            raise IndexError(index)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "MultiLeRobotDataset", FakeMultiLeRobotDataset)

    data_config = _config.DataConfig(
        repo_id=["dataset_a", "dataset_b"],
        dataset_weights=[0.5, 0.5],
        pad_at_episode_end=True,
    )
    model_config = pi0_config.Pi0Config(action_horizon=3)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        model_config.action_horizon,
        model_config,
        distill_use_rollout_data=False,
    )

    assert len(dataset) == 12
    assert [child.num_frames for child in dataset._datasets] == [6, 6]  # noqa: SLF001
    assert all(bool(dataset[i][_data_loader.DISTILL_GT_MASK_KEY]) for i in range(len(dataset)))
    sampler = _data_loader.WeightedMultiDatasetSampler(dataset, weights=[0.5, 0.5], num_samples=10)
    assert sampler.dataset_sizes == [6, 6]


def test_weighted_multi_dataset_sampler_distribution():
    dataset = _fake_multi_lerobot_dataset(sizes=(100, 100))
    generator = torch.Generator().manual_seed(0)
    sampler = _data_loader.WeightedMultiDatasetSampler(
        dataset,
        weights={"dataset_a": 0.25, "dataset_b": 0.75},
        num_samples=10_000,
        generator=generator,
    )

    samples = list(sampler)
    dataset_b_fraction = sum(index >= 100 for index in samples) / len(samples)

    assert dataset_b_fraction == pytest.approx(0.75, abs=0.03)


def test_create_torch_dataset_uses_task_field_for_multi_dataset_prompt(monkeypatch):
    class FakeMetadata:
        fps = 10
        tasks = types.MappingProxyType({0: "wrong first dataset task"})

        def __init__(self, repo_id):
            self.repo_id = repo_id

    class FakeMultiLeRobotDataset:
        def __init__(self, repo_ids, delta_timestamps):
            self.repo_ids = repo_ids
            self.delta_timestamps = delta_timestamps

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"task_index": 0, "task": ["task from dataset a", "task from dataset b"][index]}

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "MultiLeRobotDataset", FakeMultiLeRobotDataset)

    data_config = _config.DataConfig(repo_id=["dataset_a", "dataset_b"], prompt_from_task=True)
    model_config = pi0_config.Pi0Config(action_horizon=4)

    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)

    assert dataset[1]["prompt"] == "task from dataset b"


def test_distillation_data_loader_returns_gt_mask():
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    base_dataset = _data_loader.FakeDataset(model_config, 4)

    class FakeDistillationDataset:
        def __len__(self):
            return len(base_dataset)

        def __getitem__(self, index):
            return {
                **base_dataset[index],
                _data_loader.DISTILL_GT_MASK_KEY: np.asarray(index % 2 == 0, dtype=np.bool_),
            }

    raw_loader = _data_loader.TorchDataLoader(FakeDistillationDataset(), local_batch_size=4, num_batches=1)
    loader = _data_loader.DataLoaderImpl(
        _config.DataConfig(repo_id="fake"),
        raw_loader,
        include_distill_metadata=True,
    )

    _, actions, gt_mask = next(iter(loader))
    assert actions.shape == (4, model_config.action_horizon, model_config.action_dim)
    assert np.asarray(gt_mask).dtype == np.bool_
    assert np.asarray(gt_mask).tolist() == [True, False, True, False]
