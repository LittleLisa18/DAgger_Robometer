import dataclasses
import pathlib
import types

import jax
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
    dataset._datasets = [types.SimpleNamespace(num_frames=size) for size in sizes]
    return dataset


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
        tasks = {0: "wrong first dataset task"}

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
