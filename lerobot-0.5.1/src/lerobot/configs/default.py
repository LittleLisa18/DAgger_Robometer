#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetSourceConfig:
    """One local AutoDAgger export; episode indices belong to this source only."""

    repo_id: str
    root: str
    episodes: list[int] | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.repo_id or not self.root:
            raise ValueError("Each dataset source requires repo_id and a local root")
        if self.episodes == []:
            raise ValueError("Dataset source episodes must be nonempty or null")
        _validate_episodes(self.episodes)


def _validate_episodes(episodes: list[int] | None) -> None:
    if episodes is not None:
        if any(ep < 0 for ep in episodes):
            raise ValueError(
                f"Episode indices must be non-negative, got: {[ep for ep in episodes if ep < 0]}"
            )
        if len(episodes) != len(set(episodes)):
            duplicates = sorted({ep for ep in episodes if episodes.count(ep) > 1})
            raise ValueError(f"Episode indices contain duplicates: {duplicates}")


@dataclass
class DatasetConfig:
    # Use repo_id/root for one dataset, or sources for multi-dataset distillation.
    repo_id: str | None = None
    # Root directory for a concrete local dataset tree (e.g. 'dataset/path'). If None, local datasets are
    # looked up under $HF_LEROBOT_HOME/repo_id and Hub downloads use a revision-safe cache under $HF_LEROBOT_HOME/hub.
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    sources: list[DatasetSourceConfig] | None = None

    def __post_init__(self) -> None:
        _validate_episodes(self.episodes)
        if self.sources is not None:
            if not self.sources:
                raise ValueError("dataset.sources must be nonempty")
            if any(value is not None for value in (self.repo_id, self.root, self.episodes, self.revision)):
                raise ValueError("Use either dataset.sources or top-level repo_id/root/episodes/revision")
            roots = [Path(source.root).expanduser().resolve() for source in self.sources]
            if len(roots) != len(set(roots)):
                raise ValueError("Duplicate dataset source roots would sample the same data twice")
        elif not self.repo_id:
            raise ValueError("Specify dataset.repo_id or dataset.sources")


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'
    add_tags: bool = True  # If True, save configuration as tags in the WandB run.


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
