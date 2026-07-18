"""Compute normalization statistics for a config — fast variant.

Replicates the canonical pipeline in scripts/compute_norm_stats.py for configs
that use LeRobotRCDataConfig, but reads parquet files directly instead of going
through LeRobotDataset (which decodes images we don't need for norm stats).

Output is identical to the canonical script's for single-dataset, unweighted configs because:
- RunningStats accumulation (Welford's) is batch-size invariant.
- We replicate RepackTransform key selection, RCInputs (state/actions pass-through
  for the keys that affect norm stats), DeltaActions, and action_horizon stacking
  with episode-end clamping (matching LeRobotDataset.delta_timestamps behavior).

For multi-dataset configs, repo_id can be a list. If dataset_weights is provided,
statistics are computed with dataset-level weights and uniform frame sampling inside
each dataset, matching WeightedMultiDatasetSampler's intended training distribution.
When max_frames is provided for a multi-dataset config, it is applied per dataset.

Only LeRobotRCDataConfig is supported. Use compute_norm_stats.py for everything else.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from tqdm import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as _transforms


def _resolve_columns(factory: _config.LeRobotRCDataConfig) -> tuple[str, str]:
    if factory.action_mode == "eef":
        return "observation.eef_state", "eef_action"
    return "observation.state", "action"


def _delta_mask(factory: _config.LeRobotRCDataConfig) -> np.ndarray | None:
    if not (factory.action_mode == "joint" and factory.use_delta_joint_actions):
        return None
    if factory.action_dim == 14:
        return np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))
    return np.asarray(_transforms.make_bool_mask(factory.action_dim - 1, -1))


def _repo_id_list(repo_id: str | list[str]) -> list[str]:
    if isinstance(repo_id, str):
        return [repo_id]
    if not repo_id:
        raise ValueError("data_config.repo_id list is empty")
    return list(repo_id)


def _resolve_dataset_weights(
    repo_ids: list[str], dataset_weights: dict[str, float] | list[float] | None
) -> np.ndarray | None:
    if dataset_weights is None:
        return None

    if isinstance(dataset_weights, dict):
        weights = np.asarray([dataset_weights.get(repo_id, 1.0) for repo_id in repo_ids], dtype=np.float64)
    else:
        if len(dataset_weights) != len(repo_ids):
            raise ValueError(
                f"Number of dataset weights ({len(dataset_weights)}) must match number of repo_ids ({len(repo_ids)})"
            )
        weights = np.asarray(dataset_weights, dtype=np.float64)

    if np.any(weights < 0):
        raise ValueError(f"Dataset weights must be non-negative, got: {weights.tolist()}")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("At least one dataset weight must be positive")
    return weights / total


def _stack_episode(
    df_ep: pd.DataFrame,
    state_col: str,
    action_col: str,
    action_horizon: int,
    delta_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (states, actions) for one episode.

    states: (ep_len, state_dim)
    actions: (ep_len, action_horizon, action_dim) — action_horizon entries per frame,
             clamped to last frame at episode end (matching LeRobotDataset).
    """
    df_ep = df_ep.sort_values("frame_index", kind="stable")
    ep_states = np.stack([np.asarray(s) for s in df_ep[state_col].to_numpy()])
    ep_actions = np.stack([np.asarray(a) for a in df_ep[action_col].to_numpy()])

    ep_len = len(ep_states)
    base = np.arange(ep_len)[:, None]
    horizon = np.arange(action_horizon)[None, :]
    idx = np.minimum(base + horizon, ep_len - 1)
    actions_stacked = ep_actions[idx]  # (ep_len, action_horizon, action_dim)

    if delta_mask is not None:
        dims = delta_mask.shape[-1]
        offset = np.where(delta_mask, ep_states[:, :dims], 0.0)
        actions_stacked[..., :dims] -= offset[:, None, :]

    return ep_states, actions_stacked


def _find_parquet_files(base_dir: str) -> list[str]:
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    parquet_files = []
    for root, _dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(".parquet"):
                parquet_files.append(os.path.join(root, filename))
    parquet_files.sort()
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under: {base_dir}")
    return parquet_files


def _read_dataset_arrays(
    base_dir: str,
    *,
    state_col: str,
    action_col: str,
    action_horizon: int,
    delta_mask: np.ndarray | None,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    parquet_files = _find_parquet_files(base_dir)
    print(f"Reading from: {base_dir}")
    print(f"Found {len(parquet_files)} parquet files")

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    total_frames = 0

    for file_path in tqdm(parquet_files, desc=f"Reading parquet ({Path(base_dir).name})"):
        df = pd.read_parquet(file_path)
        if state_col not in df.columns or action_col not in df.columns:
            raise ValueError(
                f"{file_path} missing required columns "
                f"({state_col!r}, {action_col!r}); found: {list(df.columns)}"
            )
        if "episode_index" not in df.columns or "frame_index" not in df.columns:
            raise ValueError(f"{file_path} missing 'episode_index' or 'frame_index' column")

        for _ep_idx, df_ep in df.groupby("episode_index", sort=True):
            ep_states, ep_actions = _stack_episode(df_ep, state_col, action_col, action_horizon, delta_mask)
            if max_frames is not None and total_frames + len(ep_states) > max_frames:
                keep = max_frames - total_frames
                ep_states = ep_states[:keep]
                ep_actions = ep_actions[:keep]

            state_chunks.append(ep_states)
            action_chunks.append(ep_actions)
            total_frames += len(ep_states)

            if max_frames is not None and total_frames >= max_frames:
                break

        if max_frames is not None and total_frames >= max_frames:
            print(f"Reached max_frames={max_frames} for {base_dir}, stopping early")
            break

    if not state_chunks:
        raise RuntimeError(f"No frames collected from: {base_dir}")

    states = np.concatenate(state_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    print(f"Collected {states.shape[0]} frames from {base_dir}")
    return states, actions


def _running_norm_stats(values: np.ndarray, batch_size: int, desc: str) -> normalize.NormStats:
    stats = normalize.RunningStats()
    for i in tqdm(range(0, len(values), batch_size), desc=desc):
        stats.update(values[i : i + batch_size])
    return stats.get_statistics()


def _weighted_quantiles(values: np.ndarray, row_weights: np.ndarray, quantiles: tuple[float, ...]) -> list[np.ndarray]:
    total_weight = float(np.sum(row_weights))
    if total_weight <= 0:
        raise ValueError("Cannot compute weighted quantiles with zero total weight")

    results = []
    for quantile in quantiles:
        target = quantile * total_weight
        columns = []
        for dim in range(values.shape[-1]):
            order = np.argsort(values[:, dim], kind="stable")
            cumulative = np.cumsum(row_weights[order])
            idx = np.searchsorted(cumulative, target, side="left")
            idx = min(idx, len(order) - 1)
            columns.append(values[order[idx], dim])
        results.append(np.asarray(columns))
    return results


def _weighted_norm_stats(
    arrays: list[np.ndarray], dataset_weights: np.ndarray, *, desc: str
) -> normalize.NormStats:
    if len(arrays) != len(dataset_weights):
        raise ValueError("Number of arrays must match number of dataset weights")

    weighted_mean = None
    weighted_mean_of_squares = None
    flat_arrays = []
    row_weights = []

    for index, (array, dataset_weight) in enumerate(zip(arrays, dataset_weights, strict=True)):
        flat = array.reshape(-1, array.shape[-1])
        if len(flat) == 0:
            if dataset_weight > 0:
                raise ValueError(f"Dataset {index} has positive weight but no rows")
            continue
        if dataset_weight == 0:
            continue

        batch_mean = np.mean(flat, axis=0)
        batch_mean_of_squares = np.mean(flat**2, axis=0)
        if weighted_mean is None:
            weighted_mean = np.zeros_like(batch_mean, dtype=np.float64)
            weighted_mean_of_squares = np.zeros_like(batch_mean_of_squares, dtype=np.float64)
        weighted_mean += dataset_weight * batch_mean
        weighted_mean_of_squares += dataset_weight * batch_mean_of_squares
        flat_arrays.append(flat)
        row_weights.append(np.full(len(flat), dataset_weight / len(flat), dtype=np.float64))

    if weighted_mean is None or weighted_mean_of_squares is None:
        raise ValueError(f"No positively weighted rows available for {desc}")

    print(f"Computing weighted {desc} quantiles")
    flat_values = np.concatenate(flat_arrays, axis=0)
    flat_weights = np.concatenate(row_weights, axis=0)
    q01, q99 = _weighted_quantiles(flat_values, flat_weights, (0.01, 0.99))
    variance = weighted_mean_of_squares - weighted_mean**2
    std = np.sqrt(np.maximum(0, variance))
    return normalize.NormStats(mean=weighted_mean, std=std, q01=q01, q99=q99)


def _compute_norm_stats(
    state_arrays: list[np.ndarray],
    action_arrays: list[np.ndarray],
    dataset_weights: np.ndarray | None,
    batch_size: int,
) -> dict[str, normalize.NormStats]:
    if dataset_weights is None:
        all_states = np.concatenate(state_arrays, axis=0)
        all_actions = np.concatenate(action_arrays, axis=0)
        print(f"state shape: {all_states.shape}, actions shape: {all_actions.shape}")
        return {
            "state": _running_norm_stats(all_states, batch_size, "state stats"),
            "actions": _running_norm_stats(all_actions, batch_size, "action stats"),
        }

    for idx, (states, actions, weight) in enumerate(zip(state_arrays, action_arrays, dataset_weights, strict=True)):
        print(f"dataset[{idx}] weight={weight:.6g}, state shape={states.shape}, actions shape={actions.shape}")
    return {
        "state": _weighted_norm_stats(state_arrays, dataset_weights, desc="state"),
        "actions": _weighted_norm_stats(action_arrays, dataset_weights, desc="action"),
    }


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    factory = config.data
    if not isinstance(factory, _config.LeRobotRCDataConfig):
        raise NotImplementedError(
            "compute_norm_stats_fast.py only supports LeRobotRCDataConfig. "
            "Use compute_norm_stats.py for other configs."
        )
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive if provided")

    data_config = factory.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("data_config.repo_id is not set")
    if data_config.asset_id is None:
        raise ValueError("data_config.asset_id is not set")

    repo_ids = _repo_id_list(data_config.repo_id)
    dataset_weights = _resolve_dataset_weights(repo_ids, data_config.dataset_weights)

    state_col, action_col = _resolve_columns(factory)
    delta_mask = _delta_mask(factory)
    action_horizon = config.model.action_horizon

    print(f"state column: {state_col}, action column: {action_col}")
    print(
        f"action_horizon: {action_horizon}, action_dim: {factory.action_dim}, "
        f"state_dim: {factory.state_dim}, delta: {delta_mask is not None}"
    )
    if dataset_weights is None:
        print("dataset weights: none; using frame-unweighted stats")
    else:
        print(f"dataset weights: {dict(zip(repo_ids, dataset_weights, strict=True))}")
        if max_frames is not None and len(repo_ids) > 1:
            print(f"max_frames={max_frames} will be applied per dataset")

    state_arrays = []
    action_arrays = []
    for repo_id in repo_ids:
        states, actions = _read_dataset_arrays(
            repo_id,
            state_col=state_col,
            action_col=action_col,
            action_horizon=action_horizon,
            delta_mask=delta_mask,
            max_frames=max_frames,
        )
        state_arrays.append(states)
        action_arrays.append(actions)

    batch_size = max(config.batch_size, 1)
    norm_stats = _compute_norm_stats(state_arrays, action_arrays, dataset_weights, batch_size)
    for key, stats in norm_stats.items():
        print(f"{key}: mean={stats.mean}\n     std={stats.std}\n     q01={stats.q01}\n     q99={stats.q99}")

    if factory.assets.assets_dir:
        assets_root = Path(factory.assets.assets_dir) / config.name
    else:
        assets_root = config.assets_dirs
    output_path = assets_root / data_config.asset_id
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"Saved norm stats to {output_path}/norm_stats.json")


if __name__ == "__main__":
    tyro.cli(main)
