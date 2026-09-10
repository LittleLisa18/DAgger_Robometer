"""Export accepted, non-test teacher-takeover episodes to LeRobot v3."""

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np


def export(root, output, repo_id, fps=10):
    root, output = Path(root), Path(output)
    # Validate provenance before creating anything that looks like a dataset.
    json.loads((root / "run.json").read_text())
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    episodes = []
    for path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        # Select whole episodes; preserve the student prefix and teacher suffix.
        if (
            meta.get("accepted_for_distillation") is True
            and meta.get("took_over") is True
            and meta.get("robometer_success") is True
            and not meta.get("test_only")
            and meta.get("end_reason") in ("environment_terminated", "step_limit")
            and meta.get("steps", 0) > 0
        ):
            episodes.append((path, meta))
    if not episodes:
        raise ValueError(
            "No non-test episodes with teacher takeover and passing final score"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after finalize() and audit sidecars succeed. A failed export
    # leaves the destination free for a clean retry, with raw episodes untouched.
    with tempfile.TemporaryDirectory(
        prefix=".autodagger-export-", dir=output.parent
    ) as temp:
        staging = Path(temp) / "dataset"
        mapping = _export(root, staging, repo_id, fps, episodes)
        staging.rename(output)
    return mapping


def _export(root, output, repo_id, fps, episodes):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "collect": {"dtype": "string", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        features=features,
        robot_type="panda",
        use_videos=False,
    )
    mapping = []
    try:
        for metadata, meta in episodes:
            index = sum(e["dataset_episode_index"] is not None for e in mapping)
            with np.load(
                metadata.parent / "trajectory.npz", allow_pickle=False
            ) as trajectory:
                if any(
                    len(trajectory[k]) != meta["steps"]
                    for k in ("image", "image2", "state", "action", "collect")
                ):
                    raise ValueError(f"Frame count mismatch: {metadata}")
                if not np.isin(trajectory["collect"], ["rollout", "teacher"]).all():
                    raise ValueError(f"Invalid collect labels: {metadata}")
                for i in range(len(trajectory["action"])):
                    dataset.add_frame(
                        {
                            "observation.images.image": trajectory["image"][i],
                            "observation.images.image2": trajectory["image2"][i],
                            "observation.state": trajectory["state"][i],
                            "action": trajectory["action"][i],
                            "collect": str(trajectory["collect"][i]),
                            "task": meta["task"],
                        }
                    )
            dataset.save_episode()
            mapping.append({"dataset_episode_index": index, **meta})
    finally:
        dataset.finalize()
    (output / "meta" / "autodagger_episodes.json").write_text(
        json.dumps(mapping, indent=2)
    )
    (output / "meta" / "autodagger_run.json").write_text(
        (root / "run.json").read_text()
    )
    return mapping


def selected_frames(root, include_rollout=False, accepted_only=True):
    """Yield raw frames without joining trajectories across collection boundaries.

    Rollout actions are student actions, never teacher distillation targets.
    """
    for path in sorted(Path(root).glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta.get("test_only") or (
            accepted_only and not meta["accepted_for_distillation"]
        ):
            continue
        if not meta["steps"]:
            continue
        with np.load(path.parent / "trajectory.npz", allow_pickle=False) as data:
            for i, label in enumerate(data["collect"]):
                if label == "teacher" or include_rollout:
                    yield meta["episode_id"], i, {k: data[k][i] for k in data.files}


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo-id", default="local/autodagger_libero")
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args()
    export(a.run, a.output, a.repo_id, a.fps)
