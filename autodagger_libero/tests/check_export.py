"""Run in the hw SmolVLA environment, not the LIBERO Python 3.8 environment."""

import tempfile
import json
import shutil
from pathlib import Path
import numpy as np
from autodagger_libero.tests.test_collection import Tests
from autodagger_libero.export_dataset import export
from lerobot.datasets.lerobot_dataset import LeRobotDataset

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp) / "raw"
    Tests().episode(str(root))
    # Build independent metadata cases around a synthetic six-frame trajectory.
    metadata = root / "episode0" / "metadata.json"
    meta = json.loads(metadata.read_text())
    meta["test_only"] = False
    metadata.write_text(json.dumps(meta))
    for name, changes in (
        ("failed", {"accepted_for_distillation": False, "robometer_success": False}),
        ("student_only", {"took_over": False}),
        ("forced_test", {"test_only": True}),
        ("interrupted", {"end_reason": "interrupted"}),
    ):
        directory = root / name
        shutil.copytree(metadata.parent, directory)
        (directory / "metadata.json").write_text(
            json.dumps({**meta, **changes, "episode_id": name})
        )
    out = Path(temp) / "dataset"
    mapping = export(root, out, "local/autodagger_export_test")
    dataset = LeRobotDataset(
        "local/autodagger_export_test",
        root=out,
        delta_timestamps={"action": [0.0, 0.1, 0.2]},
    )
    assert len(dataset) == 6
    assert dataset.fps == 10
    np.testing.assert_allclose(
        [dataset[i]["timestamp"].item() for i in range(6)],
        np.arange(6) / 10,
        atol=1e-6,
    )
    assert dataset[0]["collect"] == "rollout"
    assert dataset[2]["collect"] == "teacher"
    assert dataset[2]["action"].shape == (3, 7)
    np.testing.assert_allclose(dataset[2]["action"][:, 0], 0.7)
    assert mapping[0]["dataset_episode_index"] == 0
    assert len(mapping) == 1 and mapping[0]["episode_id"] == "episode0"
    # A metadata/trajectory mismatch must not publish a partial dataset.
    metadata = root / "episode0" / "metadata.json"
    meta = json.loads(metadata.read_text())
    meta["steps"] += 1
    metadata.write_text(json.dumps(meta))
    failed_output = Path(temp) / "failed_dataset"
    try:
        export(root, failed_output, "local/autodagger_failed_export_test")
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt frame count was accepted")
    assert not failed_output.exists()
    meta["accepted_for_distillation"] = False
    metadata.write_text(json.dumps(meta))
    try:
        export(root, failed_output, "local/autodagger_empty_export_test")
    except ValueError as error:
        assert "No non-test episodes" in str(error)
    else:
        raise AssertionError("An empty selection must not publish a dataset")
    assert not failed_output.exists()
    print(
        "PASS: accepted-takeover filtering, empty selection, export/reload, collect strings, action chunk"
    )
