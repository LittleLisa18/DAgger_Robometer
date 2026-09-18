"""Multi-source boundaries, selected-episode statistics, provenance and config decoding."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import draccus
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from test_distillation import FakeDataset

from lerobot.configs.default import DatasetConfig, DatasetSourceConfig
from lerobot.datasets.autodagger_distillation import (
    AutoDAggerDistillationDataset,
    MultiAutoDAggerDistillationDataset,
    selected_normalization_stats,
)
from lerobot.datasets.compute_stats import aggregate_stats


class NumericDataset(FakeDataset):
    def __init__(self, root, shift, episodes=None):
        root.mkdir()
        super().__init__(root)
        self.repo_id = "local/same-id"
        self.meta.camera_keys = ["observation.images.image", "observation.images.image2"]
        self.meta.features["action"]["names"] = [f"a{i}" for i in range(7)]
        values = np.arange(7, dtype=np.float64) + shift
        self.values = values
        all_stats = []
        rows = []
        for episode, indices in ((2, slice(0, 5)), (7, slice(5, 7))):
            data = values[indices]
            stats = {}
            for key, shape in (
                ("action", (7,)),
                ("observation.state", (8,)),
                *[(key, (3, 1, 1)) for key in self.meta.camera_keys],
            ):
                stats[key] = {
                    name: np.full(shape, value)
                    for name, value in {
                        "min": data.min(),
                        "max": data.max(),
                        "mean": data.mean(),
                        "std": data.std(),
                    }.items()
                }
                stats[key]["count"] = np.array([len(data)])
            all_stats.append(stats)
            rows.append(
                {
                    "episode_index": episode,
                    **{
                        f"stats/{key}/{name}": value.tolist()
                        for key, feature in stats.items()
                        for name, value in feature.items()
                    },
                }
            )
        self.meta.stats = aggregate_stats(all_stats)
        (root / "meta/episodes").mkdir()
        pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/file.parquet")
        (root / "meta/stats.json").write_text(json.dumps({"test_shift": shift}))
        self.episodes = episodes
        if episodes is not None:
            self.hf_dataset = self.hf_dataset.filter(lambda row: row["episode_index"] in episodes)
            self.num_frames = len(self.hf_dataset)
            self.num_episodes = len(episodes)

    def __getitem__(self, index):
        row = self.hf_dataset[index]
        return {
            "index": torch.tensor(row["index"]),
            "episode_index": torch.tensor(row["episode_index"]),
            "action_is_pad": torch.zeros(3, dtype=torch.bool),
            **({"collect": row["collect"]} if "collect" in row else {}),
            "task": str(self.root.name),
        }


class MultiDatasetTests(unittest.TestCase):
    def test_legacy_hwc_metadata_and_mixed_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            chw = NumericDataset(Path(temp) / "chw", 0)
            hwc = self.demonstration(Path(temp) / "hwc")
            for key in hwc.meta.camera_keys:
                hwc.meta.features[key].update(
                    dtype="video", shape=[256, 256, 3], names=["height", "width", "channel"]
                )
            original = json.loads(json.dumps(hwc.meta.features))
            sources = [
                AutoDAggerDistillationDataset(chw, 3),
                AutoDAggerDistillationDataset(hwc, 3, "demonstration"),
            ]
            ds = MultiAutoDAggerDistillationDataset(sources)
            self.assertEqual(len(ds), 14)
            self.assertTrue(ds[7]["distill_bc_mask"].all())
            self.assertEqual(hwc.meta.features, original)
            hwc.meta.features["observation.images.image"]["shape"] = [128, 256, 3]
            with self.assertRaisesRegex(ValueError, "decoded shape"):
                AutoDAggerDistillationDataset(hwc, 3, "demonstration")

    def demonstration(self, root):
        ds = NumericDataset(root, 100)
        ds.meta.features.pop("collect")
        ds.hf_dataset = ds.hf_dataset.remove_columns("collect")
        for name in ("autodagger_episodes.json", "autodagger_run.json"):
            (root / "meta" / name).unlink()
        return ds

    def test_demonstration_without_labels_or_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = self.demonstration(Path(temp) / "demo")
            ds = AutoDAggerDistillationDataset(raw, 3, supervision="demonstration")
            self.assertEqual(ds[0]["distill_bc_mask"].tolist(), [True, True, True])
            self.assertEqual(ds[4]["distill_bc_mask"].tolist(), [True, False, False])
            self.assertEqual(ds[6]["distill_bc_mask"].tolist(), [True, False, False])
            self.assertNotIn("collect", ds[0])
            with self.assertRaisesRegex(ValueError, "missing collect"):
                AutoDAggerDistillationDataset(raw, 3)

    def test_mixed_batch_in_both_source_orders(self):
        with tempfile.TemporaryDirectory() as temp:
            demo = AutoDAggerDistillationDataset(self.demonstration(Path(temp) / "demo"), 3, "demonstration")
            auto = AutoDAggerDistillationDataset(NumericDataset(Path(temp) / "auto", 0), 3)
            # Collection-only features must not prevent mixing standard demonstrations.
            auto.meta.features["collection_score"] = {"shape": [1]}
            for sources in ([auto, demo], [demo, auto]):
                ds = MultiAutoDAggerDistillationDataset(sources)
                batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=14)))
                self.assertNotIn("collect", batch)
                self.assertNotIn("collect", ds.meta.features)
                self.assertNotIn("collection_score", ds.meta.features)
                demo_start = 7 if sources[0] is auto else 0
                auto_start = 0 if sources[0] is auto else 7
                self.assertTrue(batch["distill_bc_mask"][demo_start].all())
                self.assertFalse(batch["distill_bc_mask"][auto_start].any())
                self.assertEqual(batch["distill_bc_mask"][6].tolist(), [True, False, False])

    def test_demonstration_does_not_override_existing_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = NumericDataset(Path(temp) / "labeled", 0)
            with self.assertRaisesRegex(ValueError, "must not contain collect"):
                AutoDAggerDistillationDataset(raw, 3, "demonstration")
        cfg = draccus.decode(
            DatasetConfig,
            {"sources": [{"repo_id": "local/demo", "root": "/data/demo", "supervision": "demonstration"}]},
        )
        self.assertEqual(cfg.sources[0].supervision, "demonstration")
        with self.assertRaises(ValueError):
            DatasetSourceConfig(repo_id="local/demo", root="/data/demo", supervision="typo")

    def test_demonstration_padding_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = self.demonstration(Path(temp) / "demo")
            ds = AutoDAggerDistillationDataset(raw, 3, "demonstration")
            with patch.object(
                NumericDataset,
                "__getitem__",
                return_value={"action_is_pad": torch.tensor([False, True, False])},
            ):
                self.assertEqual(ds[0]["distill_bc_mask"].tolist(), [True, False, True])
            before = ds.fingerprint
            (raw.root / "meta/stats.json").write_text('{"updated": true}')
            self.assertNotEqual(before, AutoDAggerDistillationDataset(raw, 3, "demonstration").fingerprint)
            raw.meta.episodes[2]["dataset_to_index"] = 16
            with self.assertRaisesRegex(ValueError, "metadata bounds"):
                AutoDAggerDistillationDataset(raw, 3, "demonstration")

    def test_colliding_local_indices_and_source_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = [
                AutoDAggerDistillationDataset(NumericDataset(Path(temp) / name, shift), 3)
                for name, shift in (("first", 0), ("second", 100))
            ]
            ds = MultiAutoDAggerDistillationDataset(sources)
            self.assertEqual((len(ds), ds.num_episodes), (14, 4))
            self.assertEqual(ds[6]["distill_bc_mask"].tolist(), [True, False, False])
            self.assertEqual(ds[7]["distill_bc_mask"].tolist(), [False, False, False])
            self.assertEqual(ds[9]["distill_bc_mask"].tolist(), [True, True, True])
            self.assertEqual(ds[0]["source_index"], ds[7]["source_index"])
            self.assertNotEqual(ds[0]["episode_index"], ds[7]["episode_index"])
            self.assertEqual((ds[7]["index"].item(), ds[7]["dataset_index"].item()), (7, 1))
            self.assertEqual((ds[0]["task"], ds[7]["task"]), ("first", "second"))
            self.assertEqual(ds[-1]["index"].item(), 13)
            with self.assertRaises(IndexError):
                ds[14]
            batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=14)))
            self.assertEqual(batch["dataset_index"].tolist(), [0] * 7 + [1] * 7)

    def test_selected_stats_match_actual_pooled_values(self):
        with tempfile.TemporaryDirectory() as temp:
            first = NumericDataset(Path(temp) / "a", 0, episodes=[2])
            second = NumericDataset(Path(temp) / "b", 100, episodes=[7])
            ds = MultiAutoDAggerDistillationDataset(
                [AutoDAggerDistillationDataset(first, 3), AutoDAggerDistillationDataset(second, 3)]
            )
            expected = np.concatenate([first.values[:5], second.values[5:]])
            for key in ("action", "observation.state"):
                stats = ds.meta.stats[key]
                np.testing.assert_allclose(stats["mean"], expected.mean())
                np.testing.assert_allclose(stats["std"], expected.std())
                np.testing.assert_allclose(stats["min"], expected.min())
                np.testing.assert_allclose(stats["max"], expected.max())
                self.assertEqual(stats["count"].item(), 7)
            self.assertEqual((ds.num_frames, ds.num_episodes), (7, 2))
            self.assertEqual(ds.meta.episodes[1]["dataset_from_index"], 5)
            self.assertEqual(ds[4]["distill_bc_mask"].tolist(), [True, False, False])

    def test_fingerprint_covers_order_content_and_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            a = NumericDataset(Path(temp) / "a", 0)
            b = NumericDataset(Path(temp) / "b", 100)

            def fingerprint(items):
                return MultiAutoDAggerDistillationDataset(
                    [AutoDAggerDistillationDataset(item, 3) for item in items]
                ).fingerprint

            before = fingerprint([a, b])
            self.assertEqual(before, fingerprint([a, b]))
            self.assertNotEqual(before, fingerprint([b, a]))
            (b.root / "meta/autodagger_run.json").write_text('{"changed":true}')
            changed = fingerprint([a, b])
            self.assertNotEqual(before, changed)
            b.episodes = [7]
            b.hf_dataset = b.hf_dataset.filter(lambda row: row["episode_index"] == 7)
            b.num_frames, b.num_episodes = 2, 1
            self.assertNotEqual(changed, fingerprint([a, b]))

    def test_rejects_incompatible_schema_or_missing_subset_stats(self):
        with tempfile.TemporaryDirectory() as temp:
            a = AutoDAggerDistillationDataset(NumericDataset(Path(temp) / "a", 0), 3)
            b = AutoDAggerDistillationDataset(NumericDataset(Path(temp) / "b", 100), 3)
            b.meta.features["action"]["names"] = ["different"] * 7
            with self.assertRaisesRegex(ValueError, "schemas"):
                MultiAutoDAggerDistillationDataset([a, b])
            a.dataset.episodes = [42]
            with self.assertRaisesRegex(ValueError, "Missing statistics"):
                selected_normalization_stats(a.dataset)

    def test_config_roundtrip_and_rejections(self):
        payload = {
            "sources": [
                {"repo_id": "local/shared", "root": "/data/a", "episodes": [0, 2]},
                {"repo_id": "local/shared", "root": "/data/b"},
            ]
        }
        cfg = draccus.decode(DatasetConfig, payload)
        self.assertIsInstance(cfg.sources[0], DatasetSourceConfig)
        self.assertEqual(
            draccus.encode(draccus.decode(DatasetConfig, draccus.encode(cfg))), draccus.encode(cfg)
        )
        self.assertEqual(DatasetConfig(repo_id="local/single").repo_id, "local/single")
        for bad in (
            {"sources": []},
            {**payload, "repo_id": "ambiguous"},
            {
                "sources": [payload["sources"][0], payload["sources"][0]],
            },
        ):
            with self.assertRaises(draccus.utils.ParsingError):
                draccus.decode(DatasetConfig, bad)


if __name__ == "__main__":
    unittest.main()
