"""Run without the upstream hardware fixtures: python -m unittest discover -s tests/distillation."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import torch
from websockets.sync.server import serve

from lerobot.configs.distillation import DistillationConfig
from lerobot.datasets.autodagger_distillation import AutoDAggerDistillationDataset
from lerobot.remote_inference.openpi_compat import pack_message, unpack_message
from lerobot.utils.distillation import (
    OpenPITeacherClient,
    distillation_loss,
    prepare_distillation_batch,
    teacher_observation,
)


class LocalAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    def reduce(self, value, reduction):
        return value


def raw_batch(horizon=3):
    image = torch.arange(3 * 256 * 256).reshape(1, 3, 256, 256).remainder(256).float() / 255
    return {
        "observation.images.image": image,
        "observation.images.image2": 1 - image,
        "observation.state": torch.arange(8).float().unsqueeze(0),
        "task": ["pick up the bowl"],
        "action": torch.zeros(1, horizon, 7),
        "distill_bc_mask": torch.ones(1, horizon, dtype=torch.bool),
    }


class FakeDataset:
    def __init__(self, root):
        self.root = root
        self.episodes = [2, 7]
        self.num_frames, self.num_episodes = 7, 2
        self.meta = SimpleNamespace(
            fps=10,
            features={
                "observation.images.image": {"shape": [3, 256, 256]},
                "observation.images.image2": {"shape": [3, 256, 256]},
                "observation.state": {"shape": [8]},
                "action": {"shape": [7]},
                "collect": {},
            },
        )
        self.delta_timestamps = {"action": [0.0, 0.1, 0.2]}
        self.meta.episodes = {
            2: {"dataset_from_index": 10, "dataset_to_index": 15},
            7: {"dataset_from_index": 30, "dataset_to_index": 32},
        }
        self.hf_dataset = datasets.Dataset.from_dict(
            {
                "index": [10, 11, 12, 13, 14, 30, 31],
                "episode_index": [2] * 5 + [7] * 2,
                "collect": ["rollout", "rollout", "teacher", "teacher", "teacher", "teacher", "teacher"],
            }
        )
        (root / "meta").mkdir()
        for name in ("info.json", "stats.json", "autodagger_run.json", "tasks.parquet"):
            (root / "meta" / name).write_text("{}")
        rows = [
            {
                "dataset_episode_index": ep,
                "accepted_for_distillation": True,
                "took_over": True,
                "robometer_success": True,
                "test_only": False,
                "end_reason": "step_limit",
                "steps": n,
            }
            for ep, n in ((2, 5), (7, 2))
        ]
        (root / "meta/autodagger_episodes.json").write_text(json.dumps(rows))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        remaining = (5 if i < 5 else 7) - i
        return {"action_is_pad": torch.arange(3) >= remaining}


class DatasetTests(unittest.TestCase):
    def test_masks_and_episode_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = AutoDAggerDistillationDataset(FakeDataset(Path(temp)), 3)
            expected = [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]]
            self.assertEqual([ds[i]["distill_bc_mask"].int().tolist() for i in range(7)], expected)

    def test_invalid_labels_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = FakeDataset(Path(temp))
            ds.hf_dataset = ds.hf_dataset.map(lambda row: {"collect": "unknown"})
            with self.assertRaisesRegex(ValueError, "collect"):
                AutoDAggerDistillationDataset(ds, 3)
        with tempfile.TemporaryDirectory() as temp:
            ds = FakeDataset(Path(temp))
            path = Path(temp) / "meta/autodagger_episodes.json"
            rows = json.loads(path.read_text())
            rows[0]["robometer_success"] = False
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "audit"):
                AutoDAggerDistillationDataset(ds, 3)

    def test_fingerprint_tracks_file_content(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = FakeDataset(Path(temp))
            before = AutoDAggerDistillationDataset(ds, 3).fingerprint
            (Path(temp) / "meta/stats.json").write_text('{"changed": true}')
            self.assertNotEqual(before, AutoDAggerDistillationDataset(ds, 3).fingerprint)

    def test_rejects_misaligned_episode_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = FakeDataset(Path(temp))
            ds.meta.episodes[2]["dataset_to_index"] = 16
            with self.assertRaisesRegex(ValueError, "metadata bounds"):
                AutoDAggerDistillationDataset(ds, 3)


class ClientTests(unittest.TestCase):
    def with_server(self, handler, check):
        with serve(handler, "127.0.0.1", 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            config = DistillationConfig(
                f"ws://127.0.0.1:{server.socket.getsockname()[1]}", "test", timeout_s=0.2
            )
            client = OpenPITeacherClient(config)
            try:
                check(client)
            finally:
                client.close()
                server.shutdown()
                thread.join(timeout=5)

    def test_real_wire_and_normalization(self):
        def handler(ws):
            ws.send(pack_message({"policy": "test"}))
            request = unpack_message(ws.recv())
            np.testing.assert_array_equal(request["observation/state"], np.arange(8))
            self.assertEqual(request["observation/image"].dtype, np.uint8)
            ws.send(pack_message({"actions": np.full((10, 7), 0.6, dtype=np.float32)}))

        def check(client):
            original = raw_batch()

            def normalize(batch):
                return {**batch, "action": (batch["action"] - 0.2) / 2}

            result = prepare_distillation_batch(original, normalize, client, LocalAccelerator())
            torch.testing.assert_close(result["action"][0], torch.full((3, 7), 0.2))
            torch.testing.assert_close(result["action"][1], torch.full((3, 7), -0.1))
            torch.testing.assert_close(original["action"], torch.zeros_like(original["action"]))

        self.with_server(handler, check)

    def test_bad_chunk_is_not_retried(self):
        def handler(ws):
            ws.send(pack_message({}))
            ws.recv()
            ws.send(pack_message({"actions": np.zeros((50, 7), dtype=np.float32)}))

        def check(client):
            with self.assertRaisesRegex(RuntimeError, "Expected finite teacher actions"):
                prepare_distillation_batch(raw_batch(), lambda x: x, client, LocalAccelerator())
            self.assertEqual(client.retry_count, 0)

        self.with_server(handler, check)

    def test_connection_retry(self):
        calls = []

        def handler(ws):
            calls.append(1)
            ws.send(pack_message({}))
            ws.recv()
            if len(calls) == 1:
                ws.close()
            else:
                ws.send(pack_message({"actions": np.zeros((10, 7), dtype=np.float32)}))

        def check(client):
            self.assertEqual(client.infer(teacher_observation(raw_batch(), 0)).shape, (10, 7))
            self.assertEqual(client.retry_count, 1)

        self.with_server(handler, check)

    def test_timeout_is_bounded(self):
        import time

        def handler(ws):
            ws.send(pack_message({}))
            ws.recv()
            time.sleep(0.5)

        def check(client):
            with self.assertRaises(TimeoutError):
                client.infer(teacher_observation(raw_batch(), 0))
            self.assertEqual(client.retry_count, 1)

        self.with_server(handler, check)

    def test_pixel_roundtrip_and_no_second_rotation(self):
        batch = raw_batch()
        out = teacher_observation(batch, 0)
        expected = (batch["observation.images.image"][0] * 255).round().byte().permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(expected, out["observation/image"])


class LossTests(unittest.TestCase):
    def test_smolvla_element_return_preserves_default_reductions(self):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # Exercise the real wrapper without loading a vision/language checkpoint.
        expected = torch.arange(2 * 3 * 32).reshape(2, 3, 32).float()
        stub = SimpleNamespace(
            config=SimpleNamespace(
                adapt_to_pi_aloha=False, action_feature=SimpleNamespace(shape=(7,)), max_action_dim=32
            ),
            prepare_images=lambda batch: (None, None),
            prepare_state=lambda batch: None,
            prepare_action=lambda batch: None,
            model=SimpleNamespace(forward=lambda *args: expected.clone()),
        )
        batch = {
            "observation.language.tokens": None,
            "observation.language.attention_mask": None,
            "action_is_pad": torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.bool),
        }
        elements, _ = SmolVLAPolicy.forward(stub, batch, reduction="elements")
        torch.testing.assert_close(elements, expected[:, :, :7])
        masked = elements * (~batch["action_is_pad"]).unsqueeze(-1)
        loss, _ = SmolVLAPolicy.forward(stub, batch)
        per_sample, _ = SmolVLAPolicy.forward(stub, batch, reduction="none")
        torch.testing.assert_close(loss, masked.mean())
        torch.testing.assert_close(per_sample, masked.mean((1, 2)))

    def test_valid_means_and_gradient_mask(self):
        errors = torch.ones(4, 3, 7, requires_grad=True)
        mask = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
        loss, metrics = distillation_loss(errors, mask, LocalAccelerator(), DistillationConfig("ws://x", "x"))
        self.assertEqual(loss.item(), 2)
        self.assertEqual(metrics["bc_valid_steps"], 1)
        loss.backward()
        self.assertEqual(errors.grad[2, 1:].count_nonzero().item(), 0)
        self.assertEqual(errors.grad[3].count_nonzero().item(), 0)

    def test_no_bc(self):
        errors = torch.ones(2, 3, 7, requires_grad=True)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        loss, metrics = distillation_loss(errors, mask, LocalAccelerator(), DistillationConfig("ws://x", "x"))
        self.assertEqual(loss.item(), 1)
        self.assertEqual(metrics["bc_loss"], 0)
        loss.backward()
        self.assertEqual(errors.grad[1].count_nonzero().item(), 0)

    def test_config_horizon_boundaries(self):
        p = SimpleNamespace(
            type="smolvla",
            n_obs_steps=1,
            n_action_steps=1,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
            empty_cameras=0,
            rtc_config=None,
            use_peft=False,
            pretrained_path="base",
        )
        cfg = SimpleNamespace(
            policy=p,
            use_rabc=False,
            peft=None,
            rename_map={},
            dataset=SimpleNamespace(
                streaming=False, image_transforms=SimpleNamespace(enable=False), root="data"
            ),
        )
        dc = DistillationConfig("ws://teacher", "teacher-v1")
        for horizon in (1, 5, 10):
            p.chunk_size = horizon
            dc.validate(cfg)
        for horizon in (0, 11):
            p.chunk_size = horizon
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                dc.validate(cfg)


if __name__ == "__main__":
    unittest.main()
