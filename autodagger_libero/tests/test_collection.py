import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import requests

from autodagger_libero.core import (
    Config,
    Gate,
    RunStore,
    sample_prefix,
    validate_actions,
)
from autodagger_libero.collect import run_episode, observation
from autodagger_libero.dashboard import Dashboard
from autodagger_libero.export_dataset import selected_frames


class Env:
    def __init__(self, length=6, success=True):
        self.length, self.success = length, success
        self.actions = []

    def reset(self):
        self.steps = -10

    def raw(self):
        return {
            "agentview_image": np.full((256, 256, 3), max(0, self.steps), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
            "robot0_eef_pos": np.array([max(0, self.steps), 0, 0]),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def set_init_state(self, _):
        return self.raw()

    def step(self, action):
        self.steps += 1
        if self.steps > 0:
            self.actions.append(action)
        return self.raw(), 0, self.steps >= self.length, {}

    def check_success(self):
        return self.steps >= self.length and self.success


class Policy:
    def __init__(self, value):
        self.value, self.resets, self.calls = value, 0, []

    def reset(self):
        self.resets += 1

    def infer(self, obs):
        self.calls.append(obs)
        return {"actions": np.full((5, 7), self.value, np.float32)}


class Monitor:
    source = "simulated"

    def __init__(self, fail=False, success=0.9):
        self.fail, self.success = fail, success

    def score(self, frames, task):
        if self.fail:
            raise TimeoutError("test timeout")
        return {
            "progress": 0.2,
            "success_probability": self.success,
            "latency_ms": 1,
            "progress_trace": [0.2],
            "success_trace": [self.success],
        }


class Tests(unittest.TestCase):
    def test_disable_regression_keeps_stall_gate(self):
        config = Config(regression_enabled=False)
        config.validate()
        gate = Gate(config)
        gate.update(0, 0.8, 0.1)
        for step in (10, 20, 30, 40):
            self.assertIsNone(gate.update(step, 0.2, 0.1))
        self.assertEqual(gate.update(50, 0.2, 0.1), "progress_stalled")
        with self.assertRaises(ValueError):
            dataclasses.replace(config, regression_enabled="false").validate()

    def test_gate_tracks_small_peaks_and_cumulative_improvement(self):
        gate = Gate(Config())
        gate.update(0, 0.5, 0.1)
        gate.update(10, 0.54, 0.1)
        self.assertIsNone(gate.update(20, 0.43, 0.1))
        self.assertEqual(gate.update(30, 0.43, 0.1), "progress_regression")
        gate = Gate(Config())
        for step, progress in [(0, 0.5), (10, 0.53), (20, 0.56)]:
            self.assertIsNone(gate.update(step, progress, 0.1))
        self.assertEqual(gate.last_improved, 20)

    def test_nonfinite_configuration_rejected(self):
        for key in (
            "timeout",
            "progress_epsilon",
            "regression_threshold",
            "success_threshold",
        ):
            with self.assertRaises(ValueError):
                dataclasses.replace(Config(), **{key: float("nan")}).validate()

    def test_gate_stall_regression_success(self):
        c = Config()
        g = Gate(c)
        self.assertIsNone(g.update(0, 0.5, 0.1))
        self.assertEqual(g.update(50, 0.5, 0.1), "progress_stalled")
        g = Gate(c)
        g.update(0, 0.8, 0.1)
        self.assertIsNone(g.update(10, 0.6, 0.1))
        self.assertEqual(g.update(20, 0.6, 0.1), "progress_regression")
        self.assertIsNone(g.update(100, 0.6, 0.9))
        with self.assertRaises(ValueError):
            g.update(110, float("nan"), 0.1)

    def test_sampling_and_action_validation(self):
        frames = [np.array([i]) for i in range(71)]
        selected, indices = sample_prefix(frames, 16)
        self.assertEqual((indices[0], indices[-1], len(indices)), (0, 70, 16))
        for bad in (np.ones((5, 8)), np.ones((1, 7)), np.full((5, 7), np.nan)):
            with self.assertRaises(ValueError):
                validate_actions(bad, 5)
        np.testing.assert_array_equal(
            validate_actions(np.full((5, 7), -1.009), 5), -1.0
        )

    def test_observation_rotation_state(self):
        env = Env()
        env.reset()
        raw = env.raw()
        raw["agentview_image"][0, 0] = [1, 2, 3]
        obs = observation(raw)
        np.testing.assert_array_equal(obs["image"][-1, -1], [1, 2, 3])
        self.assertEqual(obs["state"].shape, (8,))
        np.testing.assert_array_equal(obs["state"][3:6], 0)

    def episode(self, root, **kwargs):
        c = Config(output=root, monitor_every=2, force_teacher_step=2, student_replan=5)
        for key, value in kwargs.pop("config", {}).items():
            setattr(c, key, value)
        store = RunStore(root, c, {})
        dashboard = Dashboard(store, c)
        student, teacher = Policy(0.1), Policy(0.7)
        env = kwargs.pop("env", Env())
        with patch(
            "autodagger_libero.collect.payload",
            side_effect=lambda obs, task, actor: obs,
        ):
            meta = run_episode(
                env,
                None,
                "test task",
                "episode0",
                c,
                student,
                teacher,
                kwargs.pop("monitor", Monitor()),
                store,
                dashboard,
                {},
            )
        return c, store, dashboard, meta, student, teacher, env

    def test_takeover_labels_alignment_queue_final(self):
        with tempfile.TemporaryDirectory() as root:
            c, store, dash, meta, student, teacher, env = self.episode(root)
            self.assertTrue(meta["accepted_for_distillation"])
            self.assertEqual(meta["takeover_step"], 2)
            with np.load(Path(root) / "episode0/trajectory.npz") as data:
                self.assertEqual(
                    data["collect"].tolist(), ["rollout"] * 2 + ["teacher"] * 4
                )
                np.testing.assert_allclose(
                    data["action"][:, 0], [0.1, 0.1, 0.7, 0.7, 0.7, 0.7]
                )
                np.testing.assert_allclose(data["state"][:, 0], np.arange(6))
            self.assertEqual(teacher.calls[0]["state"][0], 2)
            self.assertEqual(meta["samples"][-1]["step"], 6)
            self.assertTrue(store.completed("episode0"))
            self.assertEqual(student.resets, 1)
            self.assertEqual(
                dash.snapshot()["statistics"]["completed"], 0
            )  # forced tests excluded

    def test_student_success_failure_and_audit(self):
        for force, success, probability, accepted in [
            (None, True, 0.9, False),
            (2, False, 0.9, True),
            (2, True, 0.1, False),
        ]:
            with tempfile.TemporaryDirectory() as root:
                _, _, _, meta, *_ = self.episode(
                    root,
                    config={"force_teacher_step": force},
                    env=Env(success=success),
                    monitor=Monitor(success=probability),
                )
                self.assertEqual(meta["accepted_for_distillation"], accepted)
                self.assertEqual(
                    meta["success_disagreement"], success != (probability > 0.5)
                )

    def test_monitor_failure_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, _, meta, *_ = self.episode(root, monitor=Monitor(fail=True))
            self.assertEqual(meta["end_reason"], "interrupted")
            self.assertFalse(meta["accepted_for_distillation"])
            self.assertTrue(store.completed("episode0"))

    def test_resume_recovery_and_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            c = Config(output=root)
            store = RunStore(root, c, {})
            store.checkpoint(
                "partial",
                {"step": 0, "action": np.zeros(7)},
                {
                    "episode_id": "partial",
                    "steps": 0,
                    "accepted_for_distillation": False,
                },
            )
            (Path(root) / "partial/frame_000001.tmp.npz").write_bytes(
                b"truncated archive"
            )
            recovered = RunStore(root, c, {})
            self.assertTrue(recovered.completed("partial"))
            self.assertEqual(recovered.summaries()[0]["steps"], 1)
            with self.assertRaises(ValueError):
                RunStore(root, dataclasses.replace(c, seed=9), {})

    def test_dashboard_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            c, store, dash, meta, *_ = self.episode(root)
            dash.config.dashboard_port = 0
            dash.start()
            url = f"http://127.0.0.1:{dash.server.server_port}"
            try:
                self.assertEqual(requests.get(url + "/api/state").status_code, 200)
                self.assertEqual(len(requests.get(url + "/api/episodes").json()), 1)
                self.assertEqual(
                    requests.get(url + "/api/episodes/episode0").json()["steps"], 6
                )
                self.assertEqual(
                    requests.get(url + "/frames/image.jpg").headers["Content-Type"],
                    "image/jpeg",
                )
                self.assertEqual(requests.post(url + "/api/reset").status_code, 501)
                self.assertEqual(
                    requests.get(url + "/api/episodes/../../run.json").status_code, 404
                )
            finally:
                dash.close()

    def test_dashboard_episode_reset(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, dash, *_ = self.episode(root)
            dash.publish(policy_latency_ms=123)
            dash.publish(episode_id="next", status="resetting")
            self.assertEqual(dash.frames, {})
            self.assertEqual(dash.snapshot()["samples"], [])
            self.assertNotIn("policy_latency_ms", dash.snapshot())

    def test_replay_does_not_write_run(self):
        from autodagger_libero.view_run import open_run

        with tempfile.TemporaryDirectory() as root:
            self.episode(root)
            files = {
                str(p): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()
            }
            dashboard = open_run(root, port=0)
            try:
                state = dashboard.snapshot()
                self.assertEqual(state["score_source"], "simulated")
                self.assertTrue(state["status"].startswith("replay"))
            finally:
                dashboard.close()
            self.assertEqual(
                files,
                {str(p): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()},
            )


if __name__ == "__main__":
    unittest.main()
