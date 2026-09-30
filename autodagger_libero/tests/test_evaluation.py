"""Independent rollout and scoring invariants; run under the hw LIBERO Python."""

import dataclasses
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from autodagger_libero.evaluation_common import (
    EvaluationConfig,
    EvaluationStore,
    prefix_inputs,
    score_protocol,
)
from autodagger_libero.core import atomic_json
from autodagger_libero.evaluate_success import (
    owned_service,
    request_score,
    rollout_episode,
    score_one,
)
from autodagger_libero.evaluation_report import metrics, report
from autodagger_libero.tests.test_collection import Env, Policy


class EvaluationTests(unittest.TestCase):
    def test_supported_suite_configs_and_isolated_resources(self):
        package = Path(__file__).parents[1]
        configs = [
            EvaluationConfig.load(package / ("evaluation_%s.json" % suite))
            for suite in ("object", "goal", "10")
        ]
        self.assertEqual(
            {c.suite for c in configs}, {"libero_object", "libero_goal", "libero_10"}
        )
        self.assertEqual(len({c.output for c in configs}), 3)
        self.assertEqual(
            len({p for c in configs for p in (c.student_port, c.score_port)}), 6
        )
        for c in configs:
            self.assertEqual(c.episodes * len(c.task_ids), 200)
            budgets = {"libero_object": 280, "libero_goal": 300, "libero_10": 520}
            self.assertEqual(
                (c.control_freq, c.max_steps, c.seed), (20, budgets[c.suite], 7)
            )

    def test_pipeline_drains_final_commit_and_ignores_pending(self):
        from autodagger_libero.evaluation_pipeline import drain_original
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as root:
            c, store, _ = self.episode(root)
            store.begin("unfinished", {"episode_id": "unfinished"})
            worker = Mock(pid=42)
            worker.is_alive.side_effect = [True, False]
            with patch(
                "autodagger_libero.evaluation_pipeline.score_batch"
            ) as batch, patch("autodagger_libero.evaluation_pipeline.time.sleep"):
                drain_original(c, worker, {"sha256": "test"}, 5)
            batch.assert_called_once()
            self.assertEqual(
                [m["episode_id"] for _, m, _ in batch.call_args.args[3]], ["episode"]
            )
            self.assertEqual(
                json.loads((Path(root) / "pipeline_status.json").read_text())[
                    "original_attempted"
                ],
                1,
            )

    def test_score_batch_preserves_prior_index_entries(self):
        from autodagger_libero.evaluate_success import score_batch

        with tempfile.TemporaryDirectory() as root:
            c, store, _ = self.episode(root)
            directory = Path(root) / "scores/original"
            directory.mkdir(parents=True)
            atomic_json(
                directory / "index.json",
                {
                    "model_identity": "test",
                    "protocol": score_protocol(c),
                    "episodes": {"earlier": {"complete": True}},
                },
            )
            with patch("autodagger_libero.evaluate_success.requests.get") as get, patch(
                "autodagger_libero.evaluate_success.request_score",
                return_value={"success_probability": 0.8, "progress": 0.7},
            ):
                get.return_value.json.return_value = {
                    "model_path": c.score_models["original"]
                }
                score_batch(
                    c, "original", {"sha256": "test"}, list(store.episodes(verify=True))
                )
            index = json.loads((directory / "index.json").read_text())
            self.assertEqual(set(index["episodes"]), {"earlier", "episode"})

    def config(self, root, **changes):
        c = EvaluationConfig.load(Path(__file__).parents[1] / "evaluation.json")
        return dataclasses.replace(
            c, output=str(root), task_ids=[0], episodes=1, max_steps=12, **changes
        )

    def episode(self, root, env=None, policy=None):
        c = self.config(root)
        store = EvaluationStore(root)
        identity = dict(
            episode_id="episode",
            task_id=0,
            task="test",
            suite="libero_spatial",
            initial_state_id=0,
            seed=7,
        )
        result = rollout_episode(
            env or Env(length=6), None, policy or Policy(0.1), c, store, identity
        )
        return c, store, result

    def test_success_and_true_terminal_alignment(self):
        with tempfile.TemporaryDirectory() as root:
            c, store, meta = self.episode(root)
            self.assertTrue(meta["env_success"])
            directory, _, _ = next(store.episodes(verify=True))
            with np.load(directory / "trajectory.npz") as data:
                self.assertEqual(data["action"].shape, (6, 7))
                self.assertTrue((data["collect"] == "rollout").all())
                self.assertEqual(int(data["image"][-1, 0, 0, 0]), 5)
            with np.load(directory / "terminal.npz") as data:
                self.assertEqual(int(data["image"][0, 0, 0]), 6)
                self.assertEqual(int(data["step"]), 6)
            inputs = list(prefix_inputs(directory, 6, c))
            self.assertEqual([end for end, _, _ in inputs], [0, 1, 6])
            self.assertEqual(inputs[-1][1], list(range(7)))
            self.assertEqual(inputs[-1][2][-1, 0, 0, 0], 6)

    def test_budget_failure_is_not_interruption(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, meta = self.episode(root, env=Env(length=100, success=False))
            self.assertEqual(meta["end_reason"], "step_limit")
            self.assertIs(meta["env_success"], False)
            self.assertEqual(meta["steps"], 12)
            self.assertTrue(store.completed("episode"))

    def test_timeout_retained_and_retried_as_new_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            policy = Policy(0.1)
            with patch.object(policy, "infer", side_effect=TimeoutError("offline")):
                _, store, bad = self.episode(root, policy=policy)
            self.assertIsNone(bad["env_success"])
            self.assertFalse(store.completed("episode"))
            _, store, good = self.episode(root)
            self.assertTrue(good["env_success"])
            self.assertTrue(
                (Path(root) / "episodes/episode/attempt_0001/metadata.json").exists()
            )
            self.assertEqual(next(store.episodes())[0].name, "attempt_0002")

    def test_process_exit_journal_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            store = EvaluationStore(root)
            writer, name = store.begin("episode", {"episode_id": "episode", "steps": 0})
            store.begin("episode", {"episode_id": "episode", "steps": 0})
            recovered = json.loads((writer.root / name / "metadata.json").read_text())
            self.assertEqual(recovered["end_reason"], "interrupted")
            self.assertIsNone(recovered["env_success"])

    def test_episode_reset_discards_action_queue(self):
        with tempfile.TemporaryDirectory() as root:
            policy = Policy(0.1)
            self.episode(Path(root) / "first", env=Env(length=2), policy=policy)
            policy.value = 0.8
            _, store, _ = self.episode(
                Path(root) / "second", env=Env(length=2), policy=policy
            )
            with np.load(next(store.episodes())[0] / "trajectory.npz") as data:
                np.testing.assert_allclose(data["action"], 0.8)
            self.assertEqual(policy.resets, 2)

    def test_scoring_resumes_missing_prefix_and_keeps_error(self):
        with tempfile.TemporaryDirectory() as root:
            c, store, _ = self.episode(root)
            directory, meta, identity = next(store.episodes())
            scores = Path(root) / "scoring"
            scores.mkdir()
            value = {"success_probability": 0.8, "progress": 0.7}
            with patch(
                "autodagger_libero.evaluate_success.request_score",
                side_effect=[TimeoutError("down"), dict(value), dict(value)],
            ):
                path, result = score_one(
                    c, directory, meta, identity, "original", "hash", scores
                )
            self.assertFalse(result["complete"])
            self.assertNotIn("0", result["predictions"])
            with patch(
                "autodagger_libero.evaluate_success.request_score",
                return_value=dict(value),
            ) as request:
                _, result = score_one(
                    c, directory, meta, identity, "original", "hash", scores
                )
                self.assertEqual(request.call_count, 1)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["errors"]), 1)
            with patch("autodagger_libero.evaluate_success.request_score") as request:
                score_one(c, directory, meta, identity, "original", "hash", scores)
                request.assert_not_called()

    def test_response_raw_and_no_label_sent(self):
        with tempfile.TemporaryDirectory() as root:
            raw = {
                "outputs_success": {"success_probs": [[0.1, 0.9]]},
                "outputs_progress": {"progress_pred": [[0.2, 0.8]]},
                "auxiliary": {"retained": True},
            }
            with patch("autodagger_libero.evaluate_success.requests.post") as post:
                post.return_value.json.return_value = raw
                value = request_score(
                    self.config(root),
                    np.zeros((2, 256, 256, 3), np.uint8),
                    "instruction",
                )
                self.assertEqual(value["raw_response"], raw)
                body = json.loads(post.call_args.kwargs["data"]["sample_0"])
                self.assertNotIn("env_success", json.dumps(body))
                self.assertEqual(value["success_probability"], 0.9)
            with patch("autodagger_libero.evaluate_success.requests.post") as post:
                post.return_value.json.return_value = {
                    "outputs_success": {"success_probs": [[]]}
                }
                with self.assertRaises((KeyError, ValueError)):
                    request_score(
                        self.config(root), np.zeros((2, 256, 256, 3), np.uint8), "test"
                    )
                self.assertEqual(post.call_count, 2)

    def test_metrics_ties_missing_class_and_strict_threshold(self):
        m = metrics([1, 1, 0, 0], [0.9, 0.4, 0.5, 0.1])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 0, 1, 2))
        self.assertAlmostEqual(m["roc_auc"], 0.75)
        self.assertAlmostEqual(m["pr_auc"], 19 / 24)
        self.assertAlmostEqual(m["average_precision"], 5 / 6)
        self.assertEqual(metrics([1, 0], [0.5, 0.5])["roc_auc"], 0.5)
        for labels in ([1, 1], [0, 0], []):
            result = metrics(labels, [0.9] * len(labels))
            self.assertIsNone(result["roc_auc"])
            self.assertIsNone(result["pr_auc"])
            self.assertIsNone(result["balanced_accuracy"])

    def test_report_missing_scores_not_failures(self):
        with tempfile.TemporaryDirectory() as root:
            c, store, _ = self.episode(root)
            with patch("autodagger_libero.evaluation_report.figures"):
                result = report(c)
            self.assertEqual(result["paired_final"], 0)
            self.assertEqual(result["models"]["original"]["overall"]["fn"], 0)
            self.assertFalse(result["complete"])

    def test_occupied_port_never_spawns_or_kills(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            c = self.config(root, student_port=sock.getsockname()[1])
            with patch("autodagger_libero.evaluate_success.subprocess.Popen") as spawn:
                with self.assertRaises(OSError):
                    with owned_service(c, "student"):
                        pass
                spawn.assert_not_called()

    def test_closed_service_port_can_be_reused(self):
        # Make the server actively close a real connection, leaving TIME_WAIT.
        with tempfile.TemporaryDirectory() as root, socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                connection, _ = listener.accept()
                connection.close()
                self.assertEqual(client.recv(1), b"")
            listener.close()
            c = self.config(root, student_port=port)
            with patch(
                "autodagger_libero.evaluate_success.subprocess.Popen",
                side_effect=RuntimeError("preflight passed"),
            ) as spawn:
                with self.assertRaisesRegex(RuntimeError, "preflight passed"):
                    with owned_service(c, "student"):
                        pass
                spawn.assert_called_once()

    def test_paired_report_and_input_identity_guard(self):
        with tempfile.TemporaryDirectory() as root:
            c, store, _ = self.episode(root)
            directory, meta, identity = next(store.episodes())
            paths = {}
            for model in c.score_models:
                output = Path(root) / "scores" / model
                output.mkdir(parents=True)
                with patch(
                    "autodagger_libero.evaluate_success.request_score",
                    return_value={"success_probability": 0.8, "progress": 0.7},
                ):
                    path, _ = score_one(
                        c, directory, meta, identity, model, model, output
                    )
                paths[model] = path
                atomic_json(
                    output / "index.json",
                    {
                        "model_identity": model,
                        "protocol": score_protocol(c),
                        "episodes": {
                            "episode": {
                                "file": path.name,
                                "trajectory_identity": identity,
                                "complete": True,
                            }
                        },
                    },
                )
            with patch("autodagger_libero.evaluation_report.figures"):
                result = report(c)
                self.assertTrue(result["complete"])
                self.assertEqual(result["paired_final"], 1)
                self.assertEqual(result["paired"]["original"]["overall"]["tp"], 1)
                altered = json.loads(paths["finetuned"].read_text())
                altered["predictions"][str(meta["steps"])][
                    "input_sha256"
                ] = "different-input"
                atomic_json(paths["finetuned"], altered)
                with self.assertRaisesRegex(ValueError, "identical final inputs"):
                    report(c)


if __name__ == "__main__":
    unittest.main()
