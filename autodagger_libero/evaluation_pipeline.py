"""Collect in a child process while scoring only atomically committed episodes."""

import dataclasses
import json
import logging
import multiprocessing
from pathlib import Path
import signal
import time

from .core import atomic_json
from .evaluation_common import EvaluationConfig, EvaluationStore, model_identity
from .evaluate_success import (
    initialize,
    now,
    owned_service,
    rollout,
    score,
    score_batch,
)


def stop(_signum, _frame):
    raise KeyboardInterrupt()


def collect_worker(config):
    # A spawned process owns LIBERO, its policy connection, and its student server.
    # No score or success prediction ever crosses back into the collector.
    signal.signal(signal.SIGTERM, stop)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    rollout(config)


def drain_original(config, worker, identity, poll_seconds):
    store = EvaluationStore(config.output)
    attempted = set()
    while True:
        # Observe process exit BEFORE scanning commits to avoid losing its final
        # episode if it commits and exits during this scan.
        finished = not worker.is_alive()
        batch = list(store.episodes(verify=True, exclude=attempted))
        if batch:
            score_batch(config, "original", identity, batch)
            attempted.update(meta["episode_id"] for _, meta, _ in batch)
            atomic_json(
                Path(config.output) / "pipeline_status.json",
                {
                    "stage": "collect_and_score_original",
                    "updated_at": now(),
                    "original_attempted": len(attempted),
                    "collector_pid": worker.pid,
                },
            )
        if finished:
            return
        time.sleep(poll_seconds)


def run(path):
    import fcntl

    path = Path(path).resolve()
    settings = json.loads(path.read_text())
    if set(settings) != {"evaluation_config", "score_gpu", "poll_seconds"}:
        raise ValueError(
            "Pipeline JSON requires evaluation_config, score_gpu, poll_seconds"
        )
    if (
        not isinstance(settings["score_gpu"], str)
        or not settings["score_gpu"].isdigit()
    ):
        raise ValueError("score_gpu must be a single GPU index string")
    if (
        not isinstance(settings["poll_seconds"], (int, float))
        or not 0 < settings["poll_seconds"] <= 60
    ):
        raise ValueError("poll_seconds must be in (0,60]")
    config = EvaluationConfig.load(path.parent / settings["evaluation_config"])
    scoring = dataclasses.replace(config, gpu=settings["score_gpu"])
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    signal.signal(signal.SIGTERM, stop)
    with (root / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(config)
        saved = root / "pipeline_config.json"
        if saved.exists() and json.loads(saved.read_text()) != settings:
            raise ValueError(
                "Pipeline configuration changed; use a new output directory"
            )
        atomic_json(saved, settings)
        worker = multiprocessing.get_context("spawn").Process(
            target=collect_worker, args=(config,)
        )
        worker.start()
        failure = None
        try:
            identity = model_identity(config.score_models["original"])
            with owned_service(scoring, "robometer", "original"):
                drain_original(scoring, worker, identity, settings["poll_seconds"])
            worker.join()
            atomic_json(
                root / "pipeline_status.json",
                {
                    "stage": "score_finetuned",
                    "updated_at": now(),
                    "collector_exitcode": worker.exitcode,
                },
            )
            score(scoring, models={"finetuned"})
            if worker.exitcode != 0:
                raise RuntimeError(
                    "Collector failed with exit code %s; committed data retained"
                    % worker.exitcode
                )
        except BaseException as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            if worker.is_alive():
                worker.terminate()
                # Let the collector commit its interrupted attempt and stop the
                # student it owns; never kill unrelated services on the host.
                worker.join()
            from .evaluation_report import report

            summary = report(config)
            atomic_json(
                root / "pipeline_status.json",
                {
                    "stage": (
                        "complete"
                        if summary["complete"] and failure is None
                        else "incomplete"
                    ),
                    "updated_at": now(),
                    "error": failure,
                    "collector_exitcode": worker.exitcode,
                    "completed_rollouts": summary["completed_rollouts"],
                    "paired_final": summary["paired_final"],
                },
            )
        if not summary["complete"]:
            raise RuntimeError("Evaluation incomplete; see report/coverage.json")
