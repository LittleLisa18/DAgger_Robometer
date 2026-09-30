"""Independent student rollout -> two offline scorers -> auditable report."""

import argparse
import collections
import contextlib
import dataclasses
import datetime
import io
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import requests

from .clients import PolicyClient
from .collect import observation, payload
from .core import atomic_json, validate_actions
from .evaluation_common import (
    EvaluationConfig,
    EvaluationStore,
    digest,
    file_hash,
    model_identity,
    prefix_inputs,
    score_protocol,
)

PACKAGE = Path(__file__).resolve().parent.parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


def initialize(config):
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment.json"
    values = dataclasses.asdict(config)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["config"] != values:
            raise ValueError("Run configuration differs; use a new output directory")
    else:
        source = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        manifest = {
            "schema_version": 1,
            "created_at": now(),
            "config": values,
            "source_sha256": source,
            "gpu_before": gpu_snapshot(),
        }
        atomic_json(path, manifest)
    invocation = root / "invocations"
    invocation.mkdir(exist_ok=True)
    atomic_json(
        invocation / (str(time.time_ns()) + ".json"),
        {
            "at": now(),
            "arguments": sys.argv,
            "source_sha256": {
                p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    return manifest


@contextlib.contextmanager
def owned_service(config, kind, model=None):
    """Only terminate the process group started here; never reuse occupied ports."""
    port = config.student_port if kind == "student" else config.score_port
    with socket.socket() as probe:
        # Sequential scorers reuse the port after shutdown; TIME_WAIT is harmless.
        # Without SO_REUSEPORT an existing listener still prevents this bind.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=config.gpu,
        HF_HOME=config.hf_home,
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    common = [
        str(PACKAGE),
        str(PACKAGE.parent / "better_openpi/packages/openpi-client/src"),
    ]
    if kind == "student":
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PACKAGE / "lerobot-0.5.1/src")] + common
        )
        service_config = Path(config.output) / "service_config.json"
        atomic_json(service_config, dataclasses.asdict(config))
        cmd = [
            config.student_python,
            "-m",
            "autodagger_libero.evaluation_student",
            "--config",
            str(service_config),
        ]
        cwd = PACKAGE
    else:
        env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE / "robometer")] + common)
        cmd = [
            config.robometer_python,
            "robometer/evals/eval_server.py",
            "model_path=" + config.score_models[model],
            "server_url=127.0.0.1",
            "server_port=" + str(port),
            "num_gpus=1",
        ]
        cwd = PACKAGE / "robometer"
    logs = Path(config.output) / "services"
    logs.mkdir(exist_ok=True)
    name = kind + ("_" + model if model else "") + "_" + str(time.time_ns())
    log = logs / (name + ".log")
    with log.open("w") as stream:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record = {
            "pid": proc.pid,
            "command": cmd,
            "started_at": now(),
            "gpu_before": gpu_snapshot(),
        }
        atomic_json(log.with_suffix(".json"), record)
        try:
            deadline = time.monotonic() + config.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Service exited; inspect " + str(log))
                ready = False
                try:
                    if kind == "student":
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            ready = True
                    else:
                        response = requests.get(
                            "http://127.0.0.1:%d/health" % port, timeout=2
                        )
                        ready = (
                            response.ok and response.json().get("status") == "healthy"
                        )
                except (OSError, requests.RequestException, ValueError):
                    pass
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Service startup timed out; inspect " + str(log))
                time.sleep(1)
            record.update(ready_at=now(), gpu_loaded=gpu_snapshot())
            atomic_json(log.with_suffix(".json"), record)
            logging.info("%s ready, owned PID=%s", name, proc.pid)
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            record.update(
                stopped_at=now(), returncode=proc.returncode, gpu_after=gpu_snapshot()
            )
            atomic_json(log.with_suffix(".json"), record)


def rollout_episode(env, initial_state, policy, config, store, identity):
    """No teacher or scorer arguments: rollout cannot depend on their decisions."""
    records, queue, terminal = [], collections.deque(), None
    meta = dict(
        identity,
        started_at=now(),
        end_reason="interrupted",
        env_success=None,
        steps=0,
        error=None,
        test_only=config.test_only,
        policy_latency_ms=[],
    )
    writer, attempt = store.begin(identity["episode_id"], meta)
    try:
        env.reset()
        raw = env.set_init_state(initial_state)
        for _ in range(config.settling_steps):
            raw, _, done, _ = env.step([0.0] * 6 + [-1.0])
            if done or env.check_success():
                raise RuntimeError("Environment already successful during settling")
        policy.reset()
        obs = observation(raw)
        terminal = obs
        for step in range(config.max_steps):
            if not queue:
                started = time.monotonic()
                request = payload(obs, identity["task"], "rollout")
                request["evaluation_seed"] = int(
                    digest(
                        {
                            "seed": config.seed,
                            "episode": identity["episode_id"],
                            "step": step,
                        }
                    )[:8],
                    16,
                )
                original = np.asarray(
                    policy.infer(request)["actions"], dtype=np.float32
                )
                actions = validate_actions(original, config.actions_per_chunk)
                queue.extend(zip(actions, original[: config.actions_per_chunk]))
                meta["policy_latency_ms"].append((time.monotonic() - started) * 1000)
            action, original = queue.popleft()
            raw, _, done, _ = env.step(action.tolist())
            record = dict(
                obs,
                action=action,
                policy_action=original,
                step=np.int64(step),
                collect="rollout",
            )
            records.append(record)
            obs = observation(raw)
            terminal = obs
            success = bool(env.check_success())
            meta["steps"] = len(records)
            writer.checkpoint(attempt, record, meta)
            if success:
                meta.update(end_reason="success", env_success=True)
                break
            if done:
                raise RuntimeError(
                    "Environment terminated without task success before budget"
                )
        else:
            meta.update(end_reason="step_limit", env_success=False)
    except (Exception, KeyboardInterrupt) as exc:
        meta.update(
            end_reason="interrupted",
            env_success=None,
            error="%s: %s" % (type(exc).__name__, exc),
        )
        if isinstance(exc, KeyboardInterrupt):
            meta["stop_requested"] = True
        logging.exception("Interrupted %s", identity["episode_id"])
    finally:
        meta.update(steps=len(records), finished_at=now())
        store.finish(identity["episode_id"], writer, attempt, records, meta, terminal)
    return meta


def rollout(config):
    from .prepare_libero import prepare

    os.environ.update(
        MUJOCO_GL="egl",
        LIBERO_CONFIG_PATH=str(Path(config.output) / "libero_paths"),
    )
    prepare(config.libero_root, os.environ["LIBERO_CONFIG_PATH"])
    sys.path.insert(0, config.libero_root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    root = Path(config.output)
    store = EvaluationStore(root)
    student_identity = model_identity(config.student_checkpoint)
    identity_path = root / "student_identity.json"
    if (
        identity_path.exists()
        and json.loads(identity_path.read_text())["checkpoint"] != student_identity
    ):
        raise ValueError("Student checkpoint changed since previous rollout")
    suite = benchmark.get_benchmark_dict()[config.suite]()
    count = 0
    with owned_service(config, "student"):
        policy = PolicyClient(
            "ws://127.0.0.1:%d" % config.student_port, config.timeout, config.retries
        )
        try:
            if (
                policy.metadata.get("actions_per_chunk") != config.actions_per_chunk
                or Path(policy.metadata.get("checkpoint", "")).resolve()
                != Path(config.student_checkpoint).resolve()
            ):
                raise ValueError("Student service identity mismatch")
            atomic_json(
                identity_path,
                {"checkpoint": student_identity, "service": policy.metadata},
            )
            for task_id in config.task_ids:
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                if config.initial_state_start + config.episodes > len(states):
                    raise ValueError("Not enough initial states for task %d" % task_id)
                env = OffScreenRenderEnv(
                    bddl_file_name=str(
                        Path(get_libero_path("bddl_files"))
                        / task.problem_folder
                        / task.bddl_file
                    ),
                    camera_heights=256,
                    camera_widths=256,
                    control_freq=config.control_freq,
                    ignore_done=True,
                )
                try:
                    for init in range(
                        config.initial_state_start,
                        config.initial_state_start + config.episodes,
                    ):
                        eid = "%s_t%03d_i%03d_s%d" % (
                            config.suite,
                            task_id,
                            init,
                            config.seed,
                        )
                        if store.completed(eid):
                            count += 1
                            continue
                        random.seed(config.seed)
                        np.random.seed(config.seed)
                        env.seed(config.seed)
                        identity = {
                            "episode_id": eid,
                            "suite": config.suite,
                            "task_id": task_id,
                            "initial_state_id": init,
                            "seed": config.seed,
                            "task": str(task.language),
                            "student_identity": student_identity["sha256"],
                            "config_identity": digest(dataclasses.asdict(config)),
                        }
                        result = rollout_episode(
                            env, states[init], policy, config, store, identity
                        )
                        if result["end_reason"] == "interrupted":
                            raise RuntimeError(result["error"])
                        count += 1
                        logging.info(
                            "ROLLOUT %d/%d %s steps=%d success=%s",
                            count,
                            len(config.task_ids) * config.episodes,
                            eid,
                            result["steps"],
                            result["env_success"],
                        )
                finally:
                    env.close()
        finally:
            policy.close()


def request_score(config, frames, task):
    """Preserve the complete response; send no environment label to the model."""
    buffer = io.BytesIO()
    np.save(buffer, frames, allow_pickle=False)
    sample = {
        "sample_type": "progress",
        "trajectory": {
            "frames": {"__numpy_file__": "sample_0_trajectory_frames"},
            "frames_shape": list(frames.shape),
            "task": task,
            "id": "student_evaluation",
            "metadata": {"subsequence_length": len(frames)},
            "video_embeddings": None,
        },
    }
    started = time.monotonic()
    for attempt in range(config.retries + 1):
        try:
            response = requests.post(
                "http://127.0.0.1:%d/evaluate_batch_npy" % config.score_port,
                data={"sample_0": json.dumps(sample), "use_frame_steps": "false"},
                files={
                    "sample_0_trajectory_frames": (
                        "frames.npy",
                        buffer.getvalue(),
                        "application/octet-stream",
                    )
                },
                timeout=config.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            success = np.asarray(
                raw["outputs_success"]["success_probs"][0], dtype=float
            ).reshape(-1)
            progress = np.asarray(
                raw["outputs_progress"]["progress_pred"][0], dtype=float
            ).reshape(-1)
            if (
                not len(success)
                or not len(progress)
                or not np.isfinite(success).all()
                or not np.isfinite(progress).all()
                or not ((0 <= success) & (success <= 1)).all()
            ):
                raise ValueError("Invalid success/progress output")
            # Reject nonfinite values even in auxiliary heads before JSON commit.
            json.dumps(raw, allow_nan=False)
            return {
                "success_probability": float(success[-1]),
                "progress": float(progress[-1]),
                "success_trace": success.tolist(),
                "progress_trace": progress.tolist(),
                "raw_response": raw,
                "latency_ms": (time.monotonic() - started) * 1000,
                "request_attempts": attempt + 1,
            }
        except Exception:
            if attempt == config.retries:
                raise


def score_one(config, directory, meta, trajectory_identity, model, model_id, root):
    protocol = score_protocol(config)
    identity = {
        "trajectory": trajectory_identity,
        "model": model_id,
        "protocol": protocol,
    }
    path = root / (digest(identity) + ".json")
    if path.exists():
        result = json.loads(path.read_text())
        if result["identity"] != identity:
            raise ValueError("Scoring identity mismatch")
    else:
        result = {
            "identity": identity,
            "episode_id": meta["episode_id"],
            "model": model,
            "predictions": {},
            "errors": [],
            "complete": False,
        }
    for end, indices, frames in prefix_inputs(directory, meta["steps"], config):
        key = str(end)
        if key in result["predictions"]:
            continue
        try:
            prediction = request_score(config, frames, meta["task"])
            prediction.update(
                step=end,
                sampled_steps=indices,
                input_sha256=digest(
                    {
                        "task": meta["task"],
                        "indices": indices,
                        "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                    }
                ),
            )
            result["predictions"][key] = prediction
        except Exception as exc:
            result["errors"].append(
                {
                    "step": end,
                    "at": now(),
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )
            logging.error(
                "Scoring %s %s step=%s: %s", model, meta["episode_id"], end, exc
            )
        atomic_json(path, result)
    expected = {str(max(0, meta["steps"] - x)) for x in config.tail_offsets}
    result.update(complete=expected <= result["predictions"].keys(), updated_at=now())
    atomic_json(path, result)
    return path, result


def score_batch(config, model, identity, episodes):
    """Append committed episodes using an already-owned, identity-checked service."""
    root = Path(config.output)
    directory = root / "scores" / model
    directory.mkdir(parents=True, exist_ok=True)
    info = requests.get(
        "http://127.0.0.1:%d/model_info" % config.score_port, timeout=config.timeout
    )
    info.raise_for_status()
    info = info.json()
    if (
        Path(info.get("model_path", "")).resolve()
        != Path(config.score_models[model]).resolve()
    ):
        raise ValueError("Robometer checkpoint identity mismatch")
    atomic_json(
        directory / ("model_" + identity["sha256"] + ".json"),
        {"checkpoint": identity, "service": info, "protocol": score_protocol(config)},
    )
    index_path = directory / "index.json"
    index = {
        "model_identity": identity["sha256"],
        "protocol": score_protocol(config),
        "episodes": {},
    }
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["model_identity"] != identity["sha256"] or index[
            "protocol"
        ] != score_protocol(config):
            raise ValueError("Existing score index identity mismatch")
    for episode, meta, trajectory_identity in episodes:
        path, result = score_one(
            config,
            episode,
            meta,
            trajectory_identity,
            model,
            identity["sha256"],
            directory,
        )
        index["episodes"][meta["episode_id"]] = {
            "file": path.name,
            "trajectory_identity": trajectory_identity,
            "complete": result["complete"],
        }
        atomic_json(index_path, index)
        logging.info(
            "SCORE %s %s complete=%s", model, meta["episode_id"], result["complete"]
        )


def score(config, models=None):
    root = Path(config.output)
    episodes = list(EvaluationStore(root).episodes(verify=True))
    if not episodes:
        raise ValueError("No committed rollout episodes to score")
    for model, checkpoint in config.score_models.items():
        if models is not None and model not in models:
            continue
        identity = model_identity(checkpoint)
        with owned_service(config, "robometer", model):
            score_batch(config, model, identity, episodes)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "stage", choices=("rollout", "score", "report", "all", "pipeline")
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("evaluation.json"))
    )
    args = parser.parse_args()
    if args.stage == "pipeline":
        from .evaluation_pipeline import run

        run(args.config)
        return
    config = EvaluationConfig.load(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # A second invocation must not launch models or write the same run concurrently.
    import fcntl

    Path(config.output).mkdir(parents=True, exist_ok=True)
    with (Path(config.output) / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(config)

        def stop(_signum, _frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, stop)
        try:
            if args.stage in ("rollout", "all"):
                rollout(config)
            if args.stage in ("score", "all"):
                score(config)
        finally:
            from .evaluation_report import report

            summary = report(config)
        if args.stage in ("all", "report", "score") and not summary["complete"]:
            raise RuntimeError("Evaluation incomplete; see report and coverage.json")


if __name__ == "__main__":
    main()
