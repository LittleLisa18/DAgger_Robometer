"""Run with python -m autodagger_libero.collect --config ... on hw."""

from __future__ import annotations
import argparse
import collections
import dataclasses
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
from .core import Config, Gate, RunStore, sample_prefix, validate_actions
from .dashboard import Dashboard
from .clients import PolicyClient, RobometerClient

LIMITS = {
    "libero_spatial": 440,
    "libero_object": 560,
    "libero_goal": 600,
    "libero_10": 1040,
    "libero_90": 800,
}


def observation(raw):
    # LIBERO gives xyzw quaternions; both policies expect xyz, axis-angle,
    # then the two gripper joints. Rotate rendered images here exactly once.
    quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
    w = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(1 - w * w)
    angle = (
        np.zeros(3) if denominator < 1e-8 else quat[:3] * 2 * np.arccos(w) / denominator
    )
    return {
        "image": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
        "image2": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate(
            [raw["robot0_eef_pos"], angle, raw["robot0_gripper_qpos"]]
        ).astype(np.float32),
    }


def payload(obs, task, actor):
    image, wrist = obs["image"], obs["image2"]
    if actor == "teacher":
        from openpi_client import image_tools

        image, wrist = (
            image_tools.convert_to_uint8(image_tools.resize_with_pad(x, 224, 224))
            for x in (image, wrist)
        )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": obs["state"],
        "prompt": task,
    }


def run_episode(
    env,
    initial_state,
    task,
    episode_id,
    config,
    student,
    teacher,
    monitor,
    store,
    dashboard,
    identity,
):
    gate, queue, records, frames = Gate(config), collections.deque(), [], []
    actor, step = "rollout", 0
    meta = {
        **identity,
        "episode_id": episode_id,
        "task": task,
        "samples": [],
        "takeover_step": None,
        "takeover_reason": None,
        "took_over": False,
        "env_success": False,
        "robometer_success": None,
        "success_disagreement": False,
        "accepted_for_distillation": False,
        "end_reason": "interrupted",
        "error": None,
        "test_only": config.force_teacher_step is not None,
        "score_source": getattr(monitor, "source", "robometer"),
        "steps": 0,
    }
    store.checkpoint(episode_id, None, meta)
    dashboard.publish(
        episode_id=episode_id,
        suite=config.suite,
        task=task,
        actor=actor,
        step=0,
        samples=[],
        takeover_step=None,
        takeover_reason=None,
        error=None,
        status="resetting",
        test_only=meta["test_only"],
        score_source=meta["score_source"],
    )

    def score():
        selected, indices = sample_prefix(frames, config.max_frames)
        dashboard.publish(status="scoring", step=step)
        result = monitor.score(selected, task)
        result.update(step=step, sampled_steps=indices)
        meta["samples"].append(result)
        dashboard.publish(samples=meta["samples"])
        return result

    try:
        env.reset()
        raw = env.set_init_state(initial_state)
        for _ in range(10):
            raw, _, done, _ = env.step([0.0] * 6 + [-1.0])
            if done:
                raise RuntimeError("Environment terminated during settling")
        student.reset()
        teacher.reset()
        obs = observation(raw)
        frames.append(obs["image"])
        dashboard.publish(images={k: obs[k] for k in ("image", "image2")})
        while step < LIMITS[config.suite]:
            trigger = None
            if actor == "rollout":
                if (
                    config.force_teacher_step is not None
                    and step >= config.force_teacher_step
                ):
                    trigger = "forced_test"
            if step % config.monitor_every == 0:
                result = score()
                if actor == "rollout":
                    trigger = trigger or gate.update(
                        step, result["progress"], result["success_probability"]
                    )
            if trigger:
                actor = "teacher"
                # Never execute a queued student action after the handoff.
                queue.clear()
                meta.update(took_over=True, takeover_step=step, takeover_reason=trigger)
                dashboard.publish(
                    actor=actor, takeover_step=step, takeover_reason=trigger
                )
            if not queue:
                client = teacher if actor == "teacher" else student
                count = (
                    config.teacher_replan
                    if actor == "teacher"
                    else config.student_replan
                )
                dashboard.publish(status="inferring", actor=actor)
                started = time.monotonic()
                actions = client.infer(payload(obs, task, actor))["actions"]
                clipped = validate_actions(actions, count)
                queue.extend(
                    zip(clipped, np.asarray(actions, dtype=np.float32)[:count])
                )
                dashboard.publish(policy_latency_ms=(time.monotonic() - started) * 1000)
            action, policy_action = queue.popleft()
            raw, _, done, info = env.step(action.tolist())
            # obs still describes the state BEFORE this action; raw is its result.
            record = {
                **obs,
                "action": action,
                "policy_action": policy_action,
                "step": np.int64(step),
                "collect": actor,
            }
            records.append(record)
            step += 1
            obs = observation(raw)
            frames.append(obs["image"])
            meta["steps"] = step
            meta["env_success"] = bool(env.check_success())
            store.checkpoint(episode_id, record, meta)
            dashboard.publish(
                images={k: obs[k] for k in ("image", "image2")},
                status="running",
                step=step,
                actor=actor,
            )
            if done or meta["env_success"]:
                meta["end_reason"] = "environment_terminated"
                break
        else:
            meta["end_reason"] = "step_limit"
        # The last prefix includes the terminal image even between check intervals.
        final = score()
        meta["robometer_success"] = (
            final["success_probability"] > config.success_threshold
        )
        meta["success_disagreement"] = meta["robometer_success"] != meta["env_success"]
        meta["accepted_for_distillation"] = bool(
            meta["took_over"] and meta["robometer_success"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        meta.update(
            end_reason="interrupted",
            error=f"{type(exc).__name__}: {exc}",
            accepted_for_distillation=False,
        )
        logging.exception("Episode %s interrupted", episode_id)
        if isinstance(exc, KeyboardInterrupt):
            meta["stop_requested"] = True
    finally:
        meta["steps"] = len(records)
        store.save(episode_id, records, meta)
        dashboard.refresh()
        dashboard.publish(status="saved", error=meta["error"], samples=meta["samples"])
    return meta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.json"))
    )
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text())
    # The selected JSON is the sole source of collection settings. Require all
    # fields so omitted settings cannot silently fall back to Python defaults.
    if not isinstance(values, dict):
        parser.error("Config JSON must be an object")
    expected = {field.name for field in dataclasses.fields(Config)}
    missing, unknown = expected - values.keys(), values.keys() - expected
    if missing or unknown:
        parser.error(
            f"Invalid config fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    config = Config(**values)
    config.validate()
    if config.suite not in LIMITS:
        raise ValueError(f"Unsupported suite: {config.suite}")
    logging.basicConfig(level=logging.INFO)
    random.seed(config.seed)
    np.random.seed(config.seed)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[config.suite]()
    task_ids = (
        config.task_ids if config.task_ids is not None else list(range(suite.n_tasks))
    )
    if len(set(task_ids)) != len(task_ids) or any(
        i < 0 or i >= suite.n_tasks for i in task_ids
    ):
        raise ValueError("task_ids must be unique valid task indices")
    student = teacher = dashboard = None
    try:
        student = PolicyClient(config.student_url, config.timeout, config.retries)
        teacher = PolicyClient(config.teacher_url, config.timeout, config.retries)
        monitor = RobometerClient(config.robometer_url, config.timeout, config.retries)
        models = {
            "student": student.metadata,
            "teacher": teacher.metadata,
            "robometer": monitor.health(),
        }
        store = RunStore(config.output, config, models)
        dashboard = Dashboard(store, config)
        dashboard.start()
        for task_id in task_ids:
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            if config.initial_state_start + config.episodes > len(states):
                raise ValueError(
                    f"Only {len(states)} initial states available for task {task_id}"
                )
            env = OffScreenRenderEnv(
                bddl_file_name=str(
                    Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                ),
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(config.seed)
            try:
                for init_id in range(
                    config.initial_state_start,
                    config.initial_state_start + config.episodes,
                ):
                    episode_id = (
                        f"{config.suite}_t{task_id:03d}_i{init_id:03d}_s{config.seed}"
                    )
                    if store.completed(episode_id):
                        continue
                    result = run_episode(
                        env,
                        states[init_id],
                        str(task.language),
                        episode_id,
                        config,
                        student,
                        teacher,
                        monitor,
                        store,
                        dashboard,
                        {
                            "task_id": task_id,
                            "initial_state_id": init_id,
                            "seed": config.seed,
                            "suite": config.suite,
                        },
                    )
                    if result["end_reason"] == "interrupted":
                        raise RuntimeError(result["error"])
            finally:
                env.close()
        dashboard.publish(status="complete")
        print(f"Collection complete: {Path(config.output).resolve()}")
    finally:
        if student:
            student.close()
        if teacher:
            teacher.close()
        if dashboard:
            dashboard.close()


if __name__ == "__main__":
    main()
