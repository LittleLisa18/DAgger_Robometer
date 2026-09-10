"""Real LIBERO smoke with two real policies; Robometer is explicitly simulated.

Requires both model servers. Does not validate Robometer or full AutoDAgger.
"""

import json
import os
import time
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from autodagger_libero.core import Config, RunStore
from autodagger_libero.collect import run_episode, LIMITS
from autodagger_libero.clients import PolicyClient
from autodagger_libero.dashboard import Dashboard
from autodagger_libero.tests.test_collection import Monitor

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    ),
    camera_heights=256,
    camera_widths=256,
)
env.seed(7)
student = teacher = dashboard = None
try:
    student = PolicyClient("ws://127.0.0.1:8100", 60, 1)
    teacher_url = os.environ.get("TEACHER_URL", "ws://127.0.0.1:8101")
    teacher = PolicyClient(teacher_url, 60, 1)
    root = Path("autodagger_libero/validation") / (
        "simulation_" + str(int(time.time()))
    )
    config = Config(
        output=str(root),
        teacher_url=teacher_url,
        force_teacher_step=5,
        monitor_every=5,
        dashboard_port=8088,
    )
    store = RunStore(
        root,
        config,
        {
            "student": student.metadata,
            "teacher": teacher.metadata,
            "robometer": "SIMULATED TEST ONLY",
        },
    )
    dashboard = Dashboard(store, config)
    dashboard.start()
    LIMITS["libero_spatial"] = (
        20  # test budget only, no modification of production limits
    )
    result = run_episode(
        env,
        suite.get_task_init_states(0)[0],
        str(task.language),
        "real_env_test",
        config,
        student,
        teacher,
        Monitor(success=0.1),
        store,
        dashboard,
        {"task_id": 0, "initial_state_id": 0, "seed": 7},
    )
    assert result["steps"] == 20, result
    assert result["takeover_step"] == 5, result
    assert result["end_reason"] != "interrupted", result
    assert result["test_only"] and not result["accepted_for_distillation"]
    print(
        json.dumps(
            {
                "result": "PASS",
                "steps": result["steps"],
                "takeover_step": result["takeover_step"],
                "env_success": result["env_success"],
                "run": str(root),
                "robometer": "SIMULATED",
            }
        ),
        flush=True,
    )
    # Give a separate hw browser process time to capture the actual rendered frame.
    time.sleep(int(os.environ.get("SCREENSHOT_WAIT_SECONDS", "0")))
finally:
    env.close()
    if student:
        student.close()
    if teacher:
        teacher.close()
    if dashboard:
        dashboard.close()
