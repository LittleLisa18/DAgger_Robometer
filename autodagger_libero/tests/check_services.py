"""Start a test student on hw and connect to the user's already-running teacher."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlsplit

root = Path("/home/ma-user/work/users/luyuxiang/code/DAgger_Robometer")
artifacts = root / "autodagger_libero/validation"
artifacts.mkdir(parents=True, exist_ok=True)
processes = []
logs = []


def listening(port, host="127.0.0.1"):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


try:
    teacher_url = os.environ.get("TEACHER_URL", "ws://127.0.0.1:8101")
    teacher_address = urlsplit(teacher_url)
    teacher_port = teacher_address.port or (
        443 if teacher_address.scheme == "wss" else 80
    )
    if not teacher_address.hostname or not listening(
        teacher_port, teacher_address.hostname
    ):
        raise RuntimeError(f"Start the teacher manually first: {teacher_url}")
    for port in (8100, 8088):
        if listening(port):
            raise RuntimeError(
                f"Port {port} already in use; refusing to affect another service"
            )
    for mode, gpu, port in [("student", "1", 8100)]:
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        log = (artifacts / (mode + "_server.log")).open("w")
        logs.append(log)
        child = subprocess.Popen(
            ["bash", "autodagger_libero/run_hw.sh", mode],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append(child)
    deadline = time.monotonic() + 180
    while not listening(8100):
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("Policy server exited; see server logs")
        if time.monotonic() > deadline:
            raise TimeoutError("Policy server startup timeout")
        time.sleep(1)
    env = {
        **os.environ,
        "PYTHONPATH": str(root)
        + ":/home/ma-user/work/users/luyuxiang/code/LIBERO:"
        + "/home/ma-user/work/users/luyuxiang/code/better_openpi/packages/openpi-client/src",
        "LIBERO_CONFIG_PATH": str(root / "autodagger_libero/.libero"),
        "MUJOCO_GL": "egl",
        "SCREENSHOT_WAIT_SECONDS": "20",
        "PYTHONUNBUFFERED": "1",
        "TEACHER_URL": teacher_url,
    }
    log = (artifacts / "simulation.log").open("w")
    logs.append(log)
    sim = subprocess.Popen(
        [
            os.environ.get(
                "COLLECTOR_PYTHON",
                "/home/ma-user/work/users/luyuxiang/envs/libero/bin/python",
            ),
            "-m",
            "autodagger_libero.tests.check_simulation",
        ],
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    processes.append(sim)
    deadline = time.monotonic() + 240
    while '"result": "PASS"' not in (artifacts / "simulation.log").read_text():
        if sim.poll() is not None:
            raise RuntimeError("Simulation failed; see simulation.log")
        if time.monotonic() > deadline:
            raise TimeoutError("Simulation timeout")
        time.sleep(1)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1500})
        page.goto("http://127.0.0.1:8088")
        page.wait_for_function("document.getElementById('image').naturalWidth > 0")
        page.screenshot(
            path=str(artifacts / "dashboard-real-libero.png"), full_page=True
        )
        browser.close()
    sim.wait(timeout=40)
    assert sim.returncode == 0
    print((artifacts / "simulation.log").read_text())
finally:
    for child in reversed(processes):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    for log in logs:
        log.close()
