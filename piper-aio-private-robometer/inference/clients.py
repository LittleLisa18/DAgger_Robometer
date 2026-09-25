import io
import json
import queue
import threading
import time
import urllib.request

import numpy as np
from PIL import Image
from openpi_client import image_tools, websocket_client_policy


class _DashboardPreviewSender:
    """Best-effort preview transport that never blocks policy inference."""

    def __init__(self, url: str, size=(480, 640), quality: int = 90, fps: float = 5.0) -> None:
        self.url = url
        self.size = size
        self.quality = quality
        self.min_interval = 1.0 / fps
        self.last_submit = 0.0
        self.lock = threading.Lock()
        self.frames = queue.Queue(maxsize=1)
        threading.Thread(target=self._run, name="dashboard-preview", daemon=True).start()

    def submit(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self.lock:
            if now - self.last_submit < self.min_interval:
                return
            self.last_submit = now
        try:
            self.frames.put_nowait(np.asarray(frame).copy())
        except queue.Full:
            # Keeping control latency low is more important than showing every frame.
            pass

    def _run(self) -> None:
        height, width = self.size
        while True:
            frame = self.frames.get()
            try:
                frame = image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, height, width))
                output = io.BytesIO()
                Image.fromarray(frame).save(output, format="JPEG", quality=self.quality, subsampling=0)
                request = urllib.request.Request(
                    self.url, data=output.getvalue(), method="POST", headers={"Content-Type": "image/jpeg"}
                )
                with urllib.request.urlopen(request, timeout=0.5):
                    pass
            except Exception:
                # Preview failures must not affect policy requests or robot control.
                pass
            finally:
                self.frames.task_done()


def _random_observation(image_size, prompt) -> dict:
    height, width = image_size
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, height, width), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, height, width), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, height, width), dtype=np.uint8),
        },
        "prompt": prompt,
    }


def _random_observation_rtc(image_size, prompt) -> dict:
    height, width = image_size
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, height, width), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, height, width), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, height, width), dtype=np.uint8),
        },
        "prompt": prompt,
        "action_prefix": np.ones((4, 14)),
        "delay": np.array(4),
    }


class OpenpiClient:
    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        image_size=(224, 224),
        dashboard_port: int | None = None,
    ) -> None:

        # build client to connect server policy
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.image_size = image_size
        self.prompt = prompt
        self.dashboard_reset_url = (
            f"http://{host}:{dashboard_port}/api/reset" if dashboard_port is not None else None
        )
        self.dashboard_standby_url = (
            f"http://{host}:{dashboard_port}/api/standby" if dashboard_port is not None else None
        )
        self.dashboard_state_url = (
            f"http://{host}:{dashboard_port}/api/state" if dashboard_port is not None else None
        )
        self.dashboard_pause_url = (
            f"http://{host}:{dashboard_port}/api/pause" if dashboard_port is not None else None
        )
        self.dashboard_resume_url = (
            f"http://{host}:{dashboard_port}/api/resume" if dashboard_port is not None else None
        )
        self._failure_latched = False
        self._last_failure_check = 0.0
        self.preview = (
            _DashboardPreviewSender(f"http://{host}:{dashboard_port}/api/preview")
            if dashboard_port is not None
            else None
        )

    def standby_robometer(self) -> bool:
        """Clear and freeze Robometer until the operator starts the episode."""
        if self.dashboard_standby_url is None:
            return True
        try:
            request = urllib.request.Request(self.dashboard_standby_url, data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=1.0):
                self._failure_latched = False
                self._last_failure_check = 0.0
                return True
        except Exception as exc:
            print(f"Warning: failed to put live Robometer in standby: {exc}")
            return False

    def reset_episode(self) -> bool:
        """Reset the live Robometer timeline without issuing policy inference."""
        if self.dashboard_reset_url is None:
            return True
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(self.dashboard_reset_url, data=b"", method="POST")
                with urllib.request.urlopen(request, timeout=1.0):
                    self._failure_latched = False
                    self._last_failure_check = 0.0
                    return True
            except Exception as exc:
                print(f"Warning: failed to reset live Robometer episode ({attempt}/3): {exc}")
        return False

    def pause_robometer(self) -> bool:
        """Freeze live Robometer sampling after a detected failure."""
        return self._set_robometer_paused(self.dashboard_pause_url, paused=True)

    def resume_robometer(self) -> bool:
        """Resume live Robometer sampling without resetting the episode."""
        return self._set_robometer_paused(self.dashboard_resume_url, paused=False)

    def _set_robometer_paused(self, url: str | None, paused: bool) -> bool:
        if url is None:
            return True
        try:
            request = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=1.0):
                self._failure_latched = paused
                self._last_failure_check = 0.0
                return True
        except Exception as exc:
            action = "pause" if paused else "resume"
            print(f"Warning: failed to {action} live Robometer: {exc}")
            return False

    def consume_failure_event(self, min_interval: float = 0.1) -> bool:
        """Return True once when the live Robometer first detects failure."""
        if self.dashboard_state_url is None:
            return False

        now = time.monotonic()
        if now - self._last_failure_check < min_interval:
            return False
        self._last_failure_check = now

        try:
            with urllib.request.urlopen(self.dashboard_state_url, timeout=0.2) as response:
                failure = bool(json.loads(response.read()).get("failure", False))
        except Exception:
            # Dashboard availability is handled independently from policy inference.
            return False

        if not failure:
            self._failure_latched = False
            return False
        if self._failure_latched:
            return False

        self._failure_latched = True
        return True

    def get_robometer_state(self, timeout: float = 0.2) -> dict | None:
        """Return the live Robometer dashboard state, or None when unavailable."""
        if self.dashboard_state_url is None:
            return None
        try:
            with urllib.request.urlopen(self.dashboard_state_url, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception:
            # Monitoring availability must never interrupt robot control.
            return None

    def _build_observation(self, payload) -> dict:
        images = [payload["top"], payload["left"], payload["right"]]
        if self.preview is not None:
            self.preview.submit(images[0])
        height, width = self.image_size
        images = [
            image_tools.convert_to_uint8(image_tools.resize_with_pad(img, height, width))
            for img in images
        ]
        images = [img.transpose(2, 0, 1) for img in images]

        observation = {
            "state": payload["state"],
            "images": {
                "cam_high": images[0],
                "cam_left_wrist": images[1],
                "cam_right_wrist": images[2],
            },
            "prompt": payload["instruction"],
        }

        if "step" in payload:
            observation["step"] = payload["step"]

        if "action_prefix" in payload and payload["action_prefix"] is not None:
            observation["action_prefix"] = payload["action_prefix"]
            observation["delay"] = payload["delay"]

        return observation

    def predict_action(self, payload) -> np.ndarray:
        observation = self._build_observation(payload)
        response = self.client.infer(observation)
        return response["actions"]

    def predict_action_streaming(self, payload, on_actions_ready=None) -> np.ndarray:
        """Streaming prediction — calls *on_actions_ready(step, indices, actions)*
        for each group of action indices that finish denoising early.

        Returns the full action chunk once inference completes.
        """
        observation = self._build_observation(payload)
        response = self.client.infer_streaming(observation, on_actions_ready=on_actions_ready)
        if response is not None:
            return response["actions"]

    def warmup(self, rtc: bool = False, streaming: bool = False) -> None:
        if streaming:
            self.client.infer_streaming(_random_observation(self.image_size, self.prompt))
            if rtc:
                self.client.infer_streaming(_random_observation_rtc(self.image_size, self.prompt))
        else:
            self.client.infer(_random_observation(self.image_size, self.prompt))
            if rtc:
                self.client.infer(_random_observation_rtc(self.image_size, self.prompt))


class XvlaClient(OpenpiClient):
    """X-VLA HTTP client that keeps the existing optional Robometer integration."""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        chunk_size: int,
        dashboard_port: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("X-VLA chunk_size must be positive")

        import json_numpy
        import requests

        self._json_numpy = json_numpy
        self._requests = requests
        self.url = f"http://{host}:{port}/act"
        self.prompt = prompt
        self.chunk_size = chunk_size

        # Keep the same optional dashboard contract as OpenpiClient so all
        # existing Robometer monitoring, reset, pause, and preview paths work.
        self.dashboard_reset_url = (
            f"http://{host}:{dashboard_port}/api/reset" if dashboard_port is not None else None
        )
        self.dashboard_standby_url = (
            f"http://{host}:{dashboard_port}/api/standby" if dashboard_port is not None else None
        )
        self.dashboard_state_url = (
            f"http://{host}:{dashboard_port}/api/state" if dashboard_port is not None else None
        )
        self.dashboard_pause_url = (
            f"http://{host}:{dashboard_port}/api/pause" if dashboard_port is not None else None
        )
        self.dashboard_resume_url = (
            f"http://{host}:{dashboard_port}/api/resume" if dashboard_port is not None else None
        )
        self._failure_latched = False
        self._last_failure_check = 0.0
        self.preview = (
            _DashboardPreviewSender(f"http://{host}:{dashboard_port}/api/preview")
            if dashboard_port is not None
            else None
        )
        self.reset()

    def reset(self) -> None:
        """Reset X-VLA predicted-proprio recursion for a new episode."""
        self.pred_proprio = None

    def reset_episode(self) -> bool:
        self.reset()
        return super().reset_episode()

    def _request_actions(self, payload, proprio) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float32)
        if proprio.shape != (14,) or not np.isfinite(proprio).all():
            raise ValueError("X-VLA proprio must contain 14 finite joint values")

        if self.preview is not None:
            self.preview.submit(payload["top"])

        query = {
            "proprio": self._json_numpy.dumps(proprio),
            "image0": self._json_numpy.dumps(payload["top"]),
            "image1": self._json_numpy.dumps(payload["left"]),
            "image2": self._json_numpy.dumps(payload["right"]),
            "language_instruction": payload["instruction"],
            "steps": 10,
            "domain_id": 20,
        }
        response = self._requests.post(self.url, json=query, timeout=60)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or "action" not in body:
            raise ValueError("X-VLA response is missing 'action'")

        actions = np.asarray(body["action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"X-VLA actions must have shape (T, 14), got {actions.shape}")
        if actions.shape[0] < self.chunk_size:
            raise ValueError(
                f"X-VLA action chunk length {actions.shape[0]} is smaller than {self.chunk_size}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("X-VLA actions must contain only finite values")
        return actions[: self.chunk_size].copy()

    def predict_action(self, payload) -> np.ndarray:
        proprio = payload["state"] if self.pred_proprio is None else self.pred_proprio
        actions = self._request_actions(payload, proprio)
        # Match the upstream X-VLA behavior: recurse from the raw model output
        # before task-specific action postprocessing is applied by the runner.
        # before task-specific action postprocessing is applied by the runner.
        self.pred_proprio = actions[-1].copy()
        return actions

    def warmup(self) -> None:
        payload = {
            name: np.zeros((480, 640, 3), dtype=np.uint8)
            for name in ("top", "left", "right")
        }
        payload["instruction"] = self.prompt
        self._request_actions(payload, np.zeros(14, dtype=np.float32))


class VlaAdapterClient(XvlaClient):
    """VLA-Adapter AgileX HTTP client for a single right-arm policy."""

    def _request_actions(self, payload, proprio) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float32)
        if proprio.shape != (14,) or not np.isfinite(proprio).all():
            raise ValueError("VLA-Adapter proprio must contain 14 finite joint values")

        if self.preview is not None:
            self.preview.submit(payload["top"])

        # The VLA-Adapter server accepts a json_numpy-encoded observation and
        # selects the right-arm state internally when given a 14-D state.
        query = {
            "state": proprio,
            "full_image": np.asarray(payload["top"], dtype=np.uint8),
            "wrist_image": np.asarray(payload["right"], dtype=np.uint8),
            "instruction": payload["instruction"],
        }
        response = self._requests.post(
            self.url,
            json={"encoded": self._json_numpy.dumps(query)},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(
                f"VLA-Adapter server returned HTTP {response.status_code}: {response.text}"
            )

        body = response.json()
        if isinstance(body, str):
            body = self._json_numpy.loads(body)
        right_actions = np.asarray(body, dtype=np.float32)
        if right_actions.ndim != 2 or right_actions.shape[1] != 7:
            raise ValueError(
                f"VLA-Adapter actions must have shape (T, 7), got {right_actions.shape}"
            )
        if right_actions.shape[0] < self.chunk_size:
            raise ValueError(
                "VLA-Adapter action chunk length "
                f"{right_actions.shape[0]} is smaller than {self.chunk_size}"
            )
        if not np.isfinite(right_actions).all():
            raise ValueError("VLA-Adapter actions must contain only finite values")

        right_actions = right_actions[: self.chunk_size]
        left_actions = np.broadcast_to(proprio[:7], right_actions.shape).copy()
        return np.concatenate((left_actions, right_actions), axis=1)
