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
        dashboard_port: int | None = 8080,
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
