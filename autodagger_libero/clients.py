"""Bounded network clients; retries never advance the simulator."""

import io
import json
import time
import numpy as np
import requests


class PolicyClient:
    def __init__(self, url, timeout=60, retries=1):
        self.url, self.timeout, self.retries = url, timeout, retries
        self.ws = None
        self.metadata = None
        self.connect()

    def connect(self):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy

        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.close()
        self.ws = connect(
            self.url,
            compression=None,
            max_size=16 * 1024 * 1024,
            open_timeout=self.timeout,
            close_timeout=2,
        )
        try:
            self.metadata = self.codec.unpackb(self.ws.recv(timeout=self.timeout))
        except Exception:
            self.close()
            raise

    def infer(self, observation):
        for attempt in range(self.retries + 1):
            try:
                if self.ws is None:
                    self.connect()
                self.ws.send(self.packer.pack(observation))
                reply = self.ws.recv(timeout=self.timeout)
                if isinstance(reply, str):
                    raise RuntimeError(reply)
                return self.codec.unpackb(reply)
            except Exception:
                self.close()
                if attempt == self.retries:
                    raise

    def reset(self):
        # New connection resets wrapper session; policies also reset per request.
        self.connect()

    def close(self):
        if self.ws is not None:
            ws = self.ws
            self.ws = None
            try:
                ws.close()
            except Exception:
                # Cleanup must not hide the inference/handshake failure.
                pass


class RobometerClient:
    def __init__(self, url, timeout=60, retries=1):
        self.url, self.timeout, self.retries = url.rstrip("/"), timeout, retries

    def health(self):
        r = requests.get(self.url + "/health", timeout=self.timeout)
        r.raise_for_status()
        r = requests.get(self.url + "/model_info", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def score(self, frames, task):
        stream = io.BytesIO()
        np.save(stream, frames, allow_pickle=False)
        sample = {
            "sample_type": "progress",
            "trajectory": {
                "frames": {"__numpy_file__": "sample_0_trajectory_frames"},
                "frames_shape": list(frames.shape),
                "task": task,
                "id": "autodagger",
                "metadata": {"subsequence_length": len(frames)},
                "video_embeddings": None,
            },
        }
        started = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(
                    self.url + "/evaluate_batch_npy",
                    data={"sample_0": json.dumps(sample), "use_frame_steps": "false"},
                    files={
                        "sample_0_trajectory_frames": (
                            "frames.npy",
                            stream.getvalue(),
                            "application/octet-stream",
                        )
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                out = r.json()
                progress = np.asarray(
                    out["outputs_progress"]["progress_pred"][0]
                ).reshape(-1)
                success = np.asarray(
                    out["outputs_success"]["success_probs"][0]
                ).reshape(-1)
                # The complete traces are persisted as strict JSON, so validating
                # only their final elements would still allow unsavable NaNs.
                if (
                    not progress.size
                    or not success.size
                    or not np.isfinite(progress).all()
                    or not np.isfinite(success).all()
                    or not ((success >= 0) & (success <= 1)).all()
                ):
                    raise ValueError("Invalid Robometer output")
                p, s = float(progress[-1]), float(success[-1])
                return {
                    "progress": p,
                    "success_probability": s,
                    "latency_ms": (time.monotonic() - started) * 1000,
                    "progress_trace": progress.tolist(),
                    "success_trace": success.tolist(),
                }
            except Exception:
                if attempt == self.retries:
                    raise
