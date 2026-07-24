#!/usr/bin/env python3
"""Serve an OpenPI policy and monitor its observations with Robometer.

This is intentionally self-contained: it does not modify the OpenPI policy server or
Robometer.  The robot-facing websocket protocol is the same as serve_policy.py.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import http
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
from pathlib import Path
import queue
import socket
import threading
import time
import traceback
import urllib.request
import uuid
from typing import Any

import numpy as np
from PIL import Image
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi_client import msgpack_numpy


LOGGER = logging.getLogger("openpi.robometer")


@dataclasses.dataclass(frozen=True)
class Settings:
    policy_config: str
    checkpoint: str
    prompt: str | None
    policy_port: int
    dashboard_host: str
    dashboard_port: int
    robometer_url: str
    camera: str
    max_frames: int
    monitor_interval: float
    timeout: float
    success_threshold: float
    failure_timeout: float | None
    output_dir: Path
    use_frame_steps: bool
    record_policy: bool
    num_steps: int | None


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", required=True, help="OpenPI training config, e.g. pi05_agilex")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint step directory (the parent of params/assets)")
    parser.add_argument("--prompt", default=None, help="Fallback task instruction")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--dashboard-host", default="0.0.0.0")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    parser.add_argument("--robometer-url", default="http://127.0.0.1:8001")
    parser.add_argument("--camera", default="cam_high", help="Observation camera used by Robometer")
    parser.add_argument("--max-frames", type=int, default=16, help="Evenly retained frames per Robometer request")
    parser.add_argument("--monitor-interval", type=float, default=1.0, help="Minimum seconds between monitor requests")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument(
        "--failure-timeout", type=float, default=None, metavar="SECONDS",
        help="Mark the episode as failed when progress has not increased for this many seconds (disabled by default)",
    )
    parser.add_argument("--output-dir", default="robometer_live_runs")
    parser.add_argument("--use-frame-steps", action="store_true")
    parser.add_argument("--record-policy", action="store_true")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional OpenPI sampling steps")
    args = parser.parse_args()
    if args.max_frames <= 0 or args.monitor_interval < 0 or args.timeout <= 0:
        parser.error("max-frames and timeout must be positive; monitor-interval must be non-negative")
    if not 0 <= args.success_threshold <= 1:
        parser.error("success-threshold must be in [0, 1]")
    if args.failure_timeout is not None and args.failure_timeout <= 0:
        parser.error("failure-timeout must be positive")
    return Settings(
        policy_config=args.policy_config,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        policy_port=args.policy_port,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        robometer_url=args.robometer_url,
        camera=args.camera,
        max_frames=args.max_frames,
        monitor_interval=args.monitor_interval,
        timeout=args.timeout,
        success_threshold=args.success_threshold,
        failure_timeout=args.failure_timeout,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        use_frame_steps=args.use_frame_steps,
        record_policy=args.record_policy,
        num_steps=args.num_steps,
    )


def _to_hwc_uint8(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"camera frame must have 3 dimensions, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _extract_frame(observation: dict[str, Any], camera: str) -> np.ndarray:
    images = observation.get("images")
    if isinstance(images, dict) and camera in images:
        return _to_hwc_uint8(images[camera])
    candidates = (camera, f"observation.images.{camera}", "observation/image", "image")
    for key in candidates:
        if key in observation:
            return _to_hwc_uint8(observation[key])
    available = list(images) if isinstance(images, dict) else []
    raise KeyError(f"camera {camera!r} not found; available image keys: {available}")


def _jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _multipart(frames: np.ndarray, task: str, use_frame_steps: bool) -> tuple[bytes, str]:
    boundary = "----openpi-robometer-" + uuid.uuid4().hex
    sample = {
        "sample_type": "progress",
        "trajectory": {
            "frames": {"__numpy_file__": "sample_0_trajectory_frames"},
            "frames_shape": list(frames.shape),
            "task": task,
            "id": "live",
            "metadata": {"subsequence_length": int(len(frames))},
            "video_embeddings": None,
        },
    }
    npy = io.BytesIO()
    np.save(npy, frames, allow_pickle=False)
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(), b"\r\n",
        ])

    field("sample_0", json.dumps(sample))
    field("use_frame_steps", "true" if use_frame_steps else "false")
    parts.extend([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="sample_0_trajectory_frames"; filename="frames.npy"\r\n',
        b"Content-Type: application/octet-stream\r\n\r\n",
        npy.getvalue(), b"\r\n", f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _parse_result(payload: dict[str, Any]) -> tuple[float, float, list[float], list[float]]:
    progress_rows = (payload.get("outputs_progress") or {}).get("progress_pred") or []
    success_rows = (payload.get("outputs_success") or {}).get("success_probs") or []
    progress_trace = [float(x) for x in (progress_rows[0] if progress_rows else [])]
    success_trace = [float(x) for x in (success_rows[0] if success_rows else [])]
    if not progress_trace:
        raise ValueError("Robometer response contains no progress prediction")
    return progress_trace[-1], success_trace[-1] if success_trace else 0.0, progress_trace, success_trace


class LiveState:
    # Ignore tiny prediction jitter when deciding whether progress increased.
    PROGRESS_INCREASE_EPSILON = 0.05

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.frames: list[np.ndarray] = []
        self.latest_jpeg: bytes | None = None
        self.latest_preview_jpeg: bytes | None = None
        self.latest_preview_at = 0.0
        self.task = settings.prompt or ""
        self.started_at = time.time()
        self.last_submit = 0.0
        self.episode = 1
        self.samples: list[dict[str, Any]] = []
        self.status = "waiting for observations"
        self.error: str | None = None
        self.dropped = 0
        self.policy_requests = 0
        self.best_progress: float | None = None
        self.last_progress_increase_at = self.started_at
        self.paused = False
        self.failure = False
        self.job_queue: queue.Queue[tuple[np.ndarray, str, float, int]] = queue.Queue(maxsize=1)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = settings.output_dir / f"live_{stamp}.jsonl"

    def reset(self, task: str | None = None) -> None:
        with self.lock:
            self.frames.clear()
            self.samples.clear()
            self.latest_jpeg = None
            self.latest_preview_jpeg = None
            self.latest_preview_at = 0.0
            self.started_at = time.time()
            self.last_submit = 0.0
            self.episode += 1
            self.policy_requests = 0
            self.dropped = 0
            self.best_progress = None
            self.last_progress_increase_at = self.started_at
            self.paused = False
            self.failure = False
            self.status = "episode reset"
            self.error = None
            if task is not None:
                self.task = task
            try:
                while True:
                    self.job_queue.get_nowait()
                    self.job_queue.task_done()
            except queue.Empty:
                pass

    def standby(self) -> None:
        """Clear all progress and ignore observations until the operator starts."""
        self.reset()
        with self.lock:
            self.paused = True
            self.failure = False
            self.status = "waiting for operator to start"

    def pause(self) -> None:
        """Freeze Robometer at its current result until resume or reset."""
        with self.lock:
            self.paused = True
            self.failure = True
            self.status = "paused after failure"
            try:
                while True:
                    self.job_queue.get_nowait()
                    self.job_queue.task_done()
            except queue.Empty:
                pass

    def resume(self) -> None:
        with self.lock:
            self.paused = False
            self.failure = False
            self.last_progress_increase_at = time.time()
            self.last_submit = 0.0
            self.status = "live" if self.samples else "waiting for observations"

    def set_preview(self, image: bytes) -> None:
        with self.lock:
            self.latest_preview_jpeg = image
            self.latest_preview_at = time.monotonic()

    def observe(self, observation: dict[str, Any]) -> None:
        task = str(observation.get("prompt") or self.settings.prompt or "")
        reset = bool(observation.get("robometer_reset", False))
        if reset:
            self.reset(task)
        with self.lock:
            if self.paused:
                return
        frame = _extract_frame(observation, self.settings.camera)
        now = time.time()
        with self.lock:
            if self.paused:
                return
            if task and self.task and task != self.task:
                self.frames.clear()
                self.samples.clear()
                self.started_at = now
                self.episode += 1
                self.best_progress = None
                self.last_progress_increase_at = now
            self.task = task
            self.policy_requests += 1
            self.frames.append(frame)
            if len(self.frames) > self.settings.max_frames:
                indices = np.linspace(0, len(self.frames) - 1, self.settings.max_frames, dtype=int)
                self.frames = [self.frames[i] for i in indices]
            self.latest_jpeg = _jpeg(frame)
            if now - self.last_submit < self.settings.monitor_interval:
                return
            snapshot = np.stack(self.frames)
            self.last_submit = now
            request_episode = self.episode
        job = (snapshot, task, now, request_episode)
        try:
            self.job_queue.put_nowait(job)
        except queue.Full:
            try:
                self.job_queue.get_nowait()
            except queue.Empty:
                pass
            self.job_queue.put_nowait(job)
            with self.lock:
                self.dropped += 1

    def add_result(self, progress: float, success: float, latency: float, submitted: float, request_episode: int,
                   progress_trace: list[float], success_trace: list[float]) -> None:
        now = time.time()
        clipped_progress = float(np.clip(progress, 0, 1))
        with self.lock:
            if request_episode != self.episode or self.paused:
                return
            if self.best_progress is None or clipped_progress > self.best_progress + self.PROGRESS_INCREASE_EPSILON:
                self.best_progress = clipped_progress
                self.last_progress_increase_at = now
        item = {
            "time": now, "elapsed": now - self.started_at,
            "source_elapsed": submitted - self.started_at,
            "progress": clipped_progress,
            "success_probability": float(np.clip(success, 0, 1)),
            "success": int(success >= self.settings.success_threshold),
            "latency_ms": round(latency * 1000, 1), "episode": request_episode,
            "task": self.task, "progress_trace": progress_trace, "success_trace": success_trace,
        }
        with self.lock:
            if request_episode != self.episode or self.paused:
                return
            self.samples.append(item)
            self.status = "live"
            self.error = None
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def fail(self, exc: Exception, request_episode: int | None = None) -> None:
        with self.lock:
            if self.paused or (request_episode is not None and request_episode != self.episode):
                return
            self.status = "Robometer unavailable"
            self.error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            failure_timeout = self.settings.failure_timeout
            stalled_seconds = max(0.0, time.time() - self.last_progress_increase_at)
            if not self.paused and failure_timeout is not None and self.best_progress is not None:
                self.failure = stalled_seconds >= failure_timeout
            return {
                "task": self.task, "episode": self.episode, "status": self.status, "error": self.error,
                "threshold": self.settings.success_threshold, "policy_requests": self.policy_requests,
                "dropped_monitor_jobs": self.dropped, "log_path": str(self.log_path),
                "failure": self.failure, "paused": self.paused, "failure_timeout": failure_timeout,
                "progress_stalled_seconds": stalled_seconds,
                "samples": list(self.samples),
            }


def monitor_worker(state: LiveState) -> None:
    endpoint = state.settings.robometer_url.rstrip("/") + "/evaluate_batch_npy"
    while True:
        frames, task, submitted, request_episode = state.job_queue.get()
        started = time.monotonic()
        try:
            body, content_type = _multipart(frames, task, state.settings.use_frame_steps)
            request = urllib.request.Request(endpoint, data=body, method="POST", headers={"Content-Type": content_type})
            with urllib.request.urlopen(request, timeout=state.settings.timeout) as response:
                result = json.loads(response.read())
            progress, success, progress_trace, success_trace = _parse_result(result)
            state.add_result(progress, success, time.monotonic() - started, submitted, request_episode,
                             progress_trace, success_trace)
        except Exception as exc:
            state.fail(exc, request_episode)
            LOGGER.warning("Robometer monitoring request failed: %s", exc)
        finally:
            state.job_queue.task_done()


DASHBOARD_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenPI + Robometer Live</title>
<style>body{margin:0;background:#12161e;color:#eef1f6;font:15px system-ui}main{max-width:1500px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:20px;align-items:center}.muted{color:#9da6b5}.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:18px}.panel{background:#1d232e;border-radius:16px;padding:18px;margin-top:18px}.video-panel{border:4px solid transparent;transition:border-color .2s,box-shadow .2s}.video-panel.failure{border-color:#ff4545;box-shadow:0 0 24px #ff454577}.camera{width:100%;max-height:570px;object-fit:contain;background:#0c1016;border-radius:10px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.value{font-size:32px;font-weight:700;margin-top:6px}.blue{color:#53beff}.green{color:#68d391}.purple{color:#da86ff}canvas{width:100%;height:180px}button{background:#ffb84d;border:0;border-radius:8px;padding:10px 16px;font-weight:700}#error{color:#ff8e8e;white-space:pre-wrap}@media(max-width:900px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr}}</style></head>
<body><main><div class="top"><div><h1>OpenPI + Robometer Live</h1><div id="task" class="muted"></div></div><button onclick="resetEpisode()">Reset episode</button></div>
<div class="grid"><div id="video-panel" class="panel video-panel"><img id="camera" class="camera" src="/frame.jpg"><div id="meta" class="muted"></div><div id="error"></div></div>
<div><div class="cards"><div class="panel">Progress<div id="progress" class="value blue">—</div></div><div class="panel">Success<div id="binary" class="value green">—</div></div><div class="panel">Probability<div id="prob" class="value purple">—</div></div></div>
<div class="panel">Task Progress<canvas id="pchart"></canvas></div><div class="panel">Success / Probability<canvas id="schart"></canvas></div></div></div></main>
<script>function chart(id,a,b){const c=document.getElementById(id),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.strokeStyle='#3e4654';x.lineWidth=1;for(let i=0;i<3;i++){let y=12+i*(h-24)/2;x.beginPath();x.moveTo(35,y);x.lineTo(w-8,y);x.stroke()}function line(v,color,step){if(!v.length)return;x.strokeStyle=color;x.lineWidth=3;x.beginPath();v.forEach((q,i)=>{let xx=35+i*(w-45)/Math.max(1,v.length-1),yy=h-12-Math.max(0,Math.min(1,q))*(h-24);if(!i)x.moveTo(xx,yy);else if(step){x.lineTo(xx,py);x.lineTo(xx,yy)}else x.lineTo(xx,yy);py=yy});x.stroke()}let py=0;line(a,'#53beff',false);line(b,'#da86ff',false)}
async function update(){try{let s=await(await fetch('/api/state',{cache:'no-store'})).json(),a=s.samples,p=a.map(x=>x.progress),q=a.map(x=>x.success_probability),b=a.map(x=>x.success),v=document.getElementById('video-panel');v.classList.toggle('failure',s.failure);document.getElementById('task').textContent='Episode '+s.episode+' — '+(s.task||'No prompt');let f=s.failure?' | FAILURE: progress stalled '+s.progress_stalled_seconds.toFixed(1)+'s':'';document.getElementById('meta').textContent=s.status+f+' | policy requests '+s.policy_requests+' | dropped monitor jobs '+s.dropped_monitor_jobs+' | '+s.log_path;document.getElementById('error').textContent=s.error||'';if(a.length){let z=a[a.length-1];document.getElementById('progress').textContent=z.progress.toFixed(3);document.getElementById('prob').textContent=z.success_probability.toFixed(3);document.getElementById('binary').textContent=z.success?'YES':'NO'}else{document.getElementById('progress').textContent='—';document.getElementById('prob').textContent='—';document.getElementById('binary').textContent='—'}chart('pchart',p,[]);chart('schart',b,q);document.getElementById('camera').src='/frame.jpg?t='+Date.now()}catch(e){document.getElementById('error').textContent=e}}async function resetEpisode(){await fetch('/api/reset',{method:'POST'});update()}setInterval(update,200);update();</script></body></html>"""


def dashboard_handler(state: LiveState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/api/state"):
                self._send(json.dumps(state.snapshot(), ensure_ascii=False).encode(), "application/json")
            elif self.path.startswith("/frame.jpg"):
                with state.lock:
                    preview_is_live = time.monotonic() - state.latest_preview_at < 2.0
                    image = state.latest_preview_jpeg if preview_is_live else state.latest_jpeg
                if image is None:
                    self.send_error(404, "No camera frame yet")
                else:
                    self._send(image, "image/jpeg")
            elif self.path == "/" or self.path.startswith("/?"):
                self._send(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/api/reset":
                state.reset()
                self._send(b'{"ok":true}', "application/json")
            elif self.path == "/api/standby":
                state.standby()
                self._send(b'{"ok":true}', "application/json")
            elif self.path == "/api/pause":
                state.pause()
                self._send(b'{"ok":true}', "application/json")
            elif self.path == "/api/resume":
                state.resume()
                self._send(b'{"ok":true}', "application/json")
            elif self.path == "/api/preview":
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if content_type != "image/jpeg" or not 0 < length <= 2_000_000:
                    self.send_error(400, "Expected a JPEG no larger than 2 MB")
                    return
                state.set_preview(self.rfile.read(length))
                self._send(b'{"ok":true}', "application/json")
            else:
                self.send_error(404)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


class MonitoredPolicyServer:
    def __init__(self, policy: policy_lib.Policy, state: LiveState, port: int, metadata: dict[str, Any]):
        self.policy = policy
        self.state = state
        self.port = port
        self.metadata = metadata

    async def handler(self, websocket: websocket_server.ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        previous_total = None
        while True:
            try:
                started = time.monotonic()
                observation = msgpack_numpy.unpackb(await websocket.recv())
                policy_observation = dict(observation)
                policy_observation.pop("robometer_reset", None)
                try:
                    self.state.observe(observation)
                except Exception as exc:
                    self.state.fail(exc)
                infer_started = time.monotonic()
                result = self.policy.infer(policy_observation)
                result["server_timing"] = {"infer_ms": (time.monotonic() - infer_started) * 1000}
                if previous_total is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total * 1000
                await websocket.send(packer.pack(result))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Internal error")
                raise

    async def run(self) -> None:
        async with websocket_server.serve(self.handler, "0.0.0.0", self.port, compression=None, max_size=None,
                                          process_request=self.health) as server:
            await server.serve_forever()

    @staticmethod
    def health(connection: websocket_server.ServerConnection,
               request: websocket_server.Request) -> websocket_server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


def main() -> None:
    settings = parse_args()
    sample_kwargs = {"num_steps": settings.num_steps} if settings.num_steps is not None else {}
    config = training_config.get_config(settings.policy_config)
    policy = policy_config.create_trained_policy(
        config, settings.checkpoint, default_prompt=settings.prompt, sample_kwargs=sample_kwargs
    )
    metadata = policy.metadata
    if settings.record_policy:
        policy = policy_lib.PolicyRecorder(policy, "policy_records")

    state = LiveState(settings)
    threading.Thread(target=monitor_worker, args=(state,), daemon=True, name="robometer-worker").start()
    dashboard = ThreadingHTTPServer((settings.dashboard_host, settings.dashboard_port), dashboard_handler(state))
    threading.Thread(target=dashboard.serve_forever, daemon=True, name="dashboard").start()
    host = socket.gethostname()
    LOGGER.info("Policy websocket: ws://%s:%d", host, settings.policy_port)
    LOGGER.info("Live dashboard: http://%s:%d", host, settings.dashboard_port)
    LOGGER.info("Robometer endpoint: %s/evaluate_batch_npy", settings.robometer_url.rstrip("/"))
    LOGGER.info("JSONL results: %s", state.log_path)
    asyncio.run(MonitoredPolicyServer(policy, state, settings.policy_port, metadata).run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
