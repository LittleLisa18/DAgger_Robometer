"""Read-only local dashboard; bounded HTTP concurrency and preview storage."""

import copy
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote

from PIL import Image
from .replay import ReplayCache


class Dashboard:
    def __init__(self, store, config):
        self.store, self.config = store, config
        self.lock = threading.Lock()
        self.state = {
            "status": "starting",
            "samples": [],
            "error": None,
            "success_threshold": config.success_threshold,
        }
        self.frames = {}
        self.episodes = store.summaries()
        self.server = None
        self.replay = ReplayCache(store.root)

    def publish(self, *, images=None, **state):
        frames = {}
        if images:
            for name, pixels in images.items():
                output = io.BytesIO()
                Image.fromarray(pixels).save(output, "JPEG", quality=75)
                frames[name] = output.getvalue()
        with self.lock:
            if "episode_id" in state and state["episode_id"] != self.state.get(
                "episode_id"
            ):
                # Do not associate the previous episode's preview or latency
                # with the new instruction while the simulator is resetting.
                self.frames.clear()
                self.state = {
                    "success_threshold": self.config.success_threshold,
                    "samples": [],
                    "error": None,
                }
            self.state.update(state)
            if "samples" in state:
                self.state["samples"] = copy.deepcopy(state["samples"][-1024:])
            self.frames.update(frames)

    def refresh(self):
        episodes = self.store.summaries()
        with self.lock:
            self.episodes = episodes

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
            episodes = [e for e in self.episodes if not e.get("test_only")]
            total = len(episodes)
            state["statistics"] = {
                "completed": total,
                **{
                    key: (
                        sum(bool(e.get(field)) for e in episodes) / total
                        if total
                        else 0
                    )
                    for key, field in (
                        ("takeover_rate", "took_over"),
                        ("acceptance_rate", "accepted_for_distillation"),
                        ("env_success_rate", "env_success"),
                        ("disagreement_rate", "success_disagreement"),
                    )
                },
            }
            return state

    def start(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def reply(self, data, kind="application/json", status=200):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                route = urlsplit(self.path).path
                if route in ("/player.html", "/player.js"):
                    kind = (
                        "text/html; charset=utf-8"
                        if route.endswith("html")
                        else "text/javascript; charset=utf-8"
                    )
                    return self.reply(
                        Path(__file__).with_name(route[1:]).read_bytes(), kind
                    )
                if route.startswith("/api/replay/"):
                    parts = route[len("/api/replay/") :].split("/")
                    if len(parts) != 2:
                        return self.reply(b"{}", status=404)
                    episode_id = unquote(parts[0])
                    with dashboard.lock:
                        exists = any(
                            e["episode_id"] == episode_id for e in dashboard.episodes
                        )
                    if not exists or Path(episode_id).name != episode_id:
                        return self.reply(b"{}", status=404)
                    try:
                        data = dashboard.replay.frame(episode_id, int(parts[1]))
                        return self.reply(json.dumps(data, allow_nan=False).encode())
                    except (ValueError, IndexError, OSError, KeyError) as exc:
                        return self.reply(
                            json.dumps({"error": str(exc)}).encode(), status=400
                        )
                if route == "/":
                    return self.reply(
                        Path(__file__).with_name("dashboard.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                if route == "/api/state":
                    data = dashboard.snapshot()
                elif route in ("/api/episodes", "/api/replay-episodes"):
                    with dashboard.lock:
                        episodes = (
                            dashboard.episodes
                            if route == "/api/replay-episodes"
                            else dashboard.episodes[-100:]
                        )
                        data = copy.deepcopy(episodes[::-1])
                elif route.startswith("/api/episodes/"):
                    episode_id = route[len("/api/episodes/") :]
                    with dashboard.lock:
                        exists = any(
                            e["episode_id"] == episode_id for e in dashboard.episodes
                        )
                    if not exists:
                        return self.reply(b"{}", status=404)
                    data = json.loads(
                        (
                            dashboard.store.root / episode_id / "metadata.json"
                        ).read_text()
                    )
                elif route in ("/frames/image.jpg", "/frames/image2.jpg"):
                    with dashboard.lock:
                        frame = dashboard.frames.get(route.split("/")[-1][:-4])
                    return self.reply(frame or b"", "image/jpeg", 200 if frame else 404)
                else:
                    return self.reply(b"{}", status=404)
                self.reply(json.dumps(data, allow_nan=False).encode())

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            slots = threading.BoundedSemaphore(8)

            def process_request(self, request, address):
                # Slow/disconnected browsers cannot build an unbounded worker queue.
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, address):
                try:
                    super().process_request_thread(request, address)
                finally:
                    self.slots.release()

        self.server = Server(
            (self.config.dashboard_host, self.config.dashboard_port), Handler
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
