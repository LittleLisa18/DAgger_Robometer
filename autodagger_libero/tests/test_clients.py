import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import numpy as np
import requests
from autodagger_libero.clients import PolicyClient, RobometerClient


class ClientTests(unittest.TestCase):
    def test_invalid_prefix_score_rejected(self):
        with patch("autodagger_libero.clients.requests.post") as post:
            post.return_value.json.return_value = {
                "outputs_progress": {"progress_pred": [[float("nan"), 0.8]]},
                "outputs_success": {"success_probs": [[0.2, 0.9]]},
            }
            with self.assertRaisesRegex(ValueError, "Invalid Robometer output"):
                RobometerClient("http://unused", 1, 0).score(
                    np.zeros((2, 2, 2, 3), np.uint8), "test"
                )

    def test_robometer_multipart_roundtrip(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                seen.append(self.rfile.read(int(self.headers["Content-Length"])))
                result = {
                    "outputs_progress": {"progress_pred": [[0.1, 0.8]]},
                    "outputs_success": {"success_probs": [[0.2, 0.9]]},
                }
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            result = RobometerClient(
                f"http://127.0.0.1:{server.server_port}", 2, 0
            ).score(np.zeros((2, 256, 256, 3), np.uint8), "pick bowl")
            self.assertEqual(result["progress"], 0.8)
            self.assertEqual(result["success_probability"], 0.9)
            self.assertIn(b"NUMPY", seen[0])
            self.assertIn(b"pick bowl", seen[0])
        finally:
            server.shutdown()
            server.server_close()

    def test_robometer_timeout_retry_bounded(self):
        with patch(
            "autodagger_libero.clients.requests.post", side_effect=requests.Timeout
        ) as post:
            with self.assertRaises(requests.Timeout):
                RobometerClient("http://unused", 0.01, 1).score(
                    np.zeros((2, 2, 2, 3), np.uint8), "test"
                )
            self.assertEqual(post.call_count, 2)

    def test_websocket_protocol_retry_reset(self):
        from websockets.sync.server import serve
        from openpi_client import msgpack_numpy

        attempts = []

        def handler(ws):
            packer = msgpack_numpy.Packer()
            ws.send(packer.pack({"policy_type": "test"}))
            try:
                for message in ws:
                    obs = msgpack_numpy.unpackb(message)
                    attempts.append(obs)
                    if len(attempts) == 1:
                        ws.send("temporary test error")
                        return
                    ws.send(packer.pack({"actions": np.zeros((5, 7), np.float32)}))
            except Exception:
                pass

        server = serve(handler, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = None
        try:
            client = PolicyClient(
                f"ws://127.0.0.1:{server.socket.getsockname()[1]}", 2, 1
            )
            result = client.infer({"state": np.ones(8, np.float32)})
            self.assertEqual(result["actions"].shape, (5, 7))
            self.assertEqual(len(attempts), 2)
            client.reset()
            self.assertEqual(client.metadata["policy_type"], "test")
        finally:
            if client:
                client.close()
            server.shutdown()
