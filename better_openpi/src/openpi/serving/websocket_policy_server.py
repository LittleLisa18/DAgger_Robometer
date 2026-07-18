import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


class StreamingWebsocketPolicyServer:
    """Serves a policy with streaming action output over WebSocket.

    During adaptive denoising, actions that finish early (all remaining dt == 0)
    are sent immediately instead of waiting for the full chunk to complete.

    Protocol (per inference cycle):
      1. Server sends metadata on connection open.
      2. Client sends observation (msgpack).
      3. Server sends zero or more *partial* messages::

             {"type": "partial", "step": int, "action_indices": ndarray, "actions": ndarray}

      4. Server sends one *final* message with the complete result::

             {"type": "final", ...full_output_dict}

      5. Repeat from step 2.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        early_stop_actions: int | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._early_stop_actions = early_stop_actions
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened (streaming)")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        loop = asyncio.get_event_loop()

        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                queue: asyncio.Queue = asyncio.Queue()

                def on_actions_ready(ready_actions):
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "partial", "actions": ready_actions})

                def run_inference():
                    try:
                        result = self._policy.infer_streaming(
                            obs,
                            on_actions_ready=on_actions_ready,
                            early_stop_actions=self._early_stop_actions,
                        )
                    except Exception:
                        raise
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                    return result

                inference_task = asyncio.create_task(asyncio.to_thread(run_inference))

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    await websocket.send(packer.pack(item))

                final_result = await inference_task
                final_result["type"] = "final"
                final_result["server_timing"] = {
                    "infer_ms": (time.monotonic() - start_time) * 1000,
                }
                await websocket.send(packer.pack(final_result))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
