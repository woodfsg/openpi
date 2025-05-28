import asyncio
import logging
import traceback
import time
import sys

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._last_request_time = time.time()
        self._server = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    async def check_timeout(self):
        """Check if server has been inactive for too long."""
        while True:
            await asyncio.sleep(30)  # 每30秒检查一次
            if time.time() - self._last_request_time > 300:  # 5分钟
                logging.info("No requests received in the last 5 minutes. Shutting down.")
                if self._server:
                    self._server.close()
                    await self._server.wait_closed()
                sys.exit(0)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._server = await websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        )

        # 同时运行服务器和超时检查
        await asyncio.gather(
            self._server.serve_forever(),
            self.check_timeout()
        )

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        self._last_request_time = time.time()
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                self._last_request_time = time.time()  # 更新最后请求时间
                action = self._policy.infer(obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
