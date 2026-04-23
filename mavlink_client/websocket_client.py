from __future__ import annotations

import json
from typing import Callable, Awaitable, Dict, Any

import websockets


class BackendWebSocketClient:
    def __init__(self, url: str):
        self.url = url
        self.websocket = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self.websocket = await websockets.connect(self.url, ping_interval=None)
        self._connected = True
        print(f"[WS] Connected to {self.url}")

    async def send_json(self, payload: Dict[str, Any]) -> None:
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(json.dumps(payload))

    async def register_drone(self, drone_id: str) -> None:
        await self.send_json(
            {
                "msg_type": "drone_registration",
                "drone_id": drone_id,
            }
        )
        print(f"[WS] Registered drone: {drone_id}")

    async def receive_loop(
        self,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        try:
            async for raw_message in self.websocket:
                try:
                    payload = json.loads(raw_message)
                    await handler(payload)
                except Exception as e:
                    print(f"[WS] Failed to handle message: {e}")
        finally:
            self._connected = False

    async def close(self) -> None:
        if self.websocket:
            await self.websocket.close()
        self._connected = False
