"""
WebSocket client to the backend. Sends registration, telemetry, ping;
receives and dispatches commands (flight_arm, flight_take_off, flight_return_to_home, task, abort_task).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection

from mavlink_client.messages import (
    AbortTaskMessage,
    Capabilities,
    TaskMessage,
    TelemetryPayload,
    build_coordinate_response,
    build_registration,
    build_telemetry,
    build_task_event,
    parse_incoming,
)
from pydantic import ValidationError

if TYPE_CHECKING:
    from mavlink_client.mavlink_connection import MavlinkConnection, TelemetryState
    from mavlink_client.mission_handler import MissionHandler


class BackendClient:
    """
    Async WebSocket client. Runs in an asyncio loop; telemetry and ping
    are sent on intervals. Incoming messages are routed to the mission handler.
    """

    def __init__(
        self,
        url: str,
        drone_id: str,
        model: str,
        mavlink_connection: "MavlinkConnection",
        mission_handler: "MissionHandler",
        get_telemetry: Callable[[], "TelemetryState"],
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
    ):
        self.url = url
        self.drone_id = drone_id
        self.model = model
        self._mav = mavlink_connection
        self._mission = mission_handler
        self.get_telemetry = get_telemetry
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self._ws: Optional[ClientConnection] = None
        self._running = False
        self._telemetry_interval = 1.0
        self._ping_interval = 5.0
        self._capabilities = Capabilities(
            camera=None,
            led=None,
            spotlight=False,
            speaker=False,
            max_speed=15.0,
        )

    async def run(self) -> None:
        """Connect, send registration, then run telemetry and ping loops; process messages."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    self.url,
                    open_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    # Send registration once
                    await self._send(
                        build_registration(
                            self.drone_id, self.model, self._capabilities
                        )
                    )
                    if self.on_connected:
                        self.on_connected()
                    # Persist backend URL for next run (same as Android last-used)
                    try:
                        from mavlink_client.discovery import persist_backend_url
                        persist_backend_url(self.url)
                    except Exception:
                        pass
                    # Run sender and receiver concurrently
                    await asyncio.gather(
                        self._sender_loop(ws),
                        self._receiver_loop(ws),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    pass  # TODO: log
            finally:
                self._ws = None
                if self.on_disconnected:
                    self.on_disconnected()
            if self._running:
                await asyncio.sleep(5)  # Reconnect delay

    def stop(self) -> None:
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())

    async def _send(self, message: str) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send(message)

    async def _sender_loop(self, ws: ClientConnection) -> None:
        last_telemetry = 0.0
        last_ping = 0.0
        while self._running and not ws.closed:
            now = time.monotonic()
            if now - last_telemetry >= self._telemetry_interval:
                last_telemetry = now
                state = self.get_telemetry()
                if state.has_position or state.has_battery:
                    payload = TelemetryPayload(
                        lat=state.lat,
                        lon=state.lon,
                        alt=state.alt,
                        heading=state.heading,
                        speed=state.speed,
                        battery_percent=state.battery_percent if state.has_battery else 0,
                    )
                    await self._send(build_telemetry(self.drone_id, payload))
            if now - last_ping >= self._ping_interval:
                last_ping = now
                await self._send(json.dumps({"msg_type": "ping"}))
            await asyncio.sleep(0.2)

    async def _receiver_loop(self, ws: ClientConnection) -> None:
        while self._running and not ws.closed:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            await self._handle_message(raw)

    async def _handle_message(self, raw: str) -> None:
        try:
            data = parse_incoming(raw)
        except Exception:
            return
        msg_type = data.get("msg_type")
        if not msg_type:
            return
        if msg_type == "flight_arm":
            self._mission.arm()
        elif msg_type == "flight_take_off":
            self._mission.takeoff(10.0)
        elif msg_type == "flight_return_to_home":
            self._mission.return_to_launch()
        elif msg_type == "Coordinate_request":
            state = self.get_telemetry()
            resp = build_coordinate_response(
                state.lat, state.lon, state.alt, float(state.heading)
            )
            await self._send(resp)
        elif msg_type == "task":
            try:
                task_msg = TaskMessage.model_validate(data)
                await self._handle_task(task_msg)
            except ValidationError:
                pass
        elif msg_type == "abort_task":
            try:
                abort_msg = AbortTaskMessage.model_validate(data)
                if abort_msg.next == "go_home":
                    self._mission.return_to_launch()
                elif abort_msg.next == "land":
                    self._mission.land()
                # hover: no-op (already in position)
            except ValidationError:
                pass
        elif msg_type in ("offer", "candidate", "answer"):
            # WebRTC signaling: stub (no video stream from MAVLink client by default)
            pass

    async def _handle_task(self, msg: TaskMessage) -> None:
        """Execute one task and send task_event (task_complete or task_failed)."""
        mission_id = msg.mission_id
        index = msg.index
        task = msg.task
        try:
            if task.action == "go_to":
                ok = self._mission.go_to(
                    task.params.lat,
                    task.params.lon,
                    task.params.alt,
                    task.params.heading,
                )
            elif task.action == "go_home":
                ok = self._mission.return_to_launch()
            elif task.action == "land":
                ok = self._mission.land()
            else:
                # play_audio, led, spotlight: not implemented on MAVLink
                ok = False
            if ok:
                await self._send(
                    build_task_event(
                        self.drone_id, mission_id, index, "task_complete", ""
                    )
                )
            else:
                await self._send(
                    build_task_event(
                        self.drone_id,
                        mission_id,
                        index,
                        "task_failed",
                        "Command not supported or failed",
                    )
                )
        except Exception as e:
            await self._send(
                build_task_event(
                    self.drone_id, mission_id, index, "task_failed", str(e)
                )
            )
