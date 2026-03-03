#!/usr/bin/env python3
"""
Entrypoint for the MAVLink client. Resolves backend URL (discovery / persisted)
and MAVLink connection (auto-detect) with no hardcoded defaults. Connects to
vehicle and backend, sends registration and telemetry, handles commands.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time

from mavlink_client.config import (
    Config,
    resolve_backend_url,
    resolve_connection,
    resolve_drone_id,
)
from mavlink_client.mavlink_connection import MavlinkConnection
from mavlink_client.backend_client import BackendClient
from mavlink_client.mission_handler import MissionHandler


def main() -> int:
    config = Config.from_env()

    # Resolve backend URL: env → multicast discovery → persisted (no hardcoded default)
    backend_url = resolve_backend_url(discovery_timeout_seconds=15.0)
    if not backend_url:
        print(
            "Backend URL not found. Start the backend so it advertises on multicast, "
            "or set BACKEND_WS_URL=ws://host:port",
            file=sys.stderr,
        )
        return 1
    print("Backend:", backend_url)

    # Resolve MAVLink connection: env → auto-detect UDP/serial (no hardcoded default)
    connection_str = resolve_connection()
    if not connection_str:
        print(
            "No MAVLink vehicle found. Set MAVLINK_CONNECTION=udp:host:port or "
            "serial:/dev/ttyUSB0:57600, or ensure vehicle/SITL is running (e.g. UDP 14550).",
            file=sys.stderr,
        )
        return 1
    print("MAVLink:", connection_str)

    mav_cfg = config.mavlink
    mav_cfg.connection = connection_str
    # target_system will be updated from first heartbeat

    mav_conn = MavlinkConnection(
        connection_string=connection_str,
        target_system=mav_cfg.target_system,
        target_component=mav_cfg.target_component,
    )

    print("Connecting to vehicle...")
    try:
        mav_conn.start()
    except Exception as e:
        print("MAVLink connection failed:", e, file=sys.stderr)
        return 1

    # Wait for first heartbeat so we have system_id for drone_id if not set by env
    for _ in range(50):
        if mav_conn.get_system_id() is not None:
            break
        time.sleep(0.1)
    drone_id = resolve_drone_id()
    if not drone_id:
        drone_id = f"mavlink-{mav_conn.get_system_id() or 1}"
    model = os.environ.get("DRONE_MODEL", "MAVLink drone")

    print("Vehicle connected. Drone ID:", drone_id)
    mission = MissionHandler(mav_conn)

    backend = BackendClient(
        url=backend_url,
        drone_id=drone_id,
        model=model,
        mavlink_connection=mav_conn,
        mission_handler=mission,
        get_telemetry=lambda: mav_conn.state,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown():
        backend.stop()
        mav_conn.stop()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, lambda: loop.call_soon_threadsafe(shutdown)
            )
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(backend.run())
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
        mav_conn.stop()
        loop.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
