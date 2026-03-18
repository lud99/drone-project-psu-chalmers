import asyncio
from typing import Any

from mavlink_client.config import Config
from mavlink_client.telemetry_manager import TelemetryManager
from mavlink_client.websocket_client import BackendWebSocketClient
from mavlink_client.command_handler import CommandHandler
from mavlink_client.mission_executor import MissionExecutor
from mavlink_client.mavlink_connection import MavlinkConnectionManager
from mavlink_client.mavlink_adapter import MavlinkAdapter
from mavlink_client.mock_drone import MockDrone


async def main():
    config = Config()
    telemetry_manager = TelemetryManager(drone_id=config.drone_id)
    ws_client = BackendWebSocketClient(config.backend_ws_url)

    mock_drone: Any = None
    adapter: Any = None
    mission_executor: Any = None
    backend_connected = False

    if config.use_mock_drone:
        mock_drone = MockDrone(telemetry_manager)

        class MockMissionExecutor:
            def arm_and_takeoff(self, altitude=None):
                mock_drone.arm()
                mock_drone.takeoff(altitude or config.default_takeoff_alt)

            def fly_to_coordinate(self, lat, lon, alt, heading=None):
                mock_drone.goto(lat, lon, alt, heading)

            def return_to_home(self):
                mock_drone.rtl()

            def land(self):
                mock_drone.land()

            def abort(self):
                mock_drone.mode = "LOITER"

        print("[BOOT] Using mock drone mode")
        mission_executor = MockMissionExecutor()

    else:
        connection = MavlinkConnectionManager(
            connection_string=config.mavlink_connection_string,
            baud=config.mavlink_baud,
        )
        connection.connect()
        adapter = MavlinkAdapter(
            connection, telemetry_manager, config.drone_id)
        print("[BOOT] Using real MAVLink mode")
        mission_executor = MissionExecutor(adapter, config)
        print("[TEST] Arm and Takeoff")
        mission_executor.arm_and_takeoff()

    command_handler = CommandHandler(mission_executor, telemetry_manager)

    try:
        await ws_client.connect()
        backend_connected = True
        print(f"[WS] Connected to backend at {config.backend_ws_url}")
    except Exception as e:
        print(f"[WS] Backend not available, continuing without websocket: {e}")
        backend_connected = False

    if backend_connected:
        register_payload = {
            "msg_type": "register",
            "drone_id": config.drone_id,
            "drone_type": config.drone_type,
            "model": config.model,
            "capabilities": {
                "camera": None,
                "led": None,
                "spotlight": False,
                "speaker": False,
                "maxSpeed": int(config.max_speed_m_s),
            },
        }
        await ws_client.send_json(register_payload)

    async def telemetry_loop():
        while True:
            try:
                if config.use_mock_drone:
                    if mock_drone is not None:
                        mock_drone.tick()
                else:
                    if adapter is not None:
                        adapter.poll_telemetry()

                snapshot = telemetry_manager.snapshot()

                if backend_connected:
                    await ws_client.send_json(snapshot.to_backend_message())
                else:
                    if snapshot.lat is not None or snapshot.mode is not None:
                        print(
                            f"[TELEMETRY] lat={snapshot.lat}, lon={snapshot.lon}, "
                            f"alt={snapshot.alt}, heading={snapshot.heading}, "
                            f"speed={snapshot.speed}, battery={snapshot.battery_percent}, "
                            f"mode={snapshot.mode}, armed={snapshot.armed}, "
                            f"gps_fix={snapshot.gps_fix_type}, sats={snapshot.satellites_visible}"
                        )
            except Exception as e:
                print(f"[TELEMETRY] {e}")
            await asyncio.sleep(config.telemetry_interval_sec)

    async def heartbeat_loop():
        while True:
            try:
                if backend_connected:
                    await ws_client.send_json({
                        "msg_type": "ping",
                        "drone_id": config.drone_id,
                    })
            except Exception as e:
                print(f"[HEARTBEAT] {e}")
            await asyncio.sleep(config.heartbeat_interval_sec)

    async def receive_handler(message):
        try:
            reply = command_handler.handle(message)
            if reply and backend_connected:
                reply["drone_id"] = config.drone_id
                await ws_client.send_json(reply)
        except Exception as e:
            print(f"[COMMAND] {e}")
            if backend_connected:
                await ws_client.send_json({
                    "msg_type": "ack",
                    "drone_id": config.drone_id,
                    "command": message.get("msg_type"),
                    "status": "error",
                    "error": str(e),
                })

    telemetry_task = asyncio.create_task(telemetry_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop())

    if backend_connected:
        receive_task = asyncio.create_task(
            ws_client.receive_loop(receive_handler))
        await asyncio.gather(telemetry_task, heartbeat_task, receive_task)
    else:
        print("[WS] Receive loop skipped because backend is unavailable")
        await asyncio.gather(telemetry_task, heartbeat_task)


if __name__ == "__main__":
    asyncio.run(main())
