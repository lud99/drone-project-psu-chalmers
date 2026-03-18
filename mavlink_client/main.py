import asyncio
from typing import Any, Optional

from mavlink_client.command_handler import CommandHandler
from mavlink_client.config import Config
from mavlink_client.mavlink_adapter import MavlinkAdapter
from mavlink_client.mavlink_connection import MavlinkConnectionManager
from mavlink_client.mission_executor import MissionExecutor
from mavlink_client.mock_drone import MockDrone
from mavlink_client.telemetry_manager import TelemetryManager
from mavlink_client.websocket_client import BackendWebSocketClient

print("MAIN.PY LOADED")


class MockMissionExecutor:
    def __init__(self, mock_drone: MockDrone, config: Config) -> None:
        self.mock_drone = mock_drone
        self.config = config

    def arm_and_takeoff(self, altitude: Optional[float] = None) -> None:
        target_alt = altitude if altitude is not None else self.config.default_takeoff_alt
        self.mock_drone.arm()
        self.mock_drone.takeoff(target_alt)

    def fly_to_coordinate(
        self,
        lat: float,
        lon: float,
        alt: float,
        heading: Optional[float] = None,
    ) -> None:
        self.mock_drone.goto(lat, lon, alt, heading)

    def return_to_home(self) -> None:
        self.mock_drone.rtl()

    def land(self) -> None:
        self.mock_drone.land()

    def abort(self) -> None:
        self.mock_drone.mode = "LOITER"

    def disarm(self) -> None:
        self.mock_drone.disarm()


async def safe_cancel(task: Optional[asyncio.Task]) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def main() -> None:
    print("ENTERED ASYNC MAIN")

    config = Config()
    telemetry_manager = TelemetryManager(drone_id=config.drone_id)
    ws_client = BackendWebSocketClient(config.backend_ws_url)

    mock_drone: Optional[MockDrone] = None
    connection: Any = None
    adapter: Optional[MavlinkAdapter] = None
    mission_executor: Any = None
    backend_connected = False

    telemetry_task: Optional[asyncio.Task] = None
    heartbeat_task: Optional[asyncio.Task] = None
    receive_task: Optional[asyncio.Task] = None

    try:
        if config.use_mock_drone:
            print("[BOOT] Using mock drone mode")
            mock_drone = MockDrone(telemetry_manager)
            mission_executor = MockMissionExecutor(mock_drone, config)

        else:
            print("[BOOT] Using real MAVLink mode")
            connection = MavlinkConnectionManager(
                connection_string=config.mavlink_connection_string,
                baud=config.mavlink_baud,
            )
            connection.connect()

            adapter = MavlinkAdapter(
                connection=connection,
                telemetry_manager=telemetry_manager,
                drone_id=config.drone_id,
            )

            mission_executor = MissionExecutor(adapter, config)

            print("[BOOT] Real MAVLink connection ready. Waiting for commands.")
            print("[TEST] Starting automatic takeoff in 3 seconds...")
            await asyncio.sleep(3)

            try:
                mission_executor.arm_and_takeoff(2.0)
                print("[TEST] Takeoff command executed")
            except Exception as e:
                print(f"[TEST ERROR] {e}")

        command_handler = CommandHandler(mission_executor, telemetry_manager)

        try:
            print(f"[WS] Connecting to backend at {config.backend_ws_url}")
            await ws_client.connect()
            backend_connected = True
            print(f"[WS] Connected to backend at {config.backend_ws_url}")
        except Exception as e:
            backend_connected = False
            print(
                f"[WS] Backend not available, continuing without websocket: {e}")

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

        async def telemetry_loop() -> None:
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
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[TELEMETRY] {e}")

                await asyncio.sleep(config.telemetry_interval_sec)

        async def heartbeat_loop() -> None:
            while True:
                try:
                    if backend_connected:
                        await ws_client.send_json(
                            {
                                "msg_type": "ping",
                                "drone_id": config.drone_id,
                            }
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[HEARTBEAT] {e}")

                await asyncio.sleep(config.heartbeat_interval_sec)

        async def receive_handler(message: dict[str, Any]) -> None:
            try:
                reply = command_handler.handle(message)
                if reply and backend_connected:
                    reply["drone_id"] = config.drone_id
                    await ws_client.send_json(reply)
            except Exception as e:
                print(f"[COMMAND] {e}")
                if backend_connected:
                    await ws_client.send_json(
                        {
                            "msg_type": "ack",
                            "drone_id": config.drone_id,
                            "command": message.get("msg_type"),
                            "status": "error",
                            "error": str(e),
                        }
                    )

        telemetry_task = asyncio.create_task(
            telemetry_loop(), name="telemetry_loop")
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(), name="heartbeat_loop")

        if backend_connected:
            receive_task = asyncio.create_task(
                ws_client.receive_loop(receive_handler),
                name="receive_loop",
            )
            await asyncio.gather(telemetry_task, heartbeat_task, receive_task)
        else:
            print("[WS] Receive loop skipped because backend is unavailable")
            await asyncio.gather(telemetry_task, heartbeat_task)

    except KeyboardInterrupt:
        print("\n[SAFETY] KeyboardInterrupt received")

        if not config.use_mock_drone and adapter is not None:
            try:
                print("[SAFETY] Sending LAND")
                adapter.land()
                await asyncio.sleep(2)
            except Exception as e:
                print(f"[SAFETY] LAND failed: {e}")

            try:
                print("[SAFETY] Sending DISARM")
                adapter.disarm()
                await asyncio.sleep(1)
            except Exception as e:
                print(f"[SAFETY] DISARM failed: {e}")

        raise

    finally:
        await safe_cancel(telemetry_task)
        await safe_cancel(heartbeat_task)
        await safe_cancel(receive_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[EXIT] Program stopped by user")
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
