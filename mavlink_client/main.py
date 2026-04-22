import asyncio
import signal
import threading
from typing import Any, Optional

from .config import Config
from .mavlink_adapter import MavlinkAdapter
from .mavlink_connection import MavlinkConnectionManager
from .mission_executor import MissionExecutor
from .mock_drone import MockDrone
from .telemetry_manager import TelemetryManager
from .websocket_client import BackendWebSocketClient
from .drone_controller import DroneController
from .api import start_api_server

print("MAIN.PY LOADED")

TAKEOFF_ALT_METERS = 5.0  # relative to launch altitude (ground)
ARM_DELAY_SEC = 5
HOVER_DURATION_SEC = 10
LANDING_WAIT_TIMEOUT_SEC = 45
ALTITUDE_REACHED_TIMEOUT_SEC = 30


class MockMissionExecutor:
    def __init__(self, mock_drone: MockDrone, config: Config) -> None:
        self.mock_drone = mock_drone
        self.config = config

    def arm(self) -> None:
        self.mock_drone.arm()

    def takeoff(self, altitude: Optional[float] = None) -> None:
        target_alt = altitude if altitude is not None else self.config.default_takeoff_alt
        self.mock_drone.takeoff(target_alt)

    def arm_and_takeoff(self, altitude: Optional[float] = None) -> None:
        target_alt = altitude if altitude is not None else self.config.default_takeoff_alt
        self.mock_drone.arm()
        self.mock_drone.takeoff(target_alt)

    def land(self) -> None:
        self.mock_drone.land()

    def disarm(self) -> None:
        self.mock_drone.disarm()


async def safe_cancel(task: Optional[asyncio.Task]) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def telemetry_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "lat": snapshot.lat,
        "lon": snapshot.lon,
        "alt": snapshot.alt,
        "heading": snapshot.heading,
        "speed": snapshot.speed,
        "battery_percent": snapshot.battery_percent,
        "mode": snapshot.mode,
        "armed": snapshot.armed,
        "gps_fix_type": snapshot.gps_fix_type,
        "satellites_visible": snapshot.satellites_visible,
        "timestamp": getattr(snapshot, "timestamp", None),
    }


async def call_blocking_method(obj: Any, method_name: str, *args: Any) -> Any:
    method = getattr(obj, method_name, None)
    if method is None:
        raise AttributeError(
            f"{obj.__class__.__name__} does not have {method_name}()")
    return await asyncio.to_thread(method, *args)


async def arm_drone(mission_executor: Any) -> None:
    # arm_and_takeoff combines arming and takeoff
    # This function is kept for compatibility but arming happens during arm_and_takeoff
    pass


async def takeoff_drone(mission_executor: Any, altitude: float) -> None:
    if hasattr(mission_executor, "arm_and_takeoff"):
        await call_blocking_method(mission_executor, "arm_and_takeoff", altitude)
        return

    if hasattr(mission_executor, "takeoff"):
        await call_blocking_method(mission_executor, "takeoff", altitude)
        return

    raise AttributeError(
        "MissionExecutor does not support takeoff() or arm_and_takeoff()")


async def land_drone(mission_executor: Any) -> None:
    if hasattr(mission_executor, "land"):
        await call_blocking_method(mission_executor, "land")
        return
    raise AttributeError("MissionExecutor does not support land()")


async def disarm_drone(mission_executor: Any) -> None:
    if hasattr(mission_executor, "disarm"):
        await call_blocking_method(mission_executor, "disarm")
        return
    raise AttributeError("MissionExecutor does not support disarm()")


async def wait_until_altitude_reached(
    telemetry_manager: TelemetryManager,
    target_altitude: float,
    tolerance: float = 0.7,
    timeout_sec: int = ALTITUDE_REACHED_TIMEOUT_SEC,
) -> None:
    start = asyncio.get_running_loop().time()

    while True:
        snapshot = telemetry_manager.snapshot()
        current_alt = snapshot.alt

        if current_alt is not None and current_alt >= target_altitude - tolerance:
            print(f"[FLIGHT] Reached target altitude: {current_alt} m")
            return

        if asyncio.get_running_loop().time() - start > timeout_sec:
            print("[FLIGHT] Altitude wait timed out, continuing")
            return

        await asyncio.sleep(1)


async def wait_until_landed(
    telemetry_manager: TelemetryManager,
    timeout_sec: int = LANDING_WAIT_TIMEOUT_SEC,
) -> None:
    start = asyncio.get_running_loop().time()

    while True:
        snapshot = telemetry_manager.snapshot()
        current_alt = snapshot.alt
        armed = snapshot.armed

        if current_alt is not None and current_alt <= 0.3:
            print(f"[FLIGHT] Landing detected, altitude={current_alt}")
            return

        if armed is False:
            print("[FLIGHT] Drone reports disarmed during landing wait")
            return

        if asyncio.get_running_loop().time() - start > timeout_sec:
            print("[FLIGHT] Landing wait timed out, continuing")
            return

        await asyncio.sleep(1)


async def emergency_land_and_disarm(
    mission_executor: Any,
    telemetry_manager: TelemetryManager,
) -> None:
    print("[SAFETY] Emergency landing sequence started")

    if mission_executor is None:
        print("[SAFETY] Mission executor is None, skipping emergency landing")
        return

    try:
        await land_drone(mission_executor)
    except Exception as e:
        print(f"[SAFETY] Failed to send land command: {e}")

    try:
        await wait_until_landed(telemetry_manager, timeout_sec=20)
    except Exception as e:
        print(f"[SAFETY] Error while waiting for landing: {e}")

    try:
        await disarm_drone(mission_executor)
        print("[SAFETY] Drone disarmed")
    except Exception as e:
        print(f"[SAFETY] Failed to disarm drone: {e}")


async def flight_sequence(
    mission_executor: Any,
    telemetry_manager: TelemetryManager,
    shutdown_event: asyncio.Event,
    launch_altitude: float,
) -> None:
    print("[FLIGHT] Starting autonomous sequence")

    if shutdown_event.is_set():
        return

    target_altitude = launch_altitude + TAKEOFF_ALT_METERS
    print(
        f"[FLIGHT] Arming and taking off to {target_altitude} m (launch alt {launch_altitude} m + {TAKEOFF_ALT_METERS} m)")

    try:
        await takeoff_drone(mission_executor, target_altitude)
    except Exception as e:
        print(f"[FLIGHT] Takeoff failed: {e}")
        await emergency_land_and_disarm(mission_executor, telemetry_manager)
        return

    try:
        await wait_until_altitude_reached(telemetry_manager, target_altitude)

        print(f"[FLIGHT] Hovering for {HOVER_DURATION_SEC} seconds")
        for _ in range(HOVER_DURATION_SEC):
            if shutdown_event.is_set():
                return
            await asyncio.sleep(1)

        if shutdown_event.is_set():
            return

        print("[FLIGHT] Landing drone")
        await land_drone(mission_executor)

        await wait_until_landed(telemetry_manager)

        if shutdown_event.is_set():
            return

        print("[FLIGHT] Disarming drone")
        await disarm_drone(mission_executor)

        print("[FLIGHT] Autonomous sequence complete")

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[FLIGHT] Exception during flight sequence: {e}")
        await emergency_land_and_disarm(mission_executor, telemetry_manager)
        return


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
    controller: Optional[DroneController] = None

    shutdown_event = asyncio.Event()

    async def start_emergency_shutdown() -> None:
        print("[SAFETY] Emergency shutdown triggered")
        if controller:
            await controller.emergency_land_and_disarm()

    def handle_sigint() -> None:
        shutdown_event.set()
        asyncio.create_task(start_emergency_shutdown())

    try:
        if config.use_mock_drone:
            print("[BOOT] Using mock drone mode")
            mock_drone = MockDrone(telemetry_manager)
            mission_executor = MockMissionExecutor(mock_drone, config)
        else:
            print("[BOOT] Attempting real MAVLink connection...")
            connection = MavlinkConnectionManager(
                connection_string=config.mavlink_connection_string,
                baud=config.mavlink_baud,
            )

            # Try to connect in a background thread with a timeout
            connection_result = {"success": False, "exception": None}

            def connect_in_thread():
                try:
                    connection.connect()
                    connection_result["success"] = True
                except Exception as e:
                    connection_result["exception"] = e

            connection_thread = threading.Thread(
                target=connect_in_thread, daemon=True)
            connection_thread.start()
            connection_thread.join(timeout=10.0)  # Wait up to 10 seconds

            if connection_result["success"]:
                adapter = MavlinkAdapter(
                    connection=connection,
                    telemetry_manager=telemetry_manager,
                    drone_id=config.drone_id,
                )
                mission_executor = MissionExecutor(adapter, config)
                print("[BOOT] Real MAVLink connection ready")
                adapter.poll_telemetry()
            else:
                if connection_result["exception"]:
                    print(
                        f"[BOOT] MAVLink connection failed: {connection_result['exception']}")
                else:
                    print("[BOOT] MAVLink connection timed out (10s)")
                print("[BOOT] Falling back to mock mode")
                mock_drone = MockDrone(telemetry_manager)
                mission_executor = MockMissionExecutor(mock_drone, config)
                config.use_mock_drone = True

        controller = DroneController(
            mission_executor, telemetry_manager, config)
        await controller.initialize()

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, handle_sigint)
            loop.add_signal_handler(signal.SIGTERM, handle_sigint)
        except NotImplementedError:
            print("[WARN] Signal handlers not available on this platform")

        # Start API server early so the UI is always reachable
        api_thread = threading.Thread(
            target=start_api_server, args=(controller, config))
        api_thread.daemon = True
        api_thread.start()
        print(f"[API] Server started on port {config.api_port}")

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
            initial_snapshot = telemetry_manager.snapshot()
            register_payload = {
                "msg_type": "drone_registration",
                "drone_id": config.drone_id,
                "drone_type": config.drone_type,
                "model": config.model,
                "capabilities": {
                    "camera": None,
                    "led": None,
                    "spotlight": False,
                    "speaker": False,
                    "max_speed": int(config.max_speed_m_s),
                },
                "telemetry": telemetry_dict(initial_snapshot),
            }
            print("[DEBUG REGISTER PAYLOAD]", register_payload)
            await ws_client.send_json(register_payload)
            print("[WS] Register payload sent")

        async def telemetry_loop() -> None:
            while not shutdown_event.is_set():
                try:
                    if config.use_mock_drone:
                        if mock_drone is not None:
                            mock_drone.tick()
                            telemetry_manager.update(
                                lat=mock_drone.lat,
                                lon=mock_drone.lon,
                                alt=mock_drone.alt,
                                heading=mock_drone.heading,
                                speed=mock_drone.speed,
                                battery_percent=mock_drone.battery,
                                mode=mock_drone.mode,
                                armed=mock_drone.armed,
                            )
                    else:
                        if adapter is not None:
                            adapter.poll_telemetry()

                    snapshot = telemetry_manager.snapshot()
                    payload = {
                        "msg_type": "telemetry",
                        "drone_id": config.drone_id,
                        "telemetry": telemetry_dict(snapshot),
                    }

                    if backend_connected:
                        await ws_client.send_json(payload)
                    else:
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
            while not shutdown_event.is_set():
                await asyncio.sleep(config.heartbeat_interval_sec)

        telemetry_task = asyncio.create_task(
            telemetry_loop(), name="telemetry_loop")
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(), name="heartbeat_loop")

        # Wait for shutdown
        while not shutdown_event.is_set():
            await asyncio.sleep(0.1)

        await asyncio.sleep(2)

    except KeyboardInterrupt:
        print("[SAFETY] KeyboardInterrupt fallback triggered")
        if controller:
            try:
                await controller.emergency_land_and_disarm()
            except Exception as e:
                print(f"[SAFETY] Fallback landing failed: {e}")

    finally:
        print("[SHUTDOWN] Starting cleanup sequence")
        shutdown_event.set()

        await safe_cancel(telemetry_task)
        await safe_cancel(heartbeat_task)

        if controller:
            await controller.shutdown()

        if backend_connected:
            try:
                await ws_client.close()
                print("[SHUTDOWN] WebSocket closed")
            except Exception as e:
                print(f"[SHUTDOWN] Error closing WebSocket: {e}")

        print("[SHUTDOWN] Cleanup complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[EXIT] Program stopped by user")
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
