import asyncio
import math
from enum import Enum
from typing import Any, Optional, Dict, List, Tuple
import logging

from .config import Config
from .telemetry_manager import TelemetryManager
from .geofence import Geofence, haversine_distance
from .commands import Command, CommandType, StatusResponse, Point


logger = logging.getLogger(__name__)


class DroneState(str, Enum):
    IDLE = "idle"
    ARMING = "arming"
    TAKING_OFF = "taking_off"
    HOVERING = "hovering"
    NAVIGATING = "navigating"
    HOLDING = "holding"
    LANDING = "landing"
    DISARMING = "disarming"
    EMERGENCY = "emergency"
    ERROR = "error"


class DroneController:
    def __init__(self, mission_executor: Any, telemetry_manager: TelemetryManager, config: Config):
        self.mission_executor = mission_executor
        self.telemetry_manager = telemetry_manager
        self.config = config
        self.state = DroneState.IDLE
        self.current_command: Optional[str] = None
        self.geofence = Geofence()
        self.launch_altitude = 0.0
        self.latest_safety_message: Optional[str] = None
        self._navigation_task: Optional[asyncio.Task] = None
        self._geofence_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize controller, set launch altitude."""
        snapshot = self.telemetry_manager.snapshot()
        self.launch_altitude = snapshot.alt or 0.0
        logger.info(
            f"Controller initialized with launch altitude: {self.launch_altitude}")

    async def execute_command(self, command: Command) -> Dict[str, Any]:
        """Execute a command and return response."""
        logger.info(f"Received command: {command.type}")
        if command.type == CommandType.ARM:
            return await self._arm()
        elif command.type == CommandType.TAKEOFF_TO_RELATIVE_ALTITUDE:
            data = command.data or {}
            alt = data.get("relative_altitude_m",
                           self.config.default_takeoff_alt)
            return await self._takeoff_to_altitude(alt)
        elif command.type == CommandType.LAND:
            return await self._land()
        elif command.type == CommandType.DISARM:
            return await self._disarm()
        elif command.type == CommandType.GOTO_POINT:
            data = command.data or {}
            if data.get("latitude") is not None and data.get("longitude") is not None:
                return await self._goto_point(
                    data["latitude"], data["longitude"],
                    data.get("relative_altitude_m"),
                    data.get("yaw_deg"),
                    data.get("acceptance_radius_m",
                             self.config.goto_acceptance_radius_m)
                )
            return await self._goto_relative(
                data["distance_m"],
                data["direction"],
                data.get("relative_altitude_m"),
                data.get("yaw_deg"),
                data.get("acceptance_radius_m",
                         self.config.goto_acceptance_radius_m)
            )
        elif command.type == CommandType.SET_POLYGON_GEOFENCE:
            data = command.data or {}
            return await self._set_geofence(data["polygon"])
        elif command.type == CommandType.CLEAR_POLYGON_GEOFENCE:
            return await self._clear_geofence()
        elif command.type == CommandType.SET_CIRCULAR_GEOFENCE:
            data = command.data or {}
            return await self._set_circular_geofence(
                data["latitude"], data["longitude"], data["radius_m"]
            )
        elif command.type == CommandType.CLEAR_CIRCULAR_GEOFENCE:
            return await self._clear_circular_geofence()
        elif command.type == CommandType.HOLD:
            return await self._hold()
        elif command.type == CommandType.GET_STATUS:
            return await self._get_status()
        else:
            return {"success": False, "message": f"Unknown command: {command.type}"}

    async def _arm(self) -> Dict[str, Any]:
        if self.state not in [DroneState.IDLE, DroneState.ERROR, DroneState.HOLDING]:
            return {"success": False, "message": f"Cannot arm in state: {self.state}"}
        try:
            self.state = DroneState.ARMING
            self.current_command = "arm"
            if hasattr(self.mission_executor, "arm"):
                # Let the mission executor arm flow report concrete FC rejections.
                # Wrapping this thread call in wait_for can raise a false timeout
                # while the underlying arm sequence is still running.
                await asyncio.to_thread(self.mission_executor.arm)
            else:
                # minimal takeoff
                await asyncio.to_thread(
                    self.mission_executor.arm_and_takeoff,
                    self.launch_altitude + 1,
                )
            self.state = DroneState.IDLE
            logger.info("Drone armed successfully")
            return {"success": True, "message": "Drone armed"}
        except asyncio.TimeoutError:
            self.state = DroneState.ERROR
            logger.error("Arm failed: command timed out")
            snapshot = self.telemetry_manager.snapshot()
            mode = snapshot.mode if snapshot.mode is not None else "UNKNOWN"
            armed = snapshot.armed
            return {
                "success": False,
                "message": (
                    "Arm failed: command timed out while waiting for FC response "
                    f"(mode={mode}, armed={armed}). "
                    "If mode is LAND and does not change, clear prearm checks in FC/QGC and retry."
                ),
            }
        except Exception as e:
            self.state = DroneState.ERROR
            logger.error(f"Arm failed: {e}")
            return {"success": False, "message": f"Arm failed: {e}"}

    async def _takeoff_to_altitude(self, relative_alt: float) -> Dict[str, Any]:
        if self.state not in [DroneState.IDLE, DroneState.HOVERING, DroneState.HOLDING]:
            return {"success": False, "message": f"Cannot takeoff in state: {self.state}"}
        if not (self.config.min_relative_altitude_m <= relative_alt <= self.config.max_relative_altitude_m):
            return {"success": False, "message": f"Altitude {relative_alt} out of range [{self.config.min_relative_altitude_m}, {self.config.max_relative_altitude_m}]"}
        try:
            self.state = DroneState.TAKING_OFF
            self.current_command = f"takeoff to {relative_alt}m"
            target_alt = self.launch_altitude + relative_alt
            if hasattr(self.mission_executor, "takeoff"):
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self.mission_executor.takeoff, target_alt),
                    timeout=self.config.takeoff_command_timeout_sec,
                )
            else:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self.mission_executor.arm_and_takeoff, target_alt),
                    timeout=self.config.takeoff_command_timeout_sec,
                )
            # Wait for altitude
            await self._wait_for_altitude(target_alt)
            self.state = DroneState.HOVERING
            logger.info(f"Takeoff to {relative_alt}m completed")
            return {"success": True, "message": f"Reached altitude {relative_alt}m"}
        except asyncio.TimeoutError:
            self.state = DroneState.ERROR
            logger.error("Takeoff failed: command timed out")
            return {
                "success": False,
                "message": (
                    "Takeoff failed: command timed out while waiting for FC response "
                    "(possible serial reconnect instability)"
                ),
            }
        except Exception as e:
            self.state = DroneState.ERROR
            logger.error(f"Takeoff failed: {e}")
            return {"success": False, "message": f"Takeoff failed: {e}"}

    async def _land(self) -> Dict[str, Any]:
        if self.state in [DroneState.EMERGENCY, DroneState.LANDING]:
            return {"success": False, "message": "Already landing"}
        try:
            # Cancel any navigation
            if self._navigation_task and not self._navigation_task.done():
                self._navigation_task.cancel()
            self.state = DroneState.LANDING
            self.current_command = "land"
            await asyncio.to_thread(self.mission_executor.land)
            await self._wait_for_landed()
            self.state = DroneState.IDLE
            logger.info("Landing completed")
            return {"success": True, "message": "Landed successfully"}
        except Exception as e:
            self.state = DroneState.ERROR
            logger.error(f"Land failed: {e}")
            return {"success": False, "message": f"Land failed: {e}"}

    async def _disarm(self) -> Dict[str, Any]:
        if self.state not in [DroneState.IDLE, DroneState.ERROR]:
            return {"success": False, "message": f"Cannot disarm in state: {self.state}"}
        try:
            self.state = DroneState.DISARMING
            self.current_command = "disarm"
            await asyncio.to_thread(self.mission_executor.disarm)
            self.state = DroneState.IDLE
            logger.info("Drone disarmed")
            return {"success": True, "message": "Disarmed"}
        except Exception as e:
            self.state = DroneState.ERROR
            logger.error(f"Disarm failed: {e}")
            return {"success": False, "message": f"Disarm failed: {e}"}

    async def _goto_point(self, lat: float, lon: float, rel_alt: Optional[float] = None,
                          yaw: Optional[float] = None, radius: float = 2.0) -> Dict[str, Any]:
        if self.state not in [DroneState.HOVERING, DroneState.NAVIGATING, DroneState.HOLDING]:
            return {"success": False, "message": f"Cannot goto in state: {self.state}"}
        # Check geofence
        if not self.geofence.is_point_inside(lat, lon):
            return {"success": False, "message": "Target point outside geofence"}
        # Validate telemetry
        snapshot = self.telemetry_manager.snapshot()
        if snapshot.gps_fix_type is None or snapshot.gps_fix_type < 3:
            return {"success": False, "message": "GPS fix insufficient"}
        if snapshot.armed is not True:
            return {"success": False, "message": "Goto requires drone to be armed and airborne"}
        try:
            self.state = DroneState.NAVIGATING
            self.current_command = f"goto {lat},{lon}"
            alt = rel_alt if rel_alt is not None else (
                snapshot.alt or self.launch_altitude + 10)
            await asyncio.to_thread(self.mission_executor.fly_to_coordinate, lat, lon, alt, yaw)
            # Start monitoring
            self._navigation_task = asyncio.create_task(
                self._monitor_goto(lat, lon, alt, radius))
            logger.info(f"Goto {lat},{lon} initiated")
            return {"success": True, "message": f"Navigating to {lat},{lon}"}
        except Exception as e:
            self.state = DroneState.ERROR
            logger.error(f"Goto failed: {e}")
            return {"success": False, "message": f"Goto failed: {e}"}

    def _offset_coordinate(
        self,
        lat: float,
        lon: float,
        distance_m: float,
        direction: str,
    ) -> Tuple[float, float]:
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat))
        direction = direction.upper()

        target_lat = lat
        target_lon = lon

        if direction == "N":
            target_lat += distance_m / meters_per_degree_lat
        elif direction == "S":
            target_lat -= distance_m / meters_per_degree_lat
        elif direction == "E":
            if abs(meters_per_degree_lon) < 1e-6:
                raise ValueError(
                    "Cannot compute east/west offset near the poles")
            target_lon += distance_m / meters_per_degree_lon
        elif direction == "W":
            if abs(meters_per_degree_lon) < 1e-6:
                raise ValueError(
                    "Cannot compute east/west offset near the poles")
            target_lon -= distance_m / meters_per_degree_lon
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return target_lat, target_lon

    async def _goto_relative(
        self,
        distance_m: float,
        direction: str,
        rel_alt: Optional[float] = None,
        yaw: Optional[float] = None,
        radius: float = 2.0,
    ) -> Dict[str, Any]:
        snapshot = self.telemetry_manager.snapshot()
        current_lat = snapshot.lat
        current_lon = snapshot.lon

        if current_lat is None or current_lon is None:
            return {"success": False, "message": "Current GPS position unavailable"}

        target_lat, target_lon = self._offset_coordinate(
            current_lat,
            current_lon,
            distance_m,
            direction,
        )
        return await self._goto_point(target_lat, target_lon, rel_alt, yaw, radius)

    async def _monitor_goto(self, target_lat: float, target_lon: float, target_alt: float, radius: float):
        start_time = asyncio.get_event_loop().time()
        while not self._shutdown_event.is_set():
            snapshot = self.telemetry_manager.snapshot()
            current_lat = snapshot.lat
            current_lon = snapshot.lon
            current_alt = snapshot.alt
            armed = snapshot.armed
            if current_lat is None or current_lon is None or current_alt is None:
                continue

            # If FC reports disarmed while near ground, recover controller state.
            if armed is False and current_alt <= 0.5:
                self.state = DroneState.IDLE
                self.current_command = None
                logger.warning("Goto aborted: vehicle disarmed on ground, returning to IDLE")
                return

            dist = haversine_distance(
                current_lat, current_lon, target_lat, target_lon)
            alt_diff = abs((current_alt or 0) - target_alt)
            if dist <= radius and alt_diff <= 1.0:
                self.state = DroneState.HOVERING
                self.current_command = None
                logger.info("Goto target reached")
                return
            if asyncio.get_event_loop().time() - start_time > self.config.goto_timeout_sec:
                self.state = DroneState.HOLDING
                self.current_command = "hold (timeout)"
                logger.warning("Goto timeout, holding position")
                return
            await asyncio.sleep(1)

    def _reconcile_state_from_telemetry(self, snapshot: Any) -> None:
        """Keep controller state aligned with FC reality for recovery scenarios."""
        if snapshot.armed is False and (snapshot.alt is None or snapshot.alt <= 0.5):
            if self.state in {
                DroneState.TAKING_OFF,
                DroneState.HOVERING,
                DroneState.NAVIGATING,
                DroneState.HOLDING,
                DroneState.LANDING,
            }:
                self.state = DroneState.IDLE
                self.current_command = None

    async def _set_geofence(self, polygon: List[Dict]) -> Dict[str, Any]:
        points = [(p["latitude"], p["longitude"]) for p in polygon]
        if self.geofence.set_polygon(points):
            self._start_geofence_monitoring()
            logger.info("Polygon geofence set")
            return {"success": True, "message": "Polygon geofence set"}
        else:
            return {"success": False, "message": "Invalid polygon"}

    async def _clear_geofence(self) -> Dict[str, Any]:
        self.geofence.clear_polygon()
        if not self.geofence.has_active_fence() and self._geofence_task:
            self._geofence_task.cancel()
        logger.info("Polygon geofence cleared")
        return {"success": True, "message": "Polygon geofence cleared"}

    async def _set_circular_geofence(self, lat: float, lon: float, radius_m: float) -> Dict[str, Any]:
        if self.geofence.set_circle(lat, lon, radius_m):
            self._start_geofence_monitoring()
            logger.info(
                f"Circular geofence set: center ({lat}, {lon}), radius {radius_m} m")
            return {"success": True, "message": f"Circular geofence set: radius {radius_m} m"}
        else:
            return {"success": False, "message": "Invalid circular geofence parameters"}

    async def _clear_circular_geofence(self) -> Dict[str, Any]:
        self.geofence.clear_circle()
        if not self.geofence.has_active_fence() and self._geofence_task:
            self._geofence_task.cancel()
        logger.info("Circular geofence cleared")
        return {"success": True, "message": "Circular geofence cleared"}

    def _start_geofence_monitoring(self):
        if self._geofence_task:
            self._geofence_task.cancel()
        self._geofence_task = asyncio.create_task(self._monitor_geofence())

    async def _monitor_geofence(self):
        while not self._shutdown_event.is_set() and self.geofence.has_active_fence():
            snapshot = self.telemetry_manager.snapshot()
            lat = snapshot.lat
            lon = snapshot.lon
            if lat is not None and lon is not None:
                if not self.geofence.is_point_inside(lat, lon):
                    logger.warning("Geofence breach detected")
                    self.latest_safety_message = "Geofence breach"
                    if self.config.geofence_breach_action == "land":
                        await self._land()
                    else:
                        await self._hold()
            await asyncio.sleep(self.config.geofence_check_interval_sec)

    async def _hold(self) -> Dict[str, Any]:
        try:
            # Assume mission_executor has abort or set_mode to LOITER
            if hasattr(self.mission_executor, "abort"):
                await asyncio.to_thread(self.mission_executor.abort)
            self.state = DroneState.HOLDING
            self.current_command = "hold"
            logger.info("Hold command executed")
            return {"success": True, "message": "Holding position"}
        except Exception as e:
            logger.error(f"Hold failed: {e}")
            return {"success": False, "message": f"Hold failed: {e}"}

    async def _get_status(self) -> Dict[str, Any]:
        snapshot = self.telemetry_manager.snapshot()
        self._reconcile_state_from_telemetry(snapshot)
        polygon = None
        if self.geofence.polygon:
            polygon = [Point(latitude=lat, longitude=lon)
                       for lat, lon in self.geofence.polygon]
        return StatusResponse(
            state=self.state.value,
            current_command=self.current_command,
            geofence_active=self.geofence.has_active_fence(),
            polygon=polygon,
            circle=self.geofence.get_circle(),
            latest_safety_message=self.latest_safety_message,
            using_mock_drone=self.config.use_mock_drone,
            telemetry={
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
            }
        ).dict()

    async def _wait_for_altitude(self, target_alt: float, tolerance: float = 0.7, timeout: int = 30):
        start = asyncio.get_event_loop().time()
        while not self._shutdown_event.is_set():
            snapshot = self.telemetry_manager.snapshot()
            current_alt = snapshot.alt
            if current_alt is not None and abs(current_alt - target_alt) <= tolerance:
                return
            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError("Altitude not reached in time")
            await asyncio.sleep(1)

    async def _wait_for_landed(self, timeout: int = 45):
        start = asyncio.get_event_loop().time()
        while not self._shutdown_event.is_set():
            snapshot = self.telemetry_manager.snapshot()
            current_alt = snapshot.alt
            armed = snapshot.armed
            if current_alt is not None and current_alt <= 0.3:
                return
            if armed is False:
                return
            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError("Landing timeout")
            await asyncio.sleep(1)

    async def emergency_land_and_disarm(self):
        """Emergency shutdown."""
        self.state = DroneState.EMERGENCY
        self.latest_safety_message = "Emergency shutdown"
        try:
            await self._land()
            await self._disarm()
        except Exception as e:
            logger.error(f"Emergency land/disarm failed: {e}")

    async def shutdown(self):
        """Shutdown controller."""
        self._shutdown_event.set()

    async def shutdown(self):
        """Shutdown controller."""
        self._shutdown_event.set()
        if self._navigation_task and not self._navigation_task.done():
            self._navigation_task.cancel()
        if self._geofence_task and not self._geofence_task.done():
            self._geofence_task.cancel()
