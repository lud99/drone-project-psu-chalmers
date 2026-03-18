from mavlink_client.mavlink_connection import MavlinkConnectionManager
from mavlink_client.telemetry_manager import TelemetryManager
from typing import Any, Dict, Optional
import time


class MavlinkAdapter:
    def __init__(
        self,
        connection: MavlinkConnectionManager,
        telemetry_manager: TelemetryManager,
        drone_id: str,
    ):
        self.connection = connection
        self.telemetry_manager = telemetry_manager
        self.drone_id = drone_id
        self.last_telemetry_time = 0.0

    def registration_data(self) -> Dict[str, Any]:
        return {
            "msg_type": "register",
            "drone_id": self.drone_id,
            "drone_type": "mavlink",
            "model": "PX4/ArduPilot",
            "capabilities": {
                "camera": None,
                "led": None,
                "spotlight": False,
                "speaker": False,
                "max_speed": 8,
            },
        }

    def arm(self) -> None:
        self.connection.arm()

    def disarm(self) -> None:
        self.connection.disarm()

    def takeoff(self, altitude: float) -> None:
        self.connection.takeoff(altitude)

    def go_to(
        self,
        lat: float,
        lon: float,
        alt: float,
        heading: Optional[float] = None,
    ) -> None:
        self.connection.goto_location(lat, lon, alt, heading)

    def return_to_home(self) -> None:
        self.connection.rtl()

    def land(self) -> None:
        self.connection.land()

    def set_mode(self, mode: str) -> None:
        self.connection.set_mode(mode)

    def poll_telemetry(self) -> None:
        msg = self.connection.recv_match(blocking=False)
        if not msg:
            return

        msg_type = msg.get_type()

        if msg_type == "GLOBAL_POSITION_INT":
            self.telemetry_manager.update(
                lat=msg.lat / 1e7,
                lon=msg.lon / 1e7,
                heading=(msg.hdg / 100.0) if getattr(msg,
                                                     "hdg", 65535) != 65535 else None,
                speed=((msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2) ** 0.5) / 100.0,
                timestamp=time.time(),
            )

            if getattr(msg, "relative_alt", None) is not None:
                self.telemetry_manager.update(
                    alt=msg.relative_alt / 1000.0,
                    timestamp=time.time(),
                )

        elif msg_type == "ALTITUDE":
            alt_relative = getattr(msg, "altitude_relative", None)
            if alt_relative is not None:
                self.telemetry_manager.update(
                    alt=alt_relative,
                    timestamp=time.time(),
                )

        # elif msg_type == "LOCAL_POSITION_NED":
        #     z = getattr(msg, "z", None)
        #     if z is not None:
        #         self.telemetry_manager.update(
        #             alt=-z,
        #             timestamp=time.time(),
        #         )

        elif msg_type == "VFR_HUD":
            self.telemetry_manager.update(
                heading=getattr(msg, "heading", None),
                speed=getattr(msg, "groundspeed", None),
                timestamp=time.time(),
            )

        elif msg_type == "SYS_STATUS":
            battery = None
            if getattr(msg, "battery_remaining", -1) is not None and msg.battery_remaining >= 0:
                battery = int(msg.battery_remaining)

            self.telemetry_manager.update(
                battery_percent=battery,
                timestamp=time.time(),
            )

        elif msg_type == "GPS_RAW_INT":
            self.telemetry_manager.update(
                gps_fix_type=getattr(msg, "fix_type", None),
                satellites_visible=getattr(msg, "satellites_visible", None),
                timestamp=time.time(),
            )

        elif msg_type == "HEARTBEAT":
            mode = "UNKNOWN"
            master = self.connection.master
            if master is not None:
                try:
                    mode = master.flightmode
                except Exception:
                    pass

            armed = False
            try:
                armed = bool(msg.base_mode & 128)
            except Exception:
                pass

            self.telemetry_manager.update(
                mode=mode,
                armed=armed,
                timestamp=time.time(),
            )
