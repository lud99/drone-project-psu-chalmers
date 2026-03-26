from typing import Any, Dict, Optional
import time
import asyncio

from mavlink_client.mavlink_connection import MavlinkConnectionManager
from mavlink_client.telemetry_manager import TelemetryManager


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

    def _snapshot(self):
        return self.telemetry_manager.snapshot()

    def _current_mode(self) -> str:
        snapshot = self._snapshot()
        mode = getattr(snapshot, "mode", None)
        return mode if mode else "UNKNOWN"

    def _is_armed(self) -> bool:
        snapshot = self._snapshot()
        return bool(getattr(snapshot, "armed", False))

    def _current_alt(self) -> Optional[float]:
        snapshot = self._snapshot()
        return getattr(snapshot, "alt", None)

    def _drain_messages(self, duration: float = 0.5) -> None:
        end_time = time.time() + duration
        while time.time() < end_time:
            got_message = self.poll_telemetry_once()
            if not got_message:
                time.sleep(0.05)

    def _wait_for_mode(self, target_mode: str, timeout: float = 5.0) -> bool:
        end_time = time.time() + timeout
        target = target_mode.upper()

        while time.time() < end_time:
            self._drain_messages(duration=0.2)
            current_mode = self._current_mode().upper()
            armed = self._is_armed()
            print(f"[DEBUG] HEARTBEAT mode={current_mode} armed={armed}")

            if current_mode == target:
                print(f"[DEBUG] Mode changed to {target}")
                return True

            time.sleep(0.1)

        print(f"[WARN] Timed out waiting for mode change to {target}")
        return False

    def wait_for_valid_mode(self, timeout: float = 5.0) -> str:
        end_time = time.time() + timeout

        while time.time() < end_time:
            self._drain_messages(duration=0.2)
            mode = self._current_mode()
            if mode and mode != "UNKNOWN":
                return mode
            time.sleep(0.1)

        return "UNKNOWN"

    def _wait_for_arm_state(self, target_armed: bool, timeout: float = 8.0) -> bool:
        end_time = time.time() + timeout

        while time.time() < end_time:
            self._drain_messages(duration=0.2)
            armed = self._is_armed()
            mode = self._current_mode()
            print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")

            if armed == target_armed:
                return True

            time.sleep(0.1)

        state = "armed" if target_armed else "disarmed"
        print(f"[WARN] Timed out waiting for motors to become {state}")
        return False

    def _wait_for_altitude_gain(
        self,
        start_alt: Optional[float],
        min_gain: float = 0.7,
        timeout: float = 10.0,
    ) -> bool:
        end_time = time.time() + timeout

        while time.time() < end_time:
            self._drain_messages(duration=0.2)
            current_alt = self._current_alt()

            if current_alt is not None:
                if start_alt is None:
                    if current_alt >= min_gain:
                        return True
                else:
                    if current_alt >= start_alt + min_gain:
                        return True

            time.sleep(0.1)

        print("[WARN] Timed out waiting for altitude increase")
        return False

    def arm(self) -> bool:
        print("[DEBUG] Arming drone")
        self.connection.arm()

        if self._wait_for_arm_state(True, timeout=8.0):
            print("[DEBUG] Motors armed")
            return True

        print("[WARN] Arm command did not result in armed state")
        return False

    def disarm(self) -> bool:
        print("[DEBUG] Sending disarm command")
        self.connection.disarm()

        if self._wait_for_arm_state(False, timeout=6.0):
            print("[DEBUG] Motors disarmed")
            return True

        print("[WARN] Disarm command did not result in disarmed state")
        return False

    def takeoff(self, altitude: float) -> bool:
        start_alt = self._current_alt()

        print(f"[DEBUG] Sending takeoff command to {altitude} m")
        self.connection.takeoff(altitude)

        end_time = time.time() + 15.0
        saw_positive_climb = False

        while time.time() < end_time:
            self._drain_messages(duration=0.2)

            current_alt = self._current_alt()
            armed = self._is_armed()
            mode = self._current_mode()

            print(
                f"[DEBUG] TAKEOFF CHECK mode={mode} armed={armed} alt={current_alt}")

            if current_alt is not None:
                if start_alt is None:
                    climb = current_alt
                else:
                    climb = current_alt - start_alt

                if climb > 0.05:
                    saw_positive_climb = True

                if climb >= 0.7:
                    print("[DEBUG] Takeoff successful")
                    return True

            if not armed:
                if saw_positive_climb:
                    print("[ERROR] Drone disarmed after beginning climb")
                else:
                    print("[ERROR] Drone disarmed before meaningful climb")
                return False

            time.sleep(0.1)

        print("[WARN] No sufficient altitude increase detected")
        return False

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

    def land(self) -> bool:
        print("[DEBUG] Setting LAND mode")
        self.connection.land()
        return self._wait_for_mode("LAND", timeout=5.0)

    def set_mode(self, mode: str) -> bool:
        target_mode = mode.upper()
        print(f"[DEBUG] Setting {target_mode} mode")
        self.connection.set_mode(target_mode)
        return self._wait_for_mode(target_mode, timeout=5.0)

    def poll_telemetry(self) -> None:
        processed_any = False
        while True:
            msg = self.connection.recv_match(blocking=False)
            if not msg:
                break
            processed_any = True
            self._handle_message(msg)

        if not processed_any:
            return

    def poll_telemetry_once(self) -> bool:
        msg = self.connection.recv_match(blocking=False)
        if not msg:
            return False

        self._handle_message(msg)
        return True

    def _handle_message(self, msg: Any) -> None:
        msg_type = msg.get_type()
        now = time.time()

        if msg_type == "GLOBAL_POSITION_INT":
            raw_heading = getattr(msg, "hdg", 65535)
            heading = None if raw_heading == 65535 else int(
                round(raw_heading / 100.0))

            self.telemetry_manager.update(
                lat=msg.lat / 1e7,
                lon=msg.lon / 1e7,
                heading=heading,
                speed=((msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2) ** 0.5) / 100.0,
                timestamp=now,
            )

            if getattr(msg, "relative_alt", None) is not None:
                self.telemetry_manager.update(
                    alt=msg.relative_alt / 1000.0,
                    timestamp=now,
                )

        elif msg_type == "ALTITUDE":
            alt_relative = getattr(msg, "altitude_relative", None)
            if alt_relative is not None:
                self.telemetry_manager.update(
                    alt=alt_relative,
                    timestamp=now,
                )

        elif msg_type == "VFR_HUD":
            raw_heading = getattr(msg, "heading", None)
            heading = None if raw_heading is None else int(round(raw_heading))

            self.telemetry_manager.update(
                heading=heading,
                speed=getattr(msg, "groundspeed", None),
                timestamp=now,
            )

        elif msg_type == "SYS_STATUS":
            battery = None
            if getattr(msg, "battery_remaining", -1) is not None and msg.battery_remaining >= 0:
                battery = int(msg.battery_remaining)

            self.telemetry_manager.update(
                battery_percent=battery,
                timestamp=now,
            )

        elif msg_type == "GPS_RAW_INT":
            self.telemetry_manager.update(
                gps_fix_type=getattr(msg, "fix_type", None),
                satellites_visible=getattr(msg, "satellites_visible", None),
                timestamp=now,
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
                timestamp=now,
            )

        elif msg_type == "STATUSTEXT":
            text = getattr(msg, "text", "")
            severity = getattr(msg, "severity", None)
            print(f"[STATUSTEXT] severity={severity} text={text}")

        elif msg_type == "COMMAND_ACK":
            command = getattr(msg, "command", None)
            result = getattr(msg, "result", None)
            print(f"[DEBUG] COMMAND_ACK command={command} result={result}")
