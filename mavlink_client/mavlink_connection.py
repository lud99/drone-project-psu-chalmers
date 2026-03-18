from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional

from pymavlink import mavutil


class MavlinkConnectionManager:
    def __init__(self, connection_string: str, baud: int = 57600):
        self.connection_string = connection_string
        self.baud = baud
        self.master: Any = None
        self.heartbeat: Any = None

    def connect(self) -> None:
        self.master = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud,
        )
        self.heartbeat = self.master.wait_heartbeat(timeout=30)

        print(
            f"[MAVLINK] Connected. "
            f"sys={self.master.target_system}, comp={self.master.target_component}"
        )
        print(f"[DEBUG] autopilot={self.heartbeat.autopilot}")
        print(f"[DEBUG] vehicle_type={self.heartbeat.type}")

        mode_mapping = self.master.mode_mapping()
        print(
            f"[DEBUG] available_modes={list(mode_mapping.keys()) if mode_mapping else None}"
        )
        try:
            print(f"[DEBUG] initial flightmode={self.master.flightmode}")
        except Exception:
            print("[DEBUG] initial flightmode unavailable")

    def is_connected(self) -> bool:
        return self.master is not None

    def is_px4(self) -> bool:
        if self.heartbeat is None:
            return False
        return self.heartbeat.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4

    def get_mode_mapping(self) -> Mapping[str, Any]:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        mode_mapping = self.master.mode_mapping()
        if mode_mapping is None:
            raise RuntimeError("Mode mapping is unavailable")

        return mode_mapping

    def _resolve_mode_name(self, mode: str) -> str:
        mode_mapping = self.get_mode_mapping()
        requested_mode = mode.upper()

        if requested_mode in mode_mapping:
            return requested_mode

        if self.is_px4():
            px4_aliases = {
                "GUIDED": "LOITER",
                "AUTO.LOITER": "LOITER",
                "AUTO.RTL": "RTL",
                "AUTO.LAND": "LAND",
            }
            resolved = px4_aliases.get(requested_mode, requested_mode)
            if resolved in mode_mapping:
                return resolved

        raise ValueError(
            f"Unknown mode: {mode}. Supported modes: {list(mode_mapping.keys())}"
        )

    def set_mode(self, mode: str, wait: bool = True, timeout: float = 5.0) -> bool:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        resolved_mode = self._resolve_mode_name(mode)
        mode_mapping = self.get_mode_mapping()

        print(
            f"[DEBUG] set_mode requested={mode}, resolved={resolved_mode}, "
            f"supported={list(mode_mapping.keys())}"
        )

        # Let pymavlink handle PX4 vs ArduPilot correctly
        self.master.set_mode(resolved_mode)

        if not wait:
            return True

        start = time.time()
        while time.time() - start < timeout:
            msg = self.master.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

            elif msg_type == "HEARTBEAT":
                current_mode = "UNKNOWN"
                try:
                    current_mode = self.master.flightmode or "UNKNOWN"
                except Exception:
                    pass

                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                print(f"[DEBUG] HEARTBEAT mode={current_mode} armed={armed}")

                if current_mode.upper() == resolved_mode.upper():
                    print(f"[DEBUG] Mode changed to {current_mode}")
                    return True

        print(f"[WARN] Timed out waiting for mode change to {resolved_mode}")
        return False

    def print_status_messages(self, duration: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.master.recv_match(type="STATUSTEXT", blocking=False)
            if msg is not None:
                print(f"[PX4 STATUS] {msg.text}")
            time.sleep(0.1)

    def drain_messages(self, duration: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.master.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()
            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")
            elif msg_type == "SYS_STATUS":
                print("[DEBUG] Received SYS_STATUS")
            elif msg_type == "EKF_STATUS_REPORT":
                print("[DEBUG] Received EKF_STATUS_REPORT")
            elif msg_type == "HEARTBEAT":
                mode = "UNKNOWN"
                try:
                    mode = self.master.flightmode or "UNKNOWN"
                except Exception:
                    pass
                armed = bool(msg.base_mode &
                             mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")

    def arm(self, timeout: float = 10.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        print("[DEBUG] Sending arm command")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

        start = time.time()
        while time.time() - start < timeout:
            msg = self.master.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

            elif msg_type == "HEARTBEAT":
                armed = bool(msg.base_mode &
                             mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                mode = "UNKNOWN"
                try:
                    mode = self.master.flightmode or "UNKNOWN"
                except Exception:
                    pass
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")
                if armed:
                    print("[DEBUG] Motors armed")
                    return

        raise TimeoutError("Drone did not arm within timeout")

    def disarm(self) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        print("[DEBUG] Sending disarm command")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

        start = time.time()
        while time.time() - start < 5.0:
            msg = self.master.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

            elif msg_type == "HEARTBEAT":
                armed = bool(msg.base_mode &
                             mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                mode = "UNKNOWN"
                try:
                    mode = self.master.flightmode or "UNKNOWN"
                except Exception:
                    pass
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")
                if not armed:
                    print("[DEBUG] Motors disarmed")
                    return

        print("[WARN] Timed out waiting for disarm")

    def takeoff(self, altitude: float, timeout: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        print(f"[DEBUG] Sending MAV_CMD_NAV_TAKEOFF altitude={altitude}")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0, altitude,
        )

        start = time.time()
        while time.time() - start < timeout:
            msg = self.master.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")
                if msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        raise RuntimeError(
                            f"Takeoff command rejected with result={msg.result}"
                        )
                    print("[DEBUG] Takeoff command accepted")
                    return

            elif msg_type == "HEARTBEAT":
                mode = "UNKNOWN"
                try:
                    mode = self.master.flightmode or "UNKNOWN"
                except Exception:
                    pass
                armed = bool(msg.base_mode &
                             mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")

        raise TimeoutError("Did not receive takeoff ACK")

    def arm_and_takeoff(self, altitude: float = 2.0) -> None:
        print("[DEBUG] Setting LOITER mode")
        if not self.set_mode("LOITER"):
            raise RuntimeError("Failed to switch to LOITER mode")

        time.sleep(2)

        print("[DEBUG] Arming drone")
        self.arm()
        time.sleep(2)

        print(f"[DEBUG] Sending takeoff command to altitude={altitude}")
        self.takeoff(altitude)

    def rtl(self) -> None:
        print("[DEBUG] Setting RTL mode")
        if not self.set_mode("RTL"):
            raise RuntimeError("Failed to switch to RTL mode")

    def land(self) -> None:
        print("[DEBUG] Setting LAND mode")
        if not self.set_mode("LAND", wait=True):
            raise RuntimeError("Failed to switch to LAND mode")
        
    def goto_location(
        self,
        lat: float,
        lon: float,
        alt: float,
        yaw: Optional[float] = None,
    ) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )

        if yaw is None:
            type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE

        self.master.mav.set_position_target_global_int_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            type_mask,
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,
            0, 0, 0,
            0 if yaw is None else math.radians(yaw),
            0,
        )

        print(
            f"[DEBUG] Sent goto command lat={lat}, lon={lon}, alt={alt}, yaw={yaw}")

    def recv_match(self, *args, **kwargs):
        if self.master is None:
            return None
        return self.master.recv_match(*args, **kwargs)
