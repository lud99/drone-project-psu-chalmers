from __future__ import annotations

import math
import time
from typing import Any, Mapping

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

    def is_connected(self) -> bool:
        return self.master is not None

    def drain_messages(self, duration: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue

            msg_type = msg.get_type()
            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")
            elif msg_type == "SYS_STATUS":
                print("[DEBUG] Received SYS_STATUS")
            elif msg_type == "EKF_STATUS_REPORT":
                print("[DEBUG] Received EKF_STATUS_REPORT")

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
                "GUIDED": "TAKEOFF",
                "AUTO.TAKEOFF": "TAKEOFF",
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

    def set_mode(self, mode: str) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        mode_mapping = self.get_mode_mapping()
        resolved_mode = self._resolve_mode_name(mode)

        print(
            f"[DEBUG] set_mode requested={mode}, resolved={resolved_mode}, "
            f"supported={list(mode_mapping.keys())}"
        )

        mode_value = mode_mapping[resolved_mode]
        if isinstance(mode_value, tuple):
            mode_id = mode_value[1]
        else:
            mode_id = mode_value

        self.master.set_mode(mode_id)

    def print_status_messages(self, duration: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.master.recv_match(type="STATUSTEXT", blocking=False)
            if msg is not None:
                print(f"[PX4 STATUS] {msg.text}")
            time.sleep(0.1)

    # def arm(self) -> None:
    #     if self.master is None:
    #         raise RuntimeError("MAVLink not connected")

    #     print("[DEBUG] Sending arm command")

    #     self.master.mav.command_long_send(
    #         self.master.target_system,
    #         self.master.target_component,
    #         mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    #         0,
    #         1, 0, 0, 0, 0, 0, 0,
    #     )

    #     start = time.time()
    #     timeout = 10

    #     while time.time() - start < timeout:
    #         msg = self.master.recv_match(blocking=True, timeout=1)
    #         if msg is None:
    #             continue

    #         msg_type = msg.get_type()

    #         if msg_type == "STATUSTEXT":
    #             print(f"[PX4 STATUS] {msg.text}")

    #         elif msg_type == "COMMAND_ACK":
    #             print(
    #                 f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}"
    #             )

    #         elif msg_type == "HEARTBEAT":
    #             armed = bool(msg.base_mode &
    #                          mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    #             print(f"[DEBUG] Armed status: {armed}")
    #             if armed:
    #                 print("[DEBUG] Motors armed")
    #                 return

    #     raise TimeoutError("Drone did not arm within 10 seconds")

    def arm(self) -> None:
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
        timeout = 10

        while time.time() - start < timeout:
            msg = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue

            msg_type = msg.get_type()

            # 👇 THIS IS THE NEW PART
            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

            elif msg_type == "HEARTBEAT":
                armed = bool(msg.base_mode &
                             mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                print(f"[DEBUG] Armed status: {armed}")
                if armed:
                    print("[DEBUG] Motors armed")
                    return

        raise TimeoutError("Drone did not arm within 10 seconds")

    def arm_and_takeoff(self, altitude: float | None = None) -> None:
        alt = altitude if altitude is not None else self.config.default_takeoff_alt

        print("[DEBUG] Setting POSCTL mode")
        self.adapter.set_mode("POSCTL")
        time.sleep(1)

        print("[DEBUG] Arming drone")
        self.adapter.arm()
        time.sleep(2)

        print(f"[DEBUG] Sending takeoff command to altitude={alt}")
        self.adapter.takeoff(alt)

    def disarm(self) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
        self.master.motors_disarmed_wait()

    def takeoff(self, altitude: float) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        if self.is_px4():
            mode_mapping = self.get_mode_mapping()
            if "TAKEOFF" in mode_mapping:
                self.set_mode("TAKEOFF")
                time.sleep(1)

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0, altitude,
        )

    def rtl(self) -> None:
        self.set_mode("RTL")

    def land(self) -> None:
        self.set_mode("LAND")

    def goto_location(
        self,
        lat: float,
        lon: float,
        alt: float,
        yaw: float | None = None,
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

    def recv_match(self, *args, **kwargs):
        if self.master is None:
            return None
        return self.master.recv_match(*args, **kwargs)
