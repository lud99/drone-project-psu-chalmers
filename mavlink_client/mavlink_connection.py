from __future__ import annotations

import math
import time
from typing import List
from typing import Any, Mapping, Optional

from pymavlink import mavutil


class MavlinkConnectionManager:
    def __init__(self, connection_string: str, baud: int = 57600):
        self.connection_string = connection_string
        self.baud = baud
        self.master: Any = None
        self.heartbeat: Any = None
        self._reconnect_backoff_sec = 2.0
        self._next_reconnect_attempt_at = 0.0
        self._target_system: Optional[int] = None    # locked on first connect
        self._target_autopilot: Optional[int] = None  # locked on first connect
        self._target_component: Optional[int] = None  # locked on first connect

    def _connect_once(self, heartbeat_timeout: float = 30.0) -> None:
        self.master = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud,
        )

        # Determine which system ID + autopilot type to look for:
        #   - On reconnect: must match BOTH the previously-seen sys ID and
        #     autopilot type, so an ArduPilot source that also claims sys=1
        #     is not accepted in place of the real PX4 FC.
        #   - On first connect: any real FC (sys >= 1); sys=0 is the MAVLink
        #     broadcast/invalid address and is never a real flight controller.
        want_system = self._target_system      # None on first connect
        want_autopilot = self._target_autopilot  # None on first connect
        want_component = self._target_component  # None on first connect

        deadline = time.time() + heartbeat_timeout
        accepted_hb = None
        fallback_hb = None
        fallback_src = None
        while time.time() < deadline:
            hb = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if hb is None:
                continue
            src = hb.get_srcSystem()
            src_component = hb.get_srcComponent()
            if want_system is not None:
                # Reconnect: require exact sys ID + autopilot type, and when
                # known also the exact component id.
                component_ok = (
                    want_component in (None, 0)
                    or src_component == want_component
                )
                if (
                    src == want_system
                    and hb.autopilot == want_autopilot
                    and component_ok
                ):
                    accepted_hb = hb
                    self.master.target_system = src
                    self.master.target_component = src_component
                    break
            else:
                # First connect: skip sys=0 (broadcast/invalid).
                # Prefer PX4 if present on the link; keep a fallback so we can
                # still connect when PX4 is not available.
                if src >= 1:
                    if hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4:
                        accepted_hb = hb
                        self._target_system = src
                        self._target_autopilot = hb.autopilot
                        self._target_component = src_component
                        self.master.target_system = src
                        self.master.target_component = src_component
                        break

                    if fallback_hb is None:
                        fallback_hb = hb
                        fallback_src = src

        if accepted_hb is None and want_system is None and fallback_hb is not None:
            accepted_hb = fallback_hb
            self._target_system = fallback_src
            self._target_autopilot = fallback_hb.autopilot
            self._target_component = fallback_hb.get_srcComponent()
            self.master.target_system = fallback_src
            self.master.target_component = self._target_component

        if accepted_hb is None:
            self._close_master()
            if want_system is not None:
                raise RuntimeError(
                    f"Could not find heartbeat from expected system {want_system} "
                    f"autopilot {want_autopilot}"
                )
            else:
                raise RuntimeError("No valid heartbeat received within timeout")

        self.heartbeat = accepted_hb

        print(
            f"[MAVLINK] Connected. "
            f"sys={self.master.target_system}, comp={self.master.target_component}"
        )
        print(
            f"[DEBUG] locked target sys={self._target_system}, "
            f"comp={self._target_component}, autopilot={self._target_autopilot}"
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

    def _close_master(self) -> None:
        if self.master is None:
            return

        try:
            port = getattr(self.master, "port", None)
            if port is not None:
                port.close()
        except Exception:
            pass
        finally:
            self.master = None
            self.heartbeat = None

    def _should_treat_as_dead_device(self, error: Exception) -> bool:
        text = str(error)
        if isinstance(error, OSError):
            return True
        return (
            "Device not configured" in text
            or "is dead" in text
            or "Input/output error" in text
        )

    def _try_reconnect(self, force: bool = False) -> bool:
        now = time.time()
        if not force and now < self._next_reconnect_attempt_at:
            return False

        self._next_reconnect_attempt_at = now + self._reconnect_backoff_sec
        print(f"[MAVLINK] Reconnecting on {self.connection_string} ...")

        self._close_master()

        # macOS may take a few seconds to re-enumerate the USB serial device
        # after a transient drop.  Retry up to ~10 s before giving up.
        deadline = time.time() + 10.0
        last_exc: Exception = RuntimeError("Reconnect not attempted")
        while time.time() < deadline:
            try:
                self._connect_once(heartbeat_timeout=5.0)
                print("[MAVLINK] Reconnected")
                return True
            except Exception as e:
                last_exc = e
                no_such = "No such file" in str(e) or "could not open port" in str(e)
                if no_such:
                    print(f"[MAVLINK] Device not found yet, retrying... ({e})")
                    time.sleep(1.0)
                else:
                    break

        print(f"[MAVLINK] Reconnect failed: {last_exc}")
        return False

    def _ensure_connected(self) -> None:
        if self.master is not None:
            return

        if not self._try_reconnect(force=True):
            raise RuntimeError("MAVLink not connected")

    def _run_with_reconnect_retry(self, operation_name: str, op) -> Any:
        self._ensure_connected()

        try:
            return op()
        except Exception as e:
            if not self._should_treat_as_dead_device(e):
                raise

            print(f"[MAVLINK] {operation_name} failed: {e}")
            if not self._try_reconnect(force=True):
                raise

            print(f"[MAVLINK] Retrying {operation_name} after reconnect")
            return op()

    def connect(self) -> None:
        self._connect_once(heartbeat_timeout=30.0)

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

    def _current_mode(self) -> str:
        if self.master is None:
            return "UNKNOWN"

        try:
            return self.master.flightmode or "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _is_target_heartbeat(self, msg: Any) -> bool:
        """Accept heartbeats only from the connected flight controller."""
        if self.master is None:
            return False

        try:
            if msg.get_srcSystem() != self.master.target_system:
                return False

            # Prefer locked target component if known.
            target_component = self._target_component
            if target_component in (None, 0):
                target_component = self.master.target_component

            if target_component in (None, 0):
                return True

            return msg.get_srcComponent() == target_component
        except Exception:
            return False

    def _command_target_component(self) -> int:
        if self.master is None:
            return mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

        target_component = self.master.target_component
        if target_component in (None, 0):
            target_component = self._target_component

        if target_component in (None, 0):
            return mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

        return int(target_component)

    def _wait_for_armed_state(self, target_armed: bool, timeout: float) -> bool:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        start = time.time()
        saw_arm_ack = False
        while time.time() - start < timeout:
            msg = self.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")
                if (
                    target_armed
                    and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
                    and msg.result in (
                        mavutil.mavlink.MAV_RESULT_ACCEPTED,
                        mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
                    )
                ):
                    saw_arm_ack = True

            elif msg_type == "HEARTBEAT":
                if not self._is_target_heartbeat(msg):
                    continue

                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                mode = self._current_mode()
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")

                if armed == target_armed:
                    return True

        # Sometimes arming ACK arrives, but the armed heartbeat bit flips a
        # little later. Give it a short grace period before declaring failure.
        if target_armed and saw_arm_ack:
            grace_end = time.time() + 3.0
            while time.time() < grace_end:
                msg = self.recv_match(blocking=True, timeout=0.5)
                if msg is None or msg.get_type() != "HEARTBEAT":
                    continue
                if not self._is_target_heartbeat(msg):
                    continue
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                if armed:
                    return True

        return False

    def set_mode(self, mode: str, wait: bool = True, timeout: float = 5.0) -> bool:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        resolved_mode = self._resolve_mode_name(mode)
        mode_mapping = self.get_mode_mapping()

        print(
            f"[DEBUG] set_mode requested={mode}, resolved={resolved_mode}, "
            f"supported={list(mode_mapping.keys())}"
        )

        def _set_mode() -> None:
            if self.master is None:
                raise RuntimeError("MAVLink not connected")
            mode_id = mode_mapping[resolved_mode]
            # pymavlink mode_mapping can return either a plain integer or a
            # [base_mode_flags, custom_mode] list (PX4 style).
            if isinstance(mode_id, (list, tuple)):
                base_mode = int(mode_id[0])
                custom_mode = int(mode_id[1]) if len(mode_id) > 1 else 0
            else:
                base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                custom_mode = int(mode_id)
            self.master.mav.set_mode_send(
                self.master.target_system,
                base_mode,
                custom_mode,
            )

        self._run_with_reconnect_retry("set_mode", _set_mode)

        if not wait:
            return True

        start = time.time()
        while time.time() - start < timeout:
            msg = self.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

            elif msg_type == "HEARTBEAT":
                if not self._is_target_heartbeat(msg):
                    continue

                current_mode = self._current_mode()
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
            msg = self.recv_match(type="STATUSTEXT", blocking=False)
            if msg is not None:
                print(f"[PX4 STATUS] {msg.text}")
            time.sleep(0.1)

    def _collect_statustext(self, duration: float = 2.0) -> List[str]:
        texts: List[str] = []
        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.recv_match(type="STATUSTEXT", blocking=False)
            if msg is None:
                time.sleep(0.05)
                continue

            text = str(getattr(msg, "text", "")).strip()
            if text:
                print(f"[PX4 STATUS] {text}")
                texts.append(text)

        return texts

    def print_messages_for_duration(self, duration: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.recv_match(blocking=True, timeout=0.3)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

            elif msg_type == "HEARTBEAT":
                if not self._is_target_heartbeat(msg):
                    continue

                mode = self._current_mode()
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")

    def drain_messages(self, duration: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        end_time = time.time() + duration
        while time.time() < end_time:
            msg = self.recv_match(blocking=True, timeout=0.5)
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
                if not self._is_target_heartbeat(msg):
                    continue

                mode = self._current_mode()
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                print(f"[DEBUG] HEARTBEAT mode={mode} armed={armed}")

    def arm(self, timeout: float = 15.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        # Arming while still in LAND often gets ignored/denied by FC state
        # machines. Move to a stable hold mode first.
        current_mode = self._current_mode().upper()
        if current_mode == "LAND":
            print("[DEBUG] Current mode is LAND, trying armable modes before arm")
            switched = False
            for candidate_mode in ("LOITER", "POSCTL", "ALTCTL", "MANUAL"):
                try:
                    if self.set_mode(candidate_mode, wait=True, timeout=4.0):
                        switched = True
                        print(f"[DEBUG] Mode changed to {candidate_mode} before arm")
                        break
                except Exception as e:
                    print(f"[WARN] Could not switch to {candidate_mode} before arm: {e}")

            if not switched:
                print("[WARN] Still in LAND-like state before arm; arm may be rejected")

        print("[DEBUG] Sending arm command")

        def _send_arm() -> None:
            if self.master is None:
                raise RuntimeError("MAVLink not connected")
            self.master.mav.command_long_send(
                self.master.target_system,
                self._command_target_component(),
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0,
            )

        self._run_with_reconnect_retry("arm", _send_arm)

        # Fast-fail on explicit arm rejection instead of timing out blindly.
        ack_deadline = time.time() + 3.0
        while time.time() < ack_deadline:
            msg = self.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()
            if msg_type == "HEARTBEAT":
                if not self._is_target_heartbeat(msg):
                    continue
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                if armed:
                    print("[DEBUG] Motors armed")
                    return

            if msg_type != "COMMAND_ACK":
                continue

            if msg.command != mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                continue

            print(f"[DEBUG] ARM COMMAND_ACK result={msg.result}")
            if msg.result in (
                mavutil.mavlink.MAV_RESULT_ACCEPTED,
                mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
            ):
                break

            if msg.result == mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED:
                reasons = self._collect_statustext(duration=2.0)
                reason_text = reasons[-1] if reasons else "No STATUSTEXT reason reported"
                raise RuntimeError(
                    "Arm temporarily rejected (result=1). "
                    f"Vehicle likely not in an armable state/mode (often still LAND). "
                    f"FC reason: {reason_text}"
                )

            raise RuntimeError(f"Arm command rejected with result={msg.result}")

        if self._wait_for_armed_state(True, timeout=timeout):
            print("[DEBUG] Motors armed")
            return

        raise TimeoutError("Drone did not arm within timeout")

    def disarm(self) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        print("[DEBUG] Sending disarm command")

        def _send_disarm() -> None:
            if self.master is None:
                raise RuntimeError("MAVLink not connected")
            self.master.mav.command_long_send(
                self.master.target_system,
                self._command_target_component(),
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0, 0, 0, 0, 0, 0, 0,
            )

        self._run_with_reconnect_retry("disarm", _send_disarm)

        if self._wait_for_armed_state(False, timeout=5.0):
            print("[DEBUG] Motors disarmed")
            return

        print("[WARN] Timed out waiting for disarm")

    def takeoff(self, altitude: float, timeout: float = 5.0) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink not connected")

        current_mode = self._current_mode()
        print(
            f"[DEBUG] Sending takeoff command from current mode={current_mode}"
        )

        print(
            f"[DEBUG] Sending PX4-friendly MAV_CMD_NAV_TAKEOFF altitude={altitude}")

        nan = float("nan")

        def _send_takeoff() -> None:
            if self.master is None:
                raise RuntimeError("MAVLink not connected")
            self.master.mav.command_long_send(
                self.master.target_system,
                self._command_target_component(),
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0,      # min pitch
                0,      # empty
                0,      # empty
                nan,    # yaw
                nan,    # latitude
                nan,    # longitude
                altitude,
            )

        self._run_with_reconnect_retry("takeoff", _send_takeoff)

        start = time.time()
        while time.time() - start < timeout:
            msg = self.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":
                print(f"[PX4 STATUS] {msg.text}")

            elif msg_type == "COMMAND_ACK":
                print(
                    f"[DEBUG] COMMAND_ACK command={msg.command} result={msg.result}")

                if msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    if msg.result not in (
                        mavutil.mavlink.MAV_RESULT_ACCEPTED,
                        mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
                    ):
                        raise RuntimeError(
                            f"Takeoff command rejected with result={msg.result}"
                        )

                    print("[DEBUG] Takeoff command accepted")
                    return

            elif msg_type == "HEARTBEAT":
                if not self._is_target_heartbeat(msg):
                    continue

                mode = self._current_mode()
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
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

        print("[DEBUG] Dumping PX4 messages after takeoff command")
        self.print_messages_for_duration(5.0)

    def rtl(self) -> None:
        print("[DEBUG] Setting RTL mode")
        if not self.set_mode("RTL"):
            raise RuntimeError("Failed to switch to RTL mode")

    def land(self) -> None:
        print("[DEBUG] Setting LAND mode")
        try:
            if self.set_mode("LAND", wait=True):
                return
        except Exception as e:
            print(f"[DEBUG] set_mode(LAND) failed, falling back to MAV_CMD_NAV_LAND: {e}")

        def _send_land() -> None:
            if self.master is None:
                raise RuntimeError("MAVLink not connected")

            nan = float("nan")
            self.master.mav.command_long_send(
                self.master.target_system,
                self._command_target_component(),
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,      # abort altitude
                0,      # precision land mode
                0,      # empty
                nan,    # yaw angle
                nan,    # latitude
                nan,    # longitude
                nan,    # altitude
            )

        self._run_with_reconnect_retry("land", _send_land)
        print("[DEBUG] Sent MAV_CMD_NAV_LAND fallback command")

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

        def _send_goto() -> None:
            if self.master is None:
                raise RuntimeError("MAVLink not connected")
            self.master.mav.set_position_target_global_int_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.master.target_system,
                self._command_target_component(),
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

        self._run_with_reconnect_retry("goto", _send_goto)

        print(
            f"[DEBUG] Sent goto command lat={lat}, lon={lon}, alt={alt}, yaw={yaw}")

    def recv_match(self, *args, **kwargs):
        if self.master is None and not self._try_reconnect(force=True):
            return None

        if self.master is None:
            return None

        try:
            return self.master.recv_match(*args, **kwargs)
        except Exception as e:
            if not self._should_treat_as_dead_device(e):
                raise

            print(f"[MAVLINK] recv_match failed: {e}")
            self._try_reconnect(force=False)
            return None
