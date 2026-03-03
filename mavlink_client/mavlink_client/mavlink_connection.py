"""
MAVLink connection to the vehicle (UDP, TCP, or serial).
Exposes telemetry state from heartbeat, GLOBAL_POSITION_INT, and BATTERY_STATUS.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

# pymavlink
from pymavlink import mavutil


@dataclass
class TelemetryState:
    """Current telemetry from the vehicle. Updated by the connection thread."""

    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    heading: int = 0
    speed: float = 0.0
    battery_percent: int = -1
    connected: bool = False
    # From first HEARTBEAT (for drone_id when not set by user)
    system_id: Optional[int] = None
    # For validity checks
    has_position: bool = False
    has_battery: bool = False


class MavlinkConnection:
    """
    Connects to a MAVLink vehicle and keeps telemetry state updated.
    Runs a background thread to receive messages.
    """

    def __init__(
        self,
        connection_string: str,
        target_system: int = 1,
        target_component: int = 1,
        on_telemetry_updated: Optional[Callable[[TelemetryState], None]] = None,
    ):
        self.connection_string = connection_string
        self.target_system = target_system
        self.target_component = target_component
        self.on_telemetry_updated = on_telemetry_updated
        self._state = TelemetryState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._master = None  # mavutil.mavlink_connection

    @property
    def state(self) -> TelemetryState:
        return self._state

    def start(self) -> None:
        """Start the connection and the receive thread."""
        self._running = True
        self._master = mavutil.mavlink_connection(
            self.connection_string,
            dialect="ardupilotmega",
        )
        # Wait for heartbeat
        self._master.wait_heartbeat(
            timeout=10
        )  # blocking; in production you might do this in thread
        self._state.connected = True
        self._state.has_position = False
        self._state.has_battery = False
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the receive thread and mark disconnected."""
        self._running = False
        self._state.connected = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._master:
            try:
                self._master.close()
            except Exception:
                pass
            self._master = None

    def _receive_loop(self) -> None:
        while self._running and self._master:
            try:
                msg = self._master.recv_match(blocking=True, timeout=0.5)
                if msg is None:
                    continue
                if msg.get_type() == "HEARTBEAT":
                    if self._state.system_id is None:
                        try:
                            h = getattr(msg, "_header", None)
                            if h is not None:
                                self._state.system_id = int(h.srcSystem)
                                self.target_system = self._state.system_id
                        except (AttributeError, TypeError):
                            pass
                elif msg.get_type() == "GLOBAL_POSITION_INT":
                    # lat/lon in degE7, alt in mm (AMSL), relative_alt in mm
                    self._state.lat = msg.lat / 1e7
                    self._state.lon = msg.lon / 1e7
                    self._state.alt = msg.relative_alt / 1000.0  # metres
                    self._state.has_position = True
                    # Heading: use vx/vy for groundspeed direction or separate message
                    # VFR_HUD has heading; GLOBAL_POSITION_INT has vx, vy in cm/s
                    if hasattr(msg, "vx") and hasattr(msg, "vy"):
                        vx = msg.vx / 100.0
                        vy = msg.vy / 100.0
                        self._state.speed = (vx * vx + vy * vy) ** 0.5
                elif msg.get_type() == "VFR_HUD":
                    self._state.heading = int(msg.heading)
                    self._state.speed = getattr(msg, "groundspeed", 0.0) or 0.0
                elif msg.get_type() == "BATTERY_STATUS":
                    # percent_remaining in 0-100, or use voltage
                    pct = getattr(msg, "battery_remaining", -1)
                    if pct >= 0:
                        self._state.battery_percent = pct
                        self._state.has_battery = True
                elif msg.get_type() == "SYS_STATUS":
                    # Fallback: some stacks send battery in SYS_STATUS
                    if not self._state.has_battery and hasattr(msg, "battery_remaining"):
                        self._state.battery_percent = msg.battery_remaining
                        self._state.has_battery = True
                if self.on_telemetry_updated:
                    self.on_telemetry_updated(self._state)
            except Exception:
                if self._running:
                    pass  # TODO: logging
                break

    def get_connection(self):
        """Return the underlying mavutil connection (has .mav for sending)."""
        return self._master

    def get_system_id(self) -> Optional[int]:
        """Return vehicle system ID from first HEARTBEAT (for auto drone_id)."""
        return self._state.system_id
