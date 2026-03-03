"""
Mission and command handling: arm, takeoff, RTL, go-to (waypoint).
Uses the MAVLink connection to send COMMAND_LONG and mission items.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pymavlink import mavutil

if TYPE_CHECKING:
    from mavlink_client.mavlink_connection import MavlinkConnection

# MAV_CMD and MAV_COMPONENT
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_DO_SET_MODE = 176
MAV_MODE_PREFLIGHT = 0
MAV_MODE_STABILIZE_DISARMED = 0
MAV_MODE_GUIDED_ARMED = 4
MAV_MODE_AUTO_ARMED = 10
MAV_MODE_RTL = 11


class MissionHandler:
    """Send arm, takeoff, RTL, and go-to commands over MAVLink."""

    def __init__(self, connection: "MavlinkConnection"):
        self._conn = connection

    def _master(self):
        return self._conn.get_connection()

    def arm(self) -> bool:
        """Arm the vehicle. Returns True if command was sent."""
        m = self._master()
        if m is None:
            return False
        target_system = self._conn.target_system
        target_component = self._conn.target_component
        m.mav.command_long_send(
            target_system,
            target_component,
            MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm = 1
            0, 0, 0, 0, 0, 0,
        )
        return True

    def disarm(self) -> bool:
        """Disarm the vehicle."""
        m = self._master()
        if m is None:
            return False
        m.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,  # disarm
            0, 0, 0, 0, 0, 0,
        )
        return True

    def takeoff(self, altitude_m: float = 10.0) -> bool:
        """Send takeoff command (e.g. in GUIDED mode)."""
        m = self._master()
        if m is None:
            return False
        m.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0,
            0, 0,
            altitude_m,
            0,
        )
        return True

    def return_to_launch(self) -> bool:
        """Command RTL (return to launch)."""
        m = self._master()
        if m is None:
            return False
        m.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
        return True

    def land(self) -> bool:
        """Command land at current position."""
        m = self._master()
        if m is None:
            return False
        m.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            MAV_CMD_NAV_LAND,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
        return True

    def go_to(self, lat: float, lon: float, alt: float, heading_deg: Optional[int] = None) -> bool:
        """
        Go to a position (single waypoint). Uses COMMAND_INT or waypoint.
        Simple implementation: send a single waypoint command.
        """
        m = self._master()
        if m is None:
            return False
        # Use MISSION_ITEM_INT or COMMAND_INT for a single goto
        # COMMAND_INT with MAV_CMD_NAV_WAYPOINT
        # MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
        m.mav.command_int_send(
            self._conn.target_system,
            self._conn.target_component,
            3,  # frame: GLOBAL_RELATIVE_ALT
            MAV_CMD_NAV_WAYPOINT,
            0,  # current
            0,  # autocontinue
            0, 0, 0, 0,  # params 1-4
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
        )
        return True
