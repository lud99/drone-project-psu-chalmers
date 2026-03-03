"""
Auto-detect MAVLink vehicle connection (UDP listen, UDP out, serial).
Tries options in order until one receives a heartbeat within timeout.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

from pymavlink import mavutil


def _try_heartbeat(connection_string: str, timeout_seconds: float = 3.0) -> bool:
    """Return True if we get a heartbeat within timeout."""
    try:
        conn = mavutil.mavlink_connection(connection_string, dialect="ardupilotmega")
        conn.wait_heartbeat(timeout=timeout_seconds)
        conn.close()
        return True
    except Exception:
        return False


def auto_detect_connection(timeout_per_try: float = 3.0) -> Optional[str]:
    """
    Try connection strings in order; return the first that gets a heartbeat.
    Order: MAVLINK_CONNECTION env → UDP listen 14550 → UDP out 127.0.0.1:14550 → serial.
    """
    env_conn = os.environ.get("MAVLINK_CONNECTION", "").strip()
    if env_conn:
        if _try_heartbeat(env_conn, timeout_per_try):
            return env_conn
        return None  # User set env but it failed

    candidates: List[str] = [
        "udp:0.0.0.0:14550",   # Listen; SITL/vehicle sends to us
        "udp:127.0.0.1:14550",
        "udp:14550",           # Shorthand listen
    ]
    # Serial on Linux / macOS
    if sys.platform.startswith("linux"):
        for dev in ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyAMA0"]:
            if os.path.exists(dev):
                candidates.append(f"{dev}:57600")
    elif sys.platform == "darwin":
        for dev in ["/dev/tty.usbmodem1", "/dev/cu.usbmodem1", "/dev/tty.SLAB_USBtoUART"]:
            if os.path.exists(dev):
                candidates.append(f"{dev}:57600")

    for conn_str in candidates:
        if _try_heartbeat(conn_str, timeout_per_try):
            return conn_str
    return None
