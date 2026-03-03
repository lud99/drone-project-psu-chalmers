"""
Configuration for MAVLink connection and backend WebSocket.
Resolves backend URL via: env BACKEND_WS_URL → multicast discovery → persisted file.
Resolves MAVLink connection via: env MAVLINK_CONNECTION → auto-detect (UDP/serial).
Drone ID: env DRONE_ID → else "mavlink-{system_id}" once heartbeat received.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MavlinkConfig:
    """MAVLink connection settings. Use resolve_connection() to get connection string."""

    # Set by resolver or env; no hardcoded default
    connection: Optional[str] = None
    target_system: int = 1
    target_component: int = 1


@dataclass
class BackendConfig:
    """Backend WebSocket. Use resolve_backend_url() to get URL."""

    url: Optional[str] = None
    drone_id: Optional[str] = None  # None = use mavlink-{system_id} after heartbeat
    model: str = "MAVLink drone"


@dataclass
class Config:
    """Full client configuration. Call resolve() before use."""

    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            mavlink=MavlinkConfig(),
            backend=BackendConfig(),
        )


def resolve_backend_url(
    discovery_timeout_seconds: float = 15.0,
) -> Optional[str]:
    """
    Resolve backend WebSocket URL: env BACKEND_WS_URL → multicast discovery → persisted.
    Returns None if nothing available (caller should error).
    """
    from mavlink_client.discovery import (
        discover_backend,
        load_persisted_backend_url,
    )

    url = os.environ.get("BACKEND_WS_URL", "").strip()
    if url:
        if not url.startswith("ws://") and not url.startswith("wss://"):
            url = "ws://" + url
        return url
    info = discover_backend(timeout_seconds=discovery_timeout_seconds)
    if info:
        return info.ws_url
    return load_persisted_backend_url()


def resolve_connection() -> Optional[str]:
    """
    Resolve MAVLink connection: env MAVLINK_CONNECTION → auto-detect.
    Returns None if env set but failed, or auto-detect found nothing.
    """
    from mavlink_client.connection_auto import auto_detect_connection

    conn = os.environ.get("MAVLINK_CONNECTION", "").strip()
    if conn:
        return conn
    return auto_detect_connection(timeout_per_try=3.0)


def resolve_drone_id(env_only: bool = False) -> Optional[str]:
    """Drone ID from env if set; otherwise None (use mavlink-{system_id} at runtime)."""
    return os.environ.get("DRONE_ID", "").strip() or None
