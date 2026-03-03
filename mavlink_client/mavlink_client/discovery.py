"""
Backend discovery via multicast (same protocol as Android app).
Listens on 239.255.42.99:9992 for "CTH" + JSON with msg_type=backend_discovery, ip, port.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional

# Same as Android MulticastReceiver and forward_multicast
MULTICAST_GROUP = "239.255.42.99"
MULTICAST_PORT = 9992
MAGIC_PREFIX = b"CTH"


@dataclass
class BackendInfo:
    name: str
    ip: str
    port: int

    @property
    def ws_url(self) -> str:
        return f"ws://{self.ip}:{self.port}"


def discover_backend(timeout_seconds: float = 15.0) -> Optional[BackendInfo]:
    """
    Listen for one backend_discovery packet; return BackendInfo or None.
    Uses multicast join on MULTICAST_GROUP:MULTICAST_PORT.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", MULTICAST_PORT))
    except OSError:
        sock.close()
        return None
    try:
        group = socket.inet_aton(MULTICAST_GROUP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group + socket.inet_aton("0.0.0.0"))
    except OSError:
        sock.close()
        return None
    sock.settimeout(timeout_seconds)
    result = None
    try:
        data, _ = sock.recvfrom(1200)
        if not data.startswith(MAGIC_PREFIX):
            sock.close()
            return None
        msg = json.loads(data[len(MAGIC_PREFIX) :].decode("utf-8"))
        if msg.get("msg_type") == "backend_discovery":
            result = BackendInfo(
                name=msg.get("name", "backend"),
                ip=msg["ip"],
                port=int(msg["port"]),
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    finally:
        sock.close()
    return result


def persist_backend_url(url: str) -> None:
    """Save backend WebSocket URL to user config file for next run."""
    import os
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    dir_path = os.path.join(base, "drone-project-psu-chalmers")
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, "mavlink_backend_url.txt")
    with open(path, "w") as f:
        f.write(url.strip() + "\n")


def load_persisted_backend_url() -> Optional[str]:
    """Load last used backend URL from user config file."""
    import os
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    path = os.path.join(base, "drone-project-psu-chalmers", "mavlink_backend_url.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            line = f.readline().strip()
            return line if line else None
    except OSError:
        return None
