from __future__ import annotations
from threading import Lock
from typing import Optional
from mavlink_client.telemetry import Telemetry


class TelemetryManager:
    def __init__(self, drone_id: str):
        self._lock = Lock()
        self._telemetry = Telemetry(drone_id=drone_id)

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._telemetry, key):
                    setattr(self._telemetry, key, value)

    def snapshot(self) -> Telemetry:
        with self._lock:
            return Telemetry(**self._telemetry.__dict__)
