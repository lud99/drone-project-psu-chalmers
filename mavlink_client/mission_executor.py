from __future__ import annotations
from .mavlink_adapter import MavlinkAdapter
from .config import Config
import time


class MissionExecutor:
    def __init__(self, adapter: MavlinkAdapter, config: Config):
        self.adapter = adapter
        self.config = config

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

    def fly_to_coordinate(self, lat: float, lon: float, alt: float, heading: float | None = None) -> None:
        self.adapter.set_mode("TAKEOFF")
        time.sleep(1)
        self.adapter.go_to(lat, lon, alt, heading)

    def return_to_home(self) -> None:
        self.adapter.return_to_home()

    def land(self) -> None:
        self.adapter.land()

    def abort(self) -> None:
        self.adapter.set_mode("LOITER")
