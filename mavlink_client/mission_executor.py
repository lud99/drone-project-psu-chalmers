from __future__ import annotations
import time

from .config import Config
from .mavlink_adapter import MavlinkAdapter


class MissionExecutor:
    def __init__(self, adapter: MavlinkAdapter, config: Config):
        self.adapter = adapter
        self.config = config

    def arm(self) -> None:
        # Keep arming as a standalone action for command-driven UI flows.
        if not self.adapter.arm():
            raise RuntimeError("Arm command rejected by flight controller")

    def takeoff(self, altitude: float | None = None) -> None:
        alt = altitude if altitude is not None else self.config.default_takeoff_alt

        if not self.adapter._is_armed():
            if not self.adapter.arm():
                raise RuntimeError("Arm command rejected by flight controller")

        if not self.adapter.takeoff(alt):
            raise RuntimeError(
                "Takeoff failed or vehicle disarmed during climb")

    def arm_and_takeoff(self, altitude: float | None = None) -> None:
        alt = altitude if altitude is not None else self.config.default_takeoff_alt
        print("[DEBUG] PX4 arm + takeoff")
        self.arm()
        print("[DEBUG] Waiting to confirm armed state is stable")
        time.sleep(2.0)

        if not self.adapter._is_armed():
            raise RuntimeError("Vehicle disarmed immediately after arming")

        self.takeoff(alt)

    def fly_to_coordinate(
        self,
        lat: float,
        lon: float,
        alt: float,
        heading: float | None = None,
    ) -> None:
        # Some FC setups may reject POSCTL transiently even when navigation is
        # still possible from another position-hold mode.
        mode_set = False
        for candidate_mode in ("POSCTL", "LOITER"):
            try:
                if self.adapter.set_mode(candidate_mode):
                    mode_set = True
                    break
            except Exception:
                continue

        if not mode_set:
            raise RuntimeError("Failed to switch to POSCTL/LOITER mode")

        time.sleep(1.0)
        self.adapter.go_to(lat, lon, alt, heading)

    def return_to_home(self) -> None:
        self.adapter.return_to_home()

    def land(self) -> None:
        self.adapter.land()

    def disarm(self) -> None:
        self.adapter.disarm()

    def abort(self) -> None:
        self.adapter.set_mode("LOITER")
