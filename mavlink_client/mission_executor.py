from __future__ import annotations
import time

from .config import Config
from .mavlink_adapter import MavlinkAdapter


class MissionExecutor:
    def __init__(self, adapter: MavlinkAdapter, config: Config):
        self.adapter = adapter
        self.config = config

    def arm_and_takeoff(self, altitude: float | None = None) -> None:
        alt = altitude if altitude is not None else self.config.default_takeoff_alt

        print("[DEBUG] PX4 arm + takeoff")

        current_mode = self.adapter.wait_for_valid_mode()
        print(f"[DEBUG] Current mode before arm: {current_mode}")

        if current_mode not in ["LOITER", "POSCTL"]:
            if not self.adapter.set_mode("LOITER"):
                raise RuntimeError(
                    f"Vehicle is currently in {current_mode} and failed to switch to LOITER"
                )
            time.sleep(1.0)

            current_mode = self.adapter.wait_for_valid_mode()
            print(f"[DEBUG] Mode after switch attempt: {current_mode}")

            if current_mode not in ["LOITER", "POSCTL"]:
                raise RuntimeError(
                    f"Vehicle is currently in {current_mode}, not safe to auto-arm from this state"
                )

        if not self.adapter.arm():
            raise RuntimeError("Arm command rejected by flight controller")

        time.sleep(1.0)

        if not self.adapter.takeoff(alt):
            raise RuntimeError("Takeoff failed or no altitude increase")

    def fly_to_coordinate(
        self,
        lat: float,
        lon: float,
        alt: float,
        heading: float | None = None,
    ) -> None:
        if not self.adapter.set_mode("POSCTL"):
            raise RuntimeError("Failed to switch to POSCTL mode")

        time.sleep(1.0)
        self.adapter.go_to(lat, lon, alt, heading)

    def return_to_home(self) -> None:
        self.adapter.return_to_home()

    def land(self) -> None:
        self.adapter.land()

    def abort(self) -> None:
        self.adapter.set_mode("LOITER")
