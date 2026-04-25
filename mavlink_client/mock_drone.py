from __future__ import annotations
import random
import time
import math
from .telemetry_manager import TelemetryManager


class MockDrone:
    def __init__(self, telemetry_manager: TelemetryManager):
        self.telemetry_manager = telemetry_manager
        self.lat = 57.777531
        self.lon = 12.781457
        self.alt = 0.0
        self.heading = 90.0
        self.speed = 0.0
        self.battery = 95
        self.armed = False
        self.mode = "STANDBY"
        self.target_lat = None
        self.target_lon = None
        self.target_alt = None
        self.moving = False

    def arm(self):
        self.armed = True
        self.mode = "GUIDED"

    def takeoff(self, alt: float):
        self.armed = True
        self.mode = "TAKEOFF"
        self.target_alt = alt
        self.moving = True

    def goto(self, lat: float, lon: float, alt: float, heading=None):
        self.mode = "GUIDED"
        self.target_lat = lat
        self.target_lon = lon
        self.target_alt = alt
        self.heading = heading if heading is not None else self.heading
        self.moving = True
        self.speed = 5.0

    def rtl(self):
        self.mode = "RTL"
        self.moving = False
        self.speed = 4.0

    def land(self):
        self.mode = "LAND"
        self.target_alt = 0.0
        self.moving = True
        self.speed = 1.0

    def tick(self):
        self.battery = max(0, self.battery - random.randint(0, 1))

        if self.moving:
            # Move towards target
            if self.target_lat is not None and self.target_lon is not None:
                dlat = self.target_lat - self.lat
                dlon = self.target_lon - self.lon
                dist = math.sqrt(dlat**2 + dlon**2)
                if dist > 0.00001:  # Close enough
                    step = 0.000005  # Small step
                    self.lat += (dlat / dist) * step
                    self.lon += (dlon / dist) * step
                else:
                    self.target_lat = None
                    self.target_lon = None
                    self.moving = False
                    self.speed = 0.0

            if self.target_alt is not None:
                if abs(self.alt - self.target_alt) > 0.1:
                    self.alt += (self.target_alt - self.alt) * 0.1
                else:
                    self.alt = self.target_alt
                    self.target_alt = None
                    if self.mode == "LAND":
                        self.armed = False
                        self.moving = False

        self.telemetry_manager.update(
            lat=self.lat + random.uniform(-0.00001, 0.00001),
            lon=self.lon + random.uniform(-0.00001, 0.00001),
            alt=self.alt,
            heading=self.heading,
            speed=self.speed,
            battery_percent=self.battery,
            mode=self.mode,
            armed=self.armed,
            gps_fix_type=3,
            satellites_visible=14,
            timestamp=time.time(),
        )

    def disarm(self):
        self.armed = False
        self.moving = False
        print("[MOCK] Disarmed")
