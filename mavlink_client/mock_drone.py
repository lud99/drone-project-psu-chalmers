from __future__ import annotations
import random
import time
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

    def arm(self):
        self.armed = True
        self.mode = "GUIDED"

    def takeoff(self, alt: float):
        self.armed = True
        self.mode = "TAKEOFF"
        self.alt = alt
        self.speed = 3.0

    def goto(self, lat: float, lon: float, alt: float, heading=None):
        self.mode = "GUIDED"
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.heading = heading if heading is not None else self.heading
        self.speed = 5.0

    def rtl(self):
        self.mode = "RTL"
        self.speed = 4.0

    def land(self):
        self.mode = "LAND"
        self.alt = 0.0
        self.speed = 1.0
        self.armed = False

    def tick(self):
        self.battery = max(0, self.battery - random.randint(0, 1))
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
        print("[MOCK] Disarmed")
