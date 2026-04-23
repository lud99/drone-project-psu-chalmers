from typing import Any, Dict
from mavlink_client.telemetry_manager import TelemetryManager


class CommandHandler:
    def __init__(self, mission_executor: Any, telemetry_manager: TelemetryManager):
        self.mission_executor = mission_executor
        self.telemetry_manager = telemetry_manager
        self.pending_target = None

    def handle(self, message: Dict):
        msg_type = message.get("msg_type")

        if msg_type == "ping":
            return {"msg_type": "pong"}

        if msg_type == "flight_arm":
            self.mission_executor.arm_and_takeoff()
            return {"msg_type": "ack", "command": "flight_arm", "status": "ok"}

        if msg_type == "flight_take_off":
            if self.pending_target:
                self.mission_executor.fly_to_coordinate(
                    lat=self.pending_target["lat"],
                    lon=self.pending_target["lon"],
                    alt=self.pending_target["alt"],
                    heading=self.pending_target.get("heading"),
                )
            else:
                self.mission_executor.arm_and_takeoff()
            return {"msg_type": "ack", "command": "flight_take_off", "status": "ok"}

        if msg_type == "flight_return_to_home":
            self.mission_executor.return_to_home()
            return {"msg_type": "ack", "command": "flight_return_to_home", "status": "ok"}

        if msg_type == "land":
            self.mission_executor.land()
            return {"msg_type": "ack", "command": "land", "status": "ok"}

        if msg_type == "abort":
            self.mission_executor.abort()
            return {"msg_type": "ack", "command": "abort", "status": "ok"}

        if msg_type in {"Coordinate_request", "mission_assignment"}:
            lat = message.get("lat")
            lon = message.get("lon")
            alt = message.get("alt", 15.0)
            heading = message.get("heading")

            if lat is not None and lon is not None:
                self.pending_target = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt": float(alt),
                    "heading": heading,
                }
                return {
                    "msg_type": "ack",
                    "command": msg_type,
                    "status": "ok",
                    "target_loaded": True,
                }

        if msg_type == "goto":
            self.mission_executor.fly_to_coordinate(
                lat=float(message["lat"]),
                lon=float(message["lon"]),
                alt=float(message.get("alt", 15.0)),
                heading=message.get("heading"),
            )
            return {"msg_type": "ack", "command": "goto", "status": "ok"}

        return {"msg_type": "ack", "command": msg_type, "status": "ignored"}
