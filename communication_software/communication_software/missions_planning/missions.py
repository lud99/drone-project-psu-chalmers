from abc import ABC, abstractmethod
from .drone_specs import DroneSpecs
from .mission_status import MissionStatus
import uuid


DEFAULT_ALTITUDE = 10  # meters, placeholder
DEFAULT_HOVER_TIME = 2  # seconds, placeholder
RETURN_HOME = True  # placeholder


class Mission(ABC):
    def __init__(self, drone: DroneSpecs, coordinates: dict):
        self.drone = drone
        self.coordinates = coordinates
        self.mission_id = str(uuid.uuid4())
        self.status = MissionStatus.PENDING

    @abstractmethod
    def can_execute(self) -> bool:
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        pass

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "drone_id": self.drone.id,
            "mission_type": self.__class__.__name__,
            "status": self.status.value,
            "coordinates": self.coordinates,
            "parameters": self.get_parameters(),
        }


class GotoAndSound(Mission):
    def can_execute(self) -> bool:
        return self.drone.speaker

    def get_parameters(self) -> dict:
        return {
            "tasks": [
                {
                    "action": "FLY_TO",
                    "coordinates": self.coordinates,
                    "altitude": DEFAULT_ALTITUDE,
                },
                {"action": "HOVER", "duration": DEFAULT_HOVER_TIME},
                {"action": "PLAY_SOUND", "sound_type": "alert", "sound_duration": 5},
                {"action": "RETURN_HOME", "enabled": RETURN_HOME},
            ]
        }


class GotoAndBlink(Mission):
    def can_execute(self) -> bool:
        return self.drone.lights

    def get_parameters(self) -> dict:
        return {
            "tasks": [
                {
                    "action": "FLY_TO",
                    "coordinates": self.coordinates,
                    "altitude": DEFAULT_ALTITUDE,
                },
                {"action": "HOVER", "duration": DEFAULT_HOVER_TIME},
                {"action": "BLINK", "blink_count": 3, "blink_interval": 1},
                {"action": "RETURN_HOME", "enabled": RETURN_HOME},
            ]
        }


class GotoOnly(Mission):
    def can_execute(self) -> bool:
        return True

    def get_parameters(self) -> dict:
        return {
            "tasks": [
                {
                    "action": "FLY_TO",
                    "coordinates": self.coordinates,
                    "altitude": DEFAULT_ALTITUDE,
                },
                {"action": "HOVER", "duration": DEFAULT_HOVER_TIME},
                {"action": "RETURN_HOME", "enabled": RETURN_HOME},
            ]
        }
