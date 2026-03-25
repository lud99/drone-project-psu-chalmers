from abc import ABC, abstractmethod
from .drone_specs import DroneSpecs
from .mission_status import MissionStatus
import uuid


DEFAULT_ALTITUDE = 10
DEFAULT_HOVER_TIME = 2
RETURN_HOME = True
DEFAULT_ACTION_DURATION = 30


class Coordinates:
    def __init__(
        self,
        lat: float,
        lon: float,
        alt: float = DEFAULT_ALTITUDE,
        heading: float | None = None,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.heading = heading

    def as_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "heading": self.heading,
        }


class NoTypeOrFileError(Exception):
    pass


class NoAudioFileError(Exception):
    pass


class MissionBuildError(Exception):
    pass


class Mission(ABC):
    tasks: list[dict] | None = None

    def __init__(self, drone: DroneSpecs, coordinates: Coordinates):
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

    @abstractmethod
    def build_tasks(self) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def get_tasks(self) -> list:
        if self.tasks is None:
            self.build_tasks()
        if self.tasks is None:
            raise MissionBuildError("Mission tasks could not be built.")
        return self.tasks

    def get_frontend_mission_proposal(self) -> dict:
        return {
            "drone_id": self.drone.drone_id,
            "mission_type": self.__class__.__name__,
            "description": repr(self),
            "tasks": self.get_tasks(),
        }

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "drone_id": self.drone.drone_id,
            "mission_type": self.__class__.__name__,
            "status": self.status.value,
            "coordinates": self.coordinates.as_dict(),
            "parameters": self.get_parameters(),
        }


class GotoAndAudio(Mission):
    tasks: list[dict] | None = None

    def __init__(
        self,
        drone: DroneSpecs,
        coordinates: Coordinates,
        audio_type: str | None = None,
        audio_file: str | None = None,
        duration_seconds: int | None = DEFAULT_ACTION_DURATION,
        volume: float = 1.0,
    ):
        if audio_file is None and audio_type is None:
            raise NoTypeOrFileError(
                "Either audio_type or audio_file must be specified."
            )
        super().__init__(drone, coordinates)
        self.audio_type = audio_type
        self.audio_file = audio_file
        self.duration_seconds = duration_seconds
        self.volume = volume

    def can_execute(self) -> bool:
        if self.drone.speaker is None:
            return False
        if self.audio_file:
            return self.drone.speaker.has_file(self.audio_file)
        if self.audio_type:
            return self.drone.speaker.has_type(self.audio_type)
        return False

    def get_audio_file(self) -> str:
        if self.drone.speaker is None:
            raise NoAudioFileError("Drone does not have a speaker subsystem.")
        if self.audio_file:
            return self.audio_file
        if self.audio_type:
            audio_file_obj = self.drone.speaker.get_single_file_by_type(self.audio_type)
            if audio_file_obj is not None:
                return audio_file_obj.audio_file
        raise NoAudioFileError("No suitable audio file found for this mission.")

    def get_audio_params(self) -> dict:
        return {
            "audio_file": self.get_audio_file(),
            "duration_seconds": self.duration_seconds,
            "volume": self.volume,
        }

    def get_parameters(self) -> dict:
        return self.get_audio_params()

    def build_tasks(self) -> None:
        tasks = [
            {"action": "go_to", "params": self.coordinates.as_dict()},
            {"action": "play_audio", "params": self.get_audio_params()},
        ]
        if self.duration_seconds is not None:
            tasks.append({"action": "go_home", "enabled": RETURN_HOME})
        self.tasks = tasks

    def __repr__(self) -> str:
        try:
            audio_file = self.get_audio_file()
        except NoAudioFileError:
            audio_file = "N/A"
        return f"GoTo and play Audio file: {audio_file}"


class GotoAndBlink(Mission):
    def __init__(
        self,
        drone: DroneSpecs,
        coordinates: Coordinates,
        duration_seconds: int = DEFAULT_ACTION_DURATION,
    ):
        super().__init__(drone, coordinates)
        self.duration_seconds = duration_seconds

    def can_execute(self) -> bool:
        return self.drone.led is not None and self.drone.led.has_type("beacon")

    def get_parameters(self) -> dict:
        return {"type": "beacon", "duration_seconds": self.duration_seconds}

    def build_tasks(self) -> None:
        tasks = [
            {"action": "go_to", "params": self.coordinates.as_dict()},
            {"action": "led", "params": self.get_parameters()},
        ]
        if self.duration_seconds is not None:
            tasks.append({"action": "go_home", "enabled": RETURN_HOME})
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"GoTo and Blink with LED type: beacon for {self.duration_seconds}s"


class GotoAndSurveil(Mission):
    def __init__(
        self,
        drone: DroneSpecs,
        coordinates: Coordinates,
        duration_seconds: int | None = None,
        camera_pitch: float | None = -90,
        camera_yaw: float | None = 0,
    ):
        super().__init__(drone, coordinates)
        self.duration_seconds = duration_seconds
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw

    def can_execute(self) -> bool:
        return self.drone.camera is not None

    def get_parameters(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "pitch": self.camera_pitch,
            "yaw": self.camera_yaw,
        }

    def build_tasks(self) -> None:
        tasks = [
            {"action": "go_to", "params": self.coordinates.as_dict()},
            {"action": "angle_camera", "params": self.get_parameters()},
            #{"action": "hover", "params": {"duration_seconds": self.duration_seconds}},
        ]
        if self.duration_seconds is not None:
            tasks.append({"action": "go_home", "enabled": RETURN_HOME})
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"GoTo and Surveil for {self.duration_seconds}s"


class GotoAndIlluminate(Mission):
    def __init__(
        self,
        drone: DroneSpecs,
        coordinates: Coordinates,
        duration_seconds: int = DEFAULT_ACTION_DURATION,
    ):
        super().__init__(drone, coordinates)
        self.duration_seconds = duration_seconds

    def can_execute(self) -> bool:
        return self.drone.spotlight

    def get_parameters(self) -> dict:
        return {"pattern": "steady", "duration_seconds": self.duration_seconds}

    def build_tasks(self) -> None:
        tasks = [
            {"action": "go_to", "params": self.coordinates.as_dict()},
            {"action": "spotlight", "params": self.get_parameters()},
        ]
        if self.duration_seconds is not None:
            tasks.append({"action": "go_home", "enabled": RETURN_HOME})
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"GoTo and Spotlight for {self.duration_seconds}s"


class GotoOnly(Mission):
    def can_execute(self) -> bool:
        return True

    def get_parameters(self) -> dict:
        return self.coordinates.as_dict()

    def build_tasks(self) -> None:
        self.tasks = [{"action": "go_to", "params": self.coordinates.as_dict()}]

    def __repr__(self) -> str:
        return f"GoTo, lat:{self.coordinates.lat:.6f}, lon:{self.coordinates.lon:.6f} alt:{self.coordinates.alt}"
