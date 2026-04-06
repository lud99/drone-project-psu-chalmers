from abc import ABC, abstractmethod
import uuid

from typing import Optional

import communication_software.common.json_schemas as json_schemas
from communication_software.missions_planning.mission_status import (
    MissionStatus,
)

from communication_software.missions_planning.drone_capability_helpers import (
    SpeakerHelper,
    LEDHelper,
)

import communication_software.missions_planning.mission_constants as mission_constants


class NoTypeOrFileError(Exception):
    pass


class NoAudioFileError(Exception):
    pass


class MissionBuildError(Exception):
    pass


class Mission(ABC):
    tasks: list[json_schemas.AnyTaskAction] = []

    def __init__(
        self,
        drone_id: str,
        capabilities: json_schemas.Capabilities,
        coordinates: json_schemas.GoToParams,
    ):
        self.drone_id = drone_id
        self.capabilities = capabilities
        self.coordinates = coordinates
        self.mission_id = str(uuid.uuid4())
        self.status = MissionStatus.PENDING

    @abstractmethod
    def can_execute(self) -> bool:
        pass

    @abstractmethod
    def get_parameters(self) -> json_schemas.AnyParams:
        pass

    @abstractmethod
    def build_tasks(self) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def get_tasks(self) -> list[json_schemas.AnyTaskAction]:
        if len(self.tasks) == 0:
            self.build_tasks()
        if len(self.tasks) == 0:
            raise MissionBuildError("Mission tasks could not be built.")
        return self.tasks

    def get_frontend_mission_proposal(self) -> dict:
        return {
            "drone_id": self.drone_id,
            "mission_id": self.mission_id,
            "mission_type": self.__class__.__name__,
            "description": repr(self),
            "tasks": self.get_tasks(),
        }

    def to_dict(self) -> dict:
        task_json = [task.model_dump() for task in self.get_tasks()]
        return {
            "mission_id": self.mission_id,
            "drone_id": self.drone_id,
            "mission_type": self.__class__.__name__,
            "status": self.status.value,
            "coordinates": self.coordinates.model_dump(),
            "tasks": task_json,
        }


class GotoAndAudio(Mission):
    def __init__(
        self,
        drone_id: str,
        capabilities: json_schemas.Capabilities,
        coordinates: json_schemas.GoToParams,
        audio_type: str | None = None,
        audio_file: str | None = None,
        duration_seconds: int | None = mission_constants.DEFAULT_ACTION_DURATION,
        volume: float = 1.0,
    ):
        if audio_file is None and audio_type is None:
            raise NoTypeOrFileError(
                "Either audio_type or audio_file must be specified."
            )
        super().__init__(drone_id, capabilities, coordinates)
        self.audio_type = audio_type
        self.audio_file = audio_file
        self.duration_seconds = duration_seconds
        self.volume = volume
        self.speaker_helper = SpeakerHelper([])
        if capabilities.speaker:
            self.speaker_helper = SpeakerHelper(
                audio_files=capabilities.speaker.audio_files
            )

    def can_execute(self) -> bool:
        if self.capabilities.speaker is None:
            return False

        if self.audio_file:
            return self.speaker_helper.has_file(self.audio_file)
        if self.audio_type:
            return self.speaker_helper.has_type(self.audio_type)
        return False

    def get_audio_file(self) -> str:
        if self.capabilities.speaker is None:
            raise NoAudioFileError("Drone does not have a speaker subsystem.")
        if self.audio_file:
            return self.audio_file
        if self.audio_type:
            audio_file_obj = self.speaker_helper.get_single_file_by_type(
                self.audio_type
            )
            if audio_file_obj is not None:
                return audio_file_obj.audio_file
        raise NoAudioFileError("No suitable audio file found for this mission.")

    def get_audio_params(self) -> json_schemas.PlayAudioParams:
        return json_schemas.PlayAudioParams(
            file=self.get_audio_file(),
            duration_seconds=self.duration_seconds,
            volume=self.volume,
        )

    def get_parameters(self) -> json_schemas.AnyParams:
        return self.get_audio_params()

    def build_tasks(self) -> None:
        json_schemas.PlayAudioTask(params=self.get_audio_params())
        tasks = [
            json_schemas.GoToTask(params=self.coordinates),
            json_schemas.PlayAudioTask(params=self.get_audio_params()),
        ]
        if (self.duration_seconds is not None) and mission_constants.RETURN_HOME:
            tasks.append(json_schemas.GoHomeTask())
        self.tasks = tasks

    def __repr__(self) -> str:
        try:
            audio_file = self.get_audio_file()
        except NoAudioFileError:
            audio_file = "N/A"
        return f"[drone {self.drone_id}] GoTo and play Audio file: {audio_file}"


class GotoAndBlink(Mission):
    def __init__(
        self,
        drone_id: str,
        capabilities: json_schemas.Capabilities,
        coordinates: json_schemas.GoToParams,
        duration_seconds: int = mission_constants.DEFAULT_ACTION_DURATION,
    ):
        super().__init__(drone_id, capabilities, coordinates)
        self.duration_seconds = duration_seconds

        self.led_helper = LEDHelper([])
        if capabilities.led:
            self.led_helper = LEDHelper(led_types=capabilities.led.types)

    def can_execute(self) -> bool:
        return self.capabilities.led is not None and self.led_helper.has_type("beacon")

    def get_parameters(self) -> json_schemas.LEDParams:
        return json_schemas.LEDParams(
            type="beacon", duration_seconds=self.duration_seconds
        )

    def build_tasks(self) -> None:
        tasks = [
            json_schemas.GoToTask(params=self.coordinates),
            json_schemas.LEDTask(params=self.get_parameters()),
        ]
        if (self.duration_seconds is not None) and mission_constants.RETURN_HOME:
            tasks.append(json_schemas.GoHomeTask())
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"[drone {self.drone_id}] GoTo and Blink with LED type: beacon for {self.duration_seconds}s"


class GotoAndSurveil(Mission):
    def __init__(
        self,
        drone_id: str,
        capabilities: json_schemas.Capabilities,
        coordinates: json_schemas.GoToParams,
        duration_seconds: Optional[float] = None,
        camera_pitch: float = -90,
        camera_yaw: float = 0,
    ):
        super().__init__(drone_id, capabilities, coordinates)
        self.duration_seconds = duration_seconds
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw

    def can_execute(self) -> bool:
        return self.capabilities.camera is not None

    def get_parameters(self) -> json_schemas.AngleCameraParams:
        return json_schemas.AngleCameraParams(
            duration_seconds=self.duration_seconds,
            pitch=self.camera_pitch,
            yaw=self.camera_yaw,
        )

    def build_tasks(self) -> None:
        tasks = [
            json_schemas.GoToTask(params=self.coordinates),
            json_schemas.AngleCameraTask(params=self.get_parameters()),
        ]
        # {"action": "hover", "params": {"duration_seconds": self.duration_seconds}},

        if (self.duration_seconds is not None) and mission_constants.RETURN_HOME:
            tasks.append(json_schemas.GoHomeTask())
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"[drone {self.drone_id}] GoTo and Surveil for {self.duration_seconds}s"


class GotoAndIlluminate(Mission):
    def __init__(
        self,
        drone_id: str,
        capabilities: json_schemas.Capabilities,
        coordinates: json_schemas.GoToParams,
        duration_seconds: int = mission_constants.DEFAULT_ACTION_DURATION,
    ):
        super().__init__(drone_id, capabilities, coordinates)
        self.duration_seconds = duration_seconds

    def can_execute(self) -> bool:
        return self.capabilities.spotlight

    def get_parameters(self) -> json_schemas.SpotlightParams:
        return json_schemas.SpotlightParams(
            pattern="stready", duration_seconds=self.duration_seconds
        )

    def build_tasks(self) -> None:
        tasks = [
            json_schemas.GoToTask(params=self.coordinates),
            json_schemas.SpotlightTask(params=self.get_parameters()),
        ]
        if (self.duration_seconds is not None) and mission_constants.RETURN_HOME:
            tasks.append(json_schemas.GoHomeTask())
        self.tasks = tasks

    def __repr__(self) -> str:
        return (
            f"[drone {self.drone_id}] GoTo and Spotlight for {self.duration_seconds}s"
        )


class GotoOnly(Mission):
    def can_execute(self) -> bool:
        return True

    def get_parameters(self) -> json_schemas.GoToParams:
        return self.coordinates

    def build_tasks(self) -> None:
        self.tasks = [
            json_schemas.GoToTask(params=self.coordinates),
        ]

    def __repr__(self) -> str:
        return f"[drone {self.drone_id}] GoTo, lat:{self.coordinates.lat:.6f}, lon:{self.coordinates.lon:.6f} alt:{self.coordinates.alt}"
