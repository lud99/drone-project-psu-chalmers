from __future__ import annotations
from typing import Annotated, Union, Literal, TypeVar, Generic, Optional
from pydantic import BaseModel, RootModel, Field, TypeAdapter
### Capabilities and telemetry


class CameraCapabilities(BaseModel):
    aspect_ratio: float
    horizontal_fov: float
    resolution_height: int
    resolution_width: int


class LEDCapabilities(BaseModel):
    colors: list[str]  # will likely have to change


class Capabilities(BaseModel):
    camera: Union[CameraCapabilities, None]
    led: Union[LEDCapabilities, None]
    spotlight: bool
    speaker: bool
    max_speed: float


class Telemetry(BaseModel):
    lat: float
    lon: float
    alt: float
    heading: int
    speed: float
    battery_percent: int


### Drone message schemas


# Sub-models for Tasks
TaskEvents = Literal["task_complete", "task_failed"]


# Specific task definitions
class GoToParams(BaseModel):
    lat: float
    lon: float
    alt: float
    heading: int


class PlayAudioParams(BaseModel):
    file: str
    volume: float = 1.0
    duration_seconds: int


class LEDParams(BaseModel):
    color: str
    pattern: str
    duration_seconds: float


class SpotlightParams(BaseModel):
    pattern: str
    duration_seconds: float


# The specific Task types


class GoToTask(BaseModel):
    action: Literal["go_to"]
    params: GoToParams


class PlayAudioTask(BaseModel):
    action: Literal["play_audio"]
    params: PlayAudioParams


class LEDTask(BaseModel):
    action: Literal["led"]
    params: LEDParams


class SpotlightTask(BaseModel):
    action: Literal["spotlight"]
    params: SpotlightParams


# This variable holds the "one of these" logic
AnyTaskAction = Union[GoToTask, PlayAudioTask, LEDTask, SpotlightTask]


MsgTypeT = TypeVar("MsgTypeT", bound=str)


class DroneMessage(BaseModel, Generic[MsgTypeT]):
    msg_type: MsgTypeT


class BackendToDroneMessage(DroneMessage):
    drone_id: str


# Backend -> app
class TaskMessage(BackendToDroneMessage):
    msg_type: Literal["task"]
    mission_id: str
    index: int
    # This field now enforces strict structure based on the 'action' string
    task: AnyTaskAction = Field(..., discriminator="action")


class DebugMessage(DroneMessage):
    msg_type: Literal["debug"]
    message: str


# App -> backend. Upon registration
class DroneRegistrationMessage(DroneMessage):
    msg_type: Literal["drone_registration"]
    drone_type: str
    model: str
    drone_id: str
    capabilities: Capabilities


# App -> backend. Sent continuously
class TelemetryMessage(DroneMessage):
    msg_type: Literal["telemetry"]
    drone_id: str
    telemetry: Telemetry


# Backend -> app
class TaskEventMessage(BackendToDroneMessage):
    msg_type: Literal["task_event"]
    mission_id: str
    index: int
    event: TaskEvents
    message: str
    timestamp: int


# Backend -> app
class AbortTaskMessage(BackendToDroneMessage):
    msg_type: Literal["abort_task"]
    mission_id: str
    index: int
    next: Literal["go_home", "hover", "land"]


# WebRTC messages


class WebRTCCandidateMessage(DroneMessage):
    msg_type: Literal["candidate"]
    candidate: str
    id: str = "0"
    label: int = 0


class WebRTCAnswerMessage(DroneMessage):
    msg_type: Literal["answer"]
    sdp: str
    type: str


# Create a Union of all possible messages
AnyDroneMessage = Annotated[
    Union[
        TaskMessage,
        DroneRegistrationMessage,
        TelemetryMessage,
        TaskEventMessage,
        AbortTaskMessage,
        DebugMessage,
        WebRTCCandidateMessage,
        WebRTCAnswerMessage,
    ],
    Field(discriminator="msg_type"),
]


### Detections schema
class SingleDetection(BaseModel):
    gps_position: tuple[float, float]
    class_name: str
    drone_ids: Annotated[list[str], Field(min_length=1)]


class Detections(RootModel):
    root: list[SingleDetection]


### Frontend schemas


class LatLon(BaseModel):
    lat: float
    lon: float


class Points(BaseModel):
    points: list[LatLon]


class DroneInfo(BaseModel):
    drone_id: str
    capabilities: Capabilities
    telemetry: Telemetry


class FrontendMessages:
    class FrontendMessage(BaseModel, Generic[MsgTypeT]):
        msg_type: MsgTypeT

    # --- (Frontend -> Backend) ---

    class AcceptMission(FrontendMessage):
        msg_type: Literal["accept_mission"] = "accept_mission"
        mission_id: str

    class RejectMissions(FrontendMessage):
        msg_type: Literal["reject_missions"] = "reject_missions"

    class StartDrone(FrontendMessage):
        msg_type: Literal["start_drone"] = "start_drone"
        drone_id: str

    class SetWatchArea(FrontendMessage):
        msg_type: Literal["set_watch_area"] = "set_watch_area"
        area: Points

    class ProposedMissions(FrontendMessage):
        msg_type: Literal["proposed_missions"] = "proposed_missions"
        missions: list[dict]

    class ActiveMissions(FrontendMessage):
        msg_type: Literal["active_missions"] = "active_missions"
        missions: list[dict]

    class TelemetryUpdate(FrontendMessage):
        msg_type: Literal["telemetry"] = "telemetry"
        drone_id: str
        telemetry: Telemetry

    class DroneConnected(FrontendMessage):
        msg_type: Literal["drone_connected"] = "drone_connected"
        drone_id: str
        capabilities: Capabilities
        telemetry: Telemetry

    class DroneDisconnected(FrontendMessage):
        msg_type: Literal["drone_disconnected"] = "drone_disconnected"
        drone_id: str

    class GetWatchAreas(FrontendMessage):
        msg_type: Literal["get_watch_areas"] = "get_watch_areas"
        area: Points

    class ConnectedDrones(FrontendMessage):
        msg_type: Literal["connected_drones"] = "connected_drones"
        drones: list[DroneInfo]

    class ServerResponse(FrontendMessage):
        msg_type: Literal["response"] = "response"
        error: Optional[str] = None


AnyFrontendMessage = Annotated[
    Union[
        FrontendMessages.AcceptMission,
        FrontendMessages.RejectMissions,
        FrontendMessages.ProposedMissions,
        FrontendMessages.ActiveMissions,
        FrontendMessages.TelemetryUpdate,
        FrontendMessages.StartDrone,
        FrontendMessages.DroneConnected,
        FrontendMessages.DroneDisconnected,
        FrontendMessages.SetWatchArea,
        FrontendMessages.GetWatchAreas,
        FrontendMessages.ConnectedDrones,
        FrontendMessages.ServerResponse,
    ],
    Field(discriminator="msg_type"),
]


def parse_drone_message(message: str) -> AnyDroneMessage:
    # We use TypeAdapter or wrap the Union in a field to validate
    # The easiest way for a list of mixed types is TypeAdapter:

    adapter = TypeAdapter(AnyDroneMessage)
    return adapter.validate_json(message)

    # Example of handling different types
    # if isinstance(validated, TelemetryMessage):
    #     print(f" -> Drone is at {validated.lat}, {validated.lon}")
    # elif isinstance(validated, TaskMessage):
    #     print(f" -> New task: {validated.task.action}")


def parse_capabilities(message: str) -> Capabilities:
    return Capabilities.model_validate_json(message)


def parse_telemetry(message: str) -> Telemetry:
    return Telemetry.model_validate_json(message)


def parse_detections(message: str) -> Detections:
    return Detections.model_validate_json(message)


def parse_frontend_message(message: str) -> AnyFrontendMessage:
    adapter = TypeAdapter(AnyFrontendMessage)
    return adapter.validate_json(message)
