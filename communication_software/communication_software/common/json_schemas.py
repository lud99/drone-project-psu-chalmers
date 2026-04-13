from __future__ import annotations
from typing import Annotated, Union, Literal, TypeVar, Generic, Optional
from pydantic import BaseModel, RootModel, Field, TypeAdapter
### Capabilities and telemetry


class CameraCapabilities(BaseModel):
    aspect_ratio: float
    diagonal_fov: float
    resolution_height: int
    resolution_width: int


class LEDCapabilities(BaseModel):
    types: Annotated[list[str], Field(min_length=1)]


class SpeakerCapabilities(BaseModel):
    audio_files: list[str]


class Capabilities(BaseModel):
    camera: Optional[CameraCapabilities]
    led: Optional[LEDCapabilities]
    spotlight: bool
    speaker: Optional[SpeakerCapabilities]


class Telemetry(BaseModel):
    lat: float
    lon: float
    alt: float
    heading: int
    speed: float
    battery_percent: int


### Drone message schemas


# Sub-models for Tasks
TaskEvents = Literal["task_complete", "task_failed", "task_aborted"]
TaskTypes = Literal["go_to", "led", "spotlight", "play_audio"]


# Specific task definitions
class GoToParams(BaseModel):
    lat: float
    lon: float
    alt: float
    heading: Optional[int] = None


class PlayAudioParams(BaseModel):
    file: str
    volume: float = 1.0
    duration_seconds: Optional[float]


class LEDParams(BaseModel):
    type: str
    duration_seconds: Optional[float]


class SpotlightParams(BaseModel):
    pattern: str
    duration_seconds: Optional[float]


class AngleCameraParams(BaseModel):
    pitch: float
    yaw: float
    duration_seconds: Optional[float] = None


class HoverParams(BaseModel):
    duration_seconds: Optional[float] = None


AnyParams = Union[
    GoToParams,
    PlayAudioParams,
    LEDParams,
    SpotlightParams,
    AngleCameraParams,
    HoverParams,
]

# The specific Task types


class GoToTask(BaseModel):
    action: Literal["go_to"] = "go_to"
    params: GoToParams


class PlayAudioTask(BaseModel):
    action: Literal["play_audio"] = "play_audio"
    params: PlayAudioParams


class LEDTask(BaseModel):
    action: Literal["led"] = "led"
    params: LEDParams


class SpotlightTask(BaseModel):
    action: Literal["spotlight"] = "spotlight"
    params: SpotlightParams


class AngleCameraTask(BaseModel):
    action: Literal["angle_camera"] = "angle_camera"
    params: AngleCameraParams


class HoverTask(BaseModel):
    action: Literal["hover"] = "hover"
    params: HoverParams


class GoHomeTask(BaseModel):
    action: Literal["go_home"] = "go_home"


# Uppdatera AnyTaskAction
AnyTaskAction = Union[
    GoToTask,
    PlayAudioTask,
    LEDTask,
    SpotlightTask,
    AngleCameraTask,
    HoverTask,
    GoHomeTask,
]


MsgTypeT = TypeVar("MsgTypeT", bound=str)


class DroneMessage(BaseModel, Generic[MsgTypeT]):
    msg_type: MsgTypeT


class BackendToDroneMessage(DroneMessage):
    drone_id: str


# Backend -> app
class TaskMessage(BackendToDroneMessage):
    msg_type: Literal["task"] = "task"
    mission_id: str
    index: int
    # This field now enforces strict structure based on the 'action' string
    task_action: AnyTaskAction = Field(..., discriminator="action")


# Backend -> app
class TaskEventMessage(BackendToDroneMessage):
    msg_type: Literal["task_event"] = "task_event"
    mission_id: str
    index: int
    event: TaskEvents
    message: Optional[str] = None
    timestamp: int


# Backend -> app
class AbortTaskMessage(BackendToDroneMessage):
    msg_type: Literal["abort_task"] = "abort_task"
    mission_id: str
    task_action: TaskTypes


# Backend -> app
class GoHomeMessage(BackendToDroneMessage):
    msg_type: Literal["go_home"] = "go_home"
    mission_id: str


# Backend -> app
class LandMessage(BackendToDroneMessage):
    msg_type: Literal["land"] = "land"
    mission_id: str


class DebugMessage(DroneMessage):
    msg_type: Literal["debug"] = "debug"
    message: str


# App -> backend. Upon registration
class DroneRegistrationMessage(DroneMessage):
    msg_type: Literal["drone_registration"]
    drone_type: str
    model: str
    drone_id: str
    capabilities: Capabilities
    telemetry: Telemetry


# App -> backend. Sent continuously. Also sent to frontend
class TelemetryMessage(DroneMessage):
    msg_type: Literal["telemetry"] = "telemetry"
    drone_id: str
    telemetry: Telemetry


class PingMessage(DroneMessage):
    msg_type: Literal["ping"] = "ping"


class PongMessage(DroneMessage):
    msg_type: Literal["pong"] = "pong"


# WebRTC messages


class WebRTCCandidateMessage(DroneMessage):
    msg_type: Literal["candidate"] = "candidate"
    candidate: str
    id: str = "0"
    label: int = 0


class WebRTCOfferMessage(DroneMessage):
    msg_type: Literal["offer"] = "offer"
    sdp: str


class WebRTCAnswerMessage(DroneMessage):
    msg_type: Literal["answer"] = "answer"
    sdp: str
    type: Optional[str] = None


# Create a Union of all possible messages
AnyDroneMessage = Annotated[
    Union[
        TaskMessage,
        DroneRegistrationMessage,
        TelemetryMessage,
        TaskEventMessage,
        AbortTaskMessage,
        GoHomeMessage,
        LandMessage,
        DebugMessage,
        PingMessage,
        PongMessage,
        WebRTCCandidateMessage,
        WebRTCOfferMessage,
        WebRTCAnswerMessage,
    ],
    Field(discriminator="msg_type"),
]


### Detections schema
class SingleDetection(BaseModel):
    gps_position: tuple[float, float]
    object_type: str
    detection_id: int
    drone_ids: Annotated[list[str], Field(min_length=1)]
    timestamp: int  # in milliseconds


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

    class SetWatchArea(FrontendMessage):
        msg_type: Literal["set_watch_area"] = "set_watch_area"
        area: Points

    # Backend -> Frontend

    class ProposedMissions(FrontendMessage):
        msg_type: Literal["proposed_missions"] = "proposed_missions"
        missions: list[dict]

    class NoProposedMissions(FrontendMessage):
        msg_type: Literal["no_proposed_missions"] = "no_proposed_missions"
        mission_type: Optional[str]
        coordinates: GoToParams

    class ActiveMissions(FrontendMessage):
        msg_type: Literal["active_missions"] = "active_missions"
        missions: list[dict]

    # Sent via WebSockets
    class DroneConnected(FrontendMessage):
        msg_type: Literal["drone_connected"] = "drone_connected"
        drone_id: str
        capabilities: Capabilities
        telemetry: Telemetry
        error: Optional[str] = None

    # Sent via WebSockets
    class DroneDisconnected(FrontendMessage):
        msg_type: Literal["drone_disconnected"] = "drone_disconnected"
        drone_id: str
        error: Optional[str] = None

    class GetWatchAreas(FrontendMessage):
        msg_type: Literal["get_watch_areas"] = "get_watch_areas"
        area: Points

    class ConnectedDrones(FrontendMessage):
        msg_type: Literal["connected_drones"] = "connected_drones"
        drones: list[DroneInfo]

    class ServerResponse(FrontendMessage):
        msg_type: Literal["response"] = "response"
        error: Optional[str] = None

    # Sent via WebSockets
    class Error(FrontendMessage):
        msg_type: Literal["error"] = "error"
        error: str


AnyFrontendMessage = Annotated[
    Union[
        FrontendMessages.ProposedMissions,
        FrontendMessages.ActiveMissions,
        FrontendMessages.DroneConnected,
        FrontendMessages.DroneDisconnected,
        FrontendMessages.SetWatchArea,
        FrontendMessages.GetWatchAreas,
        FrontendMessages.ConnectedDrones,
        FrontendMessages.ServerResponse,
        FrontendMessages.Error,
        TelemetryMessage,
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
