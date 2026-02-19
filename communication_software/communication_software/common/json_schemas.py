from __future__ import annotations
from typing import Annotated, Union, Literal
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


# Backend -> app
class TaskMessage(BaseModel):
    msg_type: Literal["task"]
    mission_id: str
    drone_id: str
    index: int
    # This field now enforces strict structure based on the 'action' string
    task: AnyTaskAction = Field(..., discriminator="action")


# App -> backend. Upon registration
class DroneRegistrationMessage(BaseModel):
    msg_type: Literal["drone_registration"]
    drone_type: str
    model: str
    drone_id: str
    capabilities: Capabilities


# App -> backend. Sent continuously
class TelemetryMessage(BaseModel):
    msg_type: Literal["telemetry"]
    drone_id: str
    lat: float
    lon: float
    alt: float
    heading: int
    speed: float
    battery_percent: int


# Backend -> app
class TaskEventMessage(BaseModel):
    msg_type: Literal["task_event"]
    mission_id: str
    index: int
    event: TaskEvents
    message: str
    drone_id: str
    timestamp: int


# Backend -> app
class AbortTaskMessage(BaseModel):
    msg_type: Literal["abort_task"]
    mission_id: str
    index: int
    next: Literal["go_home", "hover", "land"]


# Create a Union of all possible messages
AnyMessage = Annotated[
    Union[
        TaskMessage,
        DroneRegistrationMessage,
        TelemetryMessage,
        TaskEventMessage,
        AbortTaskMessage,
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


def parse_drone_message(message: str) -> AnyMessage:
    # We use TypeAdapter or wrap the Union in a field to validate
    # The easiest way for a list of mixed types is TypeAdapter:

    adapter = TypeAdapter(AnyMessage)
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
