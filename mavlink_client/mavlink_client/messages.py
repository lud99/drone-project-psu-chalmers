"""
Wire-format message schemas aligned with the backend (app_backend_interface).
Used to build outgoing messages and parse incoming commands.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ----- Outgoing (client -> backend) -----


class TelemetryPayload(BaseModel):
    """Nested telemetry in TelemetryMessage."""

    lat: float
    lon: float
    alt: float
    heading: int
    speed: float
    battery_percent: int


class TelemetryMessage(BaseModel):
    """Sent at ~1 Hz. Backend expects msg_type 'telemetry' and nested telemetry."""

    msg_type: Literal["telemetry"] = "telemetry"
    drone_id: str
    telemetry: TelemetryPayload


class CameraCapabilities(BaseModel):
    aspect_ratio: float = 16 / 9
    horizontal_fov: float = 84.0
    resolution_width: int = 1920
    resolution_height: int = 1080


class LEDCapabilities(BaseModel):
    colors: list[str] = Field(default_factory=list)


class Capabilities(BaseModel):
    """Drone capabilities sent at registration."""

    camera: CameraCapabilities | None = None
    led: LEDCapabilities | None = None
    spotlight: bool = False
    speaker: bool = False
    max_speed: float = 15.0


class DroneRegistrationMessage(BaseModel):
    """Sent once after connecting to backend."""

    msg_type: Literal["drone_registration"] = "drone_registration"
    drone_type: Literal["MAVLink"] = "MAVLink"
    model: str
    drone_id: str
    capabilities: Capabilities


class TaskEventMessage(BaseModel):
    """Sent when a task completes or fails."""

    msg_type: Literal["task_event"] = "task_event"
    mission_id: str
    index: int
    event: Literal["task_complete", "task_failed"]
    drone_id: str
    message: str = ""
    timestamp: int = 0


# ----- Incoming (backend -> client) -----


class GoToParams(BaseModel):
    lat: float
    lon: float
    alt: float
    heading: int


class GoToTask(BaseModel):
    action: Literal["go_to"] = "go_to"
    params: GoToParams


class PlayAudioParams(BaseModel):
    file: str
    volume: float = 1.0
    duration_seconds: int


class PlayAudioTask(BaseModel):
    action: Literal["play_audio"] = "play_audio"
    params: PlayAudioParams


class LEDParams(BaseModel):
    color: str
    pattern: str
    duration_seconds: float


class LEDTask(BaseModel):
    action: Literal["led"] = "led"
    params: LEDParams


class SpotlightParams(BaseModel):
    pattern: str
    duration_seconds: float


class SpotlightTask(BaseModel):
    action: Literal["spotlight"] = "spotlight"
    params: SpotlightParams


class GoHomeTask(BaseModel):
    action: Literal["go_home"] = "go_home"
    params: dict = Field(default_factory=dict)


class LandTask(BaseModel):
    action: Literal["land"] = "land"
    params: dict = Field(default_factory=dict)


AnyTask = Union[GoToTask, PlayAudioTask, LEDTask, SpotlightTask, GoHomeTask, LandTask]


class TaskMessage(BaseModel):
    """Backend sends this to run a task."""

    msg_type: Literal["task"] = "task"
    mission_id: str
    drone_id: str
    index: int
    task: (
        GoToTask
        | PlayAudioTask
        | LEDTask
        | SpotlightTask
        | GoHomeTask
        | LandTask
    ) = Field(..., discriminator="action")


class AbortTaskMessage(BaseModel):
    """Backend sends this to abort current task."""

    msg_type: Literal["abort_task"] = "abort_task"
    mission_id: str
    index: int
    next: Literal["go_home", "hover", "land"] = "hover"


def parse_incoming(raw: str) -> dict[str, Any]:
    """
    Parse incoming JSON and return a dict with at least 'msg_type'.
    Does not validate full schema; use for routing then specific parsers if needed.
    """
    import json

    return json.loads(raw)


def build_telemetry(drone_id: str, payload: TelemetryPayload) -> str:
    return TelemetryMessage(drone_id=drone_id, telemetry=payload).model_dump_json()


def build_registration(drone_id: str, model: str, capabilities: Capabilities) -> str:
    return DroneRegistrationMessage(
        drone_id=drone_id, model=model, capabilities=capabilities
    ).model_dump_json()


def build_task_event(
    drone_id: str,
    mission_id: str,
    index: int,
    event: Literal["task_complete", "task_failed"],
    message: str = "",
) -> str:
    import time

    return TaskEventMessage(
        mission_id=mission_id,
        index=index,
        event=event,
        drone_id=drone_id,
        message=message,
        timestamp=int(time.time()),
    ).model_dump_json()


def build_coordinate_response(lat: float, lon: float, alt: float, angle: float) -> str:
    """Response to Coordinate_request (backend may send request for current pos)."""
    import json

    return json.dumps({
        "msg_type": "coordinate_response",
        "lat": str(lat),
        "lng": str(lon),
        "alt": str(alt),
        "angle": str(angle),
    })
