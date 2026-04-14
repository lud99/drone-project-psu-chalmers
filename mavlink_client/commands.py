from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class CommandType(str, Enum):
    ARM = "arm"
    TAKEOFF_TO_RELATIVE_ALTITUDE = "takeoff_to_relative_altitude"
    LAND = "land"
    DISARM = "disarm"
    GOTO_POINT = "goto_point"
    SET_POLYGON_GEOFENCE = "set_polygon_geofence"
    CLEAR_POLYGON_GEOFENCE = "clear_polygon_geofence"
    HOLD = "hold"
    GET_STATUS = "get_status"


class Point(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class PolygonPoint(Point):
    pass


class TakeoffCommand(BaseModel):
    relative_altitude_m: float = Field(..., gt=0)


class GotoCommand(BaseModel):
    latitude: float
    longitude: float
    relative_altitude_m: Optional[float] = None
    yaw_deg: Optional[float] = Field(None, ge=0, le=360)
    acceptance_radius_m: Optional[float] = Field(None, gt=0)


class GeofenceCommand(BaseModel):
    polygon: List[PolygonPoint] = Field(..., min_items=3, max_items=20)

    @validator('polygon')
    def validate_polygon(cls, v):
        if len(v) < 3:
            raise ValueError('Polygon must have at least 3 points')
        # Check if closed (first and last point same)
        if v[0] != v[-1]:
            v.append(v[0])  # auto-close
        return v


class Command(BaseModel):
    type: CommandType
    data: Optional[dict] = None

    @validator('data', pre=True, always=True)
    def validate_data(cls, v, values):
        cmd_type = values.get('type')
        if cmd_type == CommandType.TAKEOFF_TO_RELATIVE_ALTITUDE:
            return TakeoffCommand(**v).dict()
        elif cmd_type == CommandType.GOTO_POINT:
            return GotoCommand(**v).dict()
        elif cmd_type == CommandType.SET_POLYGON_GEOFENCE:
            return GeofenceCommand(**v).dict()
        return v


class StatusResponse(BaseModel):
    state: str
    current_command: Optional[str]
    geofence_active: bool
    polygon: Optional[List[Point]]
    latest_safety_message: Optional[str]
    using_mock_drone: bool
    telemetry: dict
