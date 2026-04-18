from pydantic import BaseModel, Field, field_validator, model_validator
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
    SET_CIRCULAR_GEOFENCE = "set_circular_geofence"
    CLEAR_CIRCULAR_GEOFENCE = "clear_circular_geofence"
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
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    distance_m: Optional[float] = Field(None, gt=0)
    direction: Optional[str] = None
    relative_altitude_m: Optional[float] = None
    yaw_deg: Optional[float] = Field(None, ge=0, le=360)
    acceptance_radius_m: Optional[float] = Field(None, gt=0)

    @model_validator(mode='after')
    def validate_target_definition(self):
        has_absolute_target = (
            self.latitude is not None or self.longitude is not None
        )
        has_relative_target = (
            self.distance_m is not None or self.direction is not None
        )

        if has_absolute_target and has_relative_target:
            raise ValueError(
                'Provide either latitude/longitude or distance_m/direction, not both'
            )

        if has_absolute_target:
            if self.latitude is None or self.longitude is None:
                raise ValueError(
                    'Both latitude and longitude are required for absolute goto')
            return self

        if has_relative_target:
            if self.distance_m is None or self.direction is None:
                raise ValueError(
                    'Both distance_m and direction are required for relative goto')
            direction = self.direction.upper()
            if direction not in {'N', 'S', 'E', 'W'}:
                raise ValueError('direction must be one of N, S, E, W')
            self.direction = direction
            return self

        raise ValueError(
            'Provide either latitude/longitude or distance_m/direction')


class GeofenceCommand(BaseModel):
    polygon: List[PolygonPoint] = Field(..., min_items=3, max_items=20)

    @field_validator('polygon')
    @classmethod
    def validate_polygon(cls, v):
        if len(v) < 3:
            raise ValueError('Polygon must have at least 3 points')
        # Check if closed (first and last point same)
        if v[0] != v[-1]:
            v = list(v)
            v.append(v[0])  # auto-close
        return v


class CircularGeofenceCommand(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(..., gt=0)


class Command(BaseModel):
    type: CommandType
    data: Optional[dict] = None

    @model_validator(mode='after')
    def validate_data(self):
        v = self.data
        cmd_type = self.type
        if cmd_type == CommandType.TAKEOFF_TO_RELATIVE_ALTITUDE:
            self.data = TakeoffCommand(**v).model_dump()
        elif cmd_type == CommandType.GOTO_POINT:
            self.data = GotoCommand(**v).model_dump()
        elif cmd_type == CommandType.SET_POLYGON_GEOFENCE:
            self.data = GeofenceCommand(**v).model_dump()
        elif cmd_type == CommandType.SET_CIRCULAR_GEOFENCE:
            self.data = CircularGeofenceCommand(**v).model_dump()
        return self


class StatusResponse(BaseModel):
    state: str
    current_command: Optional[str]
    geofence_active: bool
    polygon: Optional[List[Point]]
    circle: Optional[dict]
    latest_safety_message: Optional[str]
    using_mock_drone: bool
    telemetry: dict
