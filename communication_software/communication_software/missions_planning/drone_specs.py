from dataclasses import dataclass
from typing import Optional


@dataclass
class DroneSpecs:
    id: str
    model: str
    speaker: bool
    lights: bool
    camera: bool
    aspect_ratio: Optional[str] = None
    horizontal_fov: Optional[float] = None
    resolution: Optional[str] = None
