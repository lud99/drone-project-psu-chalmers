from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time


@dataclass
class Telemetry:
    drone_id: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    heading: Optional[int] = None
    speed: Optional[float] = None
    battery_percent: Optional[int] = None
    mode: Optional[str] = None
    armed: Optional[bool] = None
    gps_fix_type: Optional[int] = None
    satellites_visible: Optional[int] = None
    timestamp: float = 0.0

    def to_backend_message(self) -> Dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        payload["msg_type"] = "telemetry"

        if "timestamp" not in payload or not payload["timestamp"]:
            payload["timestamp"] = time.time()

        return payload
