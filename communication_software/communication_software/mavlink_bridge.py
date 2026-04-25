import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MavlinkBridgeError(Exception):
    """Raised when the bridge cannot reach or parse MAVLink API responses."""


@dataclass
class MavlinkBridgeConfig:
    base_url: str = os.environ.get("MAVLINK_API_BASE_URL", "http://localhost:8010")
    timeout_seconds: float = float(os.environ.get("MAVLINK_API_TIMEOUT_SEC", "5"))


class MavlinkBridgeClient:
    def __init__(self, config: MavlinkBridgeConfig | None = None) -> None:
        self.config = config or MavlinkBridgeConfig()

    def goto(
        self,
        latitude: float,
        longitude: float,
        relative_altitude_m: float,
        acceptance_radius_m: float | None = None,
        yaw_deg: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "relative_altitude_m": relative_altitude_m,
        }

        if acceptance_radius_m is not None:
            payload["acceptance_radius_m"] = acceptance_radius_m
        if yaw_deg is not None:
            payload["yaw_deg"] = yaw_deg

        return self._post_json("/api/goto", payload)

    def takeoff(self, relative_altitude_m: float) -> dict[str, Any]:
        return self._post_json(
            "/api/takeoff", {"relative_altitude_m": relative_altitude_m}
        )

    def hold(self) -> dict[str, Any]:
        return self._post_json("/api/hold", {})

    def land(self) -> dict[str, Any]:
        return self._post_json("/api/land", {})

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded_payload = json.dumps(payload).encode("utf-8")
        request = Request(
            url=f"{self.config.base_url.rstrip('/')}{endpoint}",
            data=encoded_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                if not body:
                    return {"success": True, "message": "Empty response from MAVLink API"}
                return json.loads(body)
        except HTTPError as exc:
            response_body = ""
            if exc.fp is not None:
                response_body = exc.fp.read().decode("utf-8", errors="replace")

            raise MavlinkBridgeError(
                f"MAVLink API returned HTTP {exc.code}: {response_body}"
            ) from exc
        except URLError as exc:
            raise MavlinkBridgeError(
                f"Failed to reach MAVLink API at {self.config.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise MavlinkBridgeError("Timed out waiting for MAVLink API") from exc
        except json.JSONDecodeError as exc:
            raise MavlinkBridgeError("MAVLink API returned invalid JSON") from exc
