import math
import time
import os
import json
import threading
from typing import Any, Optional
import redis
from communication_software.missions_planning.mission_status import MissionStatus
import communication_software.common.json_schemas as json_schemas
from communication_software.convex_hull_scalable import Coordinate, get_drones_location
from .missions import (
    GotoAndAudio,
    GotoAndBlink,
    GotoAndIlluminate,
    GotoAndSurveil,
    GotoOnly,
    Mission,
)
from .drone_selector import select_drone_for_mission
from communication_software.constants import DRONE_EVENT_CHANNEL
from communication_software.missions_planning.mission_registry import MissionRegistry

COOLDOWN_SECONDS = 60.0
DEDUP_DISTANCE_METERS = 10.0
GROUND_ALTITUDE = 0.0


class AutoMissionSuggester:
    """This class listens for updates in object detection
    and suggests missions based on the detected objects."""

    def __init__(self):
        """
        Starts a listener for updates in object detection.
        """
        self._frontend_ws_url = os.environ.get(
            "PROPOSED_MISSIONS_WS_URL",
            "ws://localhost:8000/api/v1/ws/drone",
        )
        self._redis = redis.Redis(
            host=os.environ.get("REDIS_URL"),
            port=os.environ.get("REDIS_PORT"),
            db=0,
            decode_responses=True,
        )
        self._cooldown_seconds = COOLDOWN_SECONDS
        self._dedupe_distance_meters = DEDUP_DISTANCE_METERS
        self._recent_detection_ids: dict[str, float] = {}
        self._recent_detection_events: list[json_schemas.SingleDetection] = []
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Signals listeners to stop gracefully."""
        self._stop_event.set()

    def clear_stop_request(self) -> None:
        """Clears stop signal before restarting listeners."""
        self._stop_event.clear()

    def _sleep_with_stop_check(self, seconds: float, step: float = 0.1) -> None:
        """Sleeps in short chunks so listeners can stop promptly."""
        deadline = time.time() + max(0.0, seconds)
        while time.time() < deadline and not self._stop_event.is_set():
            time.sleep(min(step, max(0.0, deadline - time.time())))

    def _extract_watch_area_points(self, raw_watch_area: str) -> list[dict[str, float]]:
        """Parses Redis watch area payload and returns a normalized points list."""
        payload = json.loads(raw_watch_area)

        points: Any
        if isinstance(payload, dict):
            if isinstance(payload.get("points"), list):
                points = payload["points"]
            elif isinstance(payload.get("area"), list):
                points = payload["area"]
            else:
                points = []
        elif isinstance(payload, list):
            points = payload
        else:
            points = []

        normalized_points: list[dict[str, float]] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            lat = point.get("lat")
            lon = point.get("lon")
            if lat is None or lon is None:
                continue
            normalized_points.append({"lat": float(lat), "lon": float(lon)})

        return normalized_points

    def _get_area_surveil_coordinates(
        self, points: list[dict[str, float]]
    ) -> json_schemas.GoToParams:
        hull_points = [Coordinate(lat=p["lat"], lng=p["lon"]) for p in points]
        center_lat = sum(p["lat"] for p in points) / len(points)
        center_lon = sum(p["lon"] for p in points) / len(points)
        origin = Coordinate(lat=center_lat, lng=center_lon, alt=30)

        fly_to_coords, angle = get_drones_location(
            corner_coords=hull_points,
            drone_origin=origin,
            n_drones=1,
        )
        best = fly_to_coords[0]
        return json_schemas.GoToParams(
            lat=best.lat, lon=best.lng, alt=best.alt, heading=int(angle)
        )

    @staticmethod
    def _distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Fast planar distance approximation for short-range de-dup checks."""
        m_per_deg = 111111.0
        dy = (lat2 - lat1) * m_per_deg
        dx = (lon2 - lon1) * m_per_deg * math.cos(math.radians(lat1))
        return math.sqrt(dx**2 + dy**2)

    def _should_skip_detection(self, detection: json_schemas.SingleDetection) -> bool:
        now = time.time()
        cutoff = now - self._cooldown_seconds

        self._recent_detection_ids = {
            key: ts for key, ts in self._recent_detection_ids.items() if ts >= cutoff
        }
        self._recent_detection_events = [
            event
            for event in self._recent_detection_events
            if float(event.timestamp) >= cutoff
        ]

        if detection.detection_id:
            id_key = f"{detection.object_type}:{detection.detection_id}"
            if id_key in self._recent_detection_ids:
                print(
                    f"Skipping detection {detection.detection_id}, was already detection during cooldown period"
                )
                return True

        lat = detection.gps_position[0]
        lon = detection.gps_position[1]

        for event in self._recent_detection_events:
            if event.object_type != detection.object_type:
                continue
            distance = self._distance_meters(
                float(lat),
                float(lon),
                float(event.gps_position[0]),
                float(event.gps_position[1]),
            )
            if distance <= self._dedupe_distance_meters:
                return True

        if detection.detection_id:
            self._recent_detection_ids[
                f"{detection.object_type}:{detection.detection_id}"
            ] = now

            self._recent_detection_events.append(detection)

        return False

    def area_listener(self):
        last_area_payload: str | None = None

        while not self._stop_event.is_set():
            raw_watch_area = self._redis.get("watch_area")
            if not raw_watch_area:
                self._sleep_with_stop_check(1.0)
                continue

            if raw_watch_area == last_area_payload:
                self._sleep_with_stop_check(1.0)
                continue

            # try:
            points = self._extract_watch_area_points(raw_watch_area)
            if len(points) < 3:
                print("watch_area payload does not contain enough points.")
                last_area_payload = raw_watch_area
                self._sleep_with_stop_check(1.0)
                continue

            print("[area_listener] Ny watch_area hittad, bearbetar...")
            coordinates = self._get_area_surveil_coordinates(points)

            surveil_mission = select_drone_for_mission(
                mission_type=GotoAndSurveil,
                coordinates=coordinates,
                params={"duration_seconds": None},
            )

            if surveil_mission:
                drone_id = surveil_mission.drone_id
                registry = MissionRegistry()

                # Kolla om just denna drönare redan har en DISPATCHED mission
                all_missions = registry.get_all()
                drone_busy = any(
                    m["drone_id"] == drone_id
                    and m["status"] == MissionStatus.DISPATCHED
                    for m in all_missions
                )

                if drone_busy:
                    print(
                        f"[area_listener] Drönare {drone_id} är redan aktiv - ignorerar"
                    )
                    last_area_payload = raw_watch_area
                    self._sleep_with_stop_check(1.0)
                    continue

                print("[area_listener] GotoAndSurveil - dispatchar automatiskt")
                registry.store(surveil_mission)
                first_task_raw = self._redis.lpop(
                    f"mission_{surveil_mission.mission_id}_task_queue"
                )
                if first_task_raw:
                    self._redis.publish("drone_commands", first_task_raw)
                    print(f"[area_listener] First task sent")
            else:
                print("[area_listener] No drone with camera available")
                self.send_mission_unavailable("GotoAndSurveil", coordinates)

            last_area_payload = raw_watch_area
            # except Exception as exc:
            # print(f"area_listener failed: {exc}")

            self._sleep_with_stop_check(1.0)

    def object_listener(self):
        """
        Continuously polls Redis for new detection snapshots and dispatches
        detected objects for mission suggestion.
        """
        last_processed_by_key: dict[str, str] = {}

        while not self._stop_event.is_set():
            for key in self._redis.scan_iter(match="frame_drone*_detections"):
                raw = self._redis.get(key)
                if not raw:
                    continue

                # Skip already handled frames for this key.
                if last_processed_by_key.get(key) == raw:
                    continue

                try:
                    detections = json_schemas.parse_detections(raw)
                except Exception as exc:
                    print(f"Failed to parse detections in {key}: {exc}")
                    continue

                last_processed_by_key[key] = raw

                for detection in detections.root:
                    self.handle_detected_object(detection)

            self._sleep_with_stop_check(0.5)

    def is_allowed(self, object_type: str, coordinates: dict) -> bool:
        """Check with ATOS if object is expected. Currently just a placeholder"""
        return False

    def get_offset_coordinates_for_drone(
        self,
        drone_id: str,
        object_coords: tuple[float, float],
        offset_meters: float | int,
    ) -> json_schemas.GoToParams:
        """
        Retrieve current coordinates of drone from Redis and calculate a target
        point that is `offset_meters` before the detected object and above it.
        """
        telemetry_raw = self._redis.get(f"telemetry_drone{drone_id}")
        if not telemetry_raw:
            raise ValueError(f"No telemetry found for drone '{drone_id}'")

        telemetry = json_schemas.parse_telemetry(telemetry_raw)
        lat_drone = telemetry.lat
        lon_drone = telemetry.lon
        alt_drone = telemetry.alt

        lat_object = object_coords[0]
        lon_object = object_coords[1]

        # Planar approximation in meters for short distances.
        m_per_deg = 111111.0
        dy = (lat_object - lat_drone) * m_per_deg
        dx = (lon_object - lon_drone) * m_per_deg * math.cos(math.radians(lat_drone))
        dist = math.sqrt(dx**2 + dy**2)

        if dist <= 1e-6:
            return json_schemas.GoToParams(
                lat=lat_object,
                lon=lon_object,
                alt=max(alt_drone, GROUND_ALTITUDE + offset_meters),
                heading=None,
            )

        # Clamp to [0, 1] so very small dist doesn't overshoot behind drone.
        travel = max(0.0, min(1.0, (dist - offset_meters) / dist))

        lat_new = lat_drone + travel * (lat_object - lat_drone)
        lon_new = lon_drone + travel * (lon_object - lon_drone)
        alt_new = alt_drone + offset_meters
        return json_schemas.GoToParams(
            lat=lat_new, lon=lon_new, alt=alt_new, heading=None
        )

        # TODO: This won't work as we can't get altitude from detection

    def get_coordinates_above_for_drone(
        self, object_coords: tuple[float, float], offset_meters: float | int
    ) -> json_schemas.GoToParams:
        """
        Calculate new coordinates for the drone that are directly above the object
        """
        lat = object_coords[0]
        lon = object_coords[1]
        alt = GROUND_ALTITUDE + offset_meters
        return json_schemas.GoToParams(lat=lat, lon=lon, alt=alt, heading=None)

    def send_object_notification(
        self, object_type: str, coordinates: tuple[float, float]
    ) -> None:
        """
        Sends a notification to the frontend about a detected object.
        """
        pass

    def handle_detected_object(self, detection: json_schemas.SingleDetection) -> None:
        """
        Handles a new message from the Redis channel.
        Parses the message and sends a mission suggestion to the backend.
        """

        if self._should_skip_detection(detection):
            return

        if detection.object_type == "person":
            self.handle_detected_person(detection.gps_position)
        elif detection.object_type in ["vehicle", "car", "truck", "bus"]:
            self.handle_detected_vehicle(detection.gps_position)

    def handle_detected_person(self, coordinates: tuple[float, float]) -> None:
        """
        Handles a detected person message.
        Tries missions in priority order and selects the first one with an available drone.
        """

        # Preferred missions in priority order
        missions_order = [
            (
                GotoAndAudio,
                ({"audio_type": "intruder_instructions"}, {"audio_type": "alert"}),
            ),
            (GotoAndBlink, ({"duration_seconds": None},)),
            (GotoAndIlluminate, ({"duration_seconds": None},)),
            (GotoAndSurveil, ({"duration_seconds": None},)),
            (GotoOnly, ({},)),
        ]

        # Try each mission type in order until one has an available drone
        viable_missions = []
        for mission_type, params_tuple in missions_order:
            for params in params_tuple:
                try:
                    mission = select_drone_for_mission(
                        mission_type=mission_type,
                        coordinates=json_schemas.GoToParams(
                            lat=coordinates[0], lon=coordinates[1], alt=0
                        ),
                        params=params,
                    )
                except Exception as exc:
                    print(
                        f"Skipping mission type {mission_type.__name__} for person detection: {exc}"
                    )
                    continue

                if mission:
                    # Update the coordinates of the mission to be 3m this way and 3m up from object
                    offset_missions = [GotoAndAudio, GotoAndIlluminate, GotoAndSurveil]
                    if mission_type in offset_missions:
                        new_coordinates = self.get_offset_coordinates_for_drone(
                            drone_id=mission.drone_id,
                            object_coords=coordinates,
                            offset_meters=3,
                        )
                    else:
                        # TODO: This won't work as we can't get altitude from detection
                        new_coordinates = self.get_coordinates_above_for_drone(
                            object_coords=coordinates, offset_meters=3
                        )

                    mission.coordinates = new_coordinates
                    viable_missions.append(mission)

        if viable_missions:
            self.send_proposed_missions(viable_missions)
        else:
            print(f"No drone available for any mission at {coordinates}")
            self.send_mission_unavailable(
                None,
                json_schemas.GoToParams(lat=coordinates[0], lon=coordinates[1], alt=0),
            )

            # self.send_object_notification(object_type="person", coordinates=coordinates)

    def handle_detected_vehicle(self, coordinates: tuple[float, float]) -> None:
        """
        Handles a detected vehicle message.
        Parses the message and sends a mission suggestion to the frontend.
        """
        missions_order = [
            (GotoAndAudio, ({"audio_type": "stray_car"}, {"audio_type": "alert"})),
            (GotoAndBlink, ({"duration_seconds": 10},)),
            (GotoAndIlluminate, ({"duration_seconds": 10},)),
            (GotoAndSurveil, ({"duration_seconds": None},)),
            (GotoOnly, ({},)),
        ]

        viable_missions = []
        for mission_type, params_tuple in missions_order:
            for params in params_tuple:
                try:
                    mission = select_drone_for_mission(
                        mission_type=mission_type,
                        coordinates=json_schemas.GoToParams(
                            lat=coordinates[0], lon=coordinates[1], alt=0
                        ),
                        params=params,
                    )
                except Exception as exc:
                    print(
                        f"Skipping mission type {mission_type.__name__} for vehicle detection: {exc}"
                    )
                    continue

                if mission:
                    offset_missions = [GotoAndAudio, GotoAndIlluminate, GotoAndSurveil]
                    if mission_type in offset_missions:
                        offset = 10 if mission_type == GotoAndSurveil else 5
                        new_coordinates = self.get_offset_coordinates_for_drone(
                            drone_id=mission.drone_id,
                            object_coords=coordinates,
                            offset_meters=offset,
                        )
                        mission.coordinates = new_coordinates
                    else:
                        new_coordinates = self.get_coordinates_above_for_drone(
                            object_coords=coordinates, offset_meters=5
                        )
                        mission.coordinates = new_coordinates
                    viable_missions.append(mission)
        if viable_missions:
            self.send_proposed_missions(viable_missions)
        else:
            print(f"No drone available for mission detected vehicle at {coordinates}")
            self.send_mission_unavailable(
                None,
                json_schemas.GoToParams(lat=coordinates[0], lon=coordinates[1], alt=0),
            )

            # self.send_object_notification(
            # object_type="vehicle", coordinates=coordinates
            # )

    def send_proposed_missions(self, missions: list[Mission]) -> None:

        registry = MissionRegistry()
        for mission in missions:
            registry.store(mission)

        proposed_payload = json_schemas.FrontendMessages.ProposedMissions(
            missions=[mission.get_frontend_mission_proposal() for mission in missions]
        )
        payload = proposed_payload.model_dump_json()

        self._send_proposed_missions_ws(payload)

    def send_mission_unavailable(
        self, mission_type: Optional[str], coordinates: json_schemas.GoToParams
    ) -> None:
        """
        Sends a notification to the frontend that no drone is available for a suggested mission.
        """
        try:
            self._redis.publish(
                DRONE_EVENT_CHANNEL,
                json_schemas.FrontendMessages.NoProposedMissions(
                    mission_type=mission_type, coordinates=coordinates
                ).model_dump_json(),
            )
        except Exception as exc:
            print(f"Failed to send proposed missions over redis: {exc}")

    def _send_proposed_missions_ws(self, payload: str) -> None:
        try:
            self._redis.publish(DRONE_EVENT_CHANNEL, payload)
        except Exception as exc:
            print(f"Failed to send proposed missions over redis: {exc}")
