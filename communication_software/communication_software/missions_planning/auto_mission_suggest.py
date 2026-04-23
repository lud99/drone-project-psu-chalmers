import math
import time
import os
import json
import threading
from typing import Optional
import redis
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
from communication_software.missions_planning.mission_status import MissionStatus
from communication_software.constants import DRONE_EVENT_CHANNEL, SURVEIL_AREA_CHANNEL
from communication_software.missions_planning.mission_registry import MissionRegistry

COOLDOWN_SECONDS = 60.0
DEDUP_DISTANCE_METERS = 10.0
GROUND_ALTITUDE = 0.0
BATTERY_THRESHOLD = 20.0  # Battery level threshold for replacement


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

        try:
            self._redis = redis.Redis(
                host=os.environ.get("REDIS_URL"),
                port=os.environ.get("REDIS_PORT"),
                db=0,
                decode_responses=True,
            )
            self._redis_async = redis.asyncio.Redis(
                host=os.environ.get("REDIS_URL"),
                port=os.environ.get("REDIS_PORT"),
                db=0,
                decode_responses=True,
            )
            self._redis.ping()  # Check if the connection is successful
            print("[Auto Mission Suggest] Successfully connected to Redis")
        except redis.exceptions.ConnectionError as e:
            print(f"[Auto Mission Suggest] Error connecting to Redis: {e}")
            exit()  # Exit if we can't connect

        self._cooldown_seconds = COOLDOWN_SECONDS
        self._dedupe_distance_meters = DEDUP_DISTANCE_METERS
        self._recent_detection_ids: dict[str, float] = {}
        self._recent_detection_events: list[json_schemas.SingleDetection] = []
        self._stop_event = threading.Event()
        self._battery_monitor_thread = threading.Thread(target=self.battery_monitor)
        self._battery_monitor_thread.start()

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

    def _extract_watch_area_points(self, points: dict) -> list[dict[str, float]]:
        """Parses Redis watch area payload and returns a normalized points list."""

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
        self, points: list[dict[str, float]], diagonal_fov: float
    ) -> tuple[
        json_schemas.GoToParams, list[tuple[float, float]], list[tuple[float, float]]
    ]:
        hull_points = [Coordinate(lat=p["lat"], lng=p["lon"]) for p in points]
        center_lat = sum(p["lat"] for p in points) / len(points)
        center_lon = sum(p["lon"] for p in points) / len(points)
        origin = Coordinate(lat=center_lat, lng=center_lon, alt=30)

        fly_to_coords, angle, coverage_corners, min_rect_corners = get_drones_location(
            corner_coords=hull_points,
            drone_origin=origin,
            n_drones=1,
            diagonal_fov=diagonal_fov,
        )
        best = fly_to_coords[0]
        return (
            json_schemas.GoToParams(
                lat=best.lat, lon=best.lng, alt=best.alt, heading=int(angle)
            ),
            coverage_corners,
            min_rect_corners,
        )

    @staticmethod
    def _distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Fast planar distance approximation for short-range de-dup checks."""
        m_per_deg = 111111.0
        dy = (lat2 - lat1) * m_per_deg
        dx = (lon2 - lon1) * m_per_deg * math.cos(math.radians(lat1))
        return math.sqrt(dx**2 + dy**2)

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        """Returns True if a point is inside a polygon using ray casting."""
        lat, lng = point
        x = lng
        y = lat
        inside = False
        n = len(polygon)
        for i in range(n):
            lat_i, lng_i = polygon[i]
            lat_j, lng_j = polygon[(i + 1) % n]
            xi = lng_i
            yi = lat_i
            xj = lng_j
            yj = lat_j
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-16) + xi
            )
            if intersects:
                inside = not inside
        return inside

    def _load_watch_area_min_rect(self) -> list[tuple[float, float]] | None:
        """Load the stored watch area minimum rectangle from Redis."""
        raw = self._redis.get("watch_area_min_rect")
        if not raw:
            return None
        rect = json.loads(raw)
        return [(float(lat), float(lng)) for lat, lng in rect]

    def _is_detection_within_watch_area(
        self, detection_gps_position: tuple[float, float]
    ) -> bool:
        """Check whether a detection falls inside the stored watch area rectangle."""
        polygon = self._load_watch_area_min_rect()
        if not polygon:
            return False
        return self._point_in_polygon(detection_gps_position, polygon)

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

    def send_response(
        self,
        request_id: str,
        coordinates: json_schemas.GoToParams,
        message: Optional[str] = None,
    ) -> None:
        # Notify that no drone was available
        response_key = f"response_{request_id}"
        self._redis.rpush(
            response_key,
            json.dumps({"mission": [], "coverage_corners": [], "message": message}),
        )
        self._redis.expire(response_key, 20)

        print("[area_listener] No drone with camera available")
        self.send_mission_unavailable("GotoAndSurveil", coordinates)

    async def redis_area_listener(self):

        try:
            pubsub = self._redis_async.pubsub()
            await pubsub.subscribe(SURVEIL_AREA_CHANNEL)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    request_id = data["request_id"]
                    points = data["points"]

                    points = self._extract_watch_area_points(points)
                    if len(points) < 3:
                        print("watch_area payload does not contain enough points.")
                        continue

                    print("[area_listener] Ny watch_area hittad, bearbetar...")
                    # Initial coordinates for potential error response
                    initial_coordinates, _, _ = self._get_area_surveil_coordinates(
                        points, diagonal_fov=82.6
                    )

                    current_surveil_drone = (
                        MissionRegistry.get_drone_on_surveil_mission()
                    )
                    if current_surveil_drone is not None:
                        print(
                            "[area_listener] A drone is already on a GotoAndSurveil mission, will not dispatch a new one"
                        )
                        self.send_response(
                            request_id,
                            initial_coordinates,
                            "A drone is already on a GotoAndSurveil mission, will not dispatch a new one",
                        )
                        continue

                    surveil_mission, coverage_corners, min_rect_corners, message = (
                        self._create_surveil_mission(points)
                    )

                    if surveil_mission is None:
                        self.send_response(
                            request_id,
                            initial_coordinates,
                            message,
                        )
                        continue

                    print("[area_listener] GotoAndSurveil - dispatchar automatiskt")

                    MissionRegistry.store(surveil_mission)
                    MissionRegistry.dispatch_mission(surveil_mission.mission_id)
                    surveil_mission.status = MissionStatus.DISPATCHED
                    print(f"[area_listener] First task sent")

                    self._redis.set("watch_area", json.dumps({"points": points}))
                    self._redis.set(
                        "watch_area_min_rect",
                        json.dumps(min_rect_corners),
                    )
                    self._redis.set(
                        "watch_area_coverage",
                        json.dumps(coverage_corners),
                    )

                    response_key = f"response_{request_id}"
                    print(f"Sending redis response for {request_id}")
                    self._redis.rpush(
                        response_key,
                        json.dumps(
                            {
                                "mission": surveil_mission.to_dict(),
                                "coverage_corners": coverage_corners,
                            }
                        ),
                    )
                    self._redis.expire(response_key, 20)

        except Exception as exc:
            print(f"[area_listener] Error in Redis listener: {exc}")

    def _create_surveil_mission(self, normalized_points, exclude_drone_id=None):
        """
        Creates a surveil mission for the given points, selecting an appropriate drone.
        Returns mission, coverage_corners, min_rect_corners, error_message (None if success).
        """
        if len(normalized_points) < 3:
            return None, None, None, "Not enough points"

        # Initial calculation with default FOV
        coordinates, coverage_corners, min_rect_corners = (
            self._get_area_surveil_coordinates(normalized_points, diagonal_fov=82.6)
        )

        surveil_mission = select_drone_for_mission(
            mission_type=GotoAndSurveil,
            coordinates=coordinates,
            params={"duration_seconds": None},
        )

        if surveil_mission is None:
            return None, None, None, "No drone available"

        if exclude_drone_id and surveil_mission.drone_id == exclude_drone_id:
            return None, None, None, "Only excluded drone available"

        # Recalculate with correct FOV
        drone_id = surveil_mission.drone_id
        diagonal_fov = json_schemas.parse_capabilities(
            self._redis.get(f"capabilities_drone{drone_id}")
        ).camera.diagonal_fov

        coordinates, coverage_corners, min_rect_corners = (
            self._get_area_surveil_coordinates(
                normalized_points, diagonal_fov=diagonal_fov
            )
        )

        surveil_mission = select_drone_for_mission(
            mission_type=GotoAndSurveil,
            coordinates=coordinates,
            params={"duration_seconds": None},
        )

        if surveil_mission is None or (
            exclude_drone_id and surveil_mission.drone_id == exclude_drone_id
        ):
            return None, None, None, "No suitable drone after FOV adjustment"

        return surveil_mission, coverage_corners, min_rect_corners, None

    def battery_monitor(self):
        """
        Continuously monitors battery levels of all drones.
        If the surveil drone's battery is below threshold, dispatches a new surveil mission with another drone.
        """
        while not self._stop_event.is_set():
            current_surveil_drone = MissionRegistry.get_drone_on_surveil_mission()
            if current_surveil_drone is not None:
                for key in self._redis.scan_iter(match="telemetry_drone*"):
                    drone_id = key.replace("telemetry_drone", "")
                    raw = self._redis.get(key)
                    if raw:
                        try:
                            telemetry = json_schemas.parse_telemetry(raw)
                            if (
                                telemetry.battery_percent <= BATTERY_THRESHOLD
                                and drone_id == current_surveil_drone
                            ):
                                print(
                                    f"Battery low for drone {drone_id} ({telemetry.battery_percent}%), attempting to replace surveil mission."
                                )
                                self._add_pending_surveil_mission()

                                # Send the old drone home, even if we can't find a replacement
                                print(
                                    f"[battery_monitor] Sending drone {drone_id} home"
                                )
                                go_home_mission = (
                                    MissionRegistry.abort_mission_and_go_home(drone_id)
                                )

                                self._redis.publish(
                                    DRONE_EVENT_CHANNEL,
                                    json_schemas.FrontendMessages.NewMission(
                                        mission=go_home_mission
                                    ).model_dump_json(),
                                )
                        except Exception as exc:
                            print(
                                f"Failed to parse telemetry for drone {drone_id}: {exc}"
                            )
            self._sleep_with_stop_check(5.0)  # Check every 5 seconds

    def _add_pending_surveil_mission(self):
        """
        Replaces the current surveil mission with a new one using a different drone.
        Stores the new mission in Redis and sends the old drone home.
        """
        points_raw = self._redis.get("watch_area")
        if not points_raw:
            print("No watch area stored, cannot replace surveil mission.")
            return

        points = json.loads(points_raw)["points"]
        normalized_points = self._extract_watch_area_points(points)

        current_surveil_drone = MissionRegistry.get_drone_on_surveil_mission()
        surveil_mission, _coverage_corners, _min_rect_corners, message = (
            self._create_surveil_mission(
                normalized_points, exclude_drone_id=current_surveil_drone
            )
        )

        if surveil_mission is None:
            print(f"Failed to replace surveil mission: {message}")
            return

        # Store the pending surveil mission in Redis to be dispatched after go_home completes
        MissionRegistry.store(surveil_mission)
        # Mark it as waiting so the drone appears busy
        surveil_mission_dict = MissionRegistry.update_status(
            surveil_mission.mission_id, MissionStatus.WAITING
        )
        pending_surveil_mission_message = json_schemas.FrontendMessages.NewMission(
            mission=surveil_mission_dict
        )

        self._redis.set(
            f"pending_surveil_mission",
            pending_surveil_mission_message.model_dump_json(),
        )

        print(
            f"[battery_monitor] Storing pending surveil mission {surveil_mission.mission_id} for drone {surveil_mission.drone_id}"
        )

        # Notify frontend

        self._redis.publish(
            DRONE_EVENT_CHANNEL, pending_surveil_mission_message.model_dump_json()
        )

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
                    # Only propose missions if the drone is flying
                    if MissionRegistry.is_drone_dispatched_or_waiting(
                        detection.drone_ids[0]
                    ):
                        self.handle_detected_object(detection)
                    else:
                        pass
                        # print("Found detection but drone is not dispatched")

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
        dy_new = (lat_object - lat_new) * m_per_deg
        dx_new = (lon_object - lon_new) * m_per_deg * math.cos(math.radians(lat_new))
        heading_rad = math.atan2(dx_new, dy_new)
        heading = int((math.degrees(heading_rad)))

        return json_schemas.GoToParams(
            lat=lat_new, lon=lon_new, alt=alt_new, heading=heading
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

        if not self._is_detection_within_watch_area(detection.gps_position):
            print(
                f"Detection {detection.detection_id} at {detection.gps_position[0]}, {detection.gps_position[1]} is outside watch area, skipping."
            )
            return

        print(
            f"Found new detection {detection.detection_id}, will generate and propose missions..."
        )

        if detection.object_type == "person":
            self.handle_detected_person(detection)
        elif detection.object_type in ["vehicle", "car", "truck", "bus"]:
            self.handle_detected_vehicle(detection)

    def handle_detected_person(self, detection: json_schemas.SingleDetection) -> None:
        """
        Handles a detected person message.
        Tries missions in priority order and selects the first one with an available drone.
        """

        coordinates = detection.gps_position

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
            self.send_proposed_missions(viable_missions, detection)
        else:
            print(f"No drone available for any mission at {coordinates}")
            self.send_mission_unavailable(
                None,
                json_schemas.GoToParams(lat=coordinates[0], lon=coordinates[1], alt=0),
            )

            # self.send_object_notification(object_type="person", coordinates=coordinates)

    def handle_detected_vehicle(self, detection: json_schemas.SingleDetection) -> None:
        """
        Handles a detected vehicle message.
        Parses the message and sends a mission suggestion to the frontend.
        """
        coordinates = detection.gps_position
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
            self.send_proposed_missions(viable_missions, detection)
        else:
            print(f"No drone available for mission detected vehicle at {coordinates}")
            self.send_mission_unavailable(
                None,
                json_schemas.GoToParams(lat=coordinates[0], lon=coordinates[1], alt=0),
            )

            # self.send_object_notification(
            # object_type="vehicle", coordinates=coordinates
            # )

    def send_proposed_missions(
        self, missions: list[Mission], detection: json_schemas.SingleDetection
    ) -> None:
        for mission in missions:
            MissionRegistry.store(mission)

        proposed_payload = json_schemas.FrontendMessages.ProposedMissions(
            detection_id=detection.detection_id,
            object_type=detection.object_type,
            gps_position=detection.gps_position,
            timestamp=detection.timestamp,
            missions=[
                json_schemas.ProposedMission(**mission.get_frontend_mission_proposal())
                for mission in missions
            ],
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
