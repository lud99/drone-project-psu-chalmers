"""
Automatically selects the most suitable connected drone for a given mission type

Selection consists of three steps:
    1. Availability  – drone must be connected (live telemetry + capabilities in Redis)
    2. Capability    – drone hardware must satisfy the mission requirements
    3. Ranking       – among all eligible drones, pick the highest-scoring one based on
                       a weighted combination of battery level and hardware quality
"""

from __future__ import annotations

import os
import json
import logging
import math
from dataclasses import dataclass
from typing import Optional, Type
import communication_software.common.json_schemas as json_schemas

import redis
import redis.exceptions

from .missions import (
    Mission,
    GotoAndAudio,
    GotoAndBlink,
    GotoAndIlluminate,
    GotoAndSurveil,
    GotoOnly,
)

from .mission_registry import MissionRegistry

logger = logging.getLogger(__name__)

# total must be 1
WEIGHT_BATTERY: float = 0.35  # Battery
WEIGHT_PROXIMITY: float = 0.2  # Battery
WEIGHT_HARDWARE: float = 0.45  # Mission specific hardware quality

# Maybe use a simple function that maps total number of pixels to a score instead.
"""
a = 26.731
b = -322.86
f = lambda num_pixels: a * math.log(num_pixels) + b
where num_pixels is resolution_width * resolution_height
if < 0 -> 0
if > 100 -> 100
through curvefit

"""


def resolution_score(resolution_width: int, resolution_height: int) -> float:
    """Maps resolution to a 0-100 score using a logarithmic function.
    The values (a, b) were obtained by curve fitting width*height for RESOLUTION_SCORES
    """
    num_pixels = resolution_width * resolution_height
    a = 20.094
    b = -246.0
    score = a * math.log(num_pixels) + b
    return max(0.0, min(score, 100.0))


# RESOLUTION_SCORES: dict[tuple[int, int], float] = {
#     (6016, 3384): 100.0,  # 5.4K
#     (5120, 2700): 85.0,  # 5.1K
#     (3840, 2160): 70.0,  # 4K
#     (2704, 1520): 55.0,  # 2.7K
#     (1920, 1080): 40.0,  # 1080p
#     (1280, 720): 30.0,  # 720p
#     (854, 480): 20.0,  # 480p
# }

FOV_MAX: float = 120.0  # degrees – ceiling used to normalise FOV to 0–100

DISTANCE_MAX: float = 1000.0  # meters – used to normalise distance to 0–100


# Compute the distance using the well known fact that the earth is flat
def fast_distance_meters(lat1, lon1, lat2, lon2):
    # Earth's radius in meters
    r = 6371000

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam1 = math.radians(lon1)
    lam2 = math.radians(lon2)

    x = (lam2 - lam1) * math.cos((phi1 + phi2) / 2)
    y = phi2 - phi1

    return r * math.sqrt(x**2 + y**2)


# Mission-specific hardware scoring profiles
# Each entry defines which capabilities fields matter for that mission and how
# much each one is worth.
# Sub-weights within a profile must sum to 1.0.
@dataclass
class HardwareProfile:
    """Declares which hardware dimensions matter for a given mission type."""

    # Weight for camera resolution quality  (0 → irrelevant)
    resolution: float = 0.0
    # Weight for horizontal FOV quality     (0 → irrelevant)
    fov: float = 0.0
    # Weight for having a speaker           (0 → irrelevant)
    speaker: float = 0.0
    # Weight for having lights              (0 → irrelevant)
    lights: float = 0.0
    # Weight for having no hardware. Useful for Go-to missions
    no_hardware: float = 0.0

    def __post_init__(self):
        total = (
            self.resolution + self.fov + self.speaker + self.lights + self.no_hardware
        )
        if not (0.99 < total < 1.01):  # allow small float drift
            raise ValueError(
                f"HardwareProfile weights must sum to 1.0, got {total:.3f}"
            )


# GotoAndAudio: speaker is primary; camera quality acts as a tiebreaker
PROFILE_GOTO_AND_AUDIO = HardwareProfile(
    resolution=0.5,
    fov=0.5,
)

# GotoAndBlink: lights are primary; camera quality acts as a tiebreaker
PROFILE_GOTO_AND_BLINK = HardwareProfile(
    resolution=0.5,
    fov=0.5,
)

# GotoAndIlluminate: spotlight/lighting subsystem is required
PROFILE_GOTO_AND_ILLUMINATE = HardwareProfile(
    resolution=0.5,
    fov=0.5,
)

# GotoAndSurveil: camera quality is primary
PROFILE_GOTO_AND_SURVEIL = HardwareProfile(
    resolution=0.60,
    fov=0.40,
)

# GotoOnly: here there is no special hardware needed
PROFILE_GOTO_ONLY = HardwareProfile(no_hardware=1.0)

# Registry: maps mission class → its hardware profile
HARDWARE_PROFILES: dict[Type[Mission], HardwareProfile] = {
    GotoAndAudio: PROFILE_GOTO_AND_AUDIO,
    GotoAndBlink: PROFILE_GOTO_AND_BLINK,
    GotoAndIlluminate: PROFILE_GOTO_AND_ILLUMINATE,
    GotoAndSurveil: PROFILE_GOTO_AND_SURVEIL,
    GotoOnly: PROFILE_GOTO_ONLY,
}


# Data helpers
@dataclass
class DroneCandidate:
    """Bundles everything known about one connected drone at selection time"""

    drone_id: str
    capabilities: json_schemas.Capabilities
    telemetry: json_schemas.Telemetry
    hardware_score: float  # 0 – 100  (computed)
    total_score: float = 0.0


# Step 3 scoring
def compute_hardware_score(
    drone_id: str, capabilities: json_schemas.Capabilities, mission_type: Type[Mission]
) -> float:
    """
    Returns a 0-100 score reflecting how well a drone's hardware suits a
    *specific* mission type.  Only the capabilities that the mission actually
    uses contribute to the score - everything else is ignored.
    So:
    • GotoAndAudio  → only speaker quality counts
    • GotoAndBlink  → only lights quality counts
    • GotoOnly      → camera resolution + FOV provide a tie-breaker
    """
    profile = HARDWARE_PROFILES.get(mission_type)
    if profile is None:
        # Unknown mission type – fall back to a neutral score so selection
        # still works; log a warning so the profile can be added later.
        print(
            "No HardwareProfile defined for %s!!!",
            mission_type.__name__,
        )
        return 0.0

    resolution_value = 0.0
    fov_score = 0.0

    if capabilities.camera:
        resolution_value = resolution_score(
            capabilities.camera.resolution_width,
            capabilities.camera.resolution_height,
        )

        fov_score = min(capabilities.camera.diagonal_fov / FOV_MAX, 1.0) * 100.0

    speaker_score = 100.0 if capabilities.speaker else 0.0
    lights_score = 100.0 if (capabilities.spotlight or capabilities.led) else 0.0

    total = (
        resolution_value * profile.resolution
        + fov_score * profile.fov
        + speaker_score * profile.speaker
        + lights_score * profile.lights
    )

    # This solution is a bit scuffed, but it seems to work fine for our limited use cases
    total_unweighted = (
        resolution_value + fov_score + speaker_score + lights_score
    ) / 4.0

    # Account for no_hardware factor
    total_no_hardware = (-total_unweighted) + 100

    # lerp between the total score and the reducing score. if no_hardware is 1, then total_no_hardware is used
    _total_with_reduction = total + profile.no_hardware * (total_no_hardware - total)

    # print(
    #     f"[{drone_id}] total score w/o reduction {total}, {_total_with_reduction}, {total_unweighted}"
    # )

    return round(total, 2)


def compute_total_score(
    telemetry: json_schemas.Telemetry,
    hardware_score: float,
    coordinates: json_schemas.GoToParams,
) -> float:
    """Weighted combination of battery level and hardware quality (both 0-100)."""

    travel_distance = fast_distance_meters(
        telemetry.lat, telemetry.lon, coordinates.lat, coordinates.lon
    )

    # Convert to factor between 0-100. Distances beyond DISTANCE_MAX should have the same score
    travel_score: float = min(travel_distance / DISTANCE_MAX, 1.0) * 100.0

    return round(
        WEIGHT_BATTERY * telemetry.battery_percent
        - WEIGHT_PROXIMITY
        * travel_score  # Minus since travel_score is proportional to distance
        + WEIGHT_HARDWARE * hardware_score,
        2,
    )


try:
    r = redis.Redis(
        host=os.environ.get("REDIS_URL"),
        port=os.environ.get("REDIS_PORT"),
        db=0,
        decode_responses=True,
    )
    r.ping()
    print("[Drone Selector] Successfully connected to Redis")
except redis.exceptions.ConnectionError as e:
    print(f"[Drone Selector] Error connecting to Redis: {e}")
    exit()


# Main selector
class DroneSelector:
    """
    Selects the best available drone for a requested mission type.

    Usage
    -----
    >>> selector = DroneSelector()
    >>> mission = selector.select(GotoAndAudio, coordinates={"lat": 57.7, "lng": 11.9})
    >>> if mission:
    ...     mission_registry.store(mission)
    """

    # Minimum battery level required to even be considered
    MIN_BATTERY_THRESHOLD: float = (
        30.0  # At 20 the drone will go home, to send it on a mission 30 is required
    )

    def __init__(self):
        pass

    # ── Step 1: Connected ─────────────────────────────────────────────

    def _get_connected_drones(self) -> list[DroneCandidate]:
        """
        Scans Redis for drones that have both a live telemetry entry and a
        capabilities entry.  A telemetry key without a capabilities key (or
        vice-versa) means the drone has not fully registered – it is skipped.
        """
        telemetry_keys: dict[str, str] = {}  # drone_id → redis key
        capabilities_keys: dict[str, str] = {}

        for key in r.scan_iter(match="telemetry_drone*"):
            drone_id: str = key.replace("telemetry_drone", "")
            telemetry_keys[drone_id] = key

        for key in r.scan_iter(match="capabilities_drone*"):
            drone_id: str = key.replace("capabilities_drone", "")
            capabilities_keys[drone_id] = key

        # Only drones present in both sets are considered connected
        connected_ids = set(telemetry_keys) & set(capabilities_keys)

        candidates: list[DroneCandidate] = []
        for drone_id in connected_ids:
            try:
                telemetry_json = r.get(telemetry_keys[drone_id])
                capabilities_json = r.get(capabilities_keys[drone_id])

                if telemetry_json is None or capabilities_json is None:
                    logger.warning(
                        "Drone %s: Redis data expired mid-query, skipping.", drone_id
                    )
                    continue

                telemetry = json_schemas.parse_telemetry(telemetry_json)
                capabilities = json_schemas.parse_capabilities(capabilities_json)

                if telemetry.battery_percent < self.MIN_BATTERY_THRESHOLD:
                    logger.warning(
                        "Drone %s skipped: battery %.1f%% below minimum threshold (%.1f%%).",
                        drone_id,
                        telemetry.battery_percent,
                        self.MIN_BATTERY_THRESHOLD,
                    )
                    continue

                # hardware_score is left at 0.0 here – it is computed in _rank
                # once the mission type (and therefore the scoring profile) is known.
                candidates.append(
                    DroneCandidate(
                        drone_id=drone_id,
                        capabilities=capabilities,
                        telemetry=telemetry,
                        hardware_score=0.0,
                    )
                )
                logger.debug(
                    "Drone %s connected - battery=%.1f",
                    drone_id,
                    telemetry.battery_percent,
                )

            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Drone %s: failed to parse Redis data (%s), skipping.",
                    drone_id,
                    exc,
                )

        logger.info(
            "Step 1 - %d drone(s) connected and above battery threshold.",
            len(candidates),
        )
        return candidates

    # ── Step 2: Capability filter ─────────────────────────────────────────

    def _filter_capable(
        self,
        candidates: list[DroneCandidate],
        mission_type: Type[Mission],
        coordinates: json_schemas.GoToParams,
        params: Optional[dict] = None,
    ) -> list[DroneCandidate]:
        """
        Keeps only the drones whose hardware can execute the requested mission type.
        Checks directly against the mission type rather than going through MissionFactory,
        which would instantiate every mission type just to filter for one.
        """
        capable: list[DroneCandidate] = []
        params = params or {}
        for candidate in candidates:
            if mission_type(
                candidate.drone_id, candidate.capabilities, coordinates, **params
            ).can_execute():
                capable.append(candidate)
            else:
                logger.info(
                    "Drone %s skipped: hardware does not support %s.",
                    candidate.drone_id,
                    mission_type.__name__,
                )

        logger.info(
            "Step 2 - %d drone(s) capable of executing %s.",
            len(capable),
            mission_type.__name__,
        )
        return capable

    # ── Step 3: Availability ───────────────────────────────────────────────────

    def _filter_available(
        self, candidates: list[DroneCandidate]
    ) -> list[DroneCandidate]:
        available: list[DroneCandidate] = []

        for candidate in candidates:
            if not MissionRegistry.is_drone_dispatched_or_waiting(candidate.drone_id):
                available.append(candidate)
            else:
                logger.info(
                    "Drone %s capable, but is already dispatched", candidate.drone_id
                )

        return available

    # ── Step 4: Ranking ───────────────────────────────────────────────────

    def _rank(
        self,
        candidates: list[DroneCandidate],
        mission_type: Type[Mission],
        coordinates: json_schemas.GoToParams,
    ) -> list[DroneCandidate]:
        """
        Scores each candidate using only the hardware dimensions that matter
        for the given mission type, then returns the list sorted best-first.

        Score = WEIGHT_BATTERY * battery%  +  WEIGHT_HARDWARE * hardware_score
        where hardware_score is computed against the mission's HardwareProfile.
        """
        for c in candidates:
            c.hardware_score = compute_hardware_score(
                c.drone_id, c.capabilities, mission_type
            )
            c.total_score = compute_total_score(
                c.telemetry, c.hardware_score, coordinates
            )
            logger.debug(
                "Drone %s - battery=%.1f (*%.2f) + hw=%.1f (*%.2f) → score=%.2f",
                c.drone_id,
                c.telemetry.battery_percent,
                WEIGHT_BATTERY,
                c.hardware_score,
                WEIGHT_HARDWARE,
                c.total_score,
            )

        ranked = sorted(candidates, key=lambda c: c.total_score, reverse=True)

        if ranked:
            best = ranked[0]
            logger.info(
                "Step 3 - Best drone: %s (score=%.2f, battery=%.1f%%, hw_score=%.1f)",
                best.drone_id,
                best.total_score,
                best.telemetry.battery_percent,
                best.hardware_score,
            )

        return ranked

    # Public entry point

    def select(
        self,
        mission_type: Type[Mission],
        coordinates: json_schemas.GoToParams,
        params: Optional[dict] = None,
    ) -> Optional[Mission]:
        """
        Runs the three-step selection and returns a ready-to-store Mission object
        assigned to the best drone, or None if no suitable drone is found.

        Parameters
        ----------
        mission_type : subclass of Mission
            The kind of mission to execute, e.g. GotoAndAudio, GotoAndBlink, GotoOnly.
        coordinates
            Target location, e.g. {"lat": 57.705, "lon": 11.938, "alt": 10, (optional) "heading": 0}.
        params
            Should contain following data:
                - For GotoAndAudio:
                params = {
                    "audio_type": "alert",      # optional if audio_file provided
                    "audio_file": None,         # optional if audio_type provided
                    "duration_seconds": 30,     # optional
                    "volume": 1.0,              # optional
                }

                - For GotoAndSurveil:
                params = {
                    "duration_seconds": 30,     # optional
                    "camera_pitch": -90,        # optional
                    "camera_yaw": 0,            # optional
                }

                - For GotoAndBlink:
                params = { "duration_seconds": 30 }    # optional

                - For GotoAndIlluminate:
                params = { "duration_seconds": 30 }    # optional

                - For GotoOnly: None

        Returns
        -------
        Mission | None
        """
        logger.info(
            "=== DroneSelector: selecting drone for %s ===", mission_type.__name__
        )

        params = params or {}

        # Step 1 – availability
        candidates = self._get_connected_drones()
        if not candidates:
            logger.warning("No connected drones available.")
            return None

        # Step 2 – capability
        capable = self._filter_capable(candidates, mission_type, coordinates, params)
        if not capable:
            logger.warning(
                "No drone with the required hardware for %s.", mission_type.__name__
            )
            return None

        # Step 3 - Filter out dispatched drones
        available = self._filter_available(capable)

        # Step 4 – ranking (hardware scored against this mission's profile)
        ranked = self._rank(available, mission_type, coordinates)
        if len(ranked) == 0:
            print(
                f"No available missions for connected and available drones. {len(candidates)} connected, {len(capable)} capable, {len(available)} available"
            )
            return None

        chosen = ranked[0]
        print(f"Ranked for {mission_type}: {[r.drone_id for r in ranked]}")

        # Instantiate and return the mission bound to the chosen drone
        mission = mission_type(
            chosen.drone_id, chosen.capabilities, coordinates, **params
        )
        logger.info(
            "Selected drone '%s' for mission type '%s'.",
            chosen.drone_id,
            mission_type.__name__,
        )
        return mission


# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────


def select_drone_for_mission(
    mission_type: Type[Mission],
    coordinates: json_schemas.GoToParams,
    params: Optional[dict] = None,
) -> Optional[Mission]:
    """
    Module-level shortcut so callers don't need to instantiate DroneSelector.

    Example
    -------
    >>> from communication_software.missions_planning.drone_selector import (
    ...     select_drone_for_mission
    ... )
    >>> from communication_software.missions_planning.missions import GotoAndAudio
    >>>
    >>> mission = select_drone_for_mission(
    ...     mission_type=GotoAndAudio,
    ...     coordinates=json_schemas.GoToParams(lat=57.705841, lon=11.938096, alt=20, heading=0),
    ... )
    >>> if mission:
    ...     mission_registry.store(mission)
    ... else:
    ...     print("No suitable drone found.")
    """
    selector = DroneSelector()
    return selector.select(mission_type, coordinates, params)
