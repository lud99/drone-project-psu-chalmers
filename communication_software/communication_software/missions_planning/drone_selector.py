"""
Automatically selects the most suitable connected drone for a given mission type

Selection consists of three steps:
    1. Availability  – drone must be connected (live telemetry + capabilities in Redis)
    2. Capability    – drone hardware must satisfy the mission requirements
    3. Ranking       – among all eligible drones, pick the highest-scoring one based on
                       a weighted combination of battery level and hardware quality
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional, Type

import redis
import redis.exceptions

from communication_software.missions_planning.drone_specs import DroneSpecs
from communication_software.missions_planning.missions import (
    Mission,
    GotoAndSound,
    GotoAndBlink,
    GotoOnly,
)

logger = logging.getLogger(__name__)

# total must be 1
WEIGHT_BATTERY: float = 0.55  # Battery
WEIGHT_HARDWARE: float = 0.45  # Mission specific hardware quality


# shared hardware lookup tables
RESOLUTION_SCORES: dict[str, float] = {
    "4k": 100.0,
    "2.7k": 85.0,
    "1080p": 70.0,
    "720p": 45.0,
    "480p": 20.0,
}
FOV_MAX: float = 120.0  # degrees – ceiling used to normalise FOV to 0–100


# Mission-specific hardware scoring profiles
# Each entry defines which DroneSpecs fields matter for that mission and how
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

    def __post_init__(self):
        total = self.resolution + self.fov + self.speaker + self.lights
        if not (0.99 < total < 1.01):  # allow small float drift
            raise ValueError(
                f"HardwareProfile weights must sum to 1.0, got {total:.3f}"
            )


# GotoAndSound: speaker is primary; camera quality acts as a tiebreaker
PROFILE_GOTO_AND_SOUND = HardwareProfile(
    speaker=0.70,
    resolution=0.20,
    fov=0.10,
)

# GotoAndBlink: lights are primary; camera quality acts as a tiebreaker
PROFILE_GOTO_AND_BLINK = HardwareProfile(
    lights=0.70,
    resolution=0.20,
    fov=0.10,
)

# GotoOnly: here there is no special hardware needed, but prefer a better camera if present
PROFILE_GOTO_ONLY = HardwareProfile(
    resolution=0.50,
    fov=0.50,
)

# Registry: maps mission class → its hardware profile
HARDWARE_PROFILES: dict[Type[Mission], HardwareProfile] = {
    GotoAndSound: PROFILE_GOTO_AND_SOUND,
    GotoAndBlink: PROFILE_GOTO_AND_BLINK,
    GotoOnly: PROFILE_GOTO_ONLY,
}


# Data helpers
@dataclass
class DroneCandidate:
    """Bundles everything known about one connected drone at selection time"""

    drone_id: str
    specs: DroneSpecs
    battery: float  # 0 – 100
    hardware_score: float  # 0 – 100  (computed)
    total_score: float = 0.0


def _parse_telemetry(telemetry_json: str) -> dict:
    """Returns the telemetry dict from a JSON string stored in Redis"""
    return json.loads(telemetry_json)


def _parse_capabilities(capabilities_json: str) -> DroneSpecs:
    """
    Reconstructs a DroneSpecs from the JSON stored in Redis.

    Expected Redis format (mirrors the Pydantic model used in json_schemas):
    {
        "id":             "dji-01",
        "model":          "DJI Mavic 2 Enterprise",
        "speaker":        true,
        "lights":         false,
        "camera":         true,
        "aspect_ratio":   "16:9",
        "horizontal_fov": 84.0,
        "resolution":     "1080p"
    }
    """
    data = json.loads(capabilities_json)
    return DroneSpecs(
        id=data.get("id", "unknown"),
        model=data.get("model", "unknown"),
        speaker=bool(data.get("speaker", False)),
        lights=bool(data.get("lights", False)),
        camera=data.get("camera") is not None,  # non-null camera object → has camera
        aspect_ratio=data.get("aspect_ratio"),
        horizontal_fov=data.get("horizontal_fov"),
        resolution=data.get("resolution"),
    )


# Step 3 scoring
def compute_hardware_score(specs: DroneSpecs, mission_type: Type[Mission]) -> float:
    """
    Returns a 0–100 score reflecting how well a drone's hardware suits a
    *specific* mission type.  Only the capabilities that the mission actually
    uses contribute to the score – everything else is ignored.
    So:
    • GotoAndSound  → only speaker quality counts
    • GotoAndBlink  → only lights quality counts
    • GotoOnly      → camera resolution + FOV provide a tie-breaker
    """
    profile = HARDWARE_PROFILES.get(mission_type)
    if profile is None:
        # Unknown mission type – fall back to a neutral score so selection
        # still works; log a warning so the profile can be added later.
        logger.warning(
            "No HardwareProfile defined for %s – hardware score defaulting to 50.0.",
            mission_type.__name__,
        )
        return 50.0

    resolution_score = 0.0
    if specs.resolution:
        resolution_score = RESOLUTION_SCORES.get(specs.resolution.lower(), 50.0)

    fov_score = 0.0
    if specs.horizontal_fov is not None:
        fov_score = min(specs.horizontal_fov / FOV_MAX, 1.0) * 100.0

    speaker_score = 100.0 if specs.speaker else 0.0
    lights_score = 100.0 if specs.lights else 0.0

    total = (
        resolution_score * profile.resolution
        + fov_score * profile.fov
        + speaker_score * profile.speaker
        + lights_score * profile.lights
    )
    return round(total, 2)


def compute_total_score(battery: float, hardware_score: float) -> float:
    """Weighted combination of battery level and hardware quality (both 0–100)."""
    return round(WEIGHT_BATTERY * battery + WEIGHT_HARDWARE * hardware_score, 2)


# Main selector
class DroneSelector:
    """
    Selects the best available drone for a requested mission type.

    Usage
    -----
    >>> selector = DroneSelector()
    >>> mission = selector.select(GotoAndSound, coordinates={"lat": 57.7, "lng": 11.9})
    >>> if mission:
    ...     mission_registry.store(mission)
    """

    # Minimum battery level required to even be considered
    MIN_BATTERY_THRESHOLD: float = 20.0

    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=True
        )

    # ── Step 1: Availability ─────────────────────────────────────────────

    def _get_connected_drones(self) -> list[DroneCandidate]:
        """
        Scans Redis for drones that have both a live telemetry entry and a
        capabilities entry.  A telemetry key without a capabilities key (or
        vice-versa) means the drone has not fully registered – it is skipped.
        """
        telemetry_keys: dict[str, str] = {}  # drone_id → redis key
        capabilities_keys: dict[str, str] = {}

        for key in self._redis.scan_iter(match="telemetry_drone*"):
            drone_id = key.replace("telemetry_drone", "")
            telemetry_keys[drone_id] = key

        for key in self._redis.scan_iter(match="capabilities_drone*"):
            drone_id = key.replace("capabilities_drone", "")
            capabilities_keys[drone_id] = key

        # Only drones present in both sets are considered connected
        connected_ids = set(telemetry_keys) & set(capabilities_keys)

        candidates: list[DroneCandidate] = []
        for drone_id in connected_ids:
            try:
                telemetry_json = self._redis.get(telemetry_keys[drone_id])
                capabilities_json = self._redis.get(capabilities_keys[drone_id])

                if telemetry_json is None or capabilities_json is None:
                    logger.warning(
                        "Drone %s: Redis data expired mid-query, skipping.", drone_id
                    )
                    continue

                telemetry = _parse_telemetry(telemetry_json)
                specs = _parse_capabilities(capabilities_json)

                battery = float(telemetry.get("battery", 0.0))
                if battery < self.MIN_BATTERY_THRESHOLD:
                    logger.info(
                        "Drone %s skipped: battery %.1f%% below minimum threshold (%.1f%%).",
                        drone_id,
                        battery,
                        self.MIN_BATTERY_THRESHOLD,
                    )
                    continue

                # hardware_score is left at 0.0 here – it is computed in _rank
                # once the mission type (and therefore the scoring profile) is known.
                candidates.append(
                    DroneCandidate(
                        drone_id=drone_id,
                        specs=specs,
                        battery=battery,
                        hardware_score=0.0,
                    )
                )
                logger.debug("Drone %s connected – battery=%.1f", drone_id, battery)

            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Drone %s: failed to parse Redis data (%s), skipping.",
                    drone_id,
                    exc,
                )

        logger.info(
            "Step 1 – %d drone(s) connected and above battery threshold.",
            len(candidates),
        )
        return candidates

    # ── Step 2: Capability filter ─────────────────────────────────────────

    def _filter_capable(
        self,
        candidates: list[DroneCandidate],
        mission_type: Type[Mission],
        coordinates: dict,
    ) -> list[DroneCandidate]:
        """
        Keeps only the drones whose hardware can execute the requested mission type.
        Checks directly against the mission type rather than going through MissionFactory,
        which would instantiate every mission type just to filter for one.
        """
        capable: list[DroneCandidate] = []
        for candidate in candidates:
            if mission_type(candidate.specs, coordinates).can_execute():
                capable.append(candidate)
            else:
                logger.info(
                    "Drone %s skipped: hardware does not support %s.",
                    candidate.drone_id,
                    mission_type.__name__,
                )

        logger.info(
            "Step 2 – %d drone(s) capable of executing %s.",
            len(capable),
            mission_type.__name__,
        )
        return capable

    # ── Step 3: Ranking ───────────────────────────────────────────────────

    def _rank(
        self, candidates: list[DroneCandidate], mission_type: Type[Mission]
    ) -> list[DroneCandidate]:
        """
        Scores each candidate using only the hardware dimensions that matter
        for the given mission type, then returns the list sorted best-first.

        Score = WEIGHT_BATTERY × battery%  +  WEIGHT_HARDWARE × hardware_score
        where hardware_score is computed against the mission's HardwareProfile.
        """
        for c in candidates:
            c.hardware_score = compute_hardware_score(c.specs, mission_type)
            c.total_score = compute_total_score(c.battery, c.hardware_score)
            logger.debug(
                "Drone %s – battery=%.1f (×%.2f) + hw=%.1f (×%.2f) → score=%.2f",
                c.drone_id,
                c.battery,
                WEIGHT_BATTERY,
                c.hardware_score,
                WEIGHT_HARDWARE,
                c.total_score,
            )

        ranked = sorted(candidates, key=lambda c: c.total_score, reverse=True)

        if ranked:
            best = ranked[0]
            logger.info(
                "Step 3 – Best drone: %s (score=%.2f, battery=%.1f%%, hw_score=%.1f)",
                best.drone_id,
                best.total_score,
                best.battery,
                best.hardware_score,
            )

        return ranked

    # Public entry point

    def select(
        self,
        mission_type: Type[Mission],
        coordinates: dict,
    ) -> Optional[Mission]:
        """
        Runs the three-step selection and returns a ready-to-store Mission object
        assigned to the best drone, or None if no suitable drone is found.

        Parameters
        ----------
        mission_type : subclass of Mission
            The kind of mission to execute, e.g. GotoAndSound, GotoAndBlink, GotoOnly.
        coordinates : dict
            Target location, e.g. {"lat": 57.705, "lng": 11.938}.

        Returns
        -------
        Mission | None
        """
        logger.info(
            "=== DroneSelector: selecting drone for %s ===", mission_type.__name__
        )

        # Step 1 – availability
        candidates = self._get_connected_drones()
        if not candidates:
            logger.warning("No connected drones available.")
            return None

        # Step 2 – capability
        capable = self._filter_capable(candidates, mission_type, coordinates)
        if not capable:
            logger.warning(
                "No drone with the required hardware for %s.", mission_type.__name__
            )
            return None

        # Step 3 – ranking (hardware scored against this mission's profile)
        ranked = self._rank(capable, mission_type)
        chosen = ranked[0]

        # Instantiate and return the mission bound to the chosen drone
        mission = mission_type(chosen.specs, coordinates)
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
    coordinates: dict,
    redis_host: str = "redis",
    redis_port: int = 6379,
) -> Optional[Mission]:
    """
    Module-level shortcut so callers don't need to instantiate DroneSelector.

    Example
    -------
    >>> from communication_software.missions_planning.drone_selector import (
    ...     select_drone_for_mission
    ... )
    >>> from communication_software.missions_planning.missions import GotoAndSound
    >>>
    >>> mission = select_drone_for_mission(
    ...     mission_type=GotoAndSound,
    ...     coordinates={"lat": 57.705841, "lng": 11.938096},
    ... )
    >>> if mission:
    ...     mission_registry.store(mission)
    ... else:
    ...     print("No suitable drone found.")
    """
    selector = DroneSelector(redis_host=redis_host, redis_port=redis_port)
    return selector.select(mission_type, coordinates)
