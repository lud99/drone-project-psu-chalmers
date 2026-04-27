import redis
import os
import copy
from pydantic import BaseModel
import communication_software.missions_planning.missions as missions
import communication_software.common.json_schemas as json_schemas
from communication_software.missions_planning.drone_selector import (
    select_drone_for_mission,
)

# --- Setup & Helpers ---

try:
    r = redis.Redis(
        host=os.environ.get("REDIS_URL", "localhost"),
        port=os.environ.get("REDIS_PORT", 6379),
        db=0,
        decode_responses=True,
    )
    r.ping()
    r.flushdb()
    print("✅ Connected to Redis")
except redis.exceptions.ConnectionError as e:
    print(f"❌ Redis Connection Error: {e}")
    exit()

# Central Gothenburg coordinates
LOCS = {
    "nordstan": (57.708965, 11.969438),
    "ullevi": (57.706142, 11.980153),
    "liseberg": (57.696162, 11.991556),
    "fyrfästet": (57.693093, 11.984072),  # Common target for tests
}


class Drone(BaseModel):
    id: str
    model: str
    capabilities: json_schemas.Capabilities
    telemetry: json_schemas.Telemetry


# Initializing base data
BASE_DRONES: list[Drone] = [
    Drone(
        id="dji-01",
        model="DJI Mavic 2 Enterprise",
        capabilities=json_schemas.Capabilities(
            camera=json_schemas.CameraCapabilities(
                aspect_ratio=1.77,
                diagonal_fov=84.0,
                resolution_width=1920,
                resolution_height=1080,
            ),
            speaker=json_schemas.SpeakerCapabilities(audio_files=["go_home"]),
            spotlight=True,
            led=None,
        ),
        telemetry=json_schemas.Telemetry(
            lat=LOCS["nordstan"][0],
            lon=LOCS["nordstan"][1],
            alt=20,
            heading=0,
            speed=0,
            battery_percent=100,
        ),
    ),
    Drone(
        id="dji-02",
        model="DJI Mavic 2 Enterprise",
        capabilities=json_schemas.Capabilities(
            camera=json_schemas.CameraCapabilities(
                aspect_ratio=1.77,
                diagonal_fov=84.0,
                resolution_width=1920,
                resolution_height=1080,
            ),
            led=None,
            spotlight=False,
            speaker=json_schemas.SpeakerCapabilities(
                audio_files=["stay", "horn", "restart_transponder", "go_home"]
            ),
        ),
        telemetry=json_schemas.Telemetry(
            lat=LOCS["ullevi"][0],
            lon=LOCS["ullevi"][1],
            alt=20,
            heading=0,
            speed=0,
            battery_percent=100,
        ),
    ),
    Drone(
        id="mavlink-01",
        model="Holybro Pixhawk 6X",
        capabilities=json_schemas.Capabilities(
            led=json_schemas.LEDCapabilities(types=["beacon", "strobe"]),
            spotlight=False,
            camera=None,
            speaker=None,
        ),
        telemetry=json_schemas.Telemetry(
            lat=LOCS["liseberg"][0],
            lon=LOCS["liseberg"][1],
            alt=50,
            heading=0,
            speed=0,
            battery_percent=100,
        ),
    ),
]


def update_redis(drones: list[Drone]):
    """Helper to refresh state in Redis."""
    r.flushdb()
    for drone in drones:
        r.set(f"capabilities_drone{drone.id}", drone.capabilities.model_dump_json())
        r.set(f"telemetry_drone{drone.id}", drone.telemetry.model_dump_json(), ex=60)


# --- Test Cases ---


def test_proximity_ranking():
    """Drones with identical specs: the closer one (Ullevi vs Nordstan) should be picked."""
    print("\n--- Running: Proximity Test ---")
    drones = copy.deepcopy(BASE_DRONES)
    # Target is 'fyrfästet'. Ullevi is closer than Nordstan.
    update_redis(drones)

    selected = select_drone_for_mission(
        missions.GotoOnly,
        json_schemas.GoToParams(
            lat=LOCS["fyrfästet"][0], lon=LOCS["fyrfästet"][1], alt=50
        ),
    )
    # Liseberg (mavlink-01) is actually the absolute closest to Fyrfästet.
    assert selected.drone_id == "mavlink-01", (
        f"Expected mavlink-01 (closest), got {selected.drone_id}"
    )
    print("Success: Closest drone selected.")


def test_battery_threshold():
    """Drones with < 30% battery should be ignored entirely."""
    print("\n--- Running: Battery Threshold Test ---")
    drones = copy.deepcopy(BASE_DRONES)
    # Make the closest drone have 10% battery. It should be skipped.
    for d in drones:
        if d.id == "mavlink-01":
            d.telemetry.battery_percent = 10

    update_redis(drones)
    selected = select_drone_for_mission(
        missions.GotoOnly,
        json_schemas.GoToParams(
            lat=LOCS["fyrfästet"][0], lon=LOCS["fyrfästet"][1], alt=50
        ),
    )
    assert selected.drone_id != "mavlink-01", (
        "Low battery drone was incorrectly selected."
    )
    print(f"Success: Low battery drone ignored. Selected: {selected.drone_id}")


def test_surveil_camera_quality():
    """GotoAndSurveil should prefer the drone with the higher resolution camera."""
    print("\n--- Running: Camera Quality Test ---")
    drones = copy.deepcopy(BASE_DRONES)
    # Give dji-01 a 4K camera, dji-02 stays 1080p. Position them at the same spot.
    for d in drones:
        d.telemetry.lat, d.telemetry.lon = LOCS["nordstan"]
        if d.id == "dji-01":
            d.capabilities.camera.resolution_width = 3840
            d.capabilities.camera.resolution_height = 2160

    update_redis(drones)
    selected = select_drone_for_mission(
        missions.GotoAndSurveil,
        json_schemas.GoToParams(
            lat=LOCS["fyrfästet"][0], lon=LOCS["fyrfästet"][1], alt=50
        ),
    )
    assert selected.drone_id == "dji-01", (
        f"Expected 4K drone (dji-01), got {selected.drone_id}"
    )
    print("Success: Higher resolution camera preferred for surveillance.")


def test_all_mission_types():
    """Verify that every mission type correctly filters for required hardware."""
    print("\n--- Running: All Mission Types Filter Test ---")
    drones = copy.deepcopy(BASE_DRONES)
    update_redis(drones)
    target = json_schemas.GoToParams(
        lat=LOCS["fyrfästet"][0], lon=LOCS["fyrfästet"][1], alt=30
    )

    # 1. Audio type
    m_audio = select_drone_for_mission(
        missions.GotoAndAudio, target, {"audio_type": "alert"}
    )
    assert m_audio is not None and m_audio.capabilities.speaker is not None
    assert m_audio.drone_id == "dji-02"

    # 1.5. Audio file
    m_audio = select_drone_for_mission(
        missions.GotoAndAudio, target, {"audio_file": "go_home"}
    )
    assert m_audio is not None
    assert m_audio.drone_id == "dji-01"

    # 1.6. Audio file
    # for drones[0].capabilities.speaker.audio_files.append("horn")  # Give dji-01 the "horn" file
    m_audio = select_drone_for_mission(missions.GotoAndAudio, target)
    print(m_audio)

    # 2. Blink (Requires LED 'beacon')
    m_blink = select_drone_for_mission(missions.GotoAndBlink, target)
    assert m_blink is not None
    assert m_blink.drone_id == "mavlink-01"  # Only one with beacon

    # 3. Illuminate (Requires spotlight)
    m_light = select_drone_for_mission(missions.GotoAndIlluminate, target)
    assert m_light is not None
    assert m_light.drone_id == "dji-01"  # Only one with spotlight

    # 4. Surveil (Requires camera)
    m_cam = select_drone_for_mission(missions.GotoAndSurveil, target)
    assert m_cam is not None
    assert m_cam.capabilities.camera is not None

    # 5. Goto Only (Anyone capable)
    m_goto = select_drone_for_mission(missions.GotoOnly, target)
    assert m_goto is not None
    assert m_goto.drone_id == "mavlink-01"

    print("Success: All mission types filtered correctly.")


def run_tests():
    test_proximity_ranking()
    test_battery_threshold()
    test_surveil_camera_quality()
    test_all_mission_types()
    print("\n✨ All tests passed!")
