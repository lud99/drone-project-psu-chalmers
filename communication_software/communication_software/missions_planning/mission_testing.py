import redis
import os

from pydantic import BaseModel

import communication_software.missions_planning.missions as missions

import communication_software.common.json_schemas as json_schemas

from communication_software.missions_planning.drone_selector import (
    select_drone_for_mission,
)


try:
    r = redis.Redis(
        host=os.environ.get("REDIS_URL"),
        port=os.environ.get("REDIS_PORT"),
        db=0,
        decode_responses=True,
    )
    r.ping()
    r.flushdb()  # Removes all stuff, as stopping docker containers is not enough to clear it
    print("Successfully connected to Redis (Drone Communication Server)!")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis (Drone Communication Server): {e}")
    exit()


# Central Gothenburg coordinates
coordinates = {
    "nordstan": (57.708965, 11.969438),
    "ullevi": (57.706142, 11.980153),
    "liseberg": (57.696162, 11.991556),
    "fyrfästet": (57.693093, 11.984072),
}


# Assuming a top-level Drone model to wrap your schemas
class Drone(BaseModel):
    id: str
    model: str
    capabilities: json_schemas.Capabilities
    telemetry: json_schemas.Telemetry


MOCK_DRONES: list[Drone] = [
    Drone(
        id="dji-01",
        model="DJI Mavic 2 Enterprise",
        capabilities=json_schemas.Capabilities(
            camera=json_schemas.CameraCapabilities(
                aspect_ratio=1.777,
                horizontal_fov=84.0,
                resolution_width=1920,
                resolution_height=1080,
            ),
            led=None,
            spotlight=False,
            speaker=json_schemas.SpeakerCapabilities(
                audio_files=["leave_track", "go_home"]
            ),
        ),
        telemetry=json_schemas.Telemetry(
            lat=coordinates["nordstan"][0],
            lon=coordinates["nordstan"][1],
            alt=20.5,
            heading=90,
            speed=5.5,
            battery_percent=88,
        ),
    ),
    Drone(
        id="dji-02",
        model="DJI Mavic 2 Enterprise",
        capabilities=json_schemas.Capabilities(
            camera=json_schemas.CameraCapabilities(
                aspect_ratio=1.777,
                horizontal_fov=84.0,
                resolution_width=1920,
                resolution_height=1080,
            ),
            led=None,
            spotlight=False,
            speaker=json_schemas.SpeakerCapabilities(audio_files=["horn", "stay"]),
        ),
        telemetry=json_schemas.Telemetry(
            lat=coordinates["ullevi"][0],
            lon=coordinates["ullevi"][1],
            alt=20.0,
            heading=270,
            speed=2.1,
            battery_percent=42,
        ),
    ),
    Drone(
        id="mavlink-01",
        model="Holybro Pixhawk 6X",
        capabilities=json_schemas.Capabilities(
            camera=None,
            led=json_schemas.LEDCapabilities(types=["navigation", "strobe"]),
            spotlight=False,
            speaker=None,
        ),
        telemetry=json_schemas.Telemetry(
            lat=coordinates["liseberg"][0],
            lon=coordinates["liseberg"][1],
            alt=50.0,
            heading=180,
            speed=12.0,
            battery_percent=95,
        ),
    ),
]


def update_redis():

    for drone in MOCK_DRONES:
        # Store static/semi-static capabilities
        r.set(f"capabilities_drone{drone.id}", drone.capabilities.model_dump_json())

        # Store high-frequency telemetry with a 60s TTL
        r.set(f"telemetry_drone{drone.id}", drone.telemetry.model_dump_json(), ex=60)


def run_tests():
    update_redis()

    selected_mission = select_drone_for_mission(
        missions.GotoOnly,
        json_schemas.GoToParams(
            lat=coordinates["fyrfästet"][0], lon=coordinates["fyrfästet"][1], alt=80
        ),
    )
    print(f"1. Got mission {selected_mission}")
    assert selected_mission.drone_id == "mavlink-01"

    selected_mission = select_drone_for_mission(
        missions.GotoAndAudio,
        json_schemas.GoToParams(
            lat=coordinates["fyrfästet"][0], lon=coordinates["fyrfästet"][1], alt=80
        ),
        dict({"audio_type": "alert"}),
    )

    print(f"2.Got mission {selected_mission}")
    assert selected_mission.drone_id == "dji-02"

    selected_mission = select_drone_for_mission(
        missions.GotoAndAudio,
        json_schemas.GoToParams(
            lat=coordinates["fyrfästet"][0], lon=coordinates["fyrfästet"][1], alt=80
        ),
        dict({"audio_file": "horn"}),
    )

    print(f"3.Got mission {selected_mission}")
    assert selected_mission.drone_id == "dji-02"

    selected_mission = select_drone_for_mission(
        missions.GotoAndAudio,
        json_schemas.GoToParams(
            lat=coordinates["fyrfästet"][0], lon=coordinates["fyrfästet"][1], alt=80
        ),
        dict(
            {
                "audio_type": "intruder_instructions",
                "duration_seconds": 30,
                "volume": 0.8,
            }
        ),
    )

    print(f"4.Got mission {selected_mission}")
    assert selected_mission.drone_id == "dji-01"
