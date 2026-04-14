"""
Manual integration test for AutoMissionSuggester.

Requires a Redis instance reachable at REDIS_HOST:REDIS_PORT
(defaults to localhost:6379).

Usage:
    python test_auto_mission_suggest.py [--redis-host HOST] [--redis-port PORT]

Once running, type one of the commands at the prompt:
    person  [lat] [lon]       -- inject a person detection
    car     [lat] [lon]       -- inject a car detection
    truck   [lat] [lon]       -- inject a truck detection
    bus     [lat] [lon]       -- inject a bus detection
    area    lat1,lon1 lat2,lon2 lat3,lon3 ...  -- inject a watch area (≥3 points)
    clear                     -- clear dedup state so the same object triggers again
    quit / exit               -- stop
"""

import argparse
import json
import sys
import threading
import os
from pathlib import Path

import redis

# Allow running from either:
# 1) repo root:     python communication_software/test_auto_mission_suggest.py
# 2) package dir:   cd communication_software; python test_auto_mission_suggest.py
SCRIPT_DIR = Path(__file__).resolve().parent
script_dir_str = str(SCRIPT_DIR)
if script_dir_str not in sys.path:
    sys.path.insert(0, script_dir_str)

import communication_software.common.json_schemas as json_schemas  # noqa: E402
from communication_software.missions_planning.auto_mission_suggest import (  # noqa: E402
    AutoMissionSuggester,
)

# ── Default test coordinates (Gothenburg area) ───────────────────────────────

DEFAULT_LAT = 57.7060
DEFAULT_LON = 11.9380
DEFAULT_ALT = 30.0

MOCK_DRONES = [
    {
        "id": "mock-drone-alpha",
        "lat": 57.7055,
        "lon": 11.9375,
        "alt": 50.0,
        "heading": 90,
        "speed": 0.0,
        "battery_percent": 95,
        "capabilities": json_schemas.Capabilities(
            camera=json_schemas.CameraCapabilities(
                aspect_ratio=1.77,
                horizontal_fov=84.0,
                resolution_height=1080,
                resolution_width=1920,
            ),
            led=json_schemas.LEDCapabilities(types=["front", "rear", "beacon"]),
            spotlight=True,
            speaker=json_schemas.SpeakerCapabilities(
                audio_files=["leave_track", "siren", "restart_transponder"]
            ),
        ),
    },
    {
        "id": "mock-drone-beta",
        "lat": 57.7065,
        "lon": 11.9385,
        "alt": 40.0,
        "heading": 270,
        "speed": 0.0,
        "battery_percent": 80,
        "capabilities": json_schemas.Capabilities(
            camera=None,
            led=None,
            spotlight=False,
            speaker=None,
        ),
    },
]


def seed_mock_drones(r: redis.Redis) -> None:
    """Write telemetry and capabilities for mock drones into Redis."""
    for drone in MOCK_DRONES:
        telemetry = json_schemas.Telemetry(
            lat=drone["lat"],
            lon=drone["lon"],
            alt=drone["alt"],
            heading=drone["heading"],
            speed=drone["speed"],
            battery_percent=drone["battery_percent"],
        )
        r.set(f"telemetry_drone{drone['id']}", telemetry.model_dump_json())
        r.set(
            f"capabilities_drone{drone['id']}",
            drone["capabilities"].model_dump_json(),
        )
    print(f"[seed] Seeded {len(MOCK_DRONES)} mock drones into Redis.")


def clear_test_keys(r: redis.Redis) -> None:
    """Remove stale test data so each run starts cleanly."""
    for key in r.scan_iter(match="frame_drone*_detections"):
        r.delete(key)
    r.delete("watch_area")


def inject_detection(r: redis.Redis, class_name: str, lat: float, lon: float) -> None:
    """Write a single-detection snapshot to frame_drone_mock_detections."""
    detection = json_schemas.SingleDetection(
        gps_position=(lat, lon),
        class_name=class_name,
        drone_ids=[MOCK_DRONES[0]["id"]],
    )
    payload = json_schemas.Detections(root=[detection]).model_dump_json()
    key = "frame_drone_mock_detections"
    r.set(key, payload)
    r.expire(key, 60)
    print(f"[inject] Detection: class={class_name} at ({lat:.5f}, {lon:.5f})")


def inject_watch_area(r: redis.Redis, points: list[dict[str, float]]) -> None:
    """Write a watch-area payload to Redis."""
    payload = json.dumps({"points": points})
    r.set("watch_area", payload)
    print(f"[inject] Watch area with {len(points)} points.")


def make_logging_suggester(suggester: AutoMissionSuggester) -> None:
    """Monkey-patch send_proposed_missions to print instead of sending over WS."""

    def logging_send(missions, detection):
        print(f"\n{'=' * 60}")
        print(f"[PROPOSED MISSIONS] {len(missions)} mission(s) suggested:")
        print(
            f"  detection_id={detection.detection_id}, object_type={detection.object_type}, gps={detection.gps_position}"
        )
        for i, mission in enumerate(missions, 1):
            proposal = mission.get_frontend_mission_proposal()
            print(f"  [{i}] {json.dumps(proposal, indent=4)}")
        print(f"{'=' * 60}\n")

    suggester.send_proposed_missions = logging_send  # type: ignore[method-assign]


def parse_args():
    parser = argparse.ArgumentParser(description="Test AutoMissionSuggester manually")
    parser.add_argument(
        "--redis-host", default=os.environ.get("REDIS_HOST", "localhost")
    )
    parser.add_argument(
        "--redis-port", type=int, default=int(os.environ.get("REDIS_PORT", 6379))
    )
    return parser.parse_args()


def run_interactive(r: redis.Redis, suggester: AutoMissionSuggester) -> None:
    """Read commands from stdin and drive the suggester."""
    print(
        "\nReady. Commands: person/car/truck/bus [lat lon], area lat,lon ..., clear, quit"
    )
    print("Default coordinates: lat=57.7060 lon=11.9380 (offset these per command)\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "clear":
            suggester._recent_detection_ids.clear()
            suggester._recent_detection_events.clear()
            print("[clear] Dedup state cleared.")

        elif cmd in ("person", "car", "truck", "bus", "vehicle"):
            lat = float(parts[1]) if len(parts) > 1 else DEFAULT_LAT
            lon = float(parts[2]) if len(parts) > 2 else DEFAULT_LON
            inject_detection(r, cmd, lat, lon)

        elif cmd == "area":
            if len(parts) < 4:
                print("  Usage: area lat1,lon1 lat2,lon2 lat3,lon3 ...")
                continue
            points = []
            try:
                for token in parts[1:]:
                    lat_s, lon_s = token.split(",")
                    points.append({"lat": float(lat_s), "lon": float(lon_s)})
            except ValueError:
                print(
                    "  Bad format. Use: area 57.705,11.938 57.706,11.939 57.707,11.938"
                )
                continue
            inject_watch_area(r, points)

        else:
            print(f"  Unknown command: {cmd}")


def main() -> None:
    args = parse_args()

    r = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=True,
    )

    try:
        r.ping()
    except redis.ConnectionError:
        print(
            f"ERROR: Cannot connect to Redis at {args.redis_host}:{args.redis_port}.\n"
            "Start Redis (e.g. `docker run -p 6379:6379 redis`) and retry."
        )
        sys.exit(1)

    clear_test_keys(r)
    seed_mock_drones(r)

    suggester = AutoMissionSuggester()
    make_logging_suggester(suggester)

    object_thread = threading.Thread(
        target=suggester.object_listener, daemon=True, name="object-listener"
    )
    area_thread = threading.Thread(
        target=suggester.area_listener, daemon=True, name="area-listener"
    )
    object_thread.start()
    area_thread.start()
    print("Listeners started.")

    try:
        run_interactive(r, suggester)
    finally:
        print("Stopping listeners...")
        suggester.request_stop()
        object_thread.join(timeout=2.0)
        area_thread.join(timeout=2.0)
        print("Done.")


if __name__ == "__main__":
    main()
