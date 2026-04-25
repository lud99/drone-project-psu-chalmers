import asyncio
import os
import threading
from communication_software.convex_hull_scalable import Coordinate
from communication_software.frontend_websocket import run_server
from communication_software.drone_communication import DroneCommunication
from communication_software.missions_planning.auto_mission_suggest import (
    AutoMissionSuggester,
)
import rclpy


def google_maps_coord_link(lat, lng):
    return f"https://www.google.com/maps/place/{lat},{lng}"


def print_fly_to_list(fly_to_list, angle):
    for i, fly_to in enumerate(fly_to_list):
        lat, lng, alt = fly_to.lat, fly_to.lng, fly_to.alt
        google_maps_url = google_maps_coord_link(lat, lng)
        print(
            f"Drone {i} going to: (lat, lng, alt) {lat, lng, alt}, \n angle: {angle}, link: {google_maps_url}"
        )


def get_trajectory_list(atos_communicator):
    ids = atos_communicator.get_object_ids()
    trajectory_list = {}
    for id in ids:
        coordlist = atos_communicator.get_object_traj(id)
        trajectory_list[id] = coordlist
    return trajectory_list


def is_debug_mode():
    """
    Retrieves the DEBUG_MODE environment variable and returns it as a boolean.
    Handles various string representations of boolean values.
    """
    debug_mode_str = os.getenv(
        "DEBUG_MODE", "false"
    ).lower()  # default to "false" if not set

    if debug_mode_str in ("true", "1", "t", "y", "yes"):
        return True
    elif debug_mode_str in ("false", "0", "f", "n", "no", ""):
        return False
    else:
        print(
            f"Warning: Invalid DEBUG_MODE value: {debug_mode_str}.  Treating as false."
        )
        return False


def get_origo_coords(atos_communicator) -> Coordinate:
    altitude = os.getenv("ENV_ALTITUDE")
    latitude = os.getenv("ENV_LATITUDE")
    longitude = os.getenv("ENV_LONGITUDE")
    if altitude and latitude and longitude and is_debug_mode():
        print("Using custom coords")
        return Coordinate(
            lat=float(latitude), lng=float(longitude), alt=float(altitude)
        )
    return atos_communicator.get_origin_coordinates()


def start_frontend_websocket_server(atos_communicator):
    server_thread = threading.Thread(
        target=run_server, args=(atos_communicator,), daemon=True
    )
    server_thread.start()
    print("FastAPI server started in a separate thread!")


def start_communication_websocket_server(ip):
    drone_communication = DroneCommunication()
    auto_mission_suggester = AutoMissionSuggester()
    try:
        print("Drone Communication server starting, press ctrl + c to exit")

        asyncio.run(
            run_comm_server(
                drone_communication,
                auto_mission_suggester,
                ip=ip,
            )
        )
    except KeyboardInterrupt:
        print("\nDrone Communication server interrupted!")
        auto_mission_suggester.request_stop()
        if (
            drone_communication.redis_listener_task
            and drone_communication.redis_listener_task.is_alive()
        ):
            drone_communication.redis_listener_stop_event.set()
            drone_communication.redis_listener_task.join(timeout=2)
    except OSError as e:
        print(f"OS Error starting server: {e}")
    except Exception as e:
        print(f"Unexpected error starting server: {e}")


async def run_comm_server(
    drone_communication: DroneCommunication,
    auto_mission_suggester: AutoMissionSuggester,
    ip: str,
):
    loop = asyncio.get_running_loop()
    drone_communication.loop = loop

    drone_communication.start_redis_listener_thread()

    threading.Thread(
        target=auto_mission_suggester.object_listener,
        daemon=True,
        name="object-listener",
    ).start()

    _listener_task = asyncio.create_task(auto_mission_suggester.redis_area_listener())
    print("AutoMissionSuggester listeners started.")

    await drone_communication.start_websocket_server(ip=ip)


def main_loop_exit(atos_communicator):
    # Graceful shutdown
    print("Shutting down ROS node...")
    if atos_communicator:
        atos_communicator.destroy_node()
    # if rclpy.is_initialized():
    rclpy.shutdown()
    print("Shutdown complete.")


def init_rclpy():
    if not rclpy.ok():
        print("Trying to initialize rclpy")
        rclpy.init()
