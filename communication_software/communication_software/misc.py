import asyncio
import os
import threading
from communication_software.convex_hull_scalable import Coordinate, get_drones_location
from communication_software.frontend_websocket import run_server
from communication_software.communication import Communication
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


def get_drone_coordinates(atos_communicator):
    origo = get_origo_coords(atos_communicator)

    trajectory_list = get_trajectory_list(atos_communicator)

    n_drones = int(os.environ.get("N_DRONES", 2))
    overlap = float(os.environ.get("OVERLAP", 0.5))
    fly_to_list, angle = get_drones_location(trajectory_list, origo, n_drones, overlap)

    print_fly_to_list(fly_to_list, angle)

    drone_origins = tuple([coord for coord in fly_to_list])
    angles = angle, angle
    return (drone_origins, angles)


def start_communication_websocket_server(ip, drone_origins, angles):
    communication = Communication()
    try:
        print("Communication server starting, press ctrl + c to exit")

        asyncio.run(
            run_comm_server(
                communication,
                ip=ip,
                drone_origins=drone_origins,
                angles=angles,
            )
        )
    except KeyboardInterrupt:
        print("\nCommunication server interrupted!")
        if (
            communication.redis_listener_task
            and communication.redis_listener_task.is_alive()
        ):
            communication.redis_listener_stop_event.set()
            communication.redis_listener_task.join(timeout=2)
    except OSError as e:
        print(f"OS Error starting server: {e}")
    except Exception as e:
        print(f"Unexpected error starting server: {e}")


async def run_comm_server(
    communication: Communication, ip: str, drone_origins: list, angles: list
):
    loop = asyncio.get_running_loop()
    communication.loop = loop

    communication.start_redis_listener_thread()

    await communication.send_coordinates_websocket(
        ip=ip, drone_origins=drone_origins, angles=angles
    )


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
