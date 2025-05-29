import time
import communication_software.Interface as Interface
from communication_software.ROS import AtosCommunication
from communication_software.misc import (
    start_frontend_websocket_server,
    get_drone_coordinates,
    start_communication_websocket_server,
    main_loop_exit,
    init_rclpy,
)


def main() -> None:
    Interface.print_welcome()
    init_rclpy()

    ATOScommunicator = AtosCommunication()
    try:
        ip = Interface.get_ip()
        ATOScommunicator.publish_init()
        time.sleep(1)

        (droneOrigins, angles) = get_drone_coordinates(ATOScommunicator)
        start_frontend_websocket_server(ATOScommunicator)
        start_communication_websocket_server(ip, droneOrigins, angles)

    finally:
        main_loop_exit(ATOScommunicator)


if __name__ == "__main__":
    main()
