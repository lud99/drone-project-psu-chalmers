import time
import communication_software.interface as Interface  # noqa: N812
from communication_software.ros import AtosCommunication
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

    atos_communicator = AtosCommunication()
    try:
        ip = Interface.get_ip()
        atos_communicator.publish_init()
        time.sleep(1)

        (drone_origins, angles) = get_drone_coordinates(atos_communicator)
        start_frontend_websocket_server(atos_communicator)
        start_communication_websocket_server(ip, drone_origins, angles)

    finally:
        main_loop_exit(atos_communicator)


if __name__ == "__main__":
    main()
