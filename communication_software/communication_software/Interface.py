def print_welcome() -> None:
    """Prints a menu for the user interface for the communication software.
    Askes the user if they want to continue or quit and returns a boolean.
    """
    print("------------------------------------------")
    print("~~ Welcome to the communication software~~")


def get_ip() -> str:
    """Use 0.0.0.0 to bind WebSocket on all interfaces

    Returns:
        str: a string containing the IP-adress
    """
    ip = "0.0.0.0"

    return ip
