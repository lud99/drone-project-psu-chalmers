import socket
from dotenv import set_key
from pathlib import Path

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a public IP (no packets actually sent)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    env_file_path = Path("./host_ip.env")
    # Create the file if it does not exist.
    env_file_path.touch(mode=0o600, exist_ok=True)
    # Save some values to the file.
    set_key(dotenv_path=env_file_path, key_to_set="HOST_IP", value_to_set=get_host_ip())
