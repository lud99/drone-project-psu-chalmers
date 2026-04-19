import socket
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
    host_ip = get_host_ip()
    env_file_path.write_text(f"HOST_IP={host_ip}\n", encoding="ascii")
    env_file_path.chmod(0o600)
