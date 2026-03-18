from dataclasses import dataclass
import os


@dataclass
class Config:
    drone_id: str = os.getenv("DRONE_ID", "drone2")
    drone_type: str = os.getenv("DRONE_TYPE", "mavlink")
    model: str = os.getenv("DRONE_MODEL", "PX4/ArduPilot")
    backend_ws_url: str = os.getenv("BACKEND_WS_URL", "ws://localhost:14500")
    mavlink_connection_string: str = os.getenv(
        "MAVLINK_CONNECTION_STRING", "/dev/tty.usbserial-D30JUHE4")
    mavlink_baud: int = int(os.getenv("MAVLINK_BAUD", "57600"))
    telemetry_interval_sec: float = float(
        os.getenv("TELEMETRY_INTERVAL_SEC", "1.0"))
    heartbeat_interval_sec: float = float(
        os.getenv("HEARTBEAT_INTERVAL_SEC", "5.0"))
    command_poll_interval_sec: float = float(
        os.getenv("COMMAND_POLL_INTERVAL_SEC", "0.2"))
    default_takeoff_alt: float = float(os.getenv("DEFAULT_TAKEOFF_ALT", "15"))
    max_speed_m_s: float = float(os.getenv("MAX_SPEED_M_S", "8"))
    use_mock_drone: bool = os.getenv(
        "USE_MOCK_DRONE", "false").lower() == "true"
