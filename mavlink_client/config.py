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
    default_takeoff_alt: float = float(os.getenv("DEFAULT_TAKEOFF_ALT", "2"))
    max_speed_m_s: float = float(os.getenv("MAX_SPEED_M_S", "8"))
    use_mock_drone: bool = os.getenv(
        "USE_MOCK_DRONE", "true").lower() == "true"
    # New configs
    min_relative_altitude_m: float = float(
        os.getenv("MIN_RELATIVE_ALTITUDE_M", "1.0"))
    max_relative_altitude_m: float = float(
        os.getenv("MAX_RELATIVE_ALTITUDE_M", "50.0"))
    goto_timeout_sec: int = int(os.getenv("GOTO_TIMEOUT_SEC", "60"))
    goto_acceptance_radius_m: float = float(
        os.getenv("GOTO_ACCEPTANCE_RADIUS_M", "2.0"))
    geofence_breach_action: str = os.getenv("GEOFENCE_BREACH_ACTION", "land")
    geofence_check_interval_sec: float = float(
        os.getenv("GEOFENCE_CHECK_INTERVAL_SEC", "1.0"))
    max_polygon_points: int = int(os.getenv("MAX_POLYGON_POINTS", "20"))
    default_hold_behavior: str = os.getenv("DEFAULT_HOLD_BEHAVIOR", "loiter")
    arm_command_timeout_sec: float = float(
        os.getenv("ARM_COMMAND_TIMEOUT_SEC", "20.0"))
    takeoff_command_timeout_sec: float = float(
        os.getenv("TAKEOFF_COMMAND_TIMEOUT_SEC", "45.0"))
    api_port: int = int(os.getenv("API_PORT", "8000"))
