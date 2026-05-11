import asyncio
import math
import random
import sys
import time
from websockets import connect
import websockets
import cv2
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCIceCandidate,
)
from av import VideoFrame
from typing import Optional

import communication_software.communication_software.common.json_schemas as json_schemas


# Configuration
SERVER_WS_URL = "ws://localhost:14500"
DRONE_ID = "haubits_77"
TELEMETRY_INTERVAL = 5
BATTERY_DRAIN_PER_MINUTE = 20  # Battery decreases by 20% per minute

# Battery tracking
current_battery = 90.0
battery_drain_per_interval = BATTERY_DRAIN_PER_MINUTE / (
    60 / TELEMETRY_INTERVAL
)  # Drain per 5s interval

# liseberg = (57.696162, 11.991556)
emilsborg_football_field = (57.68088480716388, 11.982436321054934)
VIDEO_PATH = "mock_drone/test_video_2024.mp4"

# Drone state tracking
current_lat = emilsborg_football_field[0] + random.uniform(-0.001, 0.001)
current_lon = emilsborg_football_field[1] + random.uniform(-0.001, 0.001)
current_alt = 0.0
current_heading = 0.0
current_speed = 0.0
home_position = (current_lat, current_lon, 0.0)


def _geo_offset(lat: float, lon: float, dx: float, dy: float) -> tuple[float, float]:
    lat_out = lat + dy / 111111.0
    lon_out = lon + dx / (111111.0 * math.cos(math.radians(lat)))
    return lat_out, lon_out


def _distance_and_heading(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float, float, int]:
    dy = (lat2 - lat1) * 111111.0
    dx = (lon2 - lon1) * 111111.0 * math.cos(math.radians(lat1))
    dist2d = math.hypot(dx, dy)
    heading = int((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)
    return dx, dy, dist2d, heading


async def _fly_to_target(
    target_lat: float,
    target_lon: float,
    target_alt: float,
    speed_mps: float = 5.0,
    update_interval: float = 0.5,
) -> None:
    global current_lat, current_lon, current_alt, current_heading, current_speed

    while True:
        dx, dy, dist2d, heading = _distance_and_heading(
            current_lat, current_lon, target_lat, target_lon
        )
        dz = target_alt - current_alt
        dist3d = math.hypot(dist2d, dz)

        if dist3d <= 2.0:
            current_lat = target_lat
            current_lon = target_lon
            current_alt = target_alt
            current_speed = 0.0
            return

        travel = min(speed_mps * update_interval, dist3d)
        fraction = travel / dist3d
        move_dx = dx * fraction
        move_dy = dy * fraction
        move_dz = dz * fraction

        current_lat, current_lon = _geo_offset(
            current_lat, current_lon, move_dx, move_dy
        )
        current_alt += move_dz
        current_speed = travel / update_interval
        current_heading = heading if dist2d > 0.1 else current_heading

        await asyncio.sleep(update_interval)


async def send_telemetry(websocket, drone_id: str):
    """Continuously sends telemetry using the TelemetryMessage class."""
    global \
        current_battery, \
        current_speed, \
        current_heading, \
        current_lat, \
        current_lon, \
        current_alt

    while True:
        current_battery = int(max(0.0, current_battery - battery_drain_per_interval))

        current_telemetry = json_schemas.Telemetry(
            lat=current_lat,
            lon=current_lon,
            alt=current_alt,
            heading=int(current_heading),
            speed=current_speed,
            battery_percent=current_battery,
        )

        msg = json_schemas.TelemetryMessage(
            msg_type="telemetry", drone_id=drone_id, telemetry=current_telemetry
        )

        await websocket.send(msg.model_dump_json())
        await asyncio.sleep(TELEMETRY_INTERVAL)


async def waiter(event: asyncio.Event):
    await event.wait()
    print("Task done")


async def send_task_complete(
    ws: websockets.ClientConnection,
    drone_id: str,
    task_message: json_schemas.TaskMessage,
):
    event_message = json_schemas.TaskEventMessage(
        mission_id=task_message.mission_id,
        drone_id=drone_id,
        index=task_message.index,
        event="task_complete",
        timestamp=int(time.time()),
    )

    await ws.send(event_message.model_dump_json())


async def do_task(
    ws: websockets.ClientConnection,
    drone_id: str,
    task_message: json_schemas.TaskMessage,
    event: asyncio.Event,
):
    print(f"Doing task {task_message.task_action.action}")

    action = task_message.task_action

    if isinstance(action, json_schemas.GoToTask):
        print("Going to a position")
        if current_alt < 18.0:
            print("Ascending to 20 meters before moving horizontally")
            await _fly_to_target(current_lat, current_lon, 20.0, speed_mps=2.0)
        print("Traveling to destination along shortest 3D path")
        await _fly_to_target(
            action.params.lat,
            action.params.lon,
            action.params.alt,
            speed_mps=5.0,
        )
        await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.PlayAudioTask):
        print(
            f"Playing sound for {action.params.file} for {action.params.duration_seconds or '∞'}s"
        )
        asyncio.create_task(waiter(event))
        if action.params.duration_seconds is not None:
            await asyncio.sleep(action.params.duration_seconds)
            event.set()
            await send_task_complete(ws, drone_id, task_message)
        else:
            await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.LEDTask):
        print(f"Activating LED for {action.params.duration_seconds or '∞'}s")
        asyncio.create_task(waiter(event))
        if action.params.duration_seconds is not None:
            await asyncio.sleep(action.params.duration_seconds)
            event.set()
            await send_task_complete(ws, drone_id, task_message)
        else:
            await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.SpotlightTask):
        print(f"Activating spotlight for {action.params.duration_seconds or '∞'}s")
        asyncio.create_task(waiter(event))
        if action.params.duration_seconds is not None:
            await asyncio.sleep(action.params.duration_seconds)
            event.set()
            await send_task_complete(ws, drone_id, task_message)
        else:
            await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.AngleCameraTask):
        print(f"Angling camera pitch={action.params.pitch} yaw={action.params.yaw}")
        await asyncio.sleep(2)
        await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.HoverTask):
        print(f"Hovering for {action.params.duration_seconds or '∞'}s")
        if action.params.duration_seconds:
            await asyncio.sleep(action.params.duration_seconds)
        else:
            await asyncio.sleep(5)
        await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.GoHomeTask):
        print("Going home")
        await _fly_to_target(
            home_position[0], home_position[1], home_position[2], speed_mps=5.0
        )
        await send_task_complete(ws, drone_id, task_message)


class VideoFileTrack(VideoStreamTrack):
    """
    A Custom Track that reads frames from an MP4 file.
    """

    def __init__(self, path):
        super().__init__()
        self.cap = cv2.VideoCapture(path)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        grabbed, frame = self.cap.read()
        if not grabbed:
            # Loop the video if it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            grabbed, frame = self.cap.read()

        # Convert BGR (OpenCV) to RGB for WebRTC
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame


async def run_drone_client(drone_id: str, video_path: Optional[str]):
    pc = RTCPeerConnection()

    if video_path:
        print(f"Streaming video from path {video_path}")
        # Add the video track to the PeerConnection
        video_track = VideoFileTrack(video_path)
        pc.addTrack(video_track)

    try:
        wait_event: asyncio.Event = asyncio.Event()
        async with connect(SERVER_WS_URL) as websocket:
            print(
                f"Connected to {SERVER_WS_URL} as drone {drone_id}, battery drain interval={battery_drain_per_interval}% every {TELEMETRY_INTERVAL}s"
            )

            await asyncio.sleep(2)

            # Test that drone id is new connection id

            reg_msg = json_schemas.DroneRegistrationMessage(
                msg_type="drone_registration",
                model="DJI-Mavic-Mock",
                drone_type="DJI",
                drone_id=drone_id,
                capabilities=json_schemas.Capabilities(
                    camera=json_schemas.CameraCapabilities(
                        aspect_ratio=1.777,
                        diagonal_fov=84.0,
                        resolution_height=1080,
                        resolution_width=1920,
                    ),
                    led=json_schemas.LEDCapabilities(types=["rear", "beacon"]),
                    spotlight=True,
                    speaker=json_schemas.SpeakerCapabilities(
                        audio_files=[
                            "horn",
                            "hello",
                            "restart_transponder",
                            "siren",
                            "warning",
                        ]
                    ),
                ),
                telemetry=json_schemas.Telemetry(
                    lat=current_lat,
                    lon=current_lon,
                    alt=current_alt,
                    heading=current_heading,
                    speed=current_speed,
                    battery_percent=int(current_battery),
                ),
            )

            await websocket.send(reg_msg.model_dump_json())
            print(f"Sent registration for {drone_id}")

            # 2. Start continuous telemetry task
            asyncio.create_task(send_telemetry(websocket, drone_id))

            # 3. Main loop to handle incoming server data
            while True:
                raw_message = await websocket.recv()

                try:
                    message = json_schemas.parse_drone_message(str(raw_message))
                    print(f"Received message: {message.msg_type}")

                    # Handle Signaling
                    msg_data = json_schemas.parse_drone_message(str(raw_message))

                    # Handle Offer from Server
                    if msg_data.msg_type == "offer":
                        print("Received Offer, sending Answer")
                        offer = RTCSessionDescription(sdp=msg_data.sdp, type="offer")
                        await pc.setRemoteDescription(offer)

                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)

                        # Send Answer back
                        answer_msg = json_schemas.WebRTCAnswerMessage(
                            msg_type="answer", sdp=pc.localDescription.sdp
                        )
                        await websocket.send(answer_msg.model_dump_json())

                    # Handle ICE Candidates from Server
                    elif msg_data.msg_type == "candidate":
                        print("Received ICE Candidate")
                        # Note: You may need to parse the candidate string based on your schema
                        # This assumes a standard candidate string
                        candidate = RTCIceCandidate(
                            sdpMid=0, sdpMLineIndex=0, candidate=msg_data.candidate
                        )
                        await pc.addIceCandidate(candidate)
                        print("Received RTC candidate")

                    if isinstance(message, json_schemas.TaskMessage):
                        wait_event = asyncio.Event()
                        asyncio.create_task(
                            do_task(websocket, message.drone_id, message, wait_event)
                        )
                    elif isinstance(message, json_schemas.AbortTaskMessage):
                        print(f"Aborting task {message.task_action}")

                        # This should abort only the specified task_action and continue running others
                        wait_event.set()
                        event_message = json_schemas.TaskEventMessage(
                            mission_id=message.mission_id,
                            drone_id=message.drone_id,
                            index=-1,
                            event="task_complete",
                            timestamp=int(time.time()),
                        )

                        await websocket.send(event_message.model_dump_json())

                    elif isinstance(message, json_schemas.LandMessage):
                        print("Aborting mission, doing land")

                        # This should abort all active task (if any are active) and land
                        wait_event.set()

                except Exception as e_inner:
                    print(f"ERROR for {raw_message}, exception: {e_inner}")

    except Exception as e:
        print(f"Client error: {e}")
    finally:
        await pc.close()


if __name__ == "__main__":
    drone_id = sys.argv[1] if len(sys.argv) > 1 else DRONE_ID
    # if arg 2 is "drain" then set battery drain to default value for testing, otherwise 0
    if len(sys.argv) <= 2 or sys.argv[2] != "drain":
        battery_drain_per_interval = 0

    asyncio.run(run_drone_client(drone_id, VIDEO_PATH))
