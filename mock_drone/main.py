import asyncio
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
VIDEO_PATH = "mock_drone/test_video_2024.mp4"


async def send_telemetry(websocket, drone_id: str):
    """Continuously sends telemetry using the TelemetryMessage class."""
    while True:
        # Create the Telemetry sub-model
        current_telemetry = json_schemas.Telemetry(
            lat=57.705 + (random.uniform(-0.001, 0.001)),
            lon=11.938 + (random.uniform(-0.001, 0.001)),
            alt=random.uniform(110, 120),
            heading=random.randint(0, 359),
            speed=random.uniform(0.0, 5.5),
            battery_percent=88,
        )

        # Wrap in the TelemetryMessage envelope
        msg = json_schemas.TelemetryMessage(
            msg_type="telemetry", drone_id=drone_id, telemetry=current_telemetry
        )

        await websocket.send(msg.model_dump_json())
        print(f"[{time.strftime('%H:%M:%S')}] Telemetry validated and sent.")
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
        await asyncio.sleep(5)
        await send_task_complete(ws, drone_id, task_message)
    elif isinstance(action, json_schemas.PlayAudioTask):
        print(f"Playing sound for {action.params.duration_seconds}s")

        asyncio.create_task(waiter(event))
        if action.params.duration_seconds is not None:
            await asyncio.sleep(action.params.duration_seconds)
            event.set()
            await send_task_complete(ws, drone_id, task_message)
        else:
            # Activating for infinite time should send complete event asap
            await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.LEDTask):
        print(f"Activating LED for {action.params.duration_seconds}s")

        asyncio.create_task(waiter(event))
        if action.params.duration_seconds is not None:
            await asyncio.sleep(action.params.duration_seconds)
            event.set()
            await send_task_complete(ws, drone_id, task_message)
        else:
            # Activating for infinite time should send complete event asap
            await send_task_complete(ws, drone_id, task_message)

    elif isinstance(action, json_schemas.SpotlightTask):
        print(f"Activating spotligt for {action.params.duration_seconds}s")

        asyncio.create_task(waiter(event))
        if action.params.duration_seconds is not None:
            await asyncio.sleep(action.params.duration_seconds)
            event.set()
            await send_task_complete(ws, drone_id, task_message)
        else:
            # Activating for infinite time should send complete event asap
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

    # Add the video track to the PeerConnection
    video_track = VideoFileTrack(video_path)
    pc.addTrack(video_track)

    try:
        wait_event: asyncio.Event = asyncio.Event()
        async with connect(SERVER_WS_URL) as websocket:
            print(f"Connected to {SERVER_WS_URL} as drone {drone_id}")

            await asyncio.sleep(2)

            # Test that drone id is new connection id

            reg_msg = json_schemas.DroneRegistrationMessage(
                msg_type="drone_registration",
                drone_type="quadcopter",
                model="DJI-Mavic-Mock",
                drone_id=drone_id,
                capabilities=json_schemas.Capabilities(
                    led=json_schemas.LEDCapabilities(types=["rear", "beacon"]),
                    spotlight=True,
                    speaker=json_schemas.SpeakerCapabilities(
                        audio_files=["hello", "world"]
                    ),
                    camera=json_schemas.CameraCapabilities(
                        aspect_ratio=1.77,
                        horizontal_fov=84.0,
                        resolution_height=1080,
                        resolution_width=1920,
                    ),
                ),
                telemetry=json_schemas.Telemetry(
                    lat=57.705 + (random.uniform(-0.001, 0.001)),
                    lon=11.938 + (random.uniform(-0.001, 0.001)),
                    alt=random.uniform(110, 120),
                    heading=random.randint(0, 359),
                    speed=random.uniform(0.0, 5.5),
                    battery_percent=88,
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
                        print("Received Offer, creating Answer...")
                        offer = RTCSessionDescription(sdp=msg_data.sdp, type="offer")
                        await pc.setRemoteDescription(offer)

                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)

                        # Send Answer back
                        answer_msg = json_schemas.WebRTCAnswerMessage(
                            msg_type="answer",
                            sdp=pc.localDescription.sdp,
                            # drone_id=drone_id,
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

                    elif isinstance(message, json_schemas.GoHomeMessage):
                        print("Aborting mission, doing go_home")

                        # This should abort all active task (if any are active) and do go_home
                        wait_event.set()

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
    video_path = sys.argv[2] if len(sys.argv) > 2 else VIDEO_PATH

    asyncio.run(run_drone_client(drone_id, video_path))
