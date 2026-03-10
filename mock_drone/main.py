import asyncio
import random
import sys
import time
from websockets import connect
import websockets

import communication_software.communication_software.common.json_schemas as json_schemas

# Configuration
SERVER_WS_URL = "ws://localhost:14500"
DRONE_ID = "haubits_77"
TELEMETRY_INTERVAL = 5


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


async def run_drone_client(drone_id: str):
    try:
        wait_event: asyncio.Event = asyncio.Event()
        async with connect(SERVER_WS_URL) as websocket:
            print(f"Connected to {SERVER_WS_URL} as drone {drone_id}")

            await asyncio.sleep(2)

            # Test that drone id is new connection id

            # TODO Add Camera support

            # cameraCapabilities(
            #             aspect_ratio=1.77,
            #             horizontal_fov=84.0,
            #             resolution_height=1080,
            #             resolution_width=1920,
            #         ),

            reg_msg = json_schemas.DroneRegistrationMessage(
                msg_type="drone_registration",
                drone_type="quadcopter",
                model="DJI-Mavic-Mock",
                drone_id=drone_id,
                capabilities=json_schemas.Capabilities(
                    camera=None,
                    led=json_schemas.LEDCapabilities(types=["rear", "beacon"]),
                    spotlight=True,
                    speaker=json_schemas.SpeakerCapabilities(
                        audio_files=["hello", "world"]
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

                    elif isinstance(message, json_schemas.AbortMissionMessage):
                        print(f"Aborting mission, doing {message.next_action}")

                        # This should abort all active task (if any are active) and do the specified next action
                        wait_event.set()

                except Exception as e_inner:
                    print(f"ERROR for {raw_message}, exception: {e_inner}")

    except Exception as e:
        print(f"Client error: {e}")


if __name__ == "__main__":
    drone_id = DRONE_ID
    if len(sys.argv) > 1:
        drone_id += "_" + sys.argv[1]
    asyncio.run(run_drone_client(drone_id))
