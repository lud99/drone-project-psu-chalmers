import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
from datetime import datetime
import redis
import redis.exceptions
import numpy as np
from communication_software.missions_planning.mission_registry import MissionRegistry
from communication_software.missions_planning.mission_status import MissionStatus


from typing import Optional

import communication_software.common.json_schemas as json_schemas

from communication_software.common.frame_utils import (
    create_not_connected_frame,
    create_error_frame,
)

try:
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    r.ping()  # Check if the connection is successful
    print("Successfully connected to Redis!")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis: {e}")
    exit()  # Exit if we can't connect

app = FastAPI()


# ATOS Simulation
class ATOSController:
    def __init__(self):
        self.test_active = False
        self.anomalies = False
        self.drone_data = {
            1: {
                "lat": 57.705841,
                "lng": 11.938096,
                "alt": 150,
                "speed": 0.0,
                "battery": 100.0,
            },
            2: {
                "lat": 57.705941,
                "lng": 11.939096,
                "alt": 150,
                "speed": 0.0,
                "battery": 100.0,
            },
        }


atos = ATOSController()


def get_telemetry_and_capabilities_key_tuples() -> list[tuple[str, str, str]]:
    # 1. Collect all relevant keys
    patterns = ["telemetry_drone*", "capabilities_drone*"]
    all_keys = []
    for p in patterns:
        all_keys.extend(r.scan_iter(match=p))

    # 2. Group keys by the ID
    drone_groups: dict[str, dict[str, Optional[str]]] = {}

    for key in all_keys:
        # Decode if keys come back as bytes
        key_str = key.decode("utf-8") if isinstance(key, bytes) else key

        # Extract drone id
        drone_id: str = key_str.replace("telemetry_drone", "").replace(
            "capabilities_drone", ""
        )

        if drone_id not in drone_groups:
            drone_groups[drone_id] = {"telemetry": None, "capabilities": None}

        if "telemetry" in key_str:
            drone_groups[drone_id]["telemetry"] = key_str
        elif "capabilities" in key_str:
            drone_groups[drone_id]["capabilities"] = key_str

    # 3. Create the list of tuples
    return [
        (drone_id, data["telemetry"], data["capabilities"])
        for drone_id, data in drone_groups.items()
        if data["telemetry"] and data["capabilities"]  # Only include if both exist
    ]


# WebSocket Endpoints
@app.websocket("/api/v1/ws/drone")
async def drone_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Drone client connected")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Message received: {data}")

            # TODO implement ws messages here
            # not sure what messages should be websocket

    except WebSocketDisconnect:
        print("Drone client disconnected")
    except Exception as e:
        print(f"Unexpected error in drone_websocket main loop: {e}")
    finally:
        print("Closing drone websocket connection.")
        # FastAPI handles closing the connection, but you can add specific cleanup here if needed.


@app.websocket("/api/v1/ws/atos")
async def atos_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("command") == "start":
                atos.test_active = True
                atos.anomalies = False
                ATOScommunicator.publish_start()
            elif data.get("command") == "stop":
                atos.test_active = False
                ATOScommunicator.publish_abort()
            await websocket.send_json(
                {
                    "status": "success",
                    "test_active": atos.test_active,
                    "anomaly": atos.anomalies,
                }
            )
    except WebSocketDisconnect:
        print("ATOS client disconnected")


COMMAND_CHANNEL = "drone_commands"

### POST routes


@app.post("/api/v1/accept_mission")
async def accept_mission(payload: str = Body(...)):
    try:
        json_schemas.parse_frontend_message(payload)
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}

    return {"msg_type": "response", "error": None}


@app.post("/api/v1/reject_missions")
async def reject_missions(payload: str = Body(...)):
    try:
        json_schemas.parse_frontend_message(payload)
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}

    return {"msg_type": "response", "error": None}


@app.post("/api/v1/start_drone")
async def start_drone(payload: str = Body(...)):
    try:
        json_schemas.parse_frontend_message(payload)
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}

    return {"msg_type": "response", "error": None}


@app.post("/api/v1/set_watch_area")
async def set_watch_area(payload: str = Body(...)):
    try:
        message = json_schemas.parse_frontend_message(payload)
        if isinstance(message, json_schemas.FrontendMessages.SetWatchArea):
            r.set("watch_area", message.area.model_dump_json())
            return {"msg_type": "response", "error": None}

    except Exception as e:
        print(e)
        return {"msg_type": "response", "error": str(e)}

    return {"msg_type": "response", "error": None}


### GET routes


@app.get("/api/v1/proposed_missions")
async def get_proposed_missions():
    try:
        # todo: get from redis
        return json_schemas.FrontendMessages.ProposedMissions(missions=[dict()])
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.get("/api/v1/active_missions")
async def get_active_missions():
    # Logic to fetch active missions
    pass


@app.get("/api/v1/get_watch_area")
async def get_watch_areas():
    try:
        data = r.get("watch_area")
        if not data:
            return {"msg_type": "response", "points": [], "error": None}
        watch_area = json.loads(data)
        return {"msg_type": "response", "points": watch_area["points"], "error": None}
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.get("/api/v1/connected_drones")
async def get_connected_drones():

    drone_list = json_schemas.FrontendMessages.ConnectedDrones(drones=[])

    try:
        for drone_id, telem_key, cap_key in get_telemetry_and_capabilities_key_tuples():
            print(f"Processing drone id {drone_id}: {telem_key} <-> {cap_key}")

            telemetry_str = r.get(telem_key)
            capabilities_str = r.get(cap_key)
            if telemetry_str is None:
                print(f"Telemetry not found for drone {drone_id}")
                raise Exception(f"Telemetry not found for drone {drone_id}")
            if capabilities_str is None:
                print(f"Telemetry not found for drone {drone_id}")
                raise Exception(f"Telemetry not found for drone {drone_id}")

            telemetry = json_schemas.parse_telemetry(telemetry_str)
            capabilities = json_schemas.parse_capabilities(capabilities_str)

            drone_list.drones.append(
                json_schemas.DroneInfo(
                    drone_id=drone_id, capabilities=capabilities, telemetry=telemetry
                )
            )

        return drone_list.model_dump_json()

    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.get("/api/v1/telemetry/{drone_id}")
async def get_telemetry(drone_id: str):
    try:
        telemetry = r.get(f"telemetry_drone{drone_id}")
        if telemetry is None:
            raise Exception(
                f"Telemetry for drone {drone_id} not found, is it connected?"
            )

        return json_schemas.FrontendMessages.TelemetryUpdate(
            msg_type="telemetry",
            drone_id=drone_id,
            telemetry=json_schemas.parse_telemetry(r.get(f"telemetry_drone{drone_id}")),
        ).model_dump_json()

    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.websocket("/api/v1/ws/flightmanager")
async def flightmanager_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Flight Manager WebSocket connected")
    try:
        while True:
            data = await websocket.receive_json()
            drone_id = data.get("drone_id")
            command = data.get("command")
            payload = data.get("payload", {})

            if drone_id is None or command is None:
                print(f"Received invalid command data: {data}")
                await websocket.send_json(
                    {"status": "error", "message": "Missing drone_id or command"}
                )
                continue

            message_to_publish = {
                "target_drone_id": drone_id,
                "command": command,
                "payload": payload,
                "timestamp": datetime.now().isoformat(),
            }
            message_str = json.dumps(message_to_publish)

            try:
                print(
                    f"Publishing command to Redis channel '{COMMAND_CHANNEL}': {message_str}"
                )
                await asyncio.to_thread(r.publish, COMMAND_CHANNEL, message_str)
                print(f"Successfully published command for drone {drone_id}")
                await websocket.send_json(
                    {
                        "drone_id": drone_id,
                        "command_sent": command,
                        "status": "published",
                    }
                )
            except redis.exceptions.RedisError as e:
                print(f"Redis error publishing command: {e}")
                await websocket.send_json(
                    {
                        "drone_id": drone_id,
                        "command_sent": command,
                        "status": "error",
                        "message": f"Redis publish error: {e}",
                    }
                )
            except Exception as e:
                print(f"Unexpected error publishing command: {e}")
                await websocket.send_json(
                    {
                        "drone_id": drone_id,
                        "command_sent": command,
                        "status": "error",
                        "message": f"Unexpected error: {e}",
                    }
                )

    except WebSocketDisconnect:
        print("Flight Manager WebSocket disconnected")
    except Exception as e:
        print(f"Error in flightmanager_websocket: {e}")
    finally:
        print("Closing flightmanager websocket")


@app.get("/api/v1/video_feed/drone1")
async def drone1_feed():
    return StreamingResponse(
        stream_drone_frames("1"), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/v1/video_feed/drone2")
async def drone2_feed():
    return StreamingResponse(
        stream_drone_frames("2"), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/v1/video_feed/drone1_annotated")
async def drone1_feed_annotated():
    return StreamingResponse(
        stream_drone_frames("1", "_annotated"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/video_feed/drone2_annotated")
async def drone2_feed_annotated():
    return StreamingResponse(
        stream_drone_frames("2", "_annotated"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/video_feed/merged")
async def merged_feed():
    return StreamingResponse(
        stream_drone_frames("merged"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


## Connect missions to the API/frontend
mission_registry = MissionRegistry()


@app.get("/api/v1/missions")
def get_missions():
    return mission_registry.get_all()


@app.post("/api/v1/missions/dispatch/{mission_id}")
def dispatch_mission(mission_id: str):
    mission = mission_registry.get(mission_id)
    if not mission:
        return {"error": "Mission not found"}

    mission_registry.update_status(mission_id, MissionStatus.DISPATCHED)
    # TODO: Forward to translation layer here

    return {"status": "dispatched", "mission": mission}


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def run_server(atos_communicator):
    global ATOScommunicator
    ATOScommunicator = atos_communicator
    uvicorn.run(
        "communication_software.frontend_websocket:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


# Video Frames Generation Based on Drone ID
async def stream_drone_frames(drone_id: str, frame_type: str = ""):

    redis_key = f"frame_drone{drone_id}${frame_type}"
    while True:
        # RTC or capture process is storing a frame in Redis.
        frame_data = await asyncio.to_thread(r.get, redis_key)
        if frame_data:
            # Might need to adjust this if you're using base64 or another format.
            frame_array = np.frombuffer(frame_data.encode("latin1"), dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                # If decoding fails, fall back to a dummy image.
                frame = create_error_frame(
                    np.zeros((480, 640, 3), dtype=np.uint8), drone_id, "invalid frame"
                )

        else:
            # No frame found in Redis, so generate a dummy frame.
            frame = create_not_connected_frame(
                np.zeros((480, 640, 3), dtype=np.uint8), drone_id
            )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            # If encoding fails, continue to try on the next iteration.
            await asyncio.sleep(0.033)
            continue

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        await asyncio.sleep(0.033)  # Approximately 30 frames per second


if __name__ == "__main__":
    uvicorn.run("frontend_websocket:app", host="0.0.0.0", port=8000, reload=True)
