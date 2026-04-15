import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import json
import os
import uuid
from datetime import datetime
import redis.exceptions
import redis
import numpy as np
import logging
from communication_software.missions_planning.mission_registry import MissionRegistry
from communication_software.missions_planning.mission_status import MissionStatus


from typing import Optional

from communication_software.constants import (
    DRONE_EVENT_CHANNEL,
    SURVEIL_AREA_CHANNEL,
)

import communication_software.common.json_schemas as json_schemas

from communication_software.common.frame_utils import (
    create_not_connected_frame,
    create_no_camera_frame,
    create_error_frame,
)

logger = logging.getLogger(__name__)


try:
    r = redis.Redis(
        host=os.environ.get("REDIS_URL"),
        port=os.environ.get("REDIS_PORT"),
        db=0,
        decode_responses=True,
    )
    r_async = redis.asyncio.Redis(
        host=os.environ.get("REDIS_URL"),
        port=os.environ.get("REDIS_PORT"),
        db=0,
        decode_responses=True,
    )
    r.ping()  # Check if the connection is successful
    print("[Frontend Server] Successfully connected to Redis")
except redis.exceptions.ConnectionError as e:
    print(f"[Frontend Server] Error connecting to Redis: {e}")
    exit()  # Exit if we can't connect


app = FastAPI()
# 1. Define the domains allowed to access your API
origins = ["http://localhost:8001"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


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


def get_connected_drones_helper():
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
        return json.dumps({"msg_type": "response", "error": str(e)})


# WebSocket Endpoints


@app.websocket("/api/v1/ws/flightmanager")
async def flightmanager_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Flight Manager WebSocket connected")

    # Forward drone events
    async def redis_listener():
        pubsub = r_async.pubsub()
        await pubsub.subscribe(DRONE_EVENT_CHANNEL)

        async for message in pubsub.listen():
            if message["type"] == "message":
                logger.debug("Received event, sending to frontend clients")
                await websocket.send_text(message["data"])

    listener_task = asyncio.create_task(redis_listener())

    # Send connected drones
    await websocket.send_text(get_connected_drones_helper())

    try:
        while True:
            data = await websocket.receive_json()
            print(f"Received {data}")

            try:
                # TODO handle some messages?
                pass

            except redis.exceptions.RedisError as e:
                error = f"Redis error publishing command: {e}"
                print(error)
                await websocket.send_text(
                    json_schemas.FrontendMessages.Error(error=error).model_dump_json()
                )

            except Exception as e:
                error = f"Unexpected error publishing command: {e}"
                print(error)
                await websocket.send_text(
                    json_schemas.FrontendMessages.Error(error=error).model_dump_json()
                )

    except WebSocketDisconnect:
        print("Flight Manager WebSocket disconnected")
    except Exception as e:
        print(f"Error in flightmanager_websocket: {e}")
    finally:
        print("Closing flightmanager websocket")
        listener_task.cancel()


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


### POST routes


@app.post("/api/v1/missions/dispatch/{mission_id}")
def dispatch_mission(mission_id: str):
    try:
        mission = MissionRegistry.get(mission_id)
        if not mission:
            return {"msg_type": "response", "error": "Mission not found"}

        MissionRegistry.dispatch_mission(mission_id)

        return {"status": "dispatched", "mission": mission}
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.post("/api/v1/missions/reject/{mission_id}")
async def reject_missions(mission_id: str):
    try:
        mission = MissionRegistry.get(mission_id)
        if not mission:
            return {"msg_type": "response", "error": "Mission not found"}

        mission_status = mission["status"]
        if mission_status != MissionStatus.PENDING.value:
            return {
                "msg_type": "response",
                "error": f"Cannot reject mission, it is in state '{mission_status}'",
            }

        mission = MissionRegistry.remove(mission_id)

        return {"msg_type": "response"}
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.post("/api/v1/set_watch_area")
async def set_watch_area(payload: str = Body(...)):
    try:
        message = json_schemas.parse_frontend_message(payload)
        if isinstance(message, json_schemas.FrontendMessages.SetWatchArea):
            request_id = str(uuid.uuid4())  # Unique ID for this specific request

            coords = [{"lat": p.lat, "lon": p.lon} for p in message.area.points]

            # Add the request_id to the payload so B knows where to send the answer
            payload_data = {"points": coords, "request_id": request_id}
            raw_watch_area = json.dumps(payload_data)

            # 1. Publish to B
            r.publish(SURVEIL_AREA_CHANNEL, raw_watch_area)

            # 2. Wait for B to respond on a unique list key (timeout after 5 seconds)
            response_key = f"response_{request_id}"
            response = r.blpop(response_key, timeout=5)
            print("response", response)

            if response:
                # response is a tuple (key, data)
                mission_data = json.loads(response[1])
                return {"msg_type": "response", "data": mission_data, "error": None}
            else:
                return {
                    "msg_type": "response",
                    "error": "Timeout waiting for drone assignment",
                }
    except Exception as e:
        return {"msg_type": "response", "error": str(e)}
    return {"msg_type": "response", "error": None}


@app.post("/api/v1/set_detections")
async def set_detections(payload: str = Body(...)):
    try:
        detections = json_schemas.parse_detections(payload).root

        for _detection in detections:
            redis_key_detections = f"frame_drone{detections[0].drone_ids[0]}_detections"
            r.set(redis_key_detections, payload)

    except Exception as e:
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


@app.get("/api/v1/missions")
def get_missions():
    return MissionRegistry.get_all()


@app.get("/api/v1/get_watch_area")
async def get_watch_area():
    try:
        data = r.get("watch_area")
        if not data:
            return {
                "msg_type": "response",
                "points": [],
                "coverage": [],
                "min_rect": [],
                "error": None,
            }

        watch_area = json.loads(data)
        coverage = []
        min_rect = []

        coverage_data = r.get("watch_area_coverage")
        if coverage_data:
            coverage = json.loads(coverage_data)

        min_rect_data = r.get("watch_area_min_rect")
        if min_rect_data:
            min_rect = json.loads(min_rect_data)

        return {
            "msg_type": "response",
            "points": watch_area["points"],
            "coverage": coverage,
            "min_rect": min_rect,
            "error": None,
        }
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

        return json_schemas.TelemetryMessage(
            drone_id=drone_id,
            telemetry=json_schemas.parse_telemetry(r.get(f"telemetry_drone{drone_id}")),
        ).model_dump_json()

    except Exception as e:
        return {"msg_type": "response", "error": str(e)}


@app.get("/api/v1/video_feed/drone{drone_id}")
async def drone2_feed(drone_id: str):
    return StreamingResponse(
        stream_drone_frames(drone_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/video_feed/drone{drone_id}_annotated")
async def drone1_feed_annotated(drone_id: str):
    return StreamingResponse(
        stream_drone_frames(drone_id, "_annotated"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/video_feed/merged")
async def merged_feed():
    return StreamingResponse(
        stream_drone_frames("merged"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/v1/missions/abort/{mission_id}")
def abort_mission(mission_id: str):
    reg = MissionRegistry()
    mission = reg.get(mission_id)
    if not mission:
        return {"error": "Mission not found"}

    r.delete(f"mission_queue:{mission_id}")
    reg.update_status(mission_id, MissionStatus.ABORTED)
    r.delete(f"drone_active:{mission['drone_id']}")

    return {"status": "aborted", "mission_id": mission_id}


@app.post("/api/v1/missions/return_home/{drone_id}")
def return_home(drone_id: str):
    reg = MissionRegistry()

    # Hitta aktivt mission för drönaren
    all_missions = reg.get_all()
    active_mission = next(
        (
            m
            for m in all_missions
            if m["drone_id"] == drone_id and m["status"] in ["DISPATCHED", "PENDING"]
        ),
        None,
    )

    if active_mission:
        mission_id = active_mission["mission_id"]
        r.delete(f"mission_queue:{mission_id}")
        reg.update_status(mission_id, MissionStatus.ABORTED)
        r.delete(f"drone_active:{drone_id}")

    go_home = json_schemas.GoHomeMessage(
        drone_id=drone_id,
        mission_id=active_mission["mission_id"] if active_mission else "manual",
    )
    r.publish("drone_commands", go_home.model_dump_json())

    return {"status": "returning_home", "drone_id": drone_id}


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

    redis_key = f"frame_drone{drone_id}{frame_type}"
    while True:
        # RTC or capture process is storing a frame in Redis.
        frame_data = await asyncio.to_thread(r.get, redis_key)
        frame = create_not_connected_frame(
            np.zeros((480, 640, 3), dtype=np.uint8), drone_id
        )
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
            telemetry, capabilities = await asyncio.to_thread(
                r.mget, f"telemetry_drone{drone_id}", f"capabilities_drone{drone_id}"
            )
            if telemetry is None:
                frame = create_not_connected_frame(
                    np.zeros((480, 640, 3), dtype=np.uint8), drone_id
                )
            elif (capabilities is not None) and (
                json_schemas.parse_capabilities(capabilities).camera is None
            ):
                frame = create_no_camera_frame(
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
