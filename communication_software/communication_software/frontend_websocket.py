import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
from datetime import datetime
import redis
import redis.exceptions
import numpy as np

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


# WebSocket Endpoints
@app.websocket("/api/v1/ws/drone")
async def drone_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Drone client connected")
    try:
        while True:
            processed_data_for_cycle = {}
            drone_id = 0
            redis_key_list = r.scan_iter(match="position_drone*")
            # print(f"redis keys: {redis_key_list}")
            for redis_key in redis_key_list:
                drone_id += 1
                json_data_string = None
                try:
                    json_data_string = r.get(redis_key)

                    if json_data_string:
                        try:
                            data_dict = json.loads(json_data_string)
                        except json.JSONDecodeError as e:
                            print(
                                f"Error decoding JSON for {redis_key}: {e}. Data: '{json_data_string}'"
                            )
                            # Optionally use last known good data if available
                            if drone_id in atos.drone_data:
                                processed_data_for_cycle[drone_id] = atos.drone_data[
                                    drone_id
                                ]
                            continue  # Skip update for this drone this cycle

                        # 3. Safely get values
                        lat = data_dict.get("latitude")
                        lng = data_dict.get("longitude")
                        alt = data_dict.get("altitude")
                        speed = data_dict.get("speed")
                        battery_percent = data_dict.get("batteryPercent")

                        if (
                            lat is None
                            or lng is None
                            or alt is None
                            or speed is None
                            or battery_percent is None
                        ):
                            print(
                                f"Warning: Missing position, battery or speed data fields in {redis_key}. Found: {data_dict}"
                            )
                            if drone_id in atos.drone_data:
                                processed_data_for_cycle[drone_id] = atos.drone_data[
                                    drone_id
                                ]
                            continue

                        if drone_id not in atos.drone_data:
                            atos.drone_data[drone_id] = {"battery": 0}
                        elif "battery" not in atos.drone_data[drone_id]:
                            atos.drone_data[drone_id]["battery"] = 0

                        atos.drone_data[drone_id].update(
                            {
                                "lat": lat,
                                "lng": lng,
                                "alt": alt,
                                "speed": speed,
                                "battery": battery_percent,
                            }
                        )
                        processed_data_for_cycle[drone_id] = atos.drone_data[drone_id]

                    else:
                        print(f"No data found in Redis for key: {redis_key}")
                        if drone_id in atos.drone_data:
                            processed_data_for_cycle[drone_id] = atos.drone_data[
                                drone_id
                            ]

                except redis.exceptions.RedisError as e:
                    print(f"Redis error while getting/processing {redis_key}: {e}")
                    if drone_id in atos.drone_data:
                        processed_data_for_cycle[drone_id] = atos.drone_data[drone_id]
                except Exception as e:
                    print(
                        f"An unexpected error occurred processing drone {drone_id}: {e}"
                    )
                    if drone_id in atos.drone_data:
                        processed_data_for_cycle[drone_id] = atos.drone_data[drone_id]

                await websocket.send_json(
                    {
                        "drone_id": drone_id,
                        **atos.drone_data[drone_id],
                        "anomaly": atos.anomalies,
                    }
                )
                # print(f"Sent data to client: {processed_data_for_cycle}") # Optional: Verbose logging
            await asyncio.sleep(0.5)

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


@app.get("/api/v1/video_feed/merged")
async def merged_feed():
    return StreamingResponse(
        stream_drone_frames("_merged_annotated"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
async def stream_drone_frames(drone_id: str):

    redis_key = f"frame_drone{drone_id}"
    while True:
        # RTC or capture process is storing a frame in Redis.
        frame_data = await asyncio.to_thread(r.get, redis_key)
        if frame_data:
            # Might need to adjust this if you're using base64 or another format.
            frame_array = np.frombuffer(frame_data.encode("latin1"), dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                # If decoding fails, fall back to a dummy image.
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"Drone {drone_id}: invalid frame",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        else:
            # No frame found in Redis, so generate a dummy frame.
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Drone {drone_id} not connected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
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
