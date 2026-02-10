import numpy as np
import cv2
import imutils
import threading
from queue import Queue
from ultralytics import YOLO
import supervision.detection.core as sv
from annotator import Annotator
import coordinateMapping
import redis
import asyncio
import os
import torch


if torch.cuda.is_available():
    print(
        f"[INFO] PyTorch CUDA detected. Available devices: {torch.cuda.device_count()}"
    )
else:
    print("[INFO] PyTorch CUDA not detected. YOLO will use CPU.")

# Global YOLO model
model = YOLO("models/best.pt")

redis_url = os.environ.get("REDIS_URL", "localhost")
# Redis connection (create a Redis client if it doesn't exist)
r = redis.StrictRedis(host=redis_url, port=6379, db=0, decode_responses=True)

## ---- HELPER FUNCTIONS ----


async def consume_async_generator(gen, queue, stop_event):
    """
    Consume an asynchronous generator and push frames into a queue.

    Args:
        gen (AsyncGenerator): The async generator yielding frames.
        queue (Queue): Queue to store the frames.
        stop_event (threading.Event): Event to stop the loop.
    """
    async for frame in gen:
        if stop_event.is_set():
            break
        queue.put(frame)
    queue.put(None)  # Signal the end of the stream


def detect_objects(frame: np.ndarray) -> sv.Detections:
    """
    Run YOLO for object detection on a frame.

    Args:
        frame (np.ndarray): Input image frame.

    Returns:
        sv.Detections: Detected objects.
    """
    results = model.track(frame, persist=True, conf=0.10, imgsz=1280)
    detections = sv.Detections.from_ultralytics(results[0])
    return detections


def get_weighted_gps(
    pixel_x: int,
    frame_width: int,
    left_gps: tuple[float, float],
    right_gps: tuple[float, float],
) -> tuple[float, float]:
    """
    Calculate a weighted GPS position based on object position in the image.

    Args:
        pixel_x (int): X coordinate of the object in pixels.
        frame_width (int): Width of the image frame.
        left_gps (tuple): GPS coordinate of the left camera.
        right_gps (tuple): GPS coordinate of the right camera.

    Returns:
        tuple: Weighted GPS position (latitude, longitude).
    """
    alpha = pixel_x / frame_width
    gps_lat = left_gps[0] * (1 - alpha) + right_gps[0] * alpha
    gps_lon = left_gps[1] * (1 - alpha) + right_gps[1] * alpha
    return (gps_lat, gps_lon)


async def set_frame(img: np.ndarray) -> None:  # Receives a frame and sends it to Redis
    """
    Store a frame in Redis as JPEG.

    Args:
        img (np.ndarray): Image to store.
    """
    try:
        # Convert to JPEG buffer
        ret, buffer = cv2.imencode(".jpg", img)
        if ret:
            frame_str = buffer.tobytes().decode("latin1")
            # Redis pipeline to save frames and set TTL
            redis_key = f"frame_drone_merged"
            with r.pipeline() as pipe:
                pipe.set(redis_key, frame_str)  # Save image
                pipe.expire(redis_key, 60)  # Set expiration (60 seconds)
                pipe.execute()  # Execute both commands together
        else:
            print(f"Misslyckades att koda sammansatta frames")
    except Exception as e:
        print(f"Fel i set_frame: {e}")


### MERGE STREAMS ###
async def stream_drone_frames(drone_id: int):
    """
    Read frames from Redis, decode and yield raw JPEG bytes.

    Args:
        drone_id (int): Identifier for the drone.

    Yields:
        bytes: JPEG encoded frame.
    """
    redis_key = f"frame_drone{drone_id}"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Pre-create dummy frame

    while True:
        frame_to_encode = None
        # Retrieve a frame from Redis
        # Ensure r.get runs in a thread as it can block
        frame_data = await asyncio.to_thread(r.get, redis_key)

        if frame_data:
            try:
                # Directly use bytes if decode_responses=False in Redis client
                # If decode_responses=True, you stored latin1 string, so encode back
                # Assuming decode_responses=True based on your original code:
                frame_bytes = frame_data.encode("latin1")

                # If you change Redis client to decode_responses=False:
                # frame_bytes = frame_data # frame_data would already be bytes

                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame_to_encode = frame
                else:
                    # If decoding fails, prepare a dummy image for encoding
                    print(
                        f"[WARNING] Failed to decode frame from Redis for drone {drone_id}"
                    )
                    dummy_frame_copy = dummy_frame.copy()  # Use a copy
                    cv2.putText(
                        dummy_frame_copy,
                        f"Drone {drone_id}: invalid frame data",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    frame_to_encode = dummy_frame_copy

            except Exception as e:
                print(
                    f"[ERROR] Error processing frame data from Redis for drone {drone_id}: {e}"
                )
                dummy_frame_copy = dummy_frame.copy()
                cv2.putText(
                    dummy_frame_copy,
                    f"Drone {drone_id}: error reading",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                frame_to_encode = dummy_frame_copy

        else:
            # If no frame exists in Redis, prepare a dummy image for encoding
            dummy_frame_copy = dummy_frame.copy()
            cv2.putText(
                dummy_frame_copy,
                f"Drone {drone_id} not connected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            frame_to_encode = dummy_frame_copy

        # Convert selected frame (real or dummy) to JPEG bytes
        ret, buffer = cv2.imencode(".jpg", frame_to_encode)
        if not ret:
            # If encoding fails, log and skip (yield None or wait?)
            # Yielding None might be better handled by the consumer
            print(f"[ERROR] Failed to encode frame to JPEG for drone {drone_id}")
            await asyncio.sleep(0.033)  # wait a bit before the next attempt
            continue  # Skip this iteration

        # *** FIX: Yield ONLY the raw JPEG bytes ***
        yield buffer.tobytes()

        await asyncio.sleep(0.033)  # Approximately 30fps


async def merge_stream(drone_ids: tuple[int, int]) -> None:
    """
    Merge video streams from two drones, detect objects, and save annotated output.

    Args:
        drone_ids (tuple): Tuple containing two drone IDs.
    """
    id1, id2 = drone_ids

    # Create queues for frames
    left_queue, right_queue = Queue(), Queue()
    stop_event = threading.Event()

    # Create async generators to consume drone streams
    frame_left = stream_drone_frames(id1)
    frame_right = stream_drone_frames(id2)

    # Standard frame size
    frame_width = 600
    frame_height = None
    overlap_width = int(frame_width * 0.495)  # Adjust if necessary

    # Camera GPS positions and altitude
    left_camera_location = (57.6900, 11.9800)  # example coordinates
    right_camera_location = (57.6901, 11.9802)  # example coordinates
    altitude = 30  # example height
    fov = 83.0
    resolution = (1920, 1080)

    # Start async tasks to consume frames
    asyncio.create_task(consume_async_generator(frame_left, left_queue, stop_event))
    asyncio.create_task(consume_async_generator(frame_right, right_queue, stop_event))

    try:
        while True:
            left_frame_data = await asyncio.to_thread(left_queue.get)
            right_frame_data = await asyncio.to_thread(right_queue.get)

            if left_frame_data is None or right_frame_data is None:
                print("[INFO] Slut på videoström.")
                stop_event.set()
                break
            # Decode frames to OpenCV

            left_frame_array = np.frombuffer(left_frame_data, dtype=np.uint8)
            left = cv2.imdecode(left_frame_array, cv2.IMREAD_COLOR)

            right_frame_array = np.frombuffer(right_frame_data, dtype=np.uint8)
            right = cv2.imdecode(right_frame_array, cv2.IMREAD_COLOR)

            # check if decoding fails
            if left is None or right is None:
                print("[INFO] Left or right image is None")
                print(f"left: {left}")
                print(f"right: {right}")
                continue  # Skip if decoding fails
            # Scale images
            if frame_height is None:
                frame_height = int(left.shape[0] * (frame_width / left.shape[1]))

            left = imutils.resize(left, width=frame_width)
            right = imutils.resize(right, width=frame_width)
            right = cv2.resize(right, (frame_width, frame_height))
            left = cv2.resize(left, (frame_width, frame_height))

            # Create blank image for stitching
            stitched_frame = np.zeros((frame_height, frame_width * 2, 3), dtype="uint8")
            stitched_frame[:, :frame_width] = left

            # Smooth transition between left and right image
            for i in range(overlap_width):
                alpha = i / overlap_width
                stitched_frame[:, frame_width - overlap_width + i] = cv2.addWeighted(
                    left[:, frame_width - overlap_width + i],
                    1 - alpha,
                    right[:, i],
                    alpha,
                    0,
                )

            # adjust the right image and put it to the end
            right_fixed = cv2.resize(
                right[:, overlap_width:], (frame_width, frame_height)
            )
            stitched_frame[:, frame_width:] = right_fixed

            # ---- OBJECT DETECTION ----
            detections = detect_objects(stitched_frame)

            # ---- GPS-CALCULATION ----
            gps_positions = []
            if detections.tracker_id is not None:  # Check if tracker_id exists
                for i, box in enumerate(detections.xyxy):
                    x_center = int((box[0] + box[2]) / 2)
                    y_center = int((box[1] + box[3]) / 2)

                    # Calculate GPS from left and right camera
                    gps_left = coordinateMapping.pixel_to_gps(
                        (x_center, y_center),
                        left_camera_location,
                        altitude,
                        fov=fov,
                        resolution=resolution,
                    )
                    gps_right = coordinateMapping.pixel_to_gps(
                        (x_center, y_center),
                        right_camera_location,
                        altitude,
                        fov=fov,
                        resolution=resolution,
                    )

                    # Weighted average calculation
                    best_gps = get_weighted_gps(
                        x_center, frame_width * 2, gps_left, gps_right
                    )
                    gps_positions.append(best_gps)

                # ---- SHOW RESULTS ----
                labels = [
                    f"ID: {d} GPS: {round(g[0], 6)}, {round(g[1], 6)}"
                    for d, g in zip(detections.tracker_id, gps_positions)
                ]
                position_labels = [
                    f"({int(d[0])}, {int(d[1])})" for d in detections.xyxy
                ]

                annotator = Annotator()  # Create an annotator
                annotated_frame = annotator.annotate_frame(
                    frame=stitched_frame,
                    detections=detections,
                    labels=labels,
                    positionLabels=position_labels,
                )
            else:
                annotated_frame = (
                    stitched_frame  # no detection, only show composite image
                )

            # Send the composite and annotated image to Redis
            annotated_frame = cv2.resize(annotated_frame, (640, 380))
            print(annotated_frame.shape)
            await set_frame(annotated_frame)
            print("Setting stitched video frame in Redis")
            # Show the annotated image in OpenCV
            # cv2.imshow("Stitched Frame with Detections", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        stop_event.set()
        cv2.destroyAllWindows()


async def main() -> None:
    print("[INFO] Startar drönarvideoprocessorer...")
    while True:
        await merge_stream((1, 2))  # Call with drone ID 1 and 2


if __name__ == "__main__":
    asyncio.run(main())
