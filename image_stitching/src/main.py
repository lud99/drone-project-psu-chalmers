import numpy as np
import cv2
import imutils
import threading
from queue import Queue
from ultralytics import YOLO
import supervision.detection.core as sv
from annotator import Annotator
import coordinate_mapping
import redis
import asyncio
import os
import torch
import io
import json

from common.frame_utils import create_not_connected_frame, create_error_frame
import common.json_schemas as json_schemas

if torch.cuda.is_available():
    print(
        f"[INFO] PyTorch CUDA detected. Available devices: {torch.cuda.device_count()}"
    )
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


async def consume_async_generator(gen, queue, stop_event, drone_id):
    """
    Consume an asynchronous generator and push frames into a queue.

    Args:
        gen (AsyncGenerator): The async generator yielding frames.
        queue (Queue): Queue to store the frames.
        stop_event (threading.Event): Event to stop the loop.
        drone_id: Id of drone, used for error logging.
    """
    async for frame, capabilities, telemetry in gen:
        if stop_event.is_set():
            break

        # For debug
        # print(capabilities, telemetry)

        if not capabilities:
            print(f"Capabilities for drone {drone_id} not found, not doing detection")
            await asyncio.sleep(0.033)  # wait a bit before the next attempt
            continue
        if not telemetry:
            print(f"Telemetry for drone {drone_id} not found, not doing detection")
            await asyncio.sleep(0.033)  # wait a bit before the next attempt
            continue

        queue.put((frame, capabilities, telemetry))
    queue.put(None)  # Signal the end of the stream


def detect_objects(frame: np.ndarray) -> sv.Detections:
    """
    Run YOLO for object detection on a frame.

    Args:
        frame (np.ndarray): Input image frame.

    Returns:
        sv.Detections: Detected objects.
    """
    results = model.track(frame, persist=True, conf=0.10, imgsz=448)
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


async def set_frame(
    drone_id: str, img: np.ndarray, detections: json_schemas.Detections
) -> None:
    """
    Store a frame in Redis as JPEG.

    Args:
        drone_id: id of drone.
        img (np.ndarray): Image to store.
        detections: List of detections
    """
    try:
        # Convert to JPEG buffer
        ret, buffer = cv2.imencode(".jpg", img)
        if ret:
            frame_str = buffer.tobytes().decode("latin1")
            # Redis pipeline to save frames and set TTL
            redis_key_frame = f"frame_drone{drone_id}_annotated"
            redis_key_detections = f"frame_drone{drone_id}_detections"
            expiration = 60

            with r.pipeline() as pipe:
                pipe.set(redis_key_frame, frame_str)  # Save image
                pipe.expire(redis_key_frame, expiration)

                # Save detections as a json string
                pipe.set(redis_key_detections, detections.model_dump_json())
                pipe.expire(redis_key_detections, expiration)
                pipe.execute()  # Execute both commands together
        else:
            print("Failed to encode frame!")
    except Exception as e:
        print(f"Exception in set_frame: {e}")


### MERGE STREAMS ###
async def stream_drone_frames(drone_id: str):
    """
    Read frames from Redis, decode and yield raw JPEG bytes.

    Args:
        drone_id (int): Identifier for the drone.

    Yields:
        bytes: JPEG encoded frame.
        capabilities: JSON string
        telemetry: JSON string
    """
    redis_keys = [
        f"frame_drone{drone_id}",
        f"capabilities_drone{drone_id}",
        f"telemetry_drone{drone_id}",
    ]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Pre-create dummy frame

    while True:
        frame_to_encode = None
        # Retrieve a frame from Redis
        # Ensure r.get runs in a thread as it can block
        frame_data, capabilities, telemetry = await asyncio.to_thread(
            r.mget, redis_keys
        )

        try:
            capabilities = (
                json_schemas.parse_capabilities(capabilities) if capabilities else {}
            )
            telemetry = json_schemas.parse_telemetry(telemetry) if telemetry else {}
        except Exception as e:
            capabilities, telemetry = None, None
            print(
                f"Failed to parse capabilities or telemetry in stream_drone_frames: {e}"
            )

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

                    frame_to_encode = create_error_frame(
                        dummy_frame.copy(), drone_id, "invalid frame data"
                    )

            except Exception as e:
                print(
                    f"[ERROR] Error processing frame data from Redis for drone {drone_id}: {e}"
                )
                frame_to_encode = create_error_frame(
                    dummy_frame.copy(), drone_id, "error reading"
                )

        else:
            # If no frame exists in Redis, prepare a dummy image for encoding
            frame_to_encode = create_not_connected_frame(dummy_frame.copy(), drone_id)

        # Convert selected frame (real or dummy) to JPEG bytes
        ret, buffer = cv2.imencode(".jpg", frame_to_encode)
        if not ret:
            # If encoding fails, log and skip (yield None or wait?)
            # Yielding None might be better handled by the consumer
            print(f"[ERROR] Failed to encode frame to JPEG for drone {drone_id}")
            await asyncio.sleep(0.033)  # wait a bit before the next attempt
            continue  # Skip this iteration

        # *** FIX: Yield ONLY the raw JPEG bytes ***
        yield buffer.tobytes(), capabilities, telemetry

        await asyncio.sleep(0.033)  # Approximately 30fps


async def merge_and_annotate_stream(drone_ids: tuple[str, str]) -> None:
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

    # Start async tasks to consume frames
    asyncio.create_task(
        consume_async_generator(frame_left, left_queue, stop_event, drone_ids[0])
    )
    asyncio.create_task(
        consume_async_generator(frame_right, right_queue, stop_event, drone_ids[1])
    )

    try:
        while True:
            (
                left_frame_data,
                left_capabilities,
                left_telemetry,
            ) = await asyncio.to_thread(left_queue.get)

            (
                right_frame_data,
                right_capabilities,
                right_telemetry,
            ) = await asyncio.to_thread(right_queue.get)

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
            right_fixed = cv2.resize(
                right[:, overlap_width:], (frame_width, frame_height)
            )
            stitched_frame[:, frame_width:] = right_fixed

            # Get telemetry and specifications
            if not left_capabilities.camera:
                print("[ERROR] No left camera specification present!")
                continue
            if not right_capabilities.camera:
                print("[ERROR] No right camera specification present!")
                continue

            # TODO: How does horizontal and vertical fov differ? what fov should be used?
            left_fov = left_capabilities.camera.horizontal_fov
            left_alt = left_telemetry.alt
            left_location = (left_telemetry.lat, left_telemetry.lon)
            right_fov = right_capabilities.camera.horizontal_fov
            right_alt = right_telemetry.alt
            right_location = (right_telemetry.lat, right_telemetry.lon)

            if left_fov != right_fov:
                print("[ERROR] Fov mismatch!")
                continue
            if left_alt != right_alt:
                print("[ERROR] Altidude mismatch!")
                continue

            (annotated_frame, detections) = detect_and_annotate_image(
                stitched_frame,
                [left_location, right_location],
                left_fov,
                left_alt,
                (frame_width, frame_height),
                [id1, id2],
            )

            # Send the composite and annotated image to Redis
            annotated_frame = cv2.resize(annotated_frame, (640, 380))
            print(annotated_frame.shape)
            await set_frame("_merged", annotated_frame, detections)
            print("Setting stitched video frame in Redis")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()


async def annotate_stream(drone_id: str) -> None:
    """
    Merge video streams from two drones, detect objects, and save annotated output.

    Args:
        drone_ids (tuple): Tuple containing two drone IDs.
    """

    # Create queues for frames
    queue = Queue()
    stop_event = threading.Event()

    # Create async generators to consume drone streams
    full_frame = stream_drone_frames(drone_id)

    # Start async tasks to consume frames
    asyncio.create_task(
        consume_async_generator(full_frame, queue, stop_event, drone_id)
    )

    try:
        while True:
            await insert_dummy_telemetry_and_capabilities(drone_id)

            frame_data, capabilities, telemetry = await asyncio.to_thread(queue.get)

            if frame_data is None:
                print("[INFO] End of videostream")
                stop_event.set()
                break

            # Decode frames to OpenCV
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            pixel_array = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            (height, width, _channels) = pixel_array.shape

            # Check if decoding fails
            if pixel_array is None:
                print("[INFO] Decoding of image failed")
                continue  # Skip if decoding fails

            # Get telemetry and specifications
            if not capabilities.camera:
                print("[ERROR] No camera specification present!")
                continue

            # TODO: How does horizontal and vertical fov differ? what fov should be used?
            fov = capabilities.camera.horizontal_fov
            alt = telemetry.alt
            location = (telemetry.lat, telemetry.lon)

            (annotated_frame, detections) = detect_and_annotate_image(
                pixel_array, [location], fov, alt, (width, height), [drone_id]
            )

            # Send the results to redis
            await set_frame(f"{drone_id}", annotated_frame, detections)
            print(f"Setting video frame and detections in Redis for drone {drone_id}")
    finally:
        stop_event.set()


def detect_and_annotate_image(
    pixel_array: np.ndarray,
    drone_coordinates: list[tuple[float, float]],
    fov: float,
    altitude: float,
    image_size: tuple[int, int],
    drone_ids: list[str],
) -> tuple[np.ndarray, json_schemas.Detections]:
    detections = detect_objects(pixel_array)

    detections_complete = json_schemas.Detections(root=[])

    detection_gps_positions = []
    if detections.tracker_id is not None:  # Check if tracker_id exists
        for i, box in enumerate(detections.xyxy):
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)

            # Calculate GPS from camera position.
            # It is a bit confusing about the resolution passed here. The image is downsampled in when running the detection,
            # but the detection positions are upscaled to the original resolution.
            # That's why it's valid to pass the original image size as the resolution.
            gps_positions = [
                coordinate_mapping.pixel_to_gps(
                    (x_center, y_center),
                    drone_coordinate,
                    altitude,
                    fov=fov,
                    resolution=image_size,
                )
                for drone_coordinate in drone_coordinates
            ]

            if len(gps_positions) == 1:
                detection_gps_positions.append(gps_positions[0])
            elif len(gps_positions) == 2:
                # Weighted average calculation
                best_gps = get_weighted_gps(
                    x_center, image_size[0] * 2, gps_positions[0], gps_positions[1]
                )
                detection_gps_positions.append(best_gps)

        # ---- SHOW RESULTS ----
        labels = [
            f"ID: {d} GPS: {round(g[0], 6)}, {round(g[1], 6)}"
            for d, g in zip(detections.tracker_id, detection_gps_positions)
        ]
        position_labels = [f"({int(d[0])}, {int(d[1])})" for d in detections.xyxy]

        for class_id, gps_position in zip(detections.class_id, detection_gps_positions):
            detections_complete.root.append(
                json_schemas.SingleDetection(
                    class_name=model.names[class_id],
                    gps_position=gps_position,
                    drone_ids=drone_ids,
                )
            )

        annotator = Annotator()
        annotated_frame = annotator.annotate_frame(
            frame=pixel_array,
            detections=detections,
            labels=labels,
            position_labels=position_labels,
        )
    else:
        annotated_frame = pixel_array  # no detection, only show image

    return (annotated_frame, detections_complete)


async def insert_dummy_telemetry_and_capabilities(drone_id: str) -> None:
    """
    Store a frame in Redis as JPEG.

    Args:
        drone_id: id of drone.
        img (np.ndarray): Image to store.
        detections: List of detections
    """
    try:
        # Redis pipeline to save frames and set TTL
        redis_key_telemetry = f"telemetry_drone{drone_id}"
        redis_key_capabilities = f"capabilities_drone{drone_id}"

        telemetry = dict(
            [
                ("lat", 57.6900),
                ("lon", 11.9800),
                ("alt", 30),
                ("heading", 10),
                ("speed", 5.0),
                ("battery_percent", 50),
            ]
        )

        capabilities = dict(
            [
                (
                    "camera",
                    dict(
                        [
                            ("aspect_ratio", 16.0 / 9.0),
                            ("horizontal_fov", 83.0),
                            ("resolution_width", 1920),
                            ("resolution_height", 1080),
                        ]
                    ),
                ),
                ("led", None),
                ("spotlight", False),
                ("speaker", False),
                ("max_speed", 15.0),
            ]
        )

        expiration = 60
        with r.pipeline() as pipe:
            pipe.set(redis_key_telemetry, json.dumps(telemetry))
            pipe.expire(redis_key_telemetry, expiration)

            pipe.set(redis_key_capabilities, json.dumps(capabilities))
            pipe.expire(redis_key_capabilities, expiration)

            pipe.execute()  # Execute both commands together
            print("Added test capabilities and telemetry data to redis")
    except Exception as e:
        print(f"Exception in test_insert_dummy_telemetry_and_capabilities: {e}")


async def adding_redis_frame(img, drone_id):
    buf = io.BytesIO()

    # 2. Save the image to the stream, specifying the format
    img.save(buf, format="JPEG")

    # 3. Retrieve the byte data

    try:
        # Convert to JPEG buffer
        jpeg_bytes = buf.getvalue()
        if True:
            frame_str = jpeg_bytes.decode("latin1")
            # Redis pipeline to save frames and set TTL
            redis_key_frame = f"frame_drone{drone_id}"
            expiration = 60

            with r.pipeline() as pipe:
                pipe.set(redis_key_frame, frame_str)  # Save image
                pipe.expire(redis_key_frame, expiration)
                pipe.execute()  # Execute both commands together
                print("Adding test frame to redis")
    except Exception as e:
        print(f"Exception in set_frame: {e}")


async def test_detect_and_annotate_image():
    from PIL import Image

    # Load and force RGB
    img = Image.open("./test2.jpg").convert("RGB")

    # Convert to NumPy
    pixel_array = np.array(img, dtype=np.uint8)

    (annotated_frame, detections) = detect_and_annotate_image(
        pixel_array, [(57.6900, 11.9800)], 83, 30, img.size, ["1"]
    )

    print(model.names)
    print(detections.model_dump_json())

    result = Image.fromarray((annotated_frame).astype(np.uint8))
    result.save("./out.jpeg")


async def test_stream_frame():
    from PIL import Image

    # Load and force RGB
    img = Image.open("./test2.jpg").convert("RGB")

    await insert_dummy_telemetry_and_capabilities("1")

    await adding_redis_frame(img, "1")

    await annotate_stream("1")

    # Don't know how to test reading the redis data here
    # But modifying the code to read detections, we can see that it works


async def test_stream_merge_frame():
    from PIL import Image

    # Load and force RGB
    img = Image.open("./test2.jpg").convert("RGB")

    await insert_dummy_telemetry_and_capabilities("1")
    await insert_dummy_telemetry_and_capabilities("2")

    await adding_redis_frame(img, "1")
    await adding_redis_frame(img, "2")

    await merge_and_annotate_stream(("1", "2"))

    # Don't know how to test reading the redis data here
    # But modifying the code to read detections, it works


async def main() -> None:
    print("[INFO] Startar drönarvideoprocessorer...")

    # await test_stream_merge_frame()
    # await test_detect_and_annotate_image()

    # await annotate_stream(1)

    await asyncio.gather(annotate_stream("1"), annotate_stream("2"))

    # await merge_stream((1, 2))  # Call with drone ID 1 and 2


if __name__ == "__main__":
    asyncio.run(main())
