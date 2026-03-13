import cv2


def create_not_connected_frame(frame, drone_id):
    cv2.putText(
        frame,
        f"Drone {drone_id}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "not connected",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return frame


def create_no_camera_frame(frame, drone_id):
    cv2.putText(
        frame,
        f"Drone {drone_id}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "no camera connected",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return frame


def create_error_frame(frame, drone_id, error):
    cv2.putText(
        frame,
        f"Drone {drone_id}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"{error}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    return frame
