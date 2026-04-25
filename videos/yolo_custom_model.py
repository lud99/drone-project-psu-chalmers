from ultralytics import YOLO
import cv2
import math

# load model
model = YOLO(
    "/home/viggof/Drone-platform-for-safety-in-testing/communication_software/communication_software/best.pt"
)

# instantiate list of object class names
class_names = [
    "person",
    "car",
    "bicycle",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# open video file
vid = cv2.VideoCapture(
    "/home/viggof/Drone-platform-for-safety-in-testing/communication_software/communication_software/Chalmers6.m4v"
)

while True:
    # get return value and frame
    ret, img = vid.read()
    # pass frame to model and get results
    results = model(img, stream=True)

    # coordinates
    for r in results:
        # get all boxes in frame
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", class_names[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, class_names[cls], org, font, font_scale, color, thickness)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
