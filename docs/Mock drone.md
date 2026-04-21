# Testing with a mock drone

To test the semi-full functionality of the system without a drone connected, a mock drone can be used. A simple one is located at ```mock_drone/main.py```.
To run the mock drone websockets have to be installed: ```pip install websockets```.
In the repo root, run ```python -m mock_drone.main```.

Multiple drones can be started, but all drones except the first one needs to be passed a unique id as the first parameter. replace <drone_id> with an id. A video will be streamed. If the second argument is ```drain```, the battery will be drained. If not provided it has infinite battery.
 ```python -m mock_drone.main <drone_id> <drain?>```


The drone registers itself with full specs, and streams video if a video file is provided. It sends telemetry messages continuously.
It answers to tasks and waits the appropriate time, sends task event messages once a task is completed and supports aborting tasks.
