# Drone communication protocol
All messages are constructed using the schemas in ```json_schemas.py```.

### Drone ID
A drone id is defined to be unique across all connected drones and persistent between restarts of the system. It is the responsibility of the drone adapter to assure these conditions are met. A non-unique id will cause the connection to be closed.

## Connection flow

When the drone adapter connects to the drone communication module, the adapter will do the following:

1. A DroneRegistrationMessage is sent, containing capabilities and telemetry
2. Every X seconds a TelemetryMessage is sent
3. Listen for messages. 
- When a ```TaskMessage``` is received, it should perform the task.
- When a ```AbortTaskMessage``` is received, it should abort the provided task
- When a ```AbortMissionMessage``` is received, it will abort the entire mission and immediately do the action specified by ```next_action```

## Message specifics
If a duration is provided for a task action (not null), it will perform the task for the specified duration and then send a ```TaskEventMessage``` with ```event_type="task_complete"```.
If a duration is not provided, it should perform the task until aborted. It will send the same ```TaskEventMessage```, but instead when the task is started.

After a task has been aborted, a ```TaskEventMessage``` with the appropriate ```event_type``` value depending on the success is sent.

If the task fails for any reason, ```event_type="task_failed"``` is set.