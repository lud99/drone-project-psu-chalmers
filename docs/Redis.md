| Key      | Value | Description |
| ----------- | ----------- | --------------------- |
| frame_drone{id}      | encoded jpeg bytes       | |
| frame_drone_merged   | encoded jpeg bytes        | |
| frame_drone{id}_annotated   | encoded jpeg bytes        | |
| frame_drone{id}\_detections   | DetectionsSchema        | |
| telemetry_drone{id}   | TelemetrySchema        | if it exists, the drone is connected |
| capabilities_drone{id}   | CapabilitiesSchema        | If it exists, the drone is connected    |
| watch_area  | Points | polygon area the drone should monitor | 
| mission_{mission_id}_active_task  | AnyTaskAction | Currently executing task| 
| mission_{mission_id}_task_queue  | AnyTaskAction | Queue of remaining tasks for a mission | 
| mission_{mission_id}  | Mission | Internal mission state | 

For schemas, see ```json_schemas.py``` and ```missions.py```