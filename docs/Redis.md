| Key      | Value | Description |
| ----------- | ----------- | --------------------- |
| frame_drone{id}      | encoded jpeg bytes       | |
| frame_drone_merged   | encoded jpeg bytes        | |
| frame_drone{id}_annotated   | encoded jpeg bytes        | |
| frame_drone{id}\_detections   | DetectionsSchema        | |
| telemetry_drone{id}   | TelemetrySchema        | if it exists, the drone is connected |
| capabilities_drone{id}   | CapabilitiesSchema        | If it exists, the drone is connected    |
| model_drone{id}   | String        | Model name for the drone     |
| watch_area  | Points | polygon area the drone should monitor | 
| watch_area_min_rect  | Points | List of 4 lat,lon corner pairs | 
| watch_area_coverage_area  | Points | List of 4 lat,lon corner pairs | 
| mission_{mission_id}_task_queue  | AnyTaskAction | Queue of remaining tasks for a mission | 
| mission_{mission_id}_state  | Mission | Internal mission state | 
| pending_surveil_mission | { mission_id, drone_id, coverage_corners } | If exists, pending surveil mission |

For schemas, see ```json_schemas.py``` and ```missions.py```