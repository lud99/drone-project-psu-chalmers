| Key      | Value |
| ----------- | ----------- |
| frame_drone{id}      | encoded jpeg bytes       |
| frame_drone_merged   | encoded jpeg bytes        |
| frame_drone{id}_annotated   | encoded jpeg bytes        |
| frame_drone{id}\_detections   | DetectionsSchema        |
| telemetry_drone{id}   | TelemetrySchema        |
| capabilities_drone{id}   | CapabilitiesSchema        |

For schemas, see ```json_schemas.py```


```
DetectionsSchema:
[
    {
        gps_position: [double, double],
        class_name: string
    }
]

```