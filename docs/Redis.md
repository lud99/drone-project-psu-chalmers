| Key      | Value |
| ----------- | ----------- |
| frame_drone{id}      | encoded jpeg bytes       |
| frame_drone_merged   | encoded jpeg bytes        |
| frame_drone{id}_annotated   | encoded jpeg bytes        |
| frame_drone{id}\_detections   | DetectionsSchema        |
| telemetry_drone{id}   | TelemetrySchema        |
| capabilities_drone{id}   | CapabilitiesSchema        |

```
TelemetrySchema:
{
    lat: double,
    lon: double,
    altitude: float,
    heading: int,
    speed: float,
    batteryPercent: int
}
```

```
CapabilitiesSchema:
{
    camera: null or {
        aspect_ratio: double,
        horizontal_fov: double,
        resolution: [int, int]
    }
    // More in the future?
}
```

```
DetectionsSchema:
[
    {
        gps_position: [double, double],
        class_name: string
    }
]

```