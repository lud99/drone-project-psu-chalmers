from .drone_specs import DroneSpecs

MOCK_DRONES = [
    DroneSpecs(
        id="dji-01",
        model="DJI Mavic 2 Enterprise",
        speaker=True,
        lights=False,
        camera=True,
        aspect_ratio="16:9",
        horizontal_fov=84.0,
        resolution="1080p",
    ),
    DroneSpecs(
        id="dji-02",
        model="DJI Mavic 2 Enterprise",
        speaker=True,
        lights=False,
        camera=True,
        aspect_ratio="16:9",
        horizontal_fov=84.0,
        resolution="1080p",
    ),
    DroneSpecs(
        id="mavlink-01",
        model="Holybro Pixhawk 6X",
        speaker=False,
        lights=True,
        camera=False,
    ),
]
