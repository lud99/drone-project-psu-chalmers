from .drone_sub_systems import Speaker, LED, Camera


class DroneSpecs:
    def __init__(
        self,
        drone_id: str,
        camera: Camera | None = None,
        speaker: Speaker | None = None,
        led: LED | None = None,
        spotlight: bool = False,
    ):
        self.drone_id = drone_id
        self.camera = camera
        self.speaker = speaker
        self.led = led
        self.spotlight = spotlight

    @classmethod
    def from_capabilities(cls, drone_id: str, capabilities) -> "DroneSpecs":
        camera = None
        if capabilities.camera is not None:
            camera = Camera(
                aspect_ratio=capabilities.camera.aspect_ratio,
                horizontal_fov=capabilities.camera.horizontal_fov,
                resolution_height=capabilities.camera.resolution_height,
                resolution_width=capabilities.camera.resolution_width,
            )

        speaker = None
        if capabilities.speaker is not None:
            speaker = Speaker(audio_files=capabilities.speaker.audio_files)

        led = None
        if capabilities.led is not None:
            led = LED(led_types=capabilities.led.types)

        return cls(
            drone_id=drone_id,
            camera=camera,
            speaker=speaker,
            led=led,
            spotlight=capabilities.spotlight,
        )
