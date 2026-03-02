from .drone_specs import DroneSpecs
from .missions import GotoAndSound, GotoAndBlink, GotoOnly, Mission


class MissionFactory:
    MISSION_TYPES = [GotoAndSound, GotoAndBlink, GotoOnly]

    @staticmethod
    def get_feasible_missions(coordinates: dict, drone: DroneSpecs) -> list[Mission]:
        return [
            mission_class(drone, coordinates)
            for mission_class in MissionFactory.MISSION_TYPES
            if mission_class(drone, coordinates).can_execute()
        ]
