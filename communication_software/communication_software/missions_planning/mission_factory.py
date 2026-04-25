from .missions import (
    GotoAndAudio,
    GotoAndBlink,
    GotoAndSurveil,
    GotoAndIlluminate,
    GotoOnly,
    Mission,
)

import communication_software.common.json_schemas as json_schemas


class MissionFactory:
    MISSION_TYPES = [
        GotoAndAudio,
        GotoAndBlink,
        GotoAndSurveil,
        GotoAndIlluminate,
        GotoOnly,
    ]

    @staticmethod
    def get_feasible_missions(
        drone_id: str,
        coordinates: json_schemas.GoToParams,
        capabilities: json_schemas.Capabilities,
    ) -> list[Mission]:
        return [
            mission_class(drone_id, capabilities, coordinates)
            for mission_class in MissionFactory.MISSION_TYPES
            if mission_class(drone_id, capabilities, coordinates).can_execute()
        ]
