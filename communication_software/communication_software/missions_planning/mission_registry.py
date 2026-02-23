import redis
import json
from .mission_status import MissionStatus


class MissionRegistry:
    def __init__(self):
        self._client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    def store(self, mission):
        self._client.set(mission.mission_id, json.dumps(mission.to_dict()))

    def get(self, mission_id: str):
        data = self._client.get(mission_id)
        return json.loads(data) if data else None

    def get_all(self) -> list:
        keys = self._client.keys("*")
        missions = []
        for key in keys:
            data = self._client.get(key)
            if data:
                missions.append(json.loads(data))
        return missions

    def update_status(self, mission_id: str, status: MissionStatus):
        mission = self.get(mission_id)
        if mission:
            mission["status"] = status.value
            self._client.set(mission_id, json.dumps(mission))

    def remove(self, mission_id: str):
        self._client.delete(mission_id)

    def clear_all(self):
        self._client.flushdb()
