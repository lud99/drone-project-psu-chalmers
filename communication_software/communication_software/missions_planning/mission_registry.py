import os
import redis
import json
from .mission_status import MissionStatus


class MissionRegistry:
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self._client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def store(self, mission):
        tasks = mission.get_tasks()
        mission_dict = mission.to_dict()
        mission_dict["tasks"] = tasks
        mission_dict["status"] = "DISPATCHED"
        self._client.set(f"mission:{mission.mission_id}", json.dumps(mission_dict))

        for i, task in enumerate(tasks):
            task_message = {
                "msg_type": "task",
                "mission_id": mission.mission_id,
                "drone_id": mission.drone.drone_id,
                "index": i,
                "task_action": task,
            }
            self._client.rpush(
                f"mission_queue:{mission.mission_id}", json.dumps(task_message)
            )

        print(f"Mission {mission.mission_id} sparad med {len(tasks)} tasks i kö")

    def get(self, mission_id: str):
        data = self._client.get(f"mission:{mission_id}")
        return json.loads(data) if data else None

    def get_all(self) -> list:
        keys = self._client.keys("mission:*")
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
            self._client.set(f"mission:{mission_id}", json.dumps(mission))

    def remove(self, mission_id: str):
        self._client.delete(f"mission:{mission_id}")
        self._client.delete(f"mission_queue:{mission_id}")

    def clear_all(self):
        self._client.flushdb()