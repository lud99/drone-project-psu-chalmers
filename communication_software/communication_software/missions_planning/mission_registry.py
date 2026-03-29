import redis
import os
import json
from .mission_status import MissionStatus

import communication_software.common.json_schemas as json_schemas
from communication_software.missions_planning.missions import Mission


class MissionRegistry:
    def __init__(
        self,
    ):
        self.r = redis.Redis(
            host=os.environ.get("REDIS_URL"),
            port=os.environ.get("REDIS_PORT"),
            decode_responses=True,
        )

    def store(self, mission: Mission):
        # tasks = mission.get_tasks()
        mission_dict = mission.to_dict()
        # mission_dict["tasks"] = tasks
        mission_dict["status"] = MissionStatus.DISPATCHED.value
        self.r.set(
            f"mission_{mission.mission_id}_active_task", json.dumps(mission_dict)
        )

        for i, task in enumerate(mission.tasks):
            task_message = json_schemas.TaskMessage(
                drone_id=mission.drone_id,
                mission_id=mission.mission_id,
                index=i,
                task_action=task,
            )
            self.r.rpush(
                f"mission_{mission.mission_id}_task_queue",
                task_message.model_dump_json(),
            )

        print(
            f"Mission {mission.mission_id} saved with {len(mission.tasks)} tasks in queue"
        )

    def get(self, mission_id: str):
        data = self.r.get(f"mission_{mission_id}_state")
        return json.loads(data) if data else None

    def get_all(self) -> list:
        keys = self.r.keys("mission_*_state")
        missions = []
        for key in keys:
            data = self.r.get(key)
            if data:
                missions.append(json.loads(data))
        return missions

    def update_status(self, mission_id: str, status: MissionStatus):
        mission = self.get(mission_id)
        if mission:
            mission["status"] = status.value
            self.r.set(f"mission_{mission_id}_state", json.dumps(mission))

    def remove(self, mission_id: str):
        self.r.delete(f"mission_{mission_id}_state")
        self.r.delete(f"mission_{mission_id}_active_task")
        self.r.delete(f"mission_{mission_id}_task_queue")
