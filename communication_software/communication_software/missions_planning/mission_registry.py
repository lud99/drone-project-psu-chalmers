import redis
import os
import json
from .mission_status import MissionStatus

import communication_software.common.json_schemas as json_schemas
from communication_software.missions_planning.missions import Mission

from communication_software.constants import DRONE_COMMANDS_CHANNEL, DRONE_EVENT_CHANNEL

try:
    r = redis.Redis(
        host=os.environ.get("REDIS_URL"),
        port=os.environ.get("REDIS_PORT"),
        db=0,
        decode_responses=True,
    )
    r.ping()
    print("[Mission Registry] Successfully connected to Redis")
except redis.exceptions.ConnectionError as e:
    print(f"[Mission Registry] Error connecting to Redis: {e}")
    exit()


class MissionRegistry:
    def __init__(self):
        pass

    @staticmethod
    def store(mission: Mission):
        # tasks = mission.get_tasks()
        mission_dict = mission.to_dict()
        r.set(f"mission_{mission.mission_id}_state", json.dumps(mission_dict))

        for i, task in enumerate(mission.tasks):
            task_message = json_schemas.TaskMessage(
                drone_id=mission.drone_id,
                mission_id=mission.mission_id,
                index=i,
                task_action=task,
            )
            r.rpush(
                f"mission_{mission.mission_id}_task_queue",
                task_message.model_dump_json(),
            )

        print(
            f"Mission {mission.mission_id} saved with {len(mission.tasks)} tasks in queue"
        )

    @staticmethod
    def dispatch_mission(mission_id: str):
        mission = json.loads(r.get(f"mission_{mission_id}_state"))

        if MissionRegistry.is_drone_dispatched(mission["drone_id"]):
            raise Exception(
                f"Cannot dispatch drone {mission['drone_id']}, it is already on a mission"
            )

        mission["status"] = MissionStatus.DISPATCHED.value

        r.set(f"mission_{mission_id}_state", json.dumps(mission))

        first_task_raw = r.lpop(f"mission_{mission_id}_task_queue")

        r.publish(DRONE_COMMANDS_CHANNEL, first_task_raw)
        r.publish(DRONE_EVENT_CHANNEL, first_task_raw)

    @staticmethod
    def abort_mission(mission_id: str):
        mission = json.loads(r.get(f"mission_{mission_id}_state"))

        if not MissionRegistry.is_drone_dispatched(mission["drone_id"]):
            raise Exception(
                f"Cannot abort mission {mission_id}, drone {mission['drone_id']} is not on a mission"
            )

        mission["status"] = MissionStatus.ABORTED.value

        r.set(f"mission_{mission_id}_state", json.dumps(mission))
        r.delete(f"mission_{mission_id}_task_queue")

        abort_message = json_schemas.AbortTaskMessage(
            mission_id=mission_id, task_action="all", drone_id=mission["drone_id"]
        )

        r.publish(DRONE_COMMANDS_CHANNEL, abort_message.model_dump_json())
        r.publish(DRONE_EVENT_CHANNEL, abort_message.model_dump_json())

    @staticmethod
    def abort_mission_and_go_home(drone_id: str):
        # Hitta aktivt mission för drönaren
        all_missions = MissionRegistry.get_all()
        active_mission = next(
            (
                m
                for m in all_missions
                if m["drone_id"] == drone_id
                and m["status"]
                in [MissionStatus.DISPATCHED.value, MissionStatus.PENDING.value]
            ),
            None,
        )

        if active_mission:
            MissionRegistry.abort_mission(active_mission["mission_id"])

        go_home = json_schemas.GoHomeMessage(
            drone_id=drone_id,
            mission_id=active_mission["mission_id"] if active_mission else "manual",
        )

        r.publish(DRONE_COMMANDS_CHANNEL, go_home.model_dump_json())
        r.publish(DRONE_EVENT_CHANNEL, go_home.model_dump_json())

    @staticmethod
    def get(mission_id: str):
        data = r.get(f"mission_{mission_id}_state")
        return json.loads(data) if data else None

    @staticmethod
    def get_all() -> list:
        keys = r.keys("mission_*_state")
        missions = []
        for key in keys:
            data = r.get(key)
            if data:
                missions.append(json.loads(data))
        return missions

    @staticmethod
    def update_status(mission_id: str, status: MissionStatus):
        mission = MissionRegistry.get(mission_id)
        if mission:
            mission["status"] = status.value
            r.set(f"mission_{mission_id}_state", json.dumps(mission))

    @staticmethod
    def is_drone_dispatched(drone_id: str) -> bool:
        for mission in MissionRegistry.get_all():
            if mission["drone_id"] == drone_id:
                if mission["status"] == MissionStatus.DISPATCHED.value:
                    return True

        return False

    @staticmethod
    def remove(mission_id: str):
        r.delete(f"mission_{mission_id}_state")
        r.delete(f"mission_{mission_id}_task_queue")
