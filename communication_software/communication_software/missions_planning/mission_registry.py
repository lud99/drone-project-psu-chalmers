import redis
import os
import json
from .mission_status import MissionStatus, TaskStatus

import communication_software.common.json_schemas as json_schemas
from communication_software.missions_planning.missions import GoHome, Mission

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
    def dispatch_mission(mission_id: str) -> dict:
        mission = json.loads(r.get(f"mission_{mission_id}_state"))

        if MissionRegistry.is_drone_dispatched(mission["drone_id"]):
            raise Exception(
                f"Cannot dispatch drone {mission['drone_id']}, it is already on a mission"
            )

        # if the drone has a waiting mission, only dispatch the mission_id if the waiting mission is the same as the mission_id, otherwise raise an exception
        if MissionRegistry.is_drone_waiting(mission["drone_id"]):
            waiting_mission = next(
                (
                    m
                    for m in MissionRegistry.get_all()
                    if m["drone_id"] == mission["drone_id"]
                    and m["status"] == MissionStatus.WAITING.value
                ),
                None,
            )
            if waiting_mission and waiting_mission["mission_id"] != mission_id:
                raise Exception(
                    f"Cannot dispatch mission {mission_id} for drone {mission['drone_id']}, it has a waiting mission {waiting_mission['mission_id']}"
                )

        mission["status"] = MissionStatus.DISPATCHED.value

        r.set(f"mission_{mission_id}_state", json.dumps(mission))

        first_task_raw = r.lpop(f"mission_{mission_id}_task_queue")

        r.publish(DRONE_COMMANDS_CHANNEL, first_task_raw)
        r.publish(DRONE_EVENT_CHANNEL, first_task_raw)

        return mission

    @staticmethod
    def abort_mission(mission_id: str, exception_on_not_dispatched: bool = True):
        mission = json.loads(r.get(f"mission_{mission_id}_state"))

        # This should actually only be to check if the drone is dispatched, but looks complicated..
        if not MissionRegistry.is_drone_dispatched(mission["drone_id"]) and (
            not MissionRegistry.is_drone_waiting(mission["drone_id"])
        ):
            if exception_on_not_dispatched:
                raise Exception(
                    f"Cannot abort mission {mission_id}, drone {mission['drone_id']} is not on a mission"
                )
            else:
                print(f"Mission {mission_id} is not dispatched, skipping abort")

        MissionRegistry.remove(mission_id)

        abort_message = json_schemas.AbortTaskMessage(
            mission_id=mission_id, task_action="all", drone_id=mission["drone_id"]
        )

        r.publish(DRONE_COMMANDS_CHANNEL, abort_message.model_dump_json())
        r.publish(DRONE_EVENT_CHANNEL, abort_message.model_dump_json())
        r.publish(
            DRONE_EVENT_CHANNEL,
            json_schemas.FrontendMessages.RemoveMission(
                mission=mission
            ).model_dump_json(),
        )

        print("[Mission Registry] Sent abort command for drone", mission["drone_id"])

    @staticmethod
    def abort_mission_and_go_home(drone_id: str) -> dict:
        # Hitta aktivt mission för drönaren
        all_missions = MissionRegistry.get_all()
        active_mission = next(
            (
                m
                for m in all_missions
                if m["drone_id"] == drone_id
                and m["status"] in [MissionStatus.DISPATCHED.value]
            ),
            None,
        )

        if active_mission:
            MissionRegistry.abort_mission(
                active_mission["mission_id"], exception_on_not_dispatched=False
            )

        go_home_mission = GoHome(
            drone_id=drone_id,
            capabilities=json_schemas.Capabilities(
                camera=None, spotlight=False, led=None, speaker=None
            ),
            coordinates=json_schemas.GoToParams(
                lat=0, lon=0, alt=0
            ),  # Just to have some coordinates, the drone ignores these and just go home
        )

        MissionRegistry.store(go_home_mission)
        go_home_mission = MissionRegistry.dispatch_mission(go_home_mission.mission_id)

        # go_home = json_schemas.TaskMessage(
        #     drone_id=drone_id,
        #     mission_id=active_mission["mission_id"]
        #     if active_mission
        #     else "no_active_mission",
        #     index=-1,
        #     task_action=json_schemas.GoHomeTask(),
        # )

        # r.publish(DRONE_COMMANDS_CHANNEL, go_home.model_dump_json())
        # r.publish(DRONE_EVENT_CHANNEL, go_home.model_dump_json())

        print("[Mission Registry] Sent go home command for drone", drone_id)
        return go_home_mission

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
    def update_status(mission_id: str, status: MissionStatus) -> dict | None:
        mission = MissionRegistry.get(mission_id)
        if mission:
            mission["status"] = status.value
            r.set(f"mission_{mission_id}_state", json.dumps(mission))
        return mission

    @staticmethod
    def update_task_status(mission_id: str, task_index: int, status: TaskStatus):
        mission = MissionRegistry.get(mission_id)
        if mission:
            mission["task_status"][task_index] = status.value
            r.set(f"mission_{mission_id}_state", json.dumps(mission))

    @staticmethod
    def is_drone_dispatched(drone_id: str) -> bool:
        for mission in MissionRegistry.get_all():
            if mission["drone_id"] == drone_id:
                if mission["status"] == MissionStatus.DISPATCHED.value:
                    return True

        return False

    @staticmethod
    def is_drone_waiting(drone_id: str) -> bool:
        for mission in MissionRegistry.get_all():
            if mission["drone_id"] == drone_id:
                if mission["status"] == MissionStatus.WAITING.value:
                    return True

        return False

    @staticmethod
    def is_drone_dispatched_or_waiting(drone_id: str) -> bool:
        for mission in MissionRegistry.get_all():
            if mission["drone_id"] == drone_id:
                if mission["status"] in [
                    MissionStatus.DISPATCHED.value,
                    MissionStatus.WAITING.value,
                ]:
                    return True

        return False

    @staticmethod
    def get_drone_on_surveil_mission() -> str | None:
        """
        Returns the drone_id of the drone currently on a GotoAndSurveil mission.
        Returns None if no drone is on a surveil mission.
        """
        for mission in MissionRegistry.get_all():
            if mission.get("mission_type") == "GotoAndSurveil" and mission[
                "status"
            ] in [MissionStatus.DISPATCHED.value]:
                return mission["drone_id"]
        return None

    @staticmethod
    def remove(mission_id: str):
        r.delete(f"mission_{mission_id}_state")
        r.delete(f"mission_{mission_id}_task_queue")

    @staticmethod
    def remove_many(mission_ids: list[str]) -> list[str]:
        removed_ids = []
        for mission_id in dict.fromkeys(mission_ids):
            MissionRegistry.remove(mission_id)
            removed_ids.append(mission_id)
        return removed_ids
