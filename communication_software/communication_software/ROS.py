from atos_interfaces.srv import *
import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import Empty
from sensor_msgs.msg import NavSatFix
from communication_software.ConvexHullScalable import Coordinate
import numpy as np


class AtosCommunication(Node):
    """A ROS client node that provides methods for publishing messages to atos/ROS topics
    and provides a way of getting the origin coordinates from ATOS
    """

    def __init__(self):
        super().__init__("atos_publisher")
        # These are the different ATOS ROS topics that could be used for changing the states
        self.QOS = rclpy.qos.QoSProfile(depth=10)
        self.init_pub = self.create_publisher(Empty, "/atos/init", self.QOS)
        self.connect_pub = self.create_publisher(Empty, "/atos/connect", self.QOS)
        self.disconnect_pub = self.create_publisher(Empty, "/atos/disconnect", self.QOS)
        self.arm_pub = self.create_publisher(Empty, "/atos/arm", self.QOS)
        self.disarm_pub = self.create_publisher(Empty, "/atos/disarm", self.QOS)
        self.start_pub = self.create_publisher(Empty, "/atos/start", self.QOS)
        self.abort_pub = self.create_publisher(Empty, "/atos/abort", self.QOS)
        self.all_clear_pub = self.create_publisher(Empty, "/atos/all_clear", self.QOS)
        self.reset_test_objects_pub = self.create_publisher(
            Empty, "/atos/reset_test_objects", self.QOS
        )
        self.reload_object_settings_pub = self.create_publisher(
            Empty, "/atos/reload_object_settings", self.QOS
        )

        self.get_id_client = self.create_client(GetObjectIds, "/atos/get_object_ids")
        self.get_ip_client = self.create_client(GetObjectIp, "/atos/get_object_ip")
        self.get_traj = self.create_client(
            GetObjectTrajectory, "/atos/get_object_trajectory"
        )

        # This is the ROS service for getting the test origin from ATOS
        self.get_test_origin_client = self.create_client(
            GetTestOrigin, "/atos/get_test_origin"
        )
        self.test_origin = None

        # This is the ROS service for getting the test state from ATOS
        self.get_object_control_state_client = self.create_client(
            GetObjectControlState, "/atos/get_object_control_state"
        )
        self.state = "UNDEFINED"
        self.lost_connection = True

        # The different possible ATOS states
        self.OBC_STATES = {
            0: "UNDEFINED",
            1: "IDLE",
            2: "INITIALIZED",
            3: "CONNECTED",
            4: "ARMED",
            5: "DISARMING",
            6: "RUNNING",
            7: "REMOTECTRL",
            8: "ERROR",
            9: "ABORTING",
            10: "CLEARING",
        }

        self.object_coordinates = {}

        # Make sure that ATOS is running before the instance can be used
        while not self.get_object_control_state_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().warn("ATOS is not running, waiting ...")
        self.get_logger().info(
            "ATOS is running, waiting 10s to make sure that everything has started"
        )
        # time.sleep(10)

    def __del__(self):
        """This does some cleanup when the instance is deleted"""
        self.destroy_node()

    def coordinate_callback(self, msg, id):
        self.object_coordinates[id] = Coordinate(msg.latitude, msg.longitude, 0)
        print(str(id) + "     " + str(msg))

    def start_coordinate_subscriber(self):
        if len(self.object_coordinates) == 0:
            self.get_object_ids()

        coordinate_subscribers = []

        for id in self.object_coordinates.keys():
            topic = "/atos/object_" + str(id) + "/gnss_fix"

            coordinate_subscribers.append(
                self.create_subscription(
                    NavSatFix,
                    topic,
                    lambda msg: self.coordinate_callback(msg, id),
                    self.QOS,
                )
            )

    def publish_init(self):
        """Method for publishing an init message to ATOS"""
        self.init_pub.publish(Empty())
        self.get_logger().info("Publishing init signal")

    def publish_connect(self):
        """Method for publishing an init message to ATOS"""
        self.connect_pub.publish(Empty())
        self.get_logger().info("Publishing connect signal")

    def publish_disconnect(self):
        """Method for publishing an init message to ATOS"""
        self.disconnect_pub.publish(Empty())
        self.get_logger().info("Publishing connect signal")

    def publish_arm(self):
        """Method for publishing an arm message to ATOS"""
        self.arm_pub.publish(Empty())
        self.get_logger().info("Publishing arm signal")

    def publish_disarm(self):
        """Method for publishing a disarm message to ATOS"""
        self.disarm_pub.publish(Empty())
        self.get_logger().info("Publishing disarm signal")

    def publish_start(self):
        """Method for publishing a start message to ATOS"""
        self.start_pub.publish(Empty())
        self.get_logger().info("Publishing start signal")

    def publish_abort(self):
        """Method for publishing an abort message to ATOS"""
        self.abort_pub.publish(Empty())
        self.get_logger().info("Publishing Abort signal")

    def get_test_origin_callback(self):
        """Help method for getting the test origin. This method does the ROS2 service call.

        Returns:
            Coordinate | None: Returns a coordinate if successful or None if some problem occurred
        """
        while not self.get_test_origin_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")

        get_test_origin_request = GetTestOrigin.Request()

        future = self.get_test_origin_client.call_async(get_test_origin_request)

        # TODO: add timeout for the following line, could spin forever
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))
            return None

        if not response.success:
            self.get_logger().error("et origin service call failed for object %u", id)
            return None
        else:
            self.test_origin = Coordinate(
                response.origin.position.latitude,
                response.origin.position.longitude,
                response.origin.position.altitude,
            )
            return self.test_origin

    def get_object_ids(self):
        """Help method for getting the test origin. This method does the ROS2 service call.

        Returns:
            Coordinate | None: Returns a coordinate if successful or None if some problem occurred
        """
        while not self.get_id_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")

        object_ids_req = GetObjectIds.Request()
        print("Requested object ids received: ", object_ids_req)
        future = self.get_id_client.call_async(object_ids_req)

        # TODO: add timeout for the following line, could spin forever
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))
            return None

        if not response.success:
            self.get_logger().error("get origin service call failed for object %u", id)
            return None
        else:
            for object_id in response.ids:
                self.object_coordinates[object_id] = None
            return list(self.object_coordinates.keys())

    def get_object_traj(self, object_id):
        object_traj_req = GetObjectTrajectory.Request(id=object_id)
        future = self.get_traj.call_async(object_traj_req)
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))
            return None

        if not response.success:
            self.get_logger().error("et origin service call failed for object %u", id)
            return None
        else:
            trajectories = []
            for point in response.trajectory.points:
                trajectories.append(
                    Coordinate(
                        point.pose.position.x,
                        point.pose.position.y,
                        point.pose.position.z,
                    )
                )
            return trajectories

    def get_object_control_state_callback(self):
        """Method for getting the ATOS state. This changes the instance variable state
        and returns the new state. If there was a problem, None will be returned

        Returns:
            str | None: Returns the new state. If there was a problem, None will be returned
        """
        while not self.get_object_control_state_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().warn("service not available, waiting again...")
            self.lost_connection = True
        self.lost_connection = False

        self.OBC_state_req = GetObjectControlState.Request()

        future = self.get_object_control_state_client.call_async(self.OBC_state_req)
        # TODO: add timeout for the following line, could spin forever
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))
            return None
        self.state = self.OBC_STATES[response.state]
        return self.state

    def get_origin_coordinates(self):
        """Method for getting the test origin. This returns a coordinate if successful
        or None if some problem occurred. ATOS has to be initialized for this to work.

        Returns:
            Coordinate | None: Returns a coordinate if successful or None if some problem occurred
        """
        self.get_object_control_state_callback()

        if self.state == "UNDEFINED":
            self.get_logger().error("Unable to get control state, ATOS not available")
            return None
        elif self.state == "ERROR" or self.state == "ABORTING":
            self.get_logger().error(
                "ATOS in state: " + self.state + ", unable to get coordinates."
            )
            return None
        elif self.state == "IDLE":
            self.get_logger().error(
                'ATOS is IDLE, the state has to be "initialized" to get coordinates.'
            )
            return None

        origin = self.get_test_origin_callback()

        return origin


def main():
    """Only for testing."""

    rclpy.init()

    atos_communicator = AtosCommunication()

    atos_communicator.publish_init()
    time.sleep(1)

    atos_communicator.publish_connect()
    time.sleep(1)

    origo = atos_communicator.get_origin_coordinates()

    print(origo.lat, origo.lng, origo.alt)
    time.sleep(1)

    # Gets the trajectories for all of the objects
    ids = atos_communicator.get_object_ids()
    trajectoryList = {}
    for id in ids:
        coordlist = atos_communicator.get_object_traj(id)
        trajectoryList[id] = coordlist

    droneOrigin, angle = getNewDroneOrigin(trajectoryList, origo)
    print(droneOrigin.lat, droneOrigin.lng, droneOrigin.alt)
    print(angle)

    # Updates the coordinates of all objects forever
    atos_communicator.start_coordinate_subscriber()

    rclpy.spin(atos_communicator)
    # del publisher


if __name__ == "__main__":
    main()
