import control_msgs.action
import rclpy
from rclpy.action import ActionClient

from skrobot.interfaces.ros2.base import ROS2RobotInterfaceBase


try:
    import franka_gripper.action
    FRANKA_GRIPPER_AVAILABLE = True
except ImportError:
    FRANKA_GRIPPER_AVAILABLE = False


WIDTH_MAX = 0.08


class PandaROS2RobotInterface(ROS2RobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        super(PandaROS2RobotInterface, self).__init__(*args, **kwargs)

        if FRANKA_GRIPPER_AVAILABLE:
            self.gripper_move = ActionClient(
                self,
                franka_gripper.action.Move,
                'franka_gripper/move')
            self.gripper_move.wait_for_server()

            self.gripper_stop = ActionClient(
                self,
                franka_gripper.action.Stop,
                'franka_gripper/stop')
            self.gripper_stop.wait_for_server()
        else:
            self.get_logger().warn("franka_gripper package not available. Gripper functions disabled.")

    @property
    def rarm_controller(self):
        pass

    def default_controller(self):
        return [self.rarm_controller]

    def grasp(self, width=0, **kwargs):
        pass

    def ungrasp(self, **kwargs):
        pass

    def move_gripper(self, width, speed=WIDTH_MAX, wait=True):
        pass

    def stop_gripper(self, wait=True):
        pass
