import actionlib
import control_msgs.msg
import rospy
import trajectory_msgs.msg

from skrobot.interfaces.ros.base import ROSRobotInterfaceBase


class NextageROSRobotInterface(ROSRobotInterfaceBase):
    def __init__(self, *args, **kwargs):
        self.on_gazebo = rospy.get_param('/gazebo/time_step', None) is not None \
                         and rospy.get_param('/torso_controller/type', None) is not None

        if self.on_gazebo:
            rospy.loginfo("Gazebo environment detected")

        self.lhand = None
        self.rhand = None

        super(NextageROSRobotInterface, self).__init__(*args, **kwargs)

    def _init_lhand(self):
        pass

    def _init_rhand(self):
        pass

    @property
    def fullbody_controller(self):
        pass

    @property
    def rarm_controller(self):
        pass

    @property
    def larm_controller(self):
        pass

    @property
    def torso_controller(self):
        pass

    @property
    def head_controller(self):
        pass

    def default_controller(self):
        if self.on_gazebo:
            return [self.larm_controller, self.rarm_controller,
                    self.torso_controller, self.head_controller]
        else:
            return [self.fullbody_controller]

    def start_grasp(self, arm='arms', **kwargs):
        pass

    def stop_grasp(self, arm='arms', **kwargs):
        pass

    def open_forceps(self, arm='arms', **kwargs):
        pass

    def close_forceps(self, arm='arms', **kwargs):
        pass

    def open_holder(self, arm='arms', **kwargs):
        pass

    def close_holder(self, arm='arms', **kwargs):
        pass


class LHandInterface:
    def __init__(self):
        self.action_client = actionlib.SimpleActionClient(
            "/lhand/position_joint_trajectory_controller/follow_joint_trajectory",
            control_msgs.msg.FollowJointTrajectoryAction
        )
        if not self.action_client.wait_for_server(rospy.Duration(5)):
            rospy.logwarn("LHand action server not available")

    def move_hand(self, grasp_angle, wait=True, tm=1.0):
        pass

    def start_grasp(self, **kwargs):
        pass

    def stop_grasp(self, **kwargs):
        pass

    def open_forceps(self, wait=False, tm=0.2):
        pass

    def close_forceps(self, wait=False, tm=0.2):
        pass


class RHandInterface:
    def __init__(self):
        self.action_client = actionlib.SimpleActionClient(
            "/rhand/position_joint_trajectory_controller/follow_joint_trajectory",
            control_msgs.msg.FollowJointTrajectoryAction
        )
        if not self.action_client.wait_for_server(rospy.Duration(5)):
            rospy.logwarn("RHand action server not available")

    def move_hand(self, grasp_angle, wait=True, tm=1.0):
        pass

    def start_grasp(self, **kwargs):
        pass

    def stop_grasp(self, **kwargs):
        pass

    def open_holder(self, wait=True, tm=0.2):
        pass

    def close_holder(self, wait=True, tm=0.2):
        pass
