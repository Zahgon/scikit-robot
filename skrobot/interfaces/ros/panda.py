import actionlib
import control_msgs.msg
import franka_gripper.msg
import franka_msgs.msg
import rospy

from skrobot.interfaces.ros.base import ROSRobotInterfaceBase


WIDTH_MAX = 0.08


class PandaROSRobotInterface(ROSRobotInterfaceBase):
    """ROS Interface for Franka Emika Panda robot.

    This class provides control interface for Panda robot including
    arm trajectory control and gripper actions (move, grasp, homing, stop).
    It also supports error state monitoring and recovery.

    Parameters
    ----------
    robot : skrobot.models.Panda
        Robot model instance.
    namespace : str or None
        ROS namespace for the robot. If specified, all topics and actions
        will be prefixed with this namespace.
    """

    def __init__(self, *args, **kwargs):
        super(PandaROSRobotInterface, self).__init__(*args, **kwargs)

        if self.namespace:
            namespace_prefix = self.namespace.strip('/') + '/'
        else:
            namespace_prefix = ''
        self._namespace_prefix = namespace_prefix

        self._init_gripper_actions(namespace_prefix)
        self._init_error_recovery_action(namespace_prefix)
        self._init_franka_state_subscriber(namespace_prefix)

    def _init_gripper_actions(self, namespace_prefix):
        """Initialize gripper action clients.

        Parameters
        ----------
        namespace_prefix : str
            Namespace prefix for action topics.
        """
        gripper_move_action = namespace_prefix + 'franka_gripper/move'
        rospy.loginfo(
            'Waiting for action server: {}'.format(gripper_move_action))
        self.gripper_move = actionlib.SimpleActionClient(
            gripper_move_action,
            franka_gripper.msg.MoveAction)
        self.gripper_move.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(gripper_move_action))

        gripper_stop_action = namespace_prefix + 'franka_gripper/stop'
        rospy.loginfo(
            'Waiting for action server: {}'.format(gripper_stop_action))
        self.gripper_stop = actionlib.SimpleActionClient(
            gripper_stop_action,
            franka_gripper.msg.StopAction)
        self.gripper_stop.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(gripper_stop_action))

        gripper_grasp_action = namespace_prefix + 'franka_gripper/grasp'
        rospy.loginfo(
            'Waiting for action server: {}'.format(gripper_grasp_action))
        self.gripper_grasp = actionlib.SimpleActionClient(
            gripper_grasp_action,
            franka_gripper.msg.GraspAction)
        self.gripper_grasp.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(gripper_grasp_action))

        gripper_homing_action = namespace_prefix + 'franka_gripper/homing'
        rospy.loginfo(
            'Waiting for action server: {}'.format(gripper_homing_action))
        self.gripper_homing = actionlib.SimpleActionClient(
            gripper_homing_action,
            franka_gripper.msg.HomingAction)
        self.gripper_homing.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(gripper_homing_action))

    def _init_error_recovery_action(self, namespace_prefix):
        """Initialize error recovery action client.

        Parameters
        ----------
        namespace_prefix : str
            Namespace prefix for action topics.
        """
        error_recovery_action = namespace_prefix + 'franka_control/error_recovery'
        rospy.loginfo(
            'Waiting for action server: {}'.format(error_recovery_action))
        self.error_recovery = actionlib.SimpleActionClient(
            error_recovery_action,
            franka_msgs.msg.ErrorRecoveryAction)
        self.error_recovery.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(error_recovery_action))

    def _init_franka_state_subscriber(self, namespace_prefix):
        """Initialize Franka state subscriber for error monitoring.

        Parameters
        ----------
        namespace_prefix : str
            Namespace prefix for topics.
        """
        self._has_error = False
        franka_state_topic = namespace_prefix + 'franka_state_controller/franka_states'
        self._franka_state_sub = rospy.Subscriber(
            franka_state_topic,
            franka_msgs.msg.FrankaState,
            self._franka_state_callback,
            queue_size=1)
        rospy.loginfo(
            'Subscribed to {}'.format(franka_state_topic))

    def _franka_state_callback(self, msg):
        """Callback for FrankaState messages.

        Parameters
        ----------
        msg : franka_msgs.msg.FrankaState
            FrankaState message.
        """
        pass

    @property
    def rarm_controller(self):
        pass

    def default_controller(self):
        return [self.rarm_controller]

    def check_error(self):
        """Check if the robot has an error.

        If this method returns True, you must call recover_error()
        before moving the robot.

        Returns
        -------
        bool
            True if the robot has an error, False otherwise.
        """
        pass

    def recover_error(self, wait=True):
        """Recover from errors and reflexes.

        This method sends an error recovery action to the robot.

        Parameters
        ----------
        wait : bool
            If True, wait until the recovery is complete.

        Returns
        -------
        bool or None
            Result of the action if wait is True, None otherwise.
        """
        pass

    def wait_recover_error(self):
        """Wait for error recovery to complete.

        Returns
        -------
        bool
            True if recovery succeeded, False otherwise.
        """
        pass

    def grasp(self, width=0, **kwargs):
        """Close the gripper to grasp an object.

        This is a simple wrapper around move_gripper for backward compatibility.

        Parameters
        ----------
        width : float
            Target width between fingers in meters.
        **kwargs
            Additional arguments passed to move_gripper.
        """
        pass

    def ungrasp(self, **kwargs):
        """Open the gripper to release an object.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to move_gripper.
        """
        pass

    def move_gripper(self, width, speed=WIDTH_MAX, wait=True):
        """Move the gripper to the target width.

        Parameters
        ----------
        width : float
            Target distance between the fingers in meters.
        speed : float
            Movement speed in meters per second.
        wait : bool
            If True, wait until the movement is complete.
        """
        pass

    def stop_gripper(self, wait=True):
        """Abort a running gripper action.

        This can be used to stop applying forces after grasping.

        Parameters
        ----------
        wait : bool
            If True, wait until the action is complete.
        """
        pass

    def grasp_gripper(self, width=0.0, speed=0.1, force=10.0,
                      epsilon_inner=0.005, epsilon_outer=0.07, wait=True):
        """Grasp an object with force control.

        Try to grasp at the desired width with the desired force
        while closing with the desired speed.

        Parameters
        ----------
        width : float
            Target distance between the fingers in meters.
        speed : float
            Closing speed in meters per second.
        force : float
            Grasping force in Newtons.
        epsilon_inner : float
            Maximum tolerated deviation when the actual grasped width is
            smaller than the commanded grasp width in meters.
        epsilon_outer : float
            Maximum tolerated deviation when the actual grasped width is
            larger than the commanded grasp width in meters.
        wait : bool
            If True, wait until the grasp is complete.

        Returns
        -------
        bool or None
            Result of the action if wait is True, None otherwise.
        """
        pass

    def start_grasp(self, width=0.0, force=80.0, speed=0.1,
                    epsilon_inner=0.005, epsilon_outer=0.06, wait=True):
        """Start grasping with force control.

        This is an alias for grasp_gripper with parameter order matching
        the EusLisp interface.

        Parameters
        ----------
        width : float
            Target distance between the fingers in meters.
        force : float
            Grasping force in Newtons.
        speed : float
            Closing speed in meters per second.
        epsilon_inner : float
            Maximum tolerated deviation when the actual grasped width is
            smaller than the commanded grasp width in meters.
        epsilon_outer : float
            Maximum tolerated deviation when the actual grasped width is
            larger than the commanded grasp width in meters.
        wait : bool
            If True, wait until the grasp is complete.

        Returns
        -------
        bool or None
            Result of the action if wait is True, None otherwise.
        """
        pass

    def stop_grasp(self, width=WIDTH_MAX, speed=WIDTH_MAX, wait=True):
        """Open the gripper to stop grasping.

        Parameters
        ----------
        width : float
            Target width to open to in meters.
        speed : float
            Movement speed in meters per second.
        wait : bool
            If True, wait until the movement is complete.
        """
        pass

    def homing_gripper(self, wait=True):
        """Home the gripper and calibrate finger width.

        This homes the gripper and updates the maximum width given
        the mounted fingers. Should be called after mounting new fingers.

        Parameters
        ----------
        wait : bool
            If True, wait until homing is complete.

        Returns
        -------
        bool or None
            Result of the action if wait is True, None otherwise.
        """
        pass
