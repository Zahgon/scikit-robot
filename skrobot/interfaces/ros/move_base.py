import actionlib
import actionlib_msgs.msg
import control_msgs.msg
import dynamic_reconfigure.msg
import dynamic_reconfigure.srv
import geometry_msgs.msg
import move_base_msgs.msg
import nav_msgs.msg
import numpy as np
import rospy
import std_srvs.srv
import trajectory_msgs.msg

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rotate_vector
from skrobot.coordinates.math import rotation_distance
from skrobot.interfaces.ros.base import ROSRobotInterfaceBase
from skrobot.interfaces.ros.tf_utils import coords_to_geometry_pose
from skrobot.interfaces.ros.tf_utils import geometry_pose_to_coords
from skrobot.interfaces.ros.tf_utils import tf_pose_to_coords
from skrobot.interfaces.ros.transform_listener import TransformListener


class ROSRobotMoveBaseInterface(ROSRobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        self.move_base_action = None
        self.move_base_trajectory_action = None
        self.move_base_goal_msg = None
        self.move_base_goal_coords = None
        self.move_base_goal_map_to_frame = None
        self.odom_topic = None
        self.go_pos_unsafe_goal_msg = None
        self.current_goal_coords = None
        self.map_frame_id = kwargs.pop(
            'map_frame_id', 'map')
        self.move_base_action_name = kwargs.pop(
            'move_base_action_name', 'move_base')
        self.base_frame_id = kwargs.pop(
            'base_frame_id',
            'base_footprint')
        self.base_controller_action_name = kwargs.pop(
            'base_controller_action_name',
            "/base_controller/follow_joint_trajectory")
        self.move_base_trajectory_joint_names = kwargs.pop(
            'base_controller_joint_names',
            ["base_link_x", "base_link_y", "base_link_pan"])
        self.move_base_simple_name = kwargs.pop(
            'move_base_simple_name',
            'move_base_simple')
        self.odom_topic = kwargs.pop('odom_topic', '/base_odometry/odom')
        self.use_tf2 = kwargs.pop('use_tf2', False)

        super(ROSRobotMoveBaseInterface, self).__init__(
            *args, **kwargs)

        self.tf_listener = TransformListener(
            use_tf2=self.use_tf2)

        self.move_base_action = actionlib.SimpleActionClient(
            self.move_base_action_name,
            move_base_msgs.msg.MoveBaseAction)

        if self.base_controller_action_name:
            self.move_base_trajectory_action = actionlib.SimpleActionClient(
                self.base_controller_action_name,
                control_msgs.msg.FollowJointTrajectoryAction)
            if self.move_base_trajectory_action.wait_for_server(
                    rospy.Duration(3)) is False:
                rospy.logwarn('{} is not found'.format(
                    self.base_controller_action_name))
                self.move_base_trajectory_action = None

        self.go_pos_unsafe_goal_msg = None

        self.move_base_simple_publisher = rospy.Publisher(
            "{}/goal".format(self.move_base_simple_name),
            geometry_msgs.msg.PoseStamped,
            queue_size=1)

        self.odom_msg = None
        self.odom_subscriber = rospy.Subscriber(
            self.odom_topic, nav_msgs.msg.Odometry,
            callback=self.odom_callback,
            queue_size=1)

    @property
    def odom(self):
        """Return Coordinates of this odom

        Returns
        -------
        odom_coords : skrobot.coordinates.Coordinates
            coordinates of odom.
        """
        pass

    def odom_callback(self, msg):
        """ROS's subscriber callback for odom.

        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            odometry message.
        """
        pass

    def go_stop(self, force_stop=True):
        """Cancel move_base.

        Parameters
        ----------
        force_stop : bool
            if force_stop is True, send go_velocity(0, 0, 0)
        """
        pass

    def move_to(self, coords, wait=True,
                frame_id=None):
        """Move Robot to target coords.

        Parameters
        ----------
        coords : str or skrobot.coordinates.Coordinates
            target tf name or target coords.
        wait : bool
            if True, wait until move end.
        frame_id : None or str

        Returns
        -------
        result : bool
            if move_base is succeeded, return True.
        """
        pass

    def move_to_send(self,
                     coords,
                     frame_id=None,
                     wait_for_server_timeout=5.0):
        """Send MoveBaseAction

        Parameters
        ----------
        coords : skrobot.coordinates.Coordinates

        Return
        ------
        result : bool
            False or True. If False, could not send MoveBaseAction

        """
        pass

    def move_to_wait(self,
                     retry=10,
                     frame_id='world'):
        pass

    def _calc_move_diff_coords(self, frame_id):
        pass

    def go_pos(self, x=0.0, y=0.0, yaw=0.0, wait=True):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        wait : bool
            if wait is True, wait until move base done.
        """
        pass

    def go_pos_unsafe(self, x=0.0, y=0.0, yaw=0.0, wait=False):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        wait : bool
            if wait is True, wait until stop go_pos_unsafe
        """
        pass

    def go_pos_unsafe_no_wait(self, x=0.0, y=0.0, yaw=0.0):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        """
        pass

    def go_pos_unsafe_wait(self, wait_counts=3):
        pass

    def go_velocity(self, x=0.0, y=0.0, yaw=0.0,
                    sec=1.0, stop=True, wait=False):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move velocity with respect to x axis. unit is [m/s].
        y : float
            move velocity with respect to y axis. unit is [m/s].
        yaw : float
            rotate angle. unit is [rad/s].
        sec : float
            time.
        wait : bool
            if wait is True, wait until move base done.
        """
        pass

    def move_trajectory_sequence(self, trajectory_points, time_list, stop=True,
                                 start_time=None, send_action=None, wait=True):
        """Move base following the trajectory points at each time points

        trajectory-points [ list of #f(x y yaw) ([m] for x, y; [rad] for yaw) ]
        time-list [list of time span [sec] ]
        stop [ stop after msec moving ]
        start-time [ robot will move at start-time [sec or ros::Time] ]
        send-action [ send message to action server, it means robot will move ]

        """
        pass

    def move_trajectory(self, x, y, yaw, sec=1.0, stop=True,
                        start_time=None,
                        send_action=None):
        """Move trajectory.

        This function call move_trajectory_sequence internally.

        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        sec : float
            time. unit is [sec].
        """
        pass

    def clear_costmap(self):
        """Send signal to clear costmap for obstacle avoidance to move_base.

        """
        pass

    def change_inflation_range(self,
                               inflation_range=0.2,
                               node_name='move_base_node',
                               costmap_name='local_costmap',
                               inflation_name='inflation'):
        """Changes inflation range of local costmap for obstacle avoidance.

        Parameters
        ----------
        inflation_range : float
            range of inflation
        node_name : str
            name of move_base_node
        costmap_name : str
            name of costmap
        inflation_name : str
            name of inflation
        """
        pass
