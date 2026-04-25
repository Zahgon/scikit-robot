from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import nextage_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class Nextage(RobotModelFromURDF):
    """
    - Nextage Open Official Information.

      https://nextage.kawadarobot.co.jp/open

    - Nextage Open Robot Description

      https://github.com/tork-a/rtmros_nextage/tree/indigo-devel/nextage_description/urdf
    """

    def __init__(self, *args, **kwargs):
        super(Nextage, self).__init__(*args, **kwargs)

        # End effector coordinates
        self.rarm_end_coords = CascadedCoords(
            pos=[-0.185, 0.0, -0.01],
            parent=self.RARM_JOINT5_Link,
            name='rarm_end_coords')
        self.rarm_end_coords.rotate(-np.pi / 2.0, 'y')

        self.larm_end_coords = CascadedCoords(
            pos=[-0.185, 0.0, -0.01],
            parent=self.LARM_JOINT5_Link,
            name='larm_end_coords')
        self.larm_end_coords.rotate(-np.pi / 2.0, 'y')

        self.head_end_coords = CascadedCoords(
            pos=[0.06, 0, 0.025],
            parent=self.HEAD_JOINT1_Link,
            name='head_end_coords')
        self.head_end_coords.rotate(np.deg2rad(90), 'y')

        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        pass

    def reset_pose(self):
        angle_vector = [
            0.0,
            0.0,
            0.0,
            np.deg2rad(0.6),
            0.0,
            np.deg2rad(-100),
            np.deg2rad(-15.2),
            np.deg2rad(9.4),
            np.deg2rad(-3.2),
            np.deg2rad(-0.6),
            0.0,
            np.deg2rad(-100),
            np.deg2rad(15.2),
            np.deg2rad(9.4),
            np.deg2rad(3.2),
        ]
        self.angle_vector(angle_vector)
        return self.angle_vector()

    def reset_manip_pose(self):
        """Reset robot to manipulation pose (same as reset_pose for Nextage)"""
        pass

    @cached_property
    def rarm(self):
        pass

    @cached_property
    def larm(self):
        pass

    @cached_property
    def head(self):
        pass

    @cached_property
    def torso(self):
        pass

    # New naming convention aliases (backward compatible)
    @property
    def right_arm(self):
        pass

    @property
    def left_arm(self):
        pass

    @property
    def right_arm_end_coords(self):
        pass

    @property
    def left_arm_end_coords(self):
        pass
