from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import kuka_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class Kuka(RobotModelFromURDF):
    """Kuka Robot Model."""

    def __init__(self, *args, **kwargs):
        super(Kuka, self).__init__(*args, **kwargs)
        self.rarm_end_coords = CascadedCoords(
            parent=self.lbr_iiwa_with_wsg50__lbr_iiwa_link_7,
            name='rarm_end_coords')
        self.rarm_end_coords.translate(
            np.array([0.0, 0.030, 0.250], dtype=np.float32))
        self.rarm_end_coords.rotate(- np.pi / 2.0, axis='y')
        self.rarm_end_coords.rotate(- np.pi / 2.0, axis='x')
        self.end_coords = [self.rarm_end_coords]

    @cached_property
    def default_urdf_path(self):
        pass

    def reset_manip_pose(self):
        pass

    @cached_property
    def rarm(self):
        pass

    def close_hand(self, av=None):
        pass

    def open_hand(self, default_angle=np.deg2rad(10), av=None):
        pass

    # New naming convention aliases (backward compatible)
    @property
    def arm(self):
        pass

    @property
    def arm_end_coords(self):
        pass
