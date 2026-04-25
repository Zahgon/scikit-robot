import importlib
import time

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.coordinates import matrix2quaternion
from skrobot.coordinates import quaternion2rpy
from skrobot.coordinates.math import xyzw2wxyz


_available = False
_import_checked = False
p = None


def _check_available(silent=False):
    global _available
    global _import_checked
    global p
    if not _import_checked:
        try:
            p = importlib.import_module('pybullet')
        except ImportError as e:
            if not silent:
                if 'numpy.core.multiarray' in str(e):
                    print('Failed to import pybullet due to a NumPy compatibility issue '  # NOQA
                          + '(numpy.core.multiarray failed to import). '
                          + 'This may be caused by an incompatible NumPy version.\n'  # NOQA
                          + 'Please try installing a different NumPy version, e.g., 1.20.3:\n\n'  # NOQA
                          + '  $ pip install numpy==1.20.3\n'
                          + 'Then reinstall pybullet:\n'
                          + '  $ pip install pybullet\n')
                else:
                    _available = False
                    print('pybullet is not installed on your environment, '
                          'so nothing will be drawn at this time. '
                          'Please install pybullet.\n\n'
                          '  $ pip install pybullet\n')
            else:
                _available = False
        except TypeError:
            _available = False
            if not silent:
                print('Unexpected TypeError occurred while importing pybullet.')
        finally:
            _import_checked = True
            if p is not None:
                _available = True
    if not _available and not silent:
        print('pybullet is unavailable. '
              'No drawing will occur until the issue is resolved.')
    return _available


class PybulletRobotInterface(Coordinates):

    """Pybullet Interface Class

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        robot model
    urdf_path : None or str
        urdf path. If this value is `None`,
        get `urdf_path` from `robot.urdf_path`.
    use_fixed_base : bool
        If this value is `True`, robot in pybullet simulator will be fixed.
    connect : int
        pybullet's connection mode. If you have already connected
        to pybullet physics server, specify the server id.
        The default value is 1 (pybullet.GUI).

    Examples
    --------
    >>> from skrobot.models import PR2
    >>> from skrobot.interfaces import PybulletRobotInterface
    >>> robot_model = PR2()
    >>> interface = PybulletRobotInterface(robot_model)

    If you have already connected to pybullet physics server

    >>> import pybullet
    >>> client_id = pybullet.connect(pybullet.GUI)
    >>> robot_model = PR2()
    >>> interface = PybulletRobotInterface(robot_model, connect=client_id)
    """

    def __init__(self, robot, urdf_path=None, use_fixed_base=False,
                 connect=1, *args, **kwargs):
        if _check_available() is False:
            raise ImportError('pybullet is not available.')
        super(PybulletRobotInterface, self).__init__(*args, **kwargs)
        if urdf_path is None:
            if robot.urdf_path is not None:
                urdf_path = robot.urdf_path
            else:
                raise ValueError('urdf_path should be given.')
        self.robot = robot
        if connect == 2:
            p.connect(connect)
        elif connect == 1:
            try:
                p.connect(connect)
            except Exception as e:
                print(e)
        self.robot_id = p.loadURDF(urdf_path, self.translation,
                                   self.quaternion_xyzw,
                                   useFixedBase=use_fixed_base)

        self.load_bullet()
        self.realtime_simulation = False

    @staticmethod
    def available():
        """Check Pybullet is available.

        Returns
        -------
        _available : bool
            If `False`, pybullet is not available.
        """
        _check_available()
        return _available

    @property
    def pose(self):
        """Getter of Pose in pybullet physics simulator.

        Wrapper of pybullet.getBasePositionAndOrientation.

        Returns
        -------
        pose : skrobot.coordinates.Coordinates
            pose of this robot in the physics simulator.
        """
        pass

    def _reset_position_and_orientation(self):
        """Reset base position and orientation.

        This function is wrapper of pybullet.resetBasePositionAndOrientation.
        """
        p.resetBasePositionAndOrientation(self.robot_id, self.translation,
                                          self.quaternion_xyzw)

    def translate(self, vec, wrt='local'):
        """Translate robot in simulator.

        For more detail,
        please see docs of skrobot.coordinates.Coordinates.translate.
        The difference between the translate, this function internally
        call pybullet.resetBasePositionAndOrientation.

        Parameters
        ----------
        vec : list or np.ndarray
            shape of (3,) translation vector. unit is [m] order.
        wrt : string or Coordinates (optional)
            translate with respect to wrt.
        """
        super(PybulletRobotInterface, self).translate(vec, wrt)
        self._reset_position_and_orientation()
        return self

    def rotate(self, theta, axis=None, wrt='local'):
        """Rotate this robot by given theta and axis.

        For more detail,
        please see docs of skrobot.coordinates.Coordinates.rotate.
        The difference between the rotate, this function internally
        call pybullet.resetBasePositionAndOrientation.

        Parameters
        ----------
        theta : float
            radian
        wrt : string or skrobot.coordinates.Coordinates
        """
        super(PybulletRobotInterface, self).rotate(theta, axis, wrt)
        self._reset_position_and_orientation()
        return self

    def transform(self, c, wrt='local'):
        """Transform this coordinate by coords based on wrt

        For more detail,
        please see docs of skrobot.coordinates.Coordinates.transform.
        The difference between the transform, this function internally
        call pybullet.resetBasePositionAndOrientation.

        Parameters
        ----------
        c : skrobot.coordinates.Coordinates
            coordinate
        wrt : string or skrobot.coordinates.Coordinates
            If wrt is 'local' or self, multiply c from the right.
            If wrt is 'world' or 'parent' or self.parent,
            transform c with respect to worldcoord.
            If wrt is Coordinates, transform c with respect to c.
        """
        super(PybulletRobotInterface, self).transform(c, wrt)
        self._reset_position_and_orientation()
        return self

    def newcoords(self, c, pos=None):
        """Update of position and orientation.

        """
        super(PybulletRobotInterface, self).newcoords(c, pos)
        self._reset_position_and_orientation()
        return self

    def load_bullet(self):
        """Load bullet configurations.

        This function internally called.
        """
        joint_num = p.getNumJoints(self.robot_id)
        joint_ids = [None] * joint_num
        joint_name_to_joint_id = {}
        for i in range(len(joint_ids)):
            joint_name = p.getJointInfo(self.robot_id, i)[1]
            try:
                idx = self.robot.joint_names.index(joint_name.decode('utf-8'))
            except ValueError:
                continue
            if idx != -1:
                joint_ids[idx] = i
                joint_name_to_joint_id[joint_name.decode('utf-8')] = i
            else:
                joint_name_to_joint_id[joint_name.decode('utf-8')] = idx
        self.joint_ids = joint_ids
        self.joint_name_to_joint_id = joint_name_to_joint_id

        self.max_velocity = 1.0
        self.position_gain = 0.1
        self.target_velocity = 0.0
        self.velocity_gain = 0.1

    def angle_vector(self, angle_vector=None, realtime_simulation=None):
        """Send a angle vector to pybullet's physics engine.

        Parameters
        ----------
        angle_vector : None or numpy.ndarray
            angle vector. If `None`, send self.robot.angle_vector()
        realtime_simulation : None or bool
            If this value is `True`, send angle_vector
            by pybullet.setJointMotorControl2.

        Returns
        -------
        angle_vector : numpy.ndarray
            return sent angle vector.
        """
        if realtime_simulation is not None and isinstance(
                realtime_simulation, bool):
            self.realtime_simulation = realtime_simulation

        if self.robot_id is None:
            return self.robot.angle_vector()
        if angle_vector is None:
            angle_vector = self.robot.angle_vector()

        for i, (joint, angle) in enumerate(
                zip(self.robot.joint_list, angle_vector)):
            idx = self.joint_name_to_joint_id[joint.name]

            joint = self.robot.joint_list[i]

            if self.realtime_simulation is False:
                p.resetJointState(self.robot_id, idx, angle)

            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                    jointIndex=idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=angle,
                                    targetVelocity=self.target_velocity,
                                    force=joint.max_joint_torque,
                                    positionGain=self.position_gain,
                                    velocityGain=self.velocity_gain,
                                    maxVelocity=self.max_velocity)

        return angle_vector

    def wait_interpolation(self, thresh=0.05, timeout=60.0):
        """Wait robot movement.

        This function usually called after self.angle_vector.
        Wait while the robot joints are moving or until time of timeout.
        This function called internally pybullet.stepSimulation().

        Parameters
        ----------
        thresh : float
            velocity threshold for detecting movement stop.
        timeout : float
            maximum time of timeout.
        """
        pass

    def sync(self):
        """Synchronize pybullet pose to robot_model.

        """
        pass


remove_user_item_indices = []
remove_body_indices = []


def draw(c,
         line_width=4,
         line_length=0.3,
         parent_link_index=0,
         color=[1, 1, 1, 1],
         radius=0.03,
         text=''):
    pass


def flush():
    pass


def create_pose_marker(position=np.array([0, 0, 0]),
                       orientation=np.array([0, 0, 0, 1]),
                       text='',
                       xColor=np.array([1, 0, 0]),
                       yColor=np.array([0, 1, 0]),
                       zColor=np.array([0, 0, 1]),
                       textColor=np.array([0, 0, 0]),
                       lineLength=0.1,
                       lineWidth=1,
                       textSize=1,
                       textPosition=np.array([0, 0, 0.1]),
                       textOrientation=None,
                       lifeTime=0,
                       parentObjectUniqueId=-1,
                       parentLinkIndex=-1):
    """Create a pose marker

    Create a pose marker that identifies a position and orientation in space
    with 3 colored lines.

    """
    pass
