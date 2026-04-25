from __future__ import division

from logging import getLogger
from math import floor
import os
import shutil
import tempfile

import filelock
import numpy as np
import pysdfgen

from skrobot._lazy_imports import _lazy_scipy
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import Coordinates
from skrobot.data import get_cache_dir
from skrobot.utils import checksum_md5


logger = getLogger(__name__)


def trimesh2sdf(mesh, **gridsdf_kwargs):
    """Convert trimesh to signed distance function.

    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        mesh object.
    gridsdf_kwargs : dict
        keyword args for skrobot.sdf.GridSDF.from_objfile

    Returns
    -------
    sdf : skrobot.sdf.SignedDistanceFunction
        converted signed distance function.
    """
    if 'shape' in mesh.metadata:
        shape = mesh.metadata['shape']
        if shape == 'box':
            extents = mesh.metadata['extents']
            sdf = BoxSDF(extents)
        elif shape == 'cylinder':
            height = mesh.metadata['height']
            radius = mesh.metadata['radius']
            sdf = CylinderSDF(radius=radius, height=height)
        elif shape == 'sphere':
            radius = mesh.metadata['radius']
            sdf = SphereSDF(radius)
        else:
            msg = "primitive type {0} is not supported".format(shape)
            raise ValueError(msg)

        if "original_primitive_origin_for_sdf" in mesh.metadata:
            # this value is set when loading a mesh from a URDF file
            origin = mesh.metadata["original_primitive_origin_for_sdf"]
            rotation_matrix = origin[:3, :3]
            translation = origin[:3, 3]
            sdf.newcoords(Coordinates(pos=translation, rot=rotation_matrix))
    else:
        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, 'tmp.obj')
        mesh.export(tmpfile)
        sdf = GridSDF.from_objfile(tmpfile, **gridsdf_kwargs)
        shutil.rmtree(tmpdir)
    return sdf


def link2sdf(link, dim_grid=30):
    """Convert Link to corresponding sdf

    Parameters
    ----------
    link : skrobot.model.Link
        link object
    dim_grid : int
        dimension of the GridSDF

    Returns
    -------
    sdf : skrobot.sdf.SignedDistanceFunction
        corresponding signed distance function to the link type.
        e.g. if Link has geometry of urdf.Box, then BoxSDF is
        created.
    """
    pass


class SignedDistanceFunction(CascadedCoords):
    """A base class for signed distance functions (SDFs).

    SDFs are used to represent and manipulate implicit surfaces.
    SDFs provide a mathematical description of a surface in terms
    of distances: a point's distance from the surface, with the
    sign indicating whether the point is inside or outside the surface.

    Notes
    -----
    The SDF operates in two coordinate frames: the sdf frame and the
    world frame. The sdf frame is the coordinate system defined by
    the SDF itself, while the world frame is a global reference frame.
    Variables and methods often use the prefixes 'sdf' or 'world' to
    indicate the frame of reference. For example, `points_world` would
    refer to points in the world frame, and `_transform_pts_world_to_sdf`
    is a method to transform points from the world frame to the sdf frame
    of the SDF.
    """

    def __init__(self, use_abs=False):
        super(SignedDistanceFunction, self).__init__()

        self.tf_sdf_to_world = self.get_transform()
        self.tf_world_to_sdf =\
            self.tf_sdf_to_world.inverse_transformation()
        self.use_abs = use_abs

    def update(self, force=False):
        # preserve the previous value of self._changed
        # as it will changed to False by super().update()
        changed = self._changed

        super(SignedDistanceFunction, self).update(force=force)

        if changed:
            # this operation must come after super().update()
            # otherwise, infinite recursion occurs
            # because get_transform() internally calls update()
            self.tf_sdf_to_world = self.get_transform()
            self.tf_world_to_sdf =\
                self.tf_sdf_to_world.inverse_transformation()

    def __call__(self, points_world):
        """Compute signed distances of input points to the implicit surface.

        Parameters
        -------
        points_world : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. world flame.

        Returns
        -------
        signed_distances : numpy.ndarray[float]
            1 dim (n_point,) array of signed distance.
        """
        points_sdf = self._transform_pts_world_to_sdf(points_world)
        sd_vals = self._signed_distance(points_sdf)
        if self.use_abs:
            return np.abs(sd_vals)
        return sd_vals

    def on_surface(self, points_world):
        """Check if points are on the surface.

        Parameters
        ----------
        points_world : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. the world frame.

        Returns
        -------
        logicals : numpy.ndarray[bool](n_point,)
            boolean vector of the on-surface predicate for `points_world`.
        sd_vals : numpy.ndarray[float](n_point,)
            signed distances corresponding to `logicals`.
        """
        pass

    def surface_points(self, n_sample=1000):
        """Sample points from the implicit surface of the sdf.

        Parameters
        ----------
        n_sample : int
            number of sample points.

        Returns
        -------
        points_world : numpy.ndarray[float](n_point, 3)
            sampled points w.r.t the world frame.
        dists : numpy.ndarray[float](n_point,)
            signed distances corresponding to points_world.
        """
        pass

    def _transform_pts_world_to_sdf(self, points_world):
        """Transform points from the world frame to the sdf frame.

        Parameters
        ----------
        points_world : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. the world frame.

        Returns
        -------
        points_sdf : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. the sdf-defined sdf frame.
        """
        pass

    def _transform_pts_sdf_to_world(self, points_sdf):
        """Transform points from the sdf to the world frame.

        Parameters
        ----------
        points_sdf : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. the sdf-defined sdf frame.

        Returns
        -------
        points_world : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. the world frame.
        """
        pass


class UnionSDF(SignedDistanceFunction):
    """One can concat multiple SDFs `sdf_list` by using this class.

    For consistency in the concatenation, it is required that
    the all SDFs to be concated are with `use_abs=False`.
    """

    def __init__(self, sdf_list):
        use_abs = False
        super(UnionSDF, self).__init__(use_abs=use_abs)

        use_abs_list = [sdf.use_abs for sdf in sdf_list]
        all_false = np.all(~np.array(use_abs_list))
        assert all_false, "use_abs for each sdf must be consistent"

        self.sdf_list = sdf_list

        threshold_list = [sdf._surface_threshold for sdf in sdf_list]
        self._surface_threshold = max(threshold_list)

    def _signed_distance(self, points_sdf):
        pass

    def _surface_points(self, n_sample=1000):
        # equally assign sample number to each sdf.surface_points()
        pass

    @classmethod
    def from_robot_model(cls, robot_model, dim_grid=50):
        """Create union sdf from a robot model

        Parameters
        ----------
        robot_model : skrobot.model.RobotModel
            Using the links of the robot_model this creates
            the UnionSDF instance.

        Returns
        -------
        union_sdf : skrobot.sdf.UnionSDF
            union sdf of robot_model
        """
        pass


class BoxSDF(SignedDistanceFunction):
    """SDF for a box specified by `width`."""

    def __init__(self, width, use_abs=False):
        super(BoxSDF, self).__init__(use_abs=use_abs)
        self._width = np.array(width)
        self._surface_threshold = np.min(self._width) * 1e-2

    def _signed_distance(self, points_sdf):
        pass

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        pass


class SphereSDF(SignedDistanceFunction):
    """SDF for a sphere specified by `radius`."""

    def __init__(self, radius, use_abs=False):
        super(SphereSDF, self).__init__(use_abs=use_abs)
        self._radius = radius
        self._surface_threshold = radius * 1e-2

    def _signed_distance(self, points_sdf):
        pass

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        pass


class CylinderSDF(SignedDistanceFunction):
    """SDF for a cylinder specified by `radius` and `height`"""

    def __init__(self, height, radius, use_abs=False):
        super(CylinderSDF, self).__init__(use_abs=use_abs)
        self._height = height
        self._radius = radius
        self._surface_threshold = min(radius, height) * 1e-2

    def _signed_distance(self, points_sdf):
        pass

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        pass


class GridSDF(SignedDistanceFunction):
    """SDF using precopmuted signed distances for gridded points."""

    def __init__(self, sdf_data, origin, resolution,
                 fill_value=np.inf, use_abs=False):

        super(GridSDF, self).__init__(use_abs=use_abs)
        # optionally use only the absolute values
        # (useful for non-closed meshes in 3D)
        self._data = np.abs(sdf_data) if use_abs else sdf_data
        self._dims = np.array(self._data.shape)
        self._resolution = resolution
        self._surface_threshold = resolution * np.sqrt(2) / 2.0

        # create regular grid interpolator
        xlin, ylin, zlin = [
            np.array(range(d)) * resolution for d in self._data.shape]
        scipy = _lazy_scipy()
        try:
            interpolator = scipy.interpolate.RegularGridInterpolator
        except AttributeError:
            # scipy<=1.8.0
            from scipy.interpolate import RegularGridInterpolator as interpolator
        self.itp = interpolator(
            (xlin, ylin, zlin),
            self._data,
            bounds_error=False,
            fill_value=fill_value)
        self.origin = origin

    def is_out_of_bounds(self, points_world):
        """check if the the input points is out of bounds

        This method checks if the the input points is out of bounds
        of RegularGridInterpolator.

        Parameters
        ----------
        points_world : numpy.ndarray[float](n_points, 3)
            points w.r.t. world to be checked.

        Returns
        -------
        is_out_arr : numpy.ndarray[bool](n_points,)
            If points is out of the interpolator's boundary,
            the corresponding element of is_out_arr is True
        """
        pass

    def _signed_distance(self, points_sdf):
        pass

    def _surface_points(self, n_sample=None):
        pass

    @staticmethod
    def from_file(filepath, **kwargs):
        """Return GridSDF instance from a .sdf file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            path of .sdf file

        Returns
        -------
        sdf_instance : skrobot.esdf.GridSDF
            instance of sdf
        """
        with open(filepath, 'r') as f:
            # dimension of each axis should all be equal for LSH
            nx, ny, nz = [int(i) for i in f.readline().split()]
            ox, oy, oz = [float(i) for i in f.readline().split()]
            dims = np.array([nx, ny, nz])
            origin = np.array([ox, oy, oz])

            # resolution of the grid cells in original mesh coords
            resolution = float(f.readline())
            sdf_data = np.fromstring(f.read(), dtype=float, sep='\n').reshape(
                *dims).transpose(2, 1, 0)
        return GridSDF(sdf_data, origin, resolution, **kwargs)

    @staticmethod
    def from_objfile(obj_filepath, dim_grid=100, padding_grid=5, **kwargs):
        """Return GridSDF instance from an .obj file.

        In the initial call of this method for an .obj file,
        the pre-process takes some time to converting it to
        a .sdf file. However, because a cache of .sdf file is
        created in the initial call, there is no overhead
        from the next call for the same .obj file.

        Parameters
        ----------
        obj_filepath : str or pathlib.Path
            path of objfile
        dim_grid : int
            dim of sdf
        padding_grid : int
            number of padding

        Returns
        -------
        sdf_instance : skrobot.sdf.GridSDF
            instance of sdf
        """
        sdf_cache_dir = os.path.join(get_cache_dir(), 'sdf')
        if not os.path.exists(sdf_cache_dir):
            os.makedirs(sdf_cache_dir)

        hashed_filename = '{}_{}_{}'.format(
            checksum_md5(obj_filepath), dim_grid, padding_grid)

        sdf_cache_path = os.path.join(sdf_cache_dir, hashed_filename + '.sdf')
        lock = filelock.FileLock(sdf_cache_path + '.lock')
        with lock:
            if not os.path.exists(sdf_cache_path):
                logger.info(
                    'trying to acquire lock for %s...', sdf_cache_path)
                logger.info(
                    'pre-computing sdf and making a cache at %s.', sdf_cache_path)
                pysdfgen.mesh2sdf(str(obj_filepath), dim_grid, padding_grid,
                                  output_filepath=sdf_cache_path)
                logger.info('finish pre-computation')
        return GridSDF.from_file(sdf_cache_path, **kwargs)


def ray_marching(pts_starts, direction_arr, f_sdf, threshold):
    pass
