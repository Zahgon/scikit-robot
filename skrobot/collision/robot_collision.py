"""Robot collision checking with geometric primitives.

This module provides collision checking for robots using spheres and capsules
to approximate link geometries. It supports both NumPy and JAX backends for
differentiable collision avoidance in trajectory optimization.

Example
-------
>>> from skrobot.collision import RobotCollisionChecker
>>> from skrobot.models import PR2
>>> robot = PR2()
>>> checker = RobotCollisionChecker(robot)
>>> checker.add_link(robot.r_gripper_palm_link)  # Auto-generate from mesh
>>> distances = checker.compute_self_collision_distances()
"""

import numpy as np

from skrobot.collision.distance import collision_distance
from skrobot.collision.geometry import Capsule
from skrobot.collision.geometry import HalfSpace
from skrobot.collision.geometry import Sphere


class LinkCollisionGeometry:
    """Collision geometry attached to a robot link.

    Parameters
    ----------
    link : Link
        Robot link to attach geometry to.
    geometry : CollisionGeometry
        Collision geometry in link-local frame.
    """

    def __init__(self, link, geometry):
        self.link = link
        self.geometry = geometry

    def get_world_geometry(self, xp=np):
        """Get geometry transformed to world frame.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        CollisionGeometry
            Geometry in world frame.
        """
        pos = self.link.worldpos()
        rot = self.link.worldrot()
        return self.geometry.transform(pos, rot, xp)


class RobotCollisionChecker:
    """Collision checker for robot links using geometric primitives.

    This class manages collision geometries attached to robot links and
    provides methods to compute collision distances for:
    - Self-collision between robot links
    - Collision with world obstacles (primitives or mesh SDF)

    Supports both NumPy and JAX backends for differentiable collision.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model instance.

    Example
    -------
    >>> from skrobot.collision import RobotCollisionChecker
    >>> from skrobot.model.primitives import Box, Sphere
    >>> checker = RobotCollisionChecker(robot)
    >>> checker.add_link(robot.r_gripper_palm_link)  # Auto from mesh
    >>> checker.add_links([robot.r_forearm_link, robot.r_upper_arm_link])
    >>> # Use skrobot.model.primitives directly (auto-converted)
    >>> sphere = Sphere(radius=0.3)
    >>> sphere.translate([1, 0, 0])
    >>> checker.add_world_obstacle(sphere)
    >>> box = Box(extents=[0.5, 0.5, 0.5])
    >>> box.translate([0.8, 0, 0.8])
    >>> checker.add_world_obstacle(box)
    >>> checker.setup_self_collision_pairs()
    >>> min_dist = checker.compute_min_distance()
    """

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self._link_geometries = []
        self._world_obstacles = []
        self._world_sdfs = []  # SDF functions for mesh obstacles
        self._self_collision_pairs = []

        # Visualization
        self._visual_spheres = []  # Visual sphere primitives for viewer
        self._visual_coords = []   # CascadedCoords for sphere positions
        self.color_normal_sphere = [250, 250, 10, 200]
        self.color_collision_sphere = [255, 0, 0, 200]

    def add_link(self, link, geometry_type='auto', n_spheres=None,
                 radius_scale=1.0, tol=0.1, aspect_threshold=1.5):
        """Add collision geometry for a link.

        Automatically generates collision geometry from the link's
        collision mesh. Can use spheres, capsule, or auto-select based
        on the mesh shape.

        Parameters
        ----------
        link : Link
            Robot link with collision_mesh attribute.
        geometry_type : str
            Type of geometry to use:
            - 'auto': Automatically select based on aspect ratio (default)
            - 'spheres': Use swept spheres (compatible with SweptSphereSdfCollisionChecker)
            - 'capsule': Use single capsule (more efficient for elongated shapes)
        n_spheres : int or None
            Number of spheres to use (only for 'spheres' type). If None,
            automatically determined based on tolerance.
        radius_scale : float
            Scale factor for computed radius.
        tol : float
            Tolerance for automatic sphere count determination.
        aspect_threshold : float
            Aspect ratio threshold for auto geometry selection.
            If length/diameter > threshold, use capsule. Default is 1.5.
        """
        pass

    def _compute_capsule_from_mesh(self, mesh):
        """Compute capsule parameters (p1, p2, radius) from mesh.

        Uses PCA to find the principal axis and computes the bounding
        capsule along that axis.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Collision mesh.

        Returns
        -------
        p1 : ndarray (3,)
            First endpoint of capsule axis.
        p2 : ndarray (3,)
            Second endpoint of capsule axis.
        radius : float
            Capsule radius.
        """
        pass

    def add_links(self, links, geometry_type='auto', n_spheres=None,
                  radius_scale=1.0, aspect_threshold=1.5):
        """Add collision geometry for multiple links.

        Parameters
        ----------
        links : list of Link
            Robot links to add.
        geometry_type : str
            Type of geometry: 'auto', 'spheres', or 'capsule'.
        n_spheres : int or None
            Number of spheres per link (only for 'spheres' type).
        radius_scale : float
            Scale factor for computed radius.
        aspect_threshold : float
            Aspect ratio threshold for auto selection.
        """
        pass

    def add_link_sphere(self, link, center_local=None, radius=0.05):
        """Add a collision sphere to a link manually.

        Parameters
        ----------
        link : Link
            Robot link.
        center_local : array-like (3,), optional
            Sphere center in link-local frame. Defaults to origin.
        radius : float
            Sphere radius.
        """
        pass

    def add_link_capsule(self, link, p1_local, p2_local, radius=0.05):
        """Add a collision capsule to a link manually.

        Parameters
        ----------
        link : Link
            Robot link.
        p1_local : array-like (3,)
            First endpoint in link-local frame.
        p2_local : array-like (3,)
            Second endpoint in link-local frame.
        radius : float
            Capsule radius.
        """
        pass

    def add_world_obstacle(self, obstacle, use_sdf=True):
        """Add a world obstacle for collision checking.

        Automatically detects the obstacle type and uses the best method:
        - Objects with .sdf attribute (when use_sdf=True): uses SDF
        - skrobot.model.primitives (Sphere, Box, Cylinder): analytical distance
        - skrobot.collision geometry: analytical distance
        - callable: treated as SDF function

        Parameters
        ----------
        obstacle : various
            Obstacle geometry in world frame. Accepts:
            - skrobot.model.primitives.Sphere, Box, Cylinder
            - skrobot.collision.Sphere, Capsule, Box, HalfSpace
            - Any object with .sdf callable attribute
            - callable SDF function: points (N, 3) -> distances (N,)
        use_sdf : bool
            If True (default), use SDF when available for more accurate
            collision checking. If False, prefer analytical primitives
            (faster, JAX-compatible).
        """
        pass

    def add_ground_plane(self, height=0.0):
        """Add a ground plane as world obstacle.

        Parameters
        ----------
        height : float
            Height of the ground plane.
        """
        pass

    def setup_self_collision_pairs(self, min_link_distance=2,
                                     ignore_pairs=None,
                                     use_urdf_adjacency=True):
        """Setup pairs of link geometries for self-collision checking.

        Parameters
        ----------
        min_link_distance : int
            Minimum kinematic chain distance for collision checking.
            Default is 2 (skip parent-child pairs).
        ignore_pairs : list of tuple, optional
            List of (link_name_a, link_name_b) pairs to ignore.
            Useful for known non-colliding pairs like torso vs arms.
        use_urdf_adjacency : bool
            If True, compute actual kinematic chain distance using
            parent-child relationships. If False, use insertion order.
        """
        pass

    def set_self_collision_pairs(self, pairs):
        """Manually set self-collision pairs.

        Parameters
        ----------
        pairs : list of tuple
            List of (i, j) index pairs into link_geometries.
        """
        pass

    def compute_world_collision_distances(self, xp=np):
        """Compute distances from all link geometries to world obstacles.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        array
            Array of signed distances.
        """
        pass

    def compute_self_collision_distances(self, xp=np):
        """Compute distances between self-collision pairs.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        array
            Array of signed distances for each collision pair.
        """
        if len(self._self_collision_pairs) == 0:
            return xp.array([])

        distances = []
        for i, j in self._self_collision_pairs:
            geom_i = self._link_geometries[i].get_world_geometry(xp)
            geom_j = self._link_geometries[j].get_world_geometry(xp)
            dist = collision_distance(geom_i, geom_j, xp)
            distances.append(dist)

        return xp.array(distances)

    def compute_all_distances(self, xp=np):
        """Compute all collision distances (world + self).

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        array
            Concatenated array of all collision distances.
        """
        pass

    def compute_min_distance(self, xp=np):
        """Compute minimum collision distance.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        float
            Minimum signed distance. Negative means collision.
        """
        pass

    def is_collision_free(self, margin=0.0, xp=np):
        """Check if robot is collision-free.

        Parameters
        ----------
        margin : float
            Safety margin. Robot is collision-free if min_distance > margin.
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        bool
            True if collision-free.
        """
        pass

    def collision_check(self, xp=np):
        """Check if any collision exists.

        Returns
        -------
        bool
            True if collision detected.
        """
        pass

    @property
    def n_feature(self):
        """Number of collision features (spheres/capsules)."""
        pass

    @property
    def link_geometries(self):
        """List of LinkCollisionGeometry objects."""
        pass

    @property
    def world_obstacles(self):
        """List of world obstacle geometries."""
        pass

    @property
    def self_collision_pairs(self):
        """List of (i, j) self-collision pairs."""
        pass

    def get_collision_spheres_world(self):
        """Get all collision spheres in world frame.

        Useful for visualization.

        Returns
        -------
        list of tuple
            List of (center, radius) for each sphere.
        """
        pass

    def add_coll_spheres_to_viewer(self, viewer):
        """Add collision geometries to viewer.

        Creates visual primitives (spheres or capsules) that follow the robot
        links. Call update_color() to update colors based on collision state.

        Parameters
        ----------
        viewer : skrobot.viewers.TrimeshSceneViewer or similar
            Viewer to add geometries to.
        """
        pass

    def delete_coll_spheres_from_viewer(self, viewer):
        """Delete collision spheres from viewer.

        Parameters
        ----------
        viewer : skrobot.viewers.TrimeshSceneViewer or similar
            Viewer to remove spheres from.
        """
        pass

    def update_color(self):
        """Update collision geometry colors based on collision state.

        Geometries in collision are colored red, others are yellow.
        Call this after robot configuration changes to update visualization.

        Returns
        -------
        array
            Array of signed distances for each collision geometry.
        """
        pass
