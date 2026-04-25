from __future__ import division

import collections
import os
from pathlib import Path
import platform
import sys
import threading

import numpy as np
from PIL import Image
import pyglet
from pyglet import compat_platform

from skrobot.pycompat import is_wsl


# WSL2 and Wayland specific fix for pyrender
# Set PYOPENGL_PLATFORM to GLX for proper OpenGL context management
if platform.system() == 'Linux':
    needs_glx = False

    # Check for WSL2 environment
    if is_wsl():
        needs_glx = True

    # Check for Wayland session (Ubuntu 24.04+ default)
    if os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
        needs_glx = True

    if needs_glx and 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = 'glx'

import inspect

import pyrender
from pyrender.trackball import Trackball
import trimesh
from trimesh import transformations
from trimesh.scene import cameras

from skrobot import model as model_module
from skrobot.coordinates import Coordinates
from skrobot.model.skeleton import SkeletonModel


# Check if pyrender supports always_on_top parameter
_from_trimesh_params = inspect.signature(pyrender.Mesh.from_trimesh).parameters
_PYRENDER_SUPPORTS_ALWAYS_ON_TOP = 'always_on_top' in _from_trimesh_params
_always_on_top_warning_shown = False


def _mesh_from_trimesh(mesh, smooth=False, always_on_top=False):
    """Create pyrender Mesh from trimesh with always_on_top support check."""
    global _always_on_top_warning_shown
    if always_on_top and not _PYRENDER_SUPPORTS_ALWAYS_ON_TOP:
        if not _always_on_top_warning_shown:
            # Use ANSI escape codes for red text
            red = "\033[91m"
            reset = "\033[0m"
            msg = (
                f"{red}always_on_top parameter is not supported by your "
                f"pyrender version.\n"
                f"To enable this feature, install scikit-robot-pyrender:\n"
                f"  pip install scikit-robot-pyrender>=0.1.50{reset}"
            )
            print(msg, file=sys.stderr)
            _always_on_top_warning_shown = True
        return pyrender.Mesh.from_trimesh(mesh, smooth=smooth)
    if _PYRENDER_SUPPORTS_ALWAYS_ON_TOP:
        return pyrender.Mesh.from_trimesh(
            mesh, smooth=smooth, always_on_top=always_on_top)
    return pyrender.Mesh.from_trimesh(mesh, smooth=smooth)


def _redraw_all_windows():
    try:
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()
            window._legacy_invalid = False
    except RuntimeError:
        pass


class PyrenderViewer(pyrender.Viewer):

    """PyrenderViewer class implemented as a Singleton.

    This ensures that only one instance of the viewer
    is created throughout the program. Any subsequent attempts to create a new
    instance will return the existing one.

    Parameters
    ----------
    resolution : tuple, optional
        The resolution of the viewer. Default is (640, 480).
    update_interval : float, optional
        The update interval (in seconds) for the viewer. Default is
        1.0 seconds.
    title : str, optional
        The title of the viewer window. Default is 'scikit-robot PyrenderViewer'.
    enable_collision_toggle : bool, optional
        Enable collision/visual mesh toggle functionality with 'v' key.
        Default is True.

    Notes
    -----
    Since this is a singleton, the __init__ method might be called
    multiple times, but only one instance is actually used.

    Keyboard Controls
    -----------------
    j : Toggle joint axes display (shows/hides joint positions and axes)
        Joint positions are displayed as blue spheres
        Joint axes are displayed as red cylinders
    v : Toggle between visual and collision meshes (if enable_collision_toggle=True)
        Collision meshes are displayed in orange/transparent color
    """

    # Class variable to hold the single instance of the class.
    _instance = None

    def __init__(self, resolution=None, update_interval=1.0,
                 render_flags=None, title=None, enable_collision_toggle=True):
        if getattr(self, '_initialized', False):
            return
        if resolution is None:
            resolution = (640, 480)

        self.thread = None
        self._visual_mesh_map = collections.OrderedDict()
        self._joint_axis_map = collections.OrderedDict()

        # Joint axis toggle functionality
        self._stored_robots = []
        self.show_joint_axes = False
        self.joint_axes_always_on_top = True

        # Skeleton visualization
        self._kinematics_models = collections.OrderedDict()

        # Collision toggle functionality
        self.enable_collision_toggle = enable_collision_toggle
        if self.enable_collision_toggle:
            self._stored_links = []
            self.show_collision = False

        self._redraw = True
        self._context_initialized = False

        refresh_rate = 1.0 / update_interval
        self._kwargs = dict(
            scene=pyrender.Scene(),
            viewport_size=resolution,
            run_in_thread=False,
            use_raymond_lighting=True,
            auto_start=False,
            render_flags=render_flags,
            refresh_rate=refresh_rate,
        )
        super(PyrenderViewer, self).__init__(**self._kwargs)
        window_title = title if title is not None else 'scikit-robot PyrenderViewer'
        self.viewer_flags['window_title'] = window_title
        self._initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PyrenderViewer, cls).__new__(cls)
        return cls._instance

    def show(self):
        if self.thread is not None and self.thread.is_alive():
            return
        # Reset rendering flag for new viewer session
        self._allow_rendering = False
        distance = self._calculate_camera_distance()
        self.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)],
                        distance=distance)
        if compat_platform == 'darwin':
            self._init_and_start_app()
            init_loop = 30
            for _ in range(init_loop):
                _redraw_all_windows()
        else:
            self.thread = threading.Thread(target=self._init_and_start_app)
            self.thread.daemon = True  # terminate when main thread exit
            self.thread.start()

    def _init_and_start_app(self):
        # Try multiple configs starting with target OpenGL version
        # and multisampling and removing these options if exception
        # Note: multisampling not available on all hardware
        from pyglet import clock
        from pyglet.gl import Config
        from pyrender.constants import MIN_OPEN_GL_MAJOR
        from pyrender.constants import MIN_OPEN_GL_MINOR
        from pyrender.constants import TARGET_OPEN_GL_MAJOR
        from pyrender.constants import TARGET_OPEN_GL_MINOR
        from pyrender.viewer import Viewer

        # Block rendering during window creation
        self._allow_rendering = False

        confs = [Config(sample_buffers=1, samples=4,
                        depth_size=24,
                        double_buffer=True,
                        major_version=TARGET_OPEN_GL_MAJOR,
                        minor_version=TARGET_OPEN_GL_MINOR),
                 Config(depth_size=24,
                        double_buffer=True,
                        major_version=TARGET_OPEN_GL_MAJOR,
                        minor_version=TARGET_OPEN_GL_MINOR),
                 Config(sample_buffers=1, samples=4,
                        depth_size=24,
                        double_buffer=True,
                        major_version=MIN_OPEN_GL_MAJOR,
                        minor_version=MIN_OPEN_GL_MINOR),
                 Config(depth_size=24,
                        double_buffer=True,
                        major_version=MIN_OPEN_GL_MAJOR,
                        minor_version=MIN_OPEN_GL_MINOR)]
        for conf in confs:
            try:
                super(Viewer, self).__init__(config=conf, resizable=True,
                                             width=self._viewport_size[0],
                                             height=self._viewport_size[1])
                break
            except (pyglet.window.NoSuchConfigException, pyglet.gl.ContextException):
                pass
            except pyglet.canvas.xlib.NoSuchDisplayException:
                print('No display found. Viewer is disabled.')
                self.has_exit = True
                return

        if not self.context:
            raise ValueError('Unable to initialize an OpenGL 3+ context')
        clock.schedule_interval(
            Viewer._time_event, 1.0 / self.viewer_flags['refresh_rate'], self
        )
        self.switch_to()
        self.set_caption(self.viewer_flags['window_title'])
        # Mark context as initialized
        self._context_initialized = True

        # Schedule _allow_rendering=True after event loop starts
        # This ensures pending on_resize events are skipped
        def enable_rendering(dt):
            pass

        if compat_platform == 'darwin':
            # On macOS, pyglet.app.run() is not called, so we enable rendering directly
            # after a short delay via multiple redraw cycles in show()
            self._allow_rendering = True
        else:
            clock.schedule_once(enable_rendering, 0.1)
            pyglet.app.run()

    def redraw(self):
        self._redraw = True
        if compat_platform == 'darwin':
            _redraw_all_windows()

    def on_draw(self):
        # Block rendering until initialization is complete
        if not getattr(self, '_allow_rendering', False):
            return

        # Ensure context is ready before drawing
        if not self.context:
            return
        try:
            self.switch_to()
        except Exception:
            # Context not ready yet, skip this draw event
            return

        with self._render_lock:
            if not self._redraw:
                super(PyrenderViewer, self).on_draw()
                return
            # apply latest angle-vector
            for link_id, (node, link) in self._visual_mesh_map.items():
                link.update(force=True)
                transform = link.worldcoords().T()
                if link.visual_mesh_changed:
                    mesh = link.concatenated_visual_mesh
                    always_on_top = getattr(link, '_always_on_top', False)
                    pyrender_mesh = _mesh_from_trimesh(
                        mesh, smooth=False, always_on_top=always_on_top)
                    self.scene.remove_node(node)
                    node = self.scene.add(pyrender_mesh, pose=transform)
                    self._visual_mesh_map[link_id] = (node, link)
                    link._visual_mesh_changed = False
                else:
                    node.matrix = transform

            # update joint axis transforms
            for joint_id, (sphere_node, axis_node, joint) in self._joint_axis_map.items():
                # Update joint position and axis
                position = joint.world_position
                axis = joint.world_axis

                # Update sphere position
                sphere_transform = np.eye(4)
                sphere_transform[:3, 3] = position
                sphere_node.matrix = sphere_transform

                # Update axis cylinder position and orientation
                if axis_node is not None and axis is not None:
                    # Calculate rotation matrix to align cylinder with axis
                    # Default cylinder is along Z-axis, need to rotate to align with joint axis
                    z_axis = np.array([0, 0, 1])
                    axis_normalized = axis / np.linalg.norm(axis)

                    # Calculate rotation axis and angle
                    rotation_axis = np.cross(z_axis, axis_normalized)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)

                    if rotation_axis_norm > 1e-6:
                        rotation_axis = rotation_axis / rotation_axis_norm
                        angle = np.arccos(np.clip(np.dot(z_axis, axis_normalized), -1.0, 1.0))
                        # Create rotation matrix using Rodrigues' formula
                        K = np.array([
                            [0, -rotation_axis[2], rotation_axis[1]],
                            [rotation_axis[2], 0, -rotation_axis[0]],
                            [-rotation_axis[1], rotation_axis[0], 0]
                        ])
                        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    else:
                        # Axis is already aligned with z-axis or opposite
                        if np.dot(z_axis, axis_normalized) > 0:
                            rotation_matrix = np.eye(3)
                        else:
                            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])

                    axis_transform = np.eye(4)
                    axis_transform[:3, :3] = rotation_matrix
                    axis_transform[:3, 3] = position
                    axis_node.matrix = axis_transform

            super(PyrenderViewer, self).on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        pass

    def on_mouse_drag(self, *args, **kwargs):
        pass

    def on_mouse_scroll(self, *args, **kwargs):
        pass

    def on_key_press(self, symbol, modifiers, *args, **kwargs):
        """Handle key press events with collision toggle support."""
        pass

    def on_resize(self, *args, **kwargs):
        # Block rendering until initialization is complete
        pass

    def _add_link(self, link):
        assert isinstance(link, model_module.Link)

        with self._render_lock:
            transform = link.worldcoords().T()
            link_id = str(id(link))
            mesh = link.concatenated_visual_mesh

            if link_id not in self._visual_mesh_map and mesh:
                node = None
                if isinstance(mesh, trimesh.path.Path3D):
                    pyrender_mesh = pyrender.Mesh(
                        primitives=[pyrender.Primitive(
                            mesh.vertices[mesh.vertex_nodes].reshape(-1, 3),
                            mode=pyrender.constants.GLTF.LINE_STRIP,
                            color_0=mesh.colors)])
                    node = self.scene.add(pyrender_mesh)
                elif isinstance(mesh, trimesh.PointCloud):
                    pyrender_mesh = pyrender.Mesh(
                        primitives=[pyrender.Primitive(
                            mesh.vertices,
                            mode=pyrender.constants.GLTF.POINTS,
                            color_0=mesh.colors)])
                    node = self.scene.add(pyrender_mesh)
                else:
                    always_on_top = getattr(link, '_always_on_top', False)
                    pyrender_mesh = _mesh_from_trimesh(
                        mesh, smooth=False, always_on_top=always_on_top)
                    # Check if the mesh has vertices
                    # before adding it to the scene
                    if len(mesh.vertices) != 0:
                        node = self.scene.add(pyrender_mesh, pose=transform)
                # Add the node and link to the
                # visual mesh map only if the node is successfully created
                if node is not None:
                    self._visual_mesh_map[link_id] = (node, link)

        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry, always_on_top=False):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
            # Store robot for joint axis toggle
            if geometry not in self._stored_robots:
                self._stored_robots.append(geometry)
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        # Set always_on_top attribute on links if requested
        if always_on_top:
            for link in links:
                link._always_on_top = True

        # Store links for collision toggle if enabled
        if self.enable_collision_toggle:
            for link in links:
                if link not in self._stored_links:
                    self._stored_links.append(link)

        for link in links:
            self._add_link(link)

        self._redraw = True

    def delete(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        with self._render_lock:
            all_links = links
            while all_links:
                link = all_links[0]
                link_id = str(id(link))
                if link_id in self._visual_mesh_map:
                    self.scene.remove_node(self._visual_mesh_map[link_id][0])
                    self._visual_mesh_map.pop(link_id)
                all_links = all_links[1:]
                all_links.extend(link.child_links)
        self._redraw = True

    def add_joint_axis(self, joint, sphere_radius=0.01, axis_length=0.1,
                       axis_radius=0.003, axis_color=None):
        """Add joint axis visualization to the scene.

        Visualizes the joint position (world_position) as a sphere and
        the joint axis (world_axis) as a cylinder.

        Parameters
        ----------
        joint : Joint
            Joint object to visualize
        sphere_radius : float, optional
            Radius of the sphere representing the joint position.
            Default is 0.01.
        axis_length : float, optional
            Length of the cylinder representing the joint axis.
            Default is 0.1.
        axis_radius : float, optional
            Radius of the cylinder representing the joint axis.
            Default is 0.003.
        axis_color : array-like, optional
            RGBA color for the axis cylinder. Default is [1.0, 0.0, 0.0, 1.0] (red).

        Returns
        -------
        None

        Examples
        --------
        >>> from skrobot.viewers import PyrenderViewer
        >>> from skrobot.models import PR2
        >>> viewer = PyrenderViewer()
        >>> robot = PR2()
        >>> viewer.add(robot)
        >>> viewer.add_joint_axis(robot.r_shoulder_pan_joint)
        >>> viewer.show()
        """
        pass

    def delete_joint_axis(self, joint):
        """Delete joint axis visualization from the scene.

        Parameters
        ----------
        joint : Joint
            Joint object whose axis visualization should be deleted

        Returns
        -------
        None

        Examples
        --------
        >>> from skrobot.viewers import PyrenderViewer
        >>> from skrobot.models import PR2
        >>> viewer = PyrenderViewer()
        >>> robot = PR2()
        >>> viewer.add(robot)
        >>> viewer.add_joint_axis(robot.r_shoulder_pan_joint)
        >>> viewer.show()
        >>> viewer.delete_joint_axis(robot.r_shoulder_pan_joint)
        """
        pass

    def _toggle_joint_axes(self):
        """Toggle joint axes display for all stored robots."""
        if self.show_joint_axes:
            for robot in self._stored_robots:
                robot_id = str(id(robot))
                if robot_id not in self._kinematics_models:
                    skeleton = SkeletonModel(robot)
                    # Add skeleton links directly to scene
                    nodes = []
                    with self._render_lock:
                        for link in skeleton.link_list:
                            mesh = link.concatenated_visual_mesh
                            if mesh is None or len(mesh.vertices) == 0:
                                continue
                            transform = link.worldcoords().T()
                            pyrender_mesh = _mesh_from_trimesh(
                                mesh, smooth=False,
                                always_on_top=self.joint_axes_always_on_top)
                            node = self.scene.add(pyrender_mesh, pose=transform)
                            nodes.append((node, link))
                    self._kinematics_models[robot_id] = (skeleton, nodes)
        else:
            for robot in self._stored_robots:
                robot_id = str(id(robot))
                if robot_id in self._kinematics_models:
                    skeleton, nodes = self._kinematics_models.pop(robot_id)
                    with self._render_lock:
                        for node, link in nodes:
                            self.scene.remove_node(node)
                    skeleton.detach()
        self._redraw = True

    def set_camera(self, angles=None, distance=None, center=None,
                   resolution=None, fov=None, coords_or_transform=None):
        if angles is None and coords_or_transform is None:
            return
        if angles is not None:
            if fov is None:
                fov = np.array([60, 45])
            rotation = transformations.euler_matrix(*angles)
            pose = cameras.look_at(
                self.scene.bounds, fov=fov, rotation=rotation,
                distance=distance, center=center)
        else:
            if isinstance(coords_or_transform, Coordinates):
                pose = coords_or_transform.worldcoords().T()
        self._camera_node.matrix = pose
        self._trackball = Trackball(
            pose=pose,
            size=self.viewport_size,
            scale=self.scene.scale,
            target=self.scene.centroid
        )

    def capture_360_images(self, output_dir, num_frames=36,
                           distance=None, center=None, fov=None,
                           lighting_config=None, camera_elevation=45,
                           distance_margin=1.2, create_gif=True,
                           gif_duration=100, gif_loop=0,
                           transparent_background=True):
        """Capture 360-degree rotation images around the scene.

        Parameters
        ----------
        output_dir : str
            Directory to save the images
        num_frames : int
            Number of images to capture (default: 36, every 10 degrees)
        distance : float, optional
            Camera distance from center
        center : array-like, optional
            Center point to rotate around
        fov : array-like, optional
            Field of view [horizontal, vertical] in degrees
        lighting_config : dict, optional
            Lighting configuration with keys: 'positions', 'colors', 'intensity'
        camera_elevation : float
            Camera elevation angle in degrees (default: 45)
        distance_margin : float
            Margin factor for automatic distance calculation (default: 1.2)
        create_gif : bool
            Whether to create a GIF animation from captured images (default: True)
        gif_duration : int
            Duration between frames in milliseconds for GIF (default: 100)
        gif_loop : int
            Number of loops for GIF (0 = infinite loop, default: 0)
        transparent_background : bool
            Whether to render with transparent background (default: True)
        """
        pass

    def _create_gif_from_images(self, image_dir, output_gif, duration=100, loop=0):
        """Create GIF animation from captured images.

        Parameters
        ----------
        image_dir : Path
            Directory containing the images
        output_gif : Path
            Output path for the GIF file
        duration : int
            Duration between frames in milliseconds
        loop : int
            Number of loops (0 = infinite loop)
        """
        pass

    def _get_default_lighting_config(self):
        """Get default lighting configuration for uniform illumination."""
        pass

    def _setup_scene_lighting(self, lighting_config=None):
        """Setup scene lighting and return list of added light nodes."""
        pass

    def _cleanup_scene_lighting(self, light_nodes):
        """Remove lighting nodes from scene and reset ambient light."""
        pass

    def _update_scene_meshes(self):
        """Update scene meshes with latest transforms."""
        pass

    def _calculate_camera_distance(self, distance_margin=1.2):
        """Calculate optimal camera distance based on scene bounds."""
        bounds = self.scene.bounds
        bbox_diagonal = np.linalg.norm(bounds[1] - bounds[0])
        return bbox_diagonal * distance_margin

    def _rebuild_scene_for_toggle(self):
        """Completely rebuild the scene with current mesh type for toggle functionality."""
        pass

    def _add_single_link_mesh_for_toggle(self, link):
        """Add a single mesh (visual or collision) for a link during toggle."""
        pass
