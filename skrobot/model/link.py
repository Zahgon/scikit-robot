try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np

import skrobot
from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates import CascadedCoords


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 collision_mesh=None,
                 visual_mesh=None,
                 mass=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self.centroid = centroid
        self.joint = None
        self._child_links = []
        self._parent_link = None
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)
        self.inertia_tensor = inertia_tensor
        self._collision_mesh = collision_mesh
        self.visual_mesh = visual_mesh
        if visual_mesh is not None:
            trimesh = _lazy_trimesh()
            self._concatenated_visual_mesh = trimesh.util.concatenate(
                self._visual_mesh)
        else:
            self._concatenated_visual_mesh = None
        self._visual_mesh_changed = False
        self._sdf = None

        # Dynamics properties
        self.mass = mass if mass is not None else 1.0  # kg

    @property
    def parent_link(self):
        pass

    @property
    def child_links(self):
        pass

    def add_joint(self, j):
        self.joint = j

    def delete_joint(self):
        pass

    def add_child_link(self, child_link):
        """Add child link."""
        if child_link is not None and child_link not in self._child_links:
            self._child_links.append(child_link)

    def del_child_link(self, link):
        self._child_links.remove(link)

    def add_parent_link(self, parent_link):
        self._parent_link = parent_link

    def del_parent_link(self):
        pass

    @property
    def collision_mesh(self):
        """Return collision mesh

        Returns
        -------
        self._collision_mesh : trimesh.base.Trimesh
            A single collision mesh for the link.
            specified in the link frame,
            or None if there is not one.
        """
        pass

    @collision_mesh.setter
    def collision_mesh(self, mesh):
        """Setter of collision mesh

        Parameters
        ----------
        mesh : trimesh.base.Trimesh
            A single collision mesh for the link.
            specified in the link frame,
            or None if there is not one.
        """
        pass

    @property
    def visual_mesh(self):
        """Return visual mesh

        Returns
        -------
        self._visual_mesh : None, trimesh.base.Trimesh, or
                            sequence of trimesh.Trimesh
            A set of visual meshes for the link in the link frame.
        """
        pass

    @visual_mesh.setter
    def visual_mesh(self, mesh):
        """Setter of visual mesh

        Parameters
        ----------
        mesh : None, trimesh.Trimesh, sequence of trimesh.Trimesh,
               trimesh.points.PointCloud or str
            A set of visual meshes for the link in the link frame.
        """
        pass

    @property
    def visual_mesh_file_paths(self):
        """Return file paths of visual meshes.

        Returns a list of file paths extracted from trimesh metadata.
        Only available when meshes were loaded from URDF files.

        Returns
        -------
        file_paths : list of str
            File paths of the visual meshes.
        """
        pass

    @property
    def concatenated_visual_mesh(self):
        """Concatenated visual mesh for visualization.

        Returns
        -------
        self._concatenated_visual_mesh : None, trimesh.base.Trimesh
            A concatenated visual meshes for the link in the link frame.
        """
        pass

    @property
    def visual_mesh_changed(self):
        pass

    @property
    def colors(self):
        pass

    def set_color(self, color):
        pass

    def reset_color(self):
        pass

    def set_alpha(self, alpha):
        """Set alpha (transparency) value for visual mesh.

        Parameters
        ----------
        alpha : float
            Alpha value between 0.0 (transparent) and 1.0 (opaque)
        """
        pass

    @property
    def sdf(self):
        """Return signed distance function.

        Returns
        -------
        self._sdf : None or skrobot.sdf.SignedDistanceFunction
            signed distance function.
        """
        return self._sdf

    @sdf.setter
    def sdf(self, sdf):
        """Setter of sdf.

        Parameters
        ----------
        sdf : skrobot.sdf.SignedDistanceFunction
            signed distance function.
        """
        if not isinstance(sdf, skrobot.sdf.SignedDistanceFunction):
            raise TypeError('sdf must be skrobot.sdf.SignedDistanceFunction.'
                            ' but is {}'.format(type(sdf)))
        self._sdf = sdf


def _find_link_path(src_link, target_link, previous_link=None,
                    include_target=False):
    if src_link == target_link:
        if include_target:
            return [target_link], True
        else:
            return [], True
    paths = []
    links = []
    if hasattr(src_link, 'parent'):
        links = [src_link.parent]
    if hasattr(src_link, '_descendants'):
        links.extend(src_link._descendants)
    for next_link in links:
        if next_link is None or next_link == previous_link:
            continue
        path, succ = _find_link_path(next_link, target_link, src_link)
        if succ is True:
            paths.append(next_link)
            paths.extend(path)
            return paths, True
    return [], False


def find_link_path(src_link, target_link, include_source=True,
                   include_target=True):
    """Find paths of src_link to target_link

    Parameters
    ----------
    src_link : skrobot.model.link.Link
        source link.
    target_link : skrobot.model.link.Link
        target link.
    include_source : bool
        If `True`, return link list includes `src_link`.
    include_target : bool
        If `True`, return link list includes `target_link`.

    Returns
    -------
    ret : tuple(List[skrobot.model.link.Link], bool)
        If the links are connected, return Link list and `True`.
        Otherwise, return an empty list and `False`.
    """
    paths, succ = _find_link_path(src_link, target_link,
                                  include_target=include_target)
    if succ:
        if include_source:
            return [src_link] + paths, succ
        else:
            return paths, succ
    else:
        return [], False
