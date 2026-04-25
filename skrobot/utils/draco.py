"""Draco compression utilities using DracoPy with color support.

This module provides Draco mesh compression with vertex color preservation,
"""

import numpy as np


def _draco_encode_with_colors(ctx):
    """
    Handle KHR_draco_mesh_compression extension for encoding a mesh primitive.

    This handler is called by trimesh *after* it has already built the
    primitive's accessors and buffer data.  We re-encode the mesh with
    DracoPy, append the compressed buffer to ``buffer_items``, and add
    the Draco extension to the primitive — reusing the accessor indices
    that trimesh already created.

    Parameters
    ----------
    ctx : dict
        PrimitiveExportContext with:
        - mesh: trimesh.Trimesh being exported
        - tree: glTF tree being built (mutable)
        - buffer_items: OrderedDict of buffer data (mutable)
        - primitive: Primitive dict being built (mutable)
        - include_normals: Whether to include normals

    Returns
    -------
    result : dict or None
        Dict with extension data, or None on failure.
    """
    pass


def _draco_decode_with_colors(ctx):
    """
    Handle KHR_draco_mesh_compression extension for decoding a glTF primitive.

    This decoder uses DracoPy which supports vertex colors.

    Parameters
    ----------
    ctx : dict
        PrimitivePreprocessContext with:
        - data: The KHR_draco_mesh_compression extension data
        - views: List of buffer views from the glTF
        - accessors: List of accessors (mutable, will be appended to)
        - primitive: The primitive dict (mutable, indices/attributes will be updated)

    Returns
    -------
    result : dict or None
        Dict with {"decompressed": True}, or None on failure.
    """
    pass


def register_dracopy_handlers():
    """Register DracoPy-based handlers with trimesh's glTF extension system.

    This registers DracoPy-based handlers that support vertex colors.
    Requires trimesh >= 4.11 with the extension registration API.
    """
    try:
        from trimesh.exchange.gltf.extensions import register_handler

        # Register decode handler for import
        register_handler("KHR_draco_mesh_compression", scope="primitive_preprocess")(
            _draco_decode_with_colors
        )

        # Register encode handler for export
        register_handler("KHR_draco_mesh_compression", scope="primitive_export")(
            _draco_encode_with_colors
        )

        return True
    except ImportError:
        return False


def is_dracopy_available():
    """Check if DracoPy is installed and compatible."""
    try:
        import DracoPy  # NOQA
        return True
    except (ImportError, ValueError):
        # ValueError can occur due to numpy ABI incompatibility
        return False


def export_glb_with_draco(meshes, filename):
    """Export meshes to GLB file with Draco compression preserving colors.

    This function creates a GLB file with KHR_draco_mesh_compression extension,
    using DracoPy for encoding which supports vertex colors.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
        Meshes to export.
    filename : str
        Output filename (should end with .glb).
    """
    import json
    import struct

    import DracoPy

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    # Build glTF structure
    gltf = {
        "asset": {"version": "2.0", "generator": "scikit-robot with DracoPy"},
        "extensionsUsed": ["KHR_draco_mesh_compression"],
        "extensionsRequired": ["KHR_draco_mesh_compression"],
        "bufferViews": [],
        "accessors": [],
        "meshes": [],
        "nodes": [],
        "scenes": [{"nodes": []}],
        "scene": 0
    }

    buffer_data = b""

    for mesh_idx, mesh in enumerate(meshes):
        # Get mesh data with colors
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        colors = None

        # Extract colors based on visual type
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            if mesh.visual.kind == 'face':
                # Convert face colors to vertex colors
                face_colors = np.asarray(mesh.visual.face_colors)[:, :3]
                vertices = vertices[faces].reshape(-1, 3).astype(np.float32)
                colors = np.repeat(face_colors, 3, axis=0).astype(np.uint8)
                faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
            elif mesh.visual.kind == 'vertex':
                vc = mesh.visual.vertex_colors
                if vc is not None:
                    vc = np.atleast_2d(np.asarray(vc))
                    if vc.shape == (1, 4) or (vc.ndim == 1 and len(vc) <= 4):
                        # Uniform color, tile to all vertices
                        c = vc.flatten()[:3].astype(np.uint8)
                        colors = np.tile(c, (len(vertices), 1))
                    elif len(vc) == len(vertices):
                        colors = vc[:, :3].astype(np.uint8)
            elif mesh.visual.kind == 'texture':
                # Some DAE files have no texture image but store
                # per-material diffuse colors.  trimesh reports these
                # as kind='texture' with image=None.  Converting to
                # vertex colors via to_color() recovers the colors.
                mat = getattr(mesh.visual, 'material', None)
                has_image = (
                    mat is not None
                    and hasattr(mat, 'image')
                    and mat.image is not None
                )
                if not has_image:
                    try:
                        color_visual = mesh.visual.to_color()
                        if hasattr(color_visual, 'vertex_colors'):
                            vc = np.atleast_2d(
                                np.asarray(color_visual.vertex_colors))
                            if vc.shape == (1, 4) or vc.ndim == 1:
                                # Uniform color, tile to all vertices
                                c = vc.flatten()[:3].astype(np.uint8)
                                colors = np.tile(c, (len(vertices), 1))
                            elif len(vc) == len(vertices):
                                colors = vc[:, :3].astype(np.uint8)
                    except Exception:
                        pass

            if colors is None and hasattr(mesh.visual, 'main_color'):
                main_color = mesh.visual.main_color
                if main_color is not None:
                    mc = np.atleast_1d(
                        np.asarray(main_color, dtype=np.uint8))
                    if len(mc) >= 3:
                        colors = np.tile(mc[:3], (len(vertices), 1))

        # Encode with DracoPy
        # DracoPy.encode uses 'points' not 'vertices'
        # Colors should be shape (N, K) not flattened
        # Use high quantization_bits (14) to preserve position precision
        # and set quantization_range based on actual mesh bounds
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_range = (v_max - v_min).max()
        if v_range < 1e-6:
            v_range = 1.0  # Avoid division by zero for degenerate meshes

        draco_buffer = DracoPy.encode(
            points=vertices,
            faces=faces.flatten(),
            colors=colors if colors is not None else None,
            quantization_bits=14,
            quantization_range=float(v_range),
            quantization_origin=v_min.tolist()
        )

        # Pad to 4-byte alignment
        padding = (4 - len(draco_buffer) % 4) % 4
        if padding > 0:
            draco_buffer = draco_buffer + b'\x00' * padding

        # Add buffer view for Draco data
        buffer_view_index = len(gltf["bufferViews"])
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": len(buffer_data),
            "byteLength": len(draco_buffer)
        })
        buffer_data += draco_buffer

        # Add accessors for mesh data (required by glTF spec even with Draco)
        # These accessors don't have bufferView since data is in Draco buffer
        n_vertices = len(vertices)
        n_faces = len(faces)

        # Position accessor
        position_accessor_index = len(gltf["accessors"])
        gltf["accessors"].append({
            "componentType": 5126,  # FLOAT
            "count": n_vertices,
            "type": "VEC3",
            "max": vertices.max(axis=0).tolist(),
            "min": vertices.min(axis=0).tolist()
        })

        # Indices accessor
        indices_accessor_index = len(gltf["accessors"])
        gltf["accessors"].append({
            "componentType": 5125,  # UNSIGNED_INT
            "count": n_faces * 3,
            "type": "SCALAR",
            "max": [int(faces.max())],
            "min": [int(faces.min())]
        })

        # Build primitive attributes and Draco extension
        # Note: DracoPy assigns unique_id based on the order attributes are added:
        # - colors (if present) gets unique_id=0
        # - points (position) gets unique_id=1 when colors present, 0 when no colors
        primitive_attributes = {"POSITION": position_accessor_index}

        if colors is not None:
            color_accessor_index = len(gltf["accessors"])
            # Normalize colors to 0-1 range for accessor metadata
            colors.astype(np.float32) / 255.0
            gltf["accessors"].append({
                "componentType": 5121,  # UNSIGNED_BYTE
                "count": n_vertices,
                "type": "VEC3",
                "normalized": True,
                "max": [1.0, 1.0, 1.0],
                "min": [0.0, 0.0, 0.0]
            })
            primitive_attributes["COLOR_0"] = color_accessor_index
            # DracoPy puts colors at unique_id=0 and position at unique_id=1
            draco_attributes = {"POSITION": 1, "COLOR_0": 0}
        else:
            # Without colors, position is at unique_id=0
            draco_attributes = {"POSITION": 0}

        # Add PBR material so renderers that don't support vertex colors
        # (like Genesis) still show correct colors via baseColorFactor.
        material_index = None
        if colors is not None:
            mean_color = colors.mean(axis=0).astype(float) / 255.0
            if "materials" not in gltf:
                gltf["materials"] = []
            material_index = len(gltf["materials"])
            gltf["materials"].append({
                "pbrMetallicRoughness": {
                    "baseColorFactor": [
                        float(mean_color[0]),
                        float(mean_color[1]),
                        float(mean_color[2]),
                        1.0,
                    ],
                    "roughnessFactor": 0.9,
                    "metallicFactor": 0.0,
                },
            })
        elif hasattr(mesh, 'visual') and mesh.visual is not None:
            mat = getattr(mesh.visual, 'material', None)
            if mat is not None:
                bcf = getattr(mat, 'baseColorFactor', None)
                if bcf is not None:
                    bcf = np.asarray(bcf, dtype=float)
                    if bcf.max() > 1.0:
                        bcf = bcf / 255.0
                    if "materials" not in gltf:
                        gltf["materials"] = []
                    material_index = len(gltf["materials"])
                    gltf["materials"].append({
                        "pbrMetallicRoughness": {
                            "baseColorFactor": bcf[:4].tolist(),
                            "roughnessFactor": 0.9,
                            "metallicFactor": 0.0,
                        },
                    })

        # Create mesh primitive
        primitive = {
            "attributes": primitive_attributes,
            "indices": indices_accessor_index,
            "extensions": {
                "KHR_draco_mesh_compression": {
                    "bufferView": buffer_view_index,
                    "attributes": draco_attributes
                }
            }
        }
        if material_index is not None:
            primitive["material"] = material_index

        # Add mesh to glTF
        gltf["meshes"].append({
            "primitives": [primitive]
        })

        # Add node
        node_index = len(gltf["nodes"])
        gltf["nodes"].append({
            "mesh": mesh_idx
        })
        gltf["scenes"][0]["nodes"].append(node_index)

    # Add buffer
    gltf["buffers"] = [{
        "byteLength": len(buffer_data)
    }]

    # Create GLB file
    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')

    # Pad JSON to 4-byte alignment
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_padding

    # GLB header
    glb_header = struct.pack('<4sII', b'glTF', 2,
                             12 + 8 + len(json_bytes) + 8 + len(buffer_data))

    # JSON chunk
    json_chunk_header = struct.pack('<II', len(json_bytes), 0x4E4F534A)  # JSON

    # BIN chunk
    bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # BIN

    # Write GLB file
    with open(filename, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk_header)
        f.write(json_bytes)
        f.write(bin_chunk_header)
        f.write(buffer_data)
