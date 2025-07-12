#!/usr/bin/env python3
"""
Mesh processing utilities for Genesis environment.
Handles convex decomposition and mesh preprocessing.
"""

import os

import trimesh


def decompress_mesh_if_needed(mesh_path: str) -> str:
    """
    Decompress a GLB mesh if it's compressed, similar to texture decompression.

    Args:
        mesh_path: Path to the original mesh file

    Returns:
        Path to the decompressed mesh file
    """
    if not mesh_path.endswith(".glb"):
        return mesh_path

    decompressed_path = f"{mesh_path}.decompressed.glb"

    if not os.path.exists(decompressed_path):
        print(f"Decompressing mesh: {mesh_path}")
        # Decompress the mesh using gltf-transform
        # https://github.com/3dlg-hcvc/hssd/issues/25#issuecomment-3004660306
        # https://github.com/Genesis-Embodied-AI/Genesis/issues/955
        os.system(f"gltf-transform ktxdecompress {mesh_path} {decompressed_path}")

    return decompressed_path


def create_convex_decomposition(
    mesh_path: str, max_hulls: int = 100, resolution: int = 100000
) -> str:
    """
    Create convex decomposition of a mesh using trimesh.

    Args:
        mesh_path: Path to the input mesh file
        max_hulls: Maximum number of convex hulls to create
        resolution: Resolution for convex decomposition

    Returns:
        Path to the decomposed mesh file
    """
    # Create output path for convex decomposition
    base_name = os.path.splitext(mesh_path)[0]
    convex_path = f"{base_name}.convex.glb"

    if os.path.exists(convex_path):
        print(f"Convex decomposition already exists: {convex_path}")
        return convex_path

    print(f"Creating convex decomposition: {mesh_path}")

    # Load the mesh
    mesh = trimesh.load_mesh(mesh_path)

    # Create convex decomposition
    convex_meshes = trimesh.decomposition.convex_decomposition(
        mesh, maxConvexHulls=max_hulls, resolution=resolution
    )

    # Create a scene with all convex meshes
    scene = trimesh.Scene()

    for i, convex_mesh_dict in enumerate(convex_meshes):
        # Convert dictionary to trimesh object
        convex_mesh = trimesh.Trimesh(
            vertices=convex_mesh_dict["vertices"], faces=convex_mesh_dict["faces"]
        )
        scene.add_geometry(convex_mesh, node_name=f"convex_hull_{i}")

    # Export the convex decomposition
    scene.export(convex_path)
    print(f"Convex decomposition saved to: {convex_path}")

    return convex_path


def process_mesh_for_physics(
    mesh_path: str, max_hulls: int = 100, resolution: int = 100000
) -> str:
    """
    Process a mesh for physics simulation, including convex decomposition if needed.

    Args:
        mesh_path: Path to the input mesh file
        max_hulls: Maximum number of convex hulls to create
        resolution: Resolution for convex decomposition

    Returns:
        Path to the processed mesh file ready for physics
    """
    if not mesh_path.endswith(".glb"):
        return mesh_path

    # First decompress if needed
    decompressed_path = decompress_mesh_if_needed(mesh_path)

    return decompressed_path
