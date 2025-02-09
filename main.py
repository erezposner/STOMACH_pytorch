import os

import argparse

import open3d as o3d
import numpy as np

import torch

from stomach_pytorch.stomach_pytorch import DPCA


def create_o3d_mesh(new_mesh_vertices: torch.Tensor, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Create an Open3D TriangleMesh from vertices and faces.

    Args:
        new_mesh_vertices (torch.Tensor): Tensor of shape (N, 3) representing vertex coordinates.
        faces (np.ndarray): Array of shape (M, 3) representing the indices of vertices forming each triangle face.

    Returns:
        o3d.geometry.TriangleMesh: Open3D mesh object.
    """
    # Convert vertices from torch.Tensor to numpy array
    vertices_np = new_mesh_vertices.detach().cpu().numpy()

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute vertex normals
    mesh.compute_vertex_normals()

    return mesh


def DPCA_main(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_comp = args.n_comp

    embeddings_file_path = args.embeddings_file
    faces_file_path = args.faces_file

    pca = DPCA(num_comp=n_comp, device=device)
    faces = np.load(faces_file_path)
    assert os.path.exists(embeddings_file_path), "PCA embeddings file is not found."
    pca.load_metadata(embeddings_file_path)  # Load embeddings
    # Generate and visualize new point clouds
    pcl = []
    num_comp = n_comp
    samples = args.n_samples_to_generate
    min_beta, max_beta = torch.tensor(-1), torch.tensor(1)
    for i in range(samples):
        torch.manual_seed(i)
        betas_leaf = torch.rand(num_comp, requires_grad=True, device=device)
        betas = (max_beta - min_beta) * betas_leaf + min_beta
        new_mesh_vertices = pca.generate_new_mesh(betas=betas)
        new_mesh_vertices.retain_grad()
        betas_recon = pca.project_mesh(new_mesh_vertices)
        betas_recon.mean().backward()
        new_mesh = create_o3d_mesh(new_mesh_vertices, faces)

        new_mesh.paint_uniform_color([1 - i / samples, i / samples, 0])
        pcl.append(new_mesh)
        o3d.visualization.draw_geometries([new_mesh])
    o3d.visualization.draw_geometries(pcl)
    print("Done")


def DPCA_main_default_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_folder",
        type=str,
        default="output/SMOL",
        help="Output folder for results",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="stomach_pytorch/embeddings/embeddings_.pkl",
        help="Path to the file for loading or saving embeddings."
             " if None, it will be join(output_folder, name(data_path)",
    )
    parser.add_argument(
        "--faces_file",
        type=str,
        default="stomach_pytorch/embeddings/faces.npy",
        help="Path to the file for loading the faces ."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., cuda:0 or cpu)",
    )

    parser.add_argument(
        "--n_comp", type=int, default=10, help="Number of components for PCA, maximum is 100"
    )

    parser.add_argument(
        "--n_samples_to_generate",
        type=int,
        default=5,
        help="Number of meshes to generate",
    )

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = DPCA_main_default_arguments()
    print(args)
    DPCA_main(args)
