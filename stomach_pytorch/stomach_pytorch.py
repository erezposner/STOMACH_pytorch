
import pickle
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import torch


class DPCA:
    def __init__(
            self, num_comp: int = None, device: torch.device = torch.device("cpu")
    ) -> None:
        self.num_comp = num_comp
        self.device = torch.device(device)
        self.avg_mesh_vertices = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.explained_variance_ratio_ = None

    @property
    def components_std(self):
        return None if self.eigen_values is None else torch.sqrt(self.eigen_values)

    def preprocess(self, vertices: torch.tensor) -> np.ndarray:
        vertices = vertices.cpu().numpy()
        flatten_vertices = self.flatten_data(vertices, vertices.shape[0])
        normalized_vertices = self.normalize_vertices(flatten_vertices)
        mean = np.mean(flatten_vertices, axis=0).reshape(-1, 3)
        self.update_vertices_mean(mean)
        return normalized_vertices

    def visualize_components_variance_explained(self):
        if self.explained_variance_ratio_ is not None:
            plt.figure(figsize=(4, 2), dpi=150)
            plt.grid()
            # Find the index where the cumulative explained variance ratio exceeds 0.95
            cumulative_var = np.cumsum(self.explained_variance_ratio_)
            index_095 = np.argmax(cumulative_var >= 0.95)

            # Plot horizontal line at y=0.95
            plt.axhline(
                y=0.95, color="k", linestyle="--", label="0.95 Explained Variance"
            )

            # Plot vertical line at corresponding x-value
            plt.axvline(
                x=index_095, color="k", linestyle="--", label=f"PCs = {index_095}"
            )

            plt.plot(
                np.arange(len(self.explained_variance_ratio_)), cumulative_var, c="g"
            )

            plt.xlabel("Principal Components")
            plt.ylabel("Cumulative Explained Variance Ratio")
            plt.title("Principal Components vs. Explained Variance Ratio")



    def flatten_data(self, vertices: np.array, number_of_observations: int):
        return np.reshape(vertices, (number_of_observations, -1))

    def normalize_vertices(self, flatten_vertices: np.array):
        scaler = preprocessing.StandardScaler(with_std=False).fit(flatten_vertices)
        normalized_vertices = scaler.transform(flatten_vertices)
        return normalized_vertices


    def __eigens_to_torch(self):
        self.eigen_values = torch.tensor(
            self.eigen_values[: self.num_comp], requires_grad=False, device=self.device
        )
        self.eigen_vectors = torch.tensor(
            self.eigen_vectors[:, : self.num_comp],
            requires_grad=False,
            device=self.device,
        )
        self.avg_mesh_vertices = torch.tensor(
            self.avg_mesh_vertices, requires_grad=False, device=self.device
        )

    def update_vertices_mean(self, vertices_mean: np.array):
        self.avg_mesh_vertices = vertices_mean




    def load_metadata(self, filename: str):
        with open(filename, "rb") as handle:
            loaded_data = pickle.load(handle)
        self.avg_mesh_vertices, self.eigen_values, self.eigen_vectors, num_comp = (
            loaded_data
        )
        if self.num_comp is not None and self.num_comp > num_comp:
            raise AssertionError(
                f"The num_comp inserted ({self.num_comp}),should be between 1 and the number of loaded"
                f" components ({num_comp})"
            )
        self.num_comp = num_comp if self.num_comp is None else self.num_comp
        self.__eigens_to_torch()

    def betas_to_principal_components(self, betas: torch.Tensor) -> torch.Tensor:
        self.verify_metadata_loaded()
        self.verify_valid_vector_length(betas)
        return self.components_std[: len(betas)] * betas

    def principal_components_to_betas(
            self, principal_components: torch.Tensor
    ) -> torch.Tensor:
        self.verify_metadata_loaded()
        self.verify_valid_vector_length(principal_components)
        components_std = self.components_std[: len(principal_components)]
        betas = torch.where(
            components_std > 1e-8, principal_components / components_std, 0.0
        )
        return betas

    def project_mesh(self, mesh_vertices: torch.Tensor) -> torch.Tensor:
        """
        input:
        mesh: Tensor of shape (N,3) where N is the number of vertices. Represents the vertices of the 3D mesh.
        output:
        betas : Tensor of shape (num_components,). Represents how many standard deviations each PCA component
        will deviate from its mean.
        """
        self.verify_metadata_loaded()
        assert (
                mesh_vertices.shape == self.avg_mesh_vertices.shape
        ), "given mesh and data's average mesh have different shapes."
        mesh_centered = mesh_vertices.flatten() - self.avg_mesh_vertices.flatten()
        mesh_projected = torch.matmul(self.eigen_vectors.T, mesh_centered)
        betas = self.principal_components_to_betas(principal_components=mesh_projected)
        return betas

    def generate_new_mesh(self, betas: torch.Tensor) -> torch.Tensor:
        self.verify_metadata_loaded()
        self.verify_valid_vector_length(betas)
        len_betas = len(betas)
        new_mesh_vertices = self.avg_mesh_vertices.flatten() + torch.matmul(
            betas * self.eigen_vectors[:, :len_betas], self.components_std[:len_betas]
        )
        new_mesh_vertices = torch.reshape(new_mesh_vertices, (-1, 3))
        return new_mesh_vertices

    def verify_metadata_loaded(self):
        assert all(
            emb is not None
            for emb in [
                self.eigen_values,
                self.eigen_vectors,
                self.avg_mesh_vertices,
                self.num_comp,
            ]
        ), (
            "Operation cannnot be performed since embeddings are not loaded."
            " Please fit the PCA or load given embeddings before calling project_mesh"
        )

    def verify_valid_vector_length(self, vector: Union[np.ndarray, torch.Tensor]):
        assert len(vector.shape) == 1, "the input must be a one dimensional vector "
        len_vector = len(vector)
        if len_vector > self.num_comp:
            raise AssertionError(
                f"The length the input is {len_vector},"
                f" but it must contain between 1 to num_comp={self.num_comp} elements"
            )




def delete_meshes_with_anomal_triangles(meshes_list: np.ndarray):
    meshes_triangles = np.array([np.array(mesh.triangles) for mesh in meshes_list])
    anomal_triangles_meshes = np.unique(
        np.where(meshes_triangles - meshes_triangles[0] != 0)[0]
    )
    meshes_list = np.delete(meshes_list, anomal_triangles_meshes)
    return meshes_list
