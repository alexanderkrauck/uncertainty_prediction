"""
This file contains the implementation of the Kernel Mixture Network (KMN)

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

# Third-party libraries
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from typing import Tuple, Optional

# Local/Application Specific
from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
)
from .loss_functions import nlll, miscalibration_area_fn
from ..data_module import TrainingDataModule  # TODO: not ideal class dependencies here. Ideally would have some sort of models module class that contains the data dependencies. But this is not a priority right now


class GaussianKMN(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden: list = [32, 32],
        initial_kernel_scales: list = [0, 1.0, 2.0, 5.0],
        center_selection_method: str = "k_means",
        dropout_rate: float = 0.2,
        trainable_scales: bool = True,
        activation_function: str = "relu",
        **kwargs,
    ):
        """

        Parameters
        ----------
        train_data_module :
            The train data module used to train the model. This is used to get the mean and std of the data to normalize the input and output. Moreover we directly use the train data module to get the centers of the kernel mixture network.
        n_hidden : list
            List of hidden layer sizes.
        initial_kernel_scales : list
            List of initial scales to use for each kernel. The scales are expected w.r.t. a normalized domain. Also scales are required to be strictly greater than 0.
        center_selection_method : str
            Method of center selection to use. Can be "random" or "kmeans" TODO: add more center selection types
        dropout_rate : float, optional
            Dropout rate for the MLP, by default 0.2
        activation_function : str, optional
            Activation function to use for the MLP, by default "relu"
        """

        super().__init__()

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        self.mean_x, self.std_x = train_data_module.train_dataset.mean_x, train_data_module.train_dataset.std_x
        self.mean_y, self.std_y = train_data_module.train_dataset.mean_y, train_data_module.train_dataset.std_y

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        self.x_size = train_data_module.train_dataset.x.shape[1]
        self.y_size = train_data_module.train_dataset.y.shape[1]

        self.center_selection_method = center_selection_method

        #Because we use softplus later and thus we have to use inverse softplus here (should be fine as long as we get valid inputs)
        initial_kernel_scales = (torch.tensor(initial_kernel_scales).exp() - 1).log()
        self.scales = nn.Parameter(
            initial_kernel_scales, requires_grad=trainable_scales
        )
        self.cluster_centers = nn.Parameter(
            torch.from_numpy(
                GaussianKMN.sample_center_points(
                    train_data_module.train_dataset.y.numpy(), self.center_selection_method , **kwargs
                ).T
            ),
            requires_grad=False,
        )

        self.n_clusters = self.cluster_centers.shape[1]
        self.n_scales = self.scales.shape[0]

        self.mlp = MLP(
            n_input=self.x_size,
            n_hidden=n_hidden,
            n_output=self.n_clusters * self.n_scales,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
        )

    def forward(self, x, y=None, normalised_output_domain: bool = False):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor to the network.
        y : torch.Tensor
            Output tensor to the network.
        normalised_output_domain : bool
            Whether the output domain is normalised or not.

        Returns:
        --------
        torch.Tensor
            Output tensor of the network.
        """

        x = (x - self.mean_x) / self.std_x

        logits = self.mlp(x)
        weights = F.softmax(logits, dim=1)

        return_dict = {"weights": weights}
        return return_dict

    def eval_output(
        self,
        y,
        output,
        normalised_output_domain: bool = False,
        reduce="mean",
        miscalibration_area_loss_weight: float = 0.0,
        weights_entropy_loss_weight: float = 0.0,
        force_alternative_loss_calculations: bool = True,
        **kwargs,
    ):
        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y
        rescaled_params = self.get_rescaled_params(y.shape[0], normalised_output_domain)
        loss = nlll(
            torch.distributions.Normal,
            y,
            **output,
            **rescaled_params,
            reduce=reduce,
            **kwargs,
        )

        metric_dict = {}
        for key, value in output.items():
            val = value.detach().cpu().numpy()
            metric_dict["mean_" + key] = (
                np.mean(val) if reduce == "mean" else np.mean(val) * y.shape[0]
            )
            metric_dict["std_" + key] = (
                np.std(val) if reduce == "mean" else np.std(val) * y.shape[0]
            )

        if normalised_output_domain:
            metric_dict["nll_loss_normalized"] = loss.item()
            if reduce == "mean":
                metric_dict["nll_loss"] = (loss + torch.log(self.std_y).sum()).item()
            else:
                metric_dict["nll_loss"] = (
                    (loss + torch.log(self.std_y).sum() * y.shape[0]) 
                ).item()
        else:
            metric_dict["nll_loss"] = loss.item()
            if reduce == "mean":
                metric_dict["nll_loss_normalized"] = (loss - torch.log(self.std_y).sum()).item()
            else:
                metric_dict["nll_loss_normalized"] = (
                    (loss - torch.log(self.std_y).sum() * y.shape[0]) 
                ).item()

        if weights_entropy_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            weights_entropy = (
                torch.distributions.categorical.Categorical(probs=output["weights"])
                .entropy()
            )
            if reduce == "mean":
                weights_entropy = weights_entropy.mean()
            else:
                weights_entropy = weights_entropy.sum()
            metric_dict["weights_entropy"] = weights_entropy.item()

        if miscalibration_area_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            miscalibration_area = miscalibration_area_fn(
                torch.distributions.Normal,
                y,
                **output,
                **rescaled_params,
                reduce=reduce,
                **kwargs,
            )

            metric_dict["misclibration_area"] = miscalibration_area.item()

        if weights_entropy_loss_weight > 0:
            loss = (
                loss - weights_entropy_loss_weight * weights_entropy
            )  # because we want to regularize for high entropy

        if miscalibration_area_loss_weight > 0:
            loss = loss + miscalibration_area_loss_weight * miscalibration_area

        if (
            normalised_output_domain
        ):  # Because we don't want to add the loss if we are not in the normalised domain as it is not meaningful
            metric_dict["loss"] = loss.item()

        return loss, metric_dict

    def get_rescaled_params(self, batch_size, normalised_output_domain=False):
        sigma = F.softplus(self.scales).unsqueeze(0).repeat(self.y_size, 1)
        if normalised_output_domain:
            mu = (self.cluster_centers - self.mean_y.unsqueeze(-1)) / self.std_y.unsqueeze(-1)
        else:
            mu = self.cluster_centers
            sigma = sigma * self.std_y.unsqueeze(-1)

        mu = (
            mu.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, -1, self.n_scales)
            .reshape(batch_size, self.y_size, -1)
        )
        sigma = (
            sigma.reshape(1, 1, *sigma.shape)
            .transpose(1, 2)
            .expand(batch_size, self.y_size, self.n_clusters, -1)
            .reshape(batch_size, self.y_size, -1)
        )

        return {"mu": mu, "sigma": sigma}

    def get_density(
        self,
        x: Tensor,
        y: Tensor,
        normalised_output_domain: bool = False,
        numeric_stability: float = 1e-6,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        if weights is None:
            weights = self(x)["weights"]
        rescaled_params = self.get_rescaled_params(x.shape[0], normalised_output_domain)
        distribution = torch.distributions.Normal(
            rescaled_params["mu"], rescaled_params["sigma"] + numeric_stability
        )  # for numerical stability if predicted sigma is too close to 0

        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        densities = (
            torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        densities = densities.sum(dim=1)  # sum over multidimensional y
        density = torch.sum(densities * weights, dim=1)
        return density

    @staticmethod
    def sample_center_points(
        y,
        method="all",
        n_centers=100,
        keep_edges=True,
        parallelize=False,
        seed=42,
        **kwargs,
    ) -> np.ndarray:
        """function to define kernel centers with various downsampling alternatives

        Parameters
        ----------
        y: np.ndarray
            numpy array from which kernel centers shall be selected - shape (n_samples,) or (n_samples, n_dim)
        method: str
            kernel center selection method - choices: [all, random, distance, k_means, agglomerative]
        k: int
            number of centers to be returned (not relevant for 'all' method)
        random_state: np.random.RandomState
            numpy.RandomState object

        Returns
        -------
        np.ndarray
            selected center points - numpy array of shape (k, n_dim). In case method is 'all' k is equal to n_samples
        """
        assert n_centers <= y.shape[0], "k must not exceed the number of samples in Y"

        n_jobs = 1
        if parallelize:
            n_jobs = -2  # use all cpu's but one

        # make sure Y is 2d array of shape (
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        assert y.ndim == 2

        # keep all points as kernel centers
        if method == "all":
            return y

        # retain outer points to ensure expressiveness at the target borders
        if keep_edges:
            ndim_y = y.shape[1]
            n_edge_points = min(2 * ndim_y, n_centers // 2)

            # select 2*n_edge_points that are the farthest away from mean
            fathest_points_idx = np.argsort(np.linalg.norm(y - y.mean(axis=0), axis=1))[
                -2 * n_edge_points :
            ]
            Y_farthest = y[np.ix_(fathest_points_idx)]

            # choose points among Y farthest so that pairwise cosine similarity maximized
            dists = cosine_distances(Y_farthest)
            selected_indices = [0]
            for _ in range(1, n_edge_points):
                idx_greatest_distance = np.argsort(
                    np.min(
                        dists[np.ix_(range(Y_farthest.shape[0]), selected_indices)],
                        axis=1,
                    ),
                    axis=0,
                )[-1]
                selected_indices.append(idx_greatest_distance)
            centers_at_edges = Y_farthest[np.ix_(selected_indices)]

            # remove selected centers from Y
            indices_to_remove = fathest_points_idx[np.ix_(selected_indices)]
            y = np.delete(y, indices_to_remove, axis=0)

            # adjust k such that the final output has size k
            n_centers -= n_edge_points

        if method == "random":
            cluster_centers = y[
                np.choice(range(y.shape[0]), n_centers, replace=False, seed=seed)
            ]

        # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
        elif method == "distance":
            dists = euclidean_distances(y)
            selected_indices = [0]
            for _ in range(1, n_centers):
                idx_greatest_distance = np.argsort(
                    np.min(dists[np.ix_(range(y.shape[0]), selected_indices)], axis=1),
                    axis=0,
                )[-1]
                selected_indices.append(idx_greatest_distance)
            cluster_centers = y[np.ix_(selected_indices)]

        # use 1-D k-means clustering
        elif method == "k_means":
            model = KMeans(n_clusters=n_centers, random_state=seed, n_init="auto")
            model.fit(y)
            cluster_centers = model.cluster_centers_

        # use agglomerative clustering
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_centers, linkage="complete")
            model.fit(y)
            labels = pd.Series(model.labels_, name="label")
            y_s = pd.DataFrame(y)
            df = pd.concat([y_s, labels], axis=1)
            cluster_centers = df.groupby("label")[np.arange(y.shape[1])].mean().values

        else:
            raise ValueError("unknown method '{}'".format(method))

        if keep_edges:
            return np.concatenate([centers_at_edges, cluster_centers], axis=0)
        else:
            return cluster_centers
