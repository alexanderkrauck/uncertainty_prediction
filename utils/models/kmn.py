from .basic_architectures import MLP, ConditionalDensityEstimator, ACTIVATION_FUNCTION_MAP, DISTRIBUTION_MAP
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from .loss_functions import nlll, miscalibration_area_fn

from ..data_module import (
    TrainingDataModule,
)  # TODO: not ideal class dependencies here. Idealy would have some sort of models module class that contains the data dependencies. But this is not a priority right now


class GaussianKMN(ConditionalDensityEstimator):

    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden: list,
        initial_kernel_scales: list,
        center_selection_type: str,
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
            List of initial scales to use for each kernel.
        center_selection_type : str
            Type of center selection to use. Can be "random" or "kmeans" TODO: add more center selection types
        n_centers : int
            Number of centers to use.
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

        self.mean_x, self.std_x = train_data_module.train_dataset.scaler_x
        self.mean_y, self.std_y = train_data_module.train_dataset.scaler_y

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        n_inputs = train_data_module.train_dataset.x.shape[1]

        self.center_selection_type = center_selection_type

        self.scales = nn.Parameter(torch.tensor(initial_kernel_scales), requires_grad=trainable_scales)
        self.cluster_centers = nn.Parameter(torch.from_numpy(GaussianKMN.sample_center_points(train_data_module.train_dataset.y.numpy(), **kwargs)).squeeze(), requires_grad=False)

        self.n_clusters = self.cluster_centers.shape[0]
        self.n_scales = self.scales.shape[0]

        self.mlp = MLP(
            n_input=n_inputs,
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
        reduce="mean",
        normalised_output_domain: bool = False,
        miscalibration_area_loss_weight: float = 0.0,
        **kwargs,
    ):
        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y
        rescaled_params = self.get_rescaled_params(y.shape[0],normalised_output_domain)
        loss = nlll(torch.distributions.Normal,y, **output,**rescaled_params, reduce=reduce, **kwargs)

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
        else:
            metric_dict["nll_loss"] = loss.item()

        if miscalibration_area_loss_weight > 0:
            miscalibration_area = miscalibration_area_fn(
                torch.distributions.Normal,y, **output, **rescaled_params, reduce=reduce, **kwargs
            )
            loss = loss + miscalibration_area_loss_weight * miscalibration_area

            metric_dict["misclibration_area"] = miscalibration_area.item()

        if normalised_output_domain:# Because we don't want to add the loss if we are not in the normalised domain as it is not meaningful
            metric_dict["loss"] = loss.item()

        return loss, metric_dict
    
    def get_rescaled_params(self, batch_size, normalised_output_domain = False):
        sigma = F.softplus(self.scales)
        if normalised_output_domain:
            mu = (self.cluster_centers - self.mean_y) / self.std_y
        else:
            mu = self.cluster_centers
            sigma = sigma * self.std_y
        
        mu = mu.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.n_scales).reshape(batch_size, -1)
        sigma = sigma.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_clusters, -1).reshape(batch_size, -1)

        return {"mu": mu, "sigma":sigma}

    def get_density(
        self,
        x: Tensor,
        y: Tensor,
        normalised_output_domain: bool = False,
        numeric_stability: float = 1e-6,
        weights: Optional[Tensor] = None
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
            torch.exp(distribution.log_prob(y)) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        density = torch.sum(densities * weights, dim=1)
        return density
        
    @staticmethod
    def sample_center_points(y, method='all', n_centers=100, keep_edges=False, parallelize=False, seed=42, **kwargs) -> np.ndarray:
        """ function to define kernel centers with various downsampling alternatives

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
        if method == 'all':
            return y

        # retain outer points to ensure expressiveness at the target borders
        if keep_edges:
            ndim_y = y.shape[1]
            n_edge_points = min(2 * ndim_y, n_centers//2)

            # select 2*n_edge_points that are the farthest away from mean
            fathest_points_idx = np.argsort(np.linalg.norm(y - y.mean(axis=0), axis=1))[-2 * n_edge_points:]
            Y_farthest = y[np.ix_(fathest_points_idx)]

            # choose points among Y farthest so that pairwise cosine similarity maximized
            dists = cosine_distances(Y_farthest)
            selected_indices = [0]
            for _ in range(1, n_edge_points):
                idx_greatest_distance = \
                np.argsort(np.min(dists[np.ix_(range(Y_farthest.shape[0]), selected_indices)], axis=1), axis=0)[-1]
                selected_indices.append(idx_greatest_distance)
            centers_at_edges = Y_farthest[np.ix_(selected_indices)]

            # remove selected centers from Y
            indices_to_remove = fathest_points_idx[np.ix_(selected_indices)]
            y = np.delete(y, indices_to_remove, axis=0)

            # adjust k such that the final output has size k
            n_centers -= n_edge_points

        if method == 'random':
            cluster_centers = y[np.choice(range(y.shape[0]), n_centers, replace=False, seed=seed)]

        # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
        elif method == 'distance':
            dists = euclidean_distances(y)
            selected_indices = [0]
            for _ in range(1, n_centers):
                idx_greatest_distance = np.argsort(np.min(dists[np.ix_(range(y.shape[0]), selected_indices)], axis=1), axis=0)[-1]
                selected_indices.append(idx_greatest_distance)
            cluster_centers = y[np.ix_(selected_indices)]


        # use 1-D k-means clustering
        elif method == 'k_means':
            model = KMeans(n_clusters=n_centers, n_jobs=n_jobs, random_state=seed)
            model.fit(y)
            cluster_centers = model.cluster_centers_

        # use agglomerative clustering
        elif method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=n_centers, linkage='complete')
            model.fit(y)
            labels = pd.Series(model.labels_, name='label')
            y_s = pd.DataFrame(y)
            df = pd.concat([y_s, labels], axis=1)
            cluster_centers = df.groupby('label')[np.arange(y.shape[1])].mean().values

        else:
            raise ValueError("unknown method '{}'".format(method))

        if keep_edges:
            return np.concatenate([centers_at_edges, cluster_centers], axis=0)
        else:
            return cluster_centers


