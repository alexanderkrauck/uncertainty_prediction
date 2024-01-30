"""
This file contains the implementation of the Mixture Density Network (MDN) architecture.

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
from torch import nn, Tensor
import torch.nn.functional as F

# Typing
from typing import Tuple, Optional

# Local/Application Specific
from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
    DISTRIBUTION_MAP,
)
from .loss_functions import nlll, miscalibration_area_fn
from ..data_module import (
    TrainingDataModule,
)  # TODO: not ideal class dependencies here. Ideally would have some sort of models module class that contains the data dependencies. But this is not a priority right now


class MDN(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden: list = [32, 32],
        n_distributions: int = 10,
        dropout_rate: float = 0.2,
        activation_function: str = "relu",
        distribution_type: str = "gaussian",
        std_stability_mode: str = "softplus",
        tanh_std_stability: float = 3.0,
        **kwargs,
    ):
        """

        Parameters
        ----------
        train_data_module :
            The train data module used to train the model. This is used to get the mean and std of the data to normalize the input and output.
        n_hidden : list
            List of hidden layer sizes.
        n_distributions : int
            Number of distributions to use.
        dropout_rate : float, optional
            Dropout rate, by default 0.2
        activation_function : str, optional
            Activation function to use, by default "relu"
        distribution_type : str, optional
            Distribution type to use, by default "gaussian"
        std_stability_mode : str, optional
            Mode to use for std stability. Can be "none", "tanh" or "softplus", by default "none"
        tanh_std_stability : Optional[float]
            Tanh std stability value, by default 3.0. Only used if std_stability_mode is "tanh"
        """
        super().__init__()

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        self.mean_x, self.std_x = (
            train_data_module.train_dataset.mean_x,
            train_data_module.train_dataset.std_x,
        )
        self.mean_y, self.std_y = (
            train_data_module.train_dataset.mean_y,
            train_data_module.train_dataset.std_y,
        )

        self.tanh_std_stability = tanh_std_stability
        self.std_stability_mode = std_stability_mode.lower()
        if self.std_stability_mode not in ["none", "tanh", "softplus"]:
            raise ValueError(
                f"std_stability_mode must be one of 'none', 'tanh' or 'softplus', but is {self.std_stability_mode}"
            )
        if self.std_stability_mode == "tanh" and self.tanh_std_stability is None:
            raise ValueError(
                "tanh_std_stability must be set if std_stability_mode is tanh"
            )

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        self.x_size = train_data_module.train_dataset.x.shape[1]
        self.y_size = train_data_module.train_dataset.y.shape[1]

        distribution_type = distribution_type.lower()
        if distribution_type in DISTRIBUTION_MAP:
            self.distribution_class = DISTRIBUTION_MAP[distribution_type.lower()]
        else:
            raise ValueError(f"Distribution type {distribution_type} not supported.")
        self.n_distributions = n_distributions
        self.split_sizes = [
            self.n_distributions,
            self.n_distributions * self.y_size,
            self.n_distributions * self.y_size,
        ]
        self.mlp = MLP(
            self.x_size,
            n_hidden,
            (n_distributions * self.y_size) * 2 + n_distributions,
            dropout_rate,
            activation_function,
            **kwargs,
        )

    def forward(self, x, y=None, normalised_output_domain: bool = False):
        x = (x - self.mean_x) / self.std_x

        mlp_out = self.mlp(x)
        logits_weights, mu, log_sigma = torch.split(mlp_out, self.split_sizes, dim=1)

        if self.std_stability_mode == "tanh":
            log_sigma = (
                F.tanh(log_sigma) * self.tanh_std_stability
            )  # TODO !!!!!!!!!!!!!!! justify that. its intended to be a hack to prevent sigmas from getting too large
            sigma = torch.exp(log_sigma)
        elif self.std_stability_mode == "softplus":
            sigma = F.softplus(log_sigma)
        else:
            sigma = torch.exp(
                log_sigma
            )  # this can lead to instability easily as sigmas can get extremely large

        weights = F.softmax(logits_weights, dim=1)

        mu = mu.reshape(-1, self.y_size, self.n_distributions)
        sigma = sigma.reshape(-1, self.y_size, self.n_distributions)

        if not normalised_output_domain:
            mu = self.std_y.unsqueeze(-1) * mu + self.mean_y.unsqueeze(-1)
            sigma = self.std_y.unsqueeze(-1) * sigma

        return_dict = {"weights": weights, "mu": mu, "sigma": sigma}
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

        loss = nlll(self.distribution_class, y, **output, reduce=reduce, **kwargs)

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
            metric_dict["nll_loss"] = (loss + torch.log(self.std_y).sum()).item()
        else:
            metric_dict["nll_loss"] = loss.item()

        if weights_entropy_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            weights_entropy = (
                torch.distributions.categorical.Categorical(probs=output["weights"])
                .entropy()
                .mean()
            )
            metric_dict["weights_entropy"] = weights_entropy.item()

        if miscalibration_area_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            miscalibration_area = miscalibration_area_fn(
                self.distribution_class, y, **output, reduce=reduce, **kwargs
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

    def get_density(
        self,
        x: Tensor,
        y: Tensor,
        normalised_output_domain: bool = False,
        numeric_stability: float = 1e-8,
        weights: Optional[Tensor] = None,
        mu: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
    ) -> Tensor:
        if weights is None or mu is None or sigma is None:
            output = self(x, y, normalised_output_domain=normalised_output_domain)
            weights, mu, sigma = output["weights"], output["mu"], output["sigma"]
        distribution = self.distribution_class(
            mu, sigma + numeric_stability
        )  # for numerical stability if predicted sigma is too close to 0

        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        densities = (
            torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        densities = densities.sum(1)
        density = torch.sum(densities * weights, dim=1)

        # if not normalised_output_domain:
        #    density = density * (1 / self.std_y).prod() #(we multiply by the jacobian determinant of the transformation from the normalised to the unnormalised domain)

        return density
